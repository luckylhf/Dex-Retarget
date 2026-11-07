"""CPU admission, process memory accounting and crash-safe compilation leases."""
import contextlib
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

GIB=1024**3
MIB=1024**2

def dynamic_mode(args):return getattr(args,'resource_mode','manual')=='dynamic'

def lower_priority():
    try:
        os.nice(max(0,10-os.nice(0)))
        return True
    except (AttributeError,OSError):
        return False  # e.g. restricted macOS hosts; CPU/memory gates still apply

def host_reserve(args,snapshot):
    explicit=getattr(args,'memory_reserve_gib',None)
    if dynamic_mode(args):
        target=getattr(args,'memory_target_percent',90.)
        return max(math.ceil(snapshot['total']*(100.-target)/100.),int((explicit or 0)*GIB))
    return int((10. if explicit is None else explicit)*GIB)

def state_root():
    return Path(os.environ.get('DEX_FRAX_STATE_DIR',Path.home()/'.cache/dex-frax')).expanduser().resolve()

def cpu_count():
    count=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else (os.cpu_count() or 1)
    if sys.platform=='linux':
        for folder in cgroup_ancestors():
            p=folder/'cpu.max'
            if p.exists():
                quota,period=p.read_text().split()
                if quota!='max':count=min(count,max(1,math.ceil(int(quota)/int(period))))
    return count

def workers(value):
    usable=cpu_count();n=usable if str(value)=='auto' else int(value)
    if not 1<=n<=usable:raise ValueError(f'workers must be between 1 and {usable}')
    return n

def cpu_order(allowed=None,sysfs=Path('/sys/devices/system/cpu')):
    """Use every physical core before assigning its SMT sibling."""
    allowed=sorted(allowed if allowed is not None else (os.sched_getaffinity(0) if hasattr(os,'sched_getaffinity') else range(os.cpu_count() or 1)))
    groups={}
    for cpu in allowed:
        try:
            root=sysfs/f'cpu{cpu}'/'topology'
            key=(int((root/'physical_package_id').read_text()),int((root/'core_id').read_text()))
        except (OSError,ValueError):key=(0,cpu)
        groups.setdefault(key,[]).append(cpu)
    cores=list(groups.values())
    return [core[i] for i in range(max(map(len,cores),default=0)) for core in cores if i<len(core)]

def cgroup_ancestors():
    p=Path('/proc/self/cgroup');base=Path('/sys/fs/cgroup')
    if not p.exists():return []
    for line in p.read_text().splitlines():
        if line.startswith('0::'):
            path=base/line[3:].lstrip('/')
            if '..' in path.parts:return []
            return [path]+[parent for parent in path.parents if parent==base or base in parent.parents]
    return []

def cgroup_clean_file_cache(folder):
    """Clean, unmapped file cache is reclaimable regardless of its LRU list.

    Full-library SHA checks can fill a scope with cache before any worker starts.
    Keep anonymous memory, tmpfs, mapped files and pending writes fully charged.
    The kernel's MemoryMax/MemoryHigh limits remain unchanged.
    """
    try:
        fields={k:int(v) for k,v in (line.split() for line in (folder/'memory.stat').read_text().splitlines())}
        clean=max(0,fields['file']-fields.get('shmem',0)-fields.get('file_mapped',0)-fields.get('file_dirty',0)-fields.get('file_writeback',0))
        return max(0,min(fields['inactive_file']+fields.get('active_file',0),clean))
    except (OSError,ValueError,KeyError):
        return 0


def memory():
    if sys.platform=='linux':
        fields={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line}
        total,available=fields['MemTotal'],fields['MemAvailable'];headroom=total;limit=None;scope_peak=None
        for folder in cgroup_ancestors():
            p=folder/'memory.max';q=folder/'memory.current'
            if p.exists() and q.exists() and p.read_text().strip()!='max':
                maximum=int(p.read_text());current=int(q.read_text())
                reclaimable=min(current,cgroup_clean_file_cache(folder))
                headroom=min(headroom,max(0,maximum-current+reclaimable));limit=min(limit or maximum,maximum)
                if scope_peak is None and (folder/'memory.peak').exists():scope_peak=int((folder/'memory.peak').read_text())
        return dict(total=total,available=available,cgroup_headroom=headroom,cgroup_limit=limit,cgroup_peak=scope_peak)
    if sys.platform=='darwin':
        import re
        total=int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True))
        text=subprocess.check_output(['vm_stat'],text=True);page=int(re.search(r'page size of (\d+)',text)[1])
        fields={m[1]:int(m[2])*page for m in re.finditer(r'^([^:\n]+):\s*(\d+)\.',text,re.M)}
        # Conservative: do not count compressed memory as available.
        available=sum(fields.get(k,0) for k in ('Pages free','Pages inactive','Pages speculative'))
        return dict(total=total,available=available,cgroup_headroom=total,cgroup_limit=None)
    raise RuntimeError('Resource-guarded batch currently supports Linux and macOS')

def process_memory(pid):
    """PSS avoids counting shared model/library pages once per worker."""
    if not pid:return 0
    if sys.platform=='linux':
        try:
            for line in Path(f'/proc/{pid}/smaps_rollup').read_text().splitlines():
                if line.startswith('Pss:'):return int(line.split()[1])*1024
        except (OSError,ValueError):pass
        try:
            for line in Path(f'/proc/{pid}/status').read_text().splitlines():
                if line.startswith('VmRSS:'):return int(line.split()[1])*1024
        except (OSError,ValueError):return 0
    try:return int(subprocess.check_output(['ps','-o','rss=','-p',str(pid)],text=True))*1024
    except (OSError,subprocess.SubprocessError,ValueError):return 0

def output_frames(path):
    frames=dt=None
    with Path(path).open(errors='strict') as stream:
        for line in stream:
            if line.strip().startswith('Frames:'):frames=int(line.split(':',1)[1])
            if line.strip().startswith('Frame Time:'):
                dt=float(line.split(':',1)[1]);break
    if frames is None or dt is None or frames<2 or not math.isfinite(dt) or dt<=0:
        raise ValueError('Invalid BVH frame header')
    return math.floor((frames-1)*dt*50+1e-8)+1

def peak_bytes(frames,base_gib=4.,bytes_per_frame=65536):
    # Conservative JAX baseline, not the old TRF/fork 256 MiB estimate.
    return int(base_gib*GIB+frames*bytes_per_frame)

def can_admit(snapshot,reserve,free_disk,disk_floor,running,candidate_peak,candidate_used=0,shared_reserve=0):
    if free_disk<disk_floor:return False,'disk_reserve'
    headroom=min(snapshot['available']-reserve,snapshot['cgroup_headroom'])
    reserved=sum(max(0,peak-used) for peak,used in running)
    if shared_reserve+reserved+max(0,candidate_peak-candidate_used)>headroom:return False,'memory_reserve'
    return True,None


def select_idle(slots,active,usage):
    """Prefer a resident JAX worker over an empty slot; credit only measured PSS."""
    idle=[s for s in slots if s not in active]
    return max(idle,key=lambda s:(bool(s.pid),
        getattr(s,'telemetry',{}).get('cache_state')=='jax_hot',usage.get(s,0)),default=None)


def reclaim_idle(slots,active,usage,snapshot,reserve,candidate=None,candidate_peak=0,shared=0):
    """Reclaim one idle process only when real headroom needs it.

    Future growth reservations alone must not evict warm workers. Under actual
    pressure return memory, preferring noncandidate/cold workers. A sole reusable
    candidate cannot make its own next job fit by being killed and respawned.
    Never credit reclaimed memory until a fresh monitor/PSS sweep observes it.
    """
    headroom=min(snapshot['available']-reserve,snapshot['cgroup_headroom'])
    pressure=headroom<0
    incremental=shared+max(0,candidate_peak-usage.get(candidate,0))
    if not pressure and (candidate is None or headroom>=incremental):return None
    idle=[s for s in slots if s not in active and s.pid and usage.get(s,0)>0
          and (pressure or s is not candidate)]
    return max(idle,key=lambda s:(s is not candidate,
        getattr(s,'telemetry',{}).get('cache_state')!='jax_hot',usage[s]),default=None)

@contextlib.contextmanager
def batch_lock():
    root=state_root();root.mkdir(parents=True,exist_ok=True)
    lock_path=Path(os.environ.get('DEX_FRAX_CPU_LEASE_FILE',root/'active_batch.lock')).expanduser()
    with lock_path.open('a') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another Dex-Frax batch holds the global CPU/memory lease') from None
        yield

def release_memory():
    import gc
    gc.collect()
    if sys.platform=='linux':
        import ctypes
        ctypes.CDLL(None).malloc_trim(0)

class CompileLease:
    """flock is released by the OS even when a worker is killed mid-compile."""
    def __init__(self,required=True):
        self.file=None;self.wait_s=0.
        root=os.environ.get('DEX_FRAX_COMPILE_LOCK_DIR')
        if not root or not required:return
        root=Path(root);root.mkdir(parents=True,exist_ok=True)
        begin=time.monotonic()
        self.file=(root/'compile.lock').open('a')
        fcntl.flock(self.file,fcntl.LOCK_EX)
        self.wait_s=time.monotonic()-begin
    def release(self):
        if self.file is not None:
            fcntl.flock(self.file,fcntl.LOCK_UN);self.file.close();self.file=None
    def __enter__(self):return self
    def __exit__(self,*_):self.release()

def contain_linux(args):
    """Parallel Linux CLI runs get a dedicated user scope, never a host-wide OOM."""
    hard_limit=getattr(args,'memory_limit_gib',None)
    if sys.platform!='linux' or (workers(args.workers)==1 and hard_limit is None and not dynamic_mode(args)):return None
    snapshot=memory();reserve=host_reserve(args,snapshot)
    # Explicit MemoryMax is a backstop, not an allocation. Host admission still
    # keeps the full reserve; even the hard ceiling keeps the emergency floor.
    ceiling=snapshot['total']-(reserve//2 if hard_limit is not None else reserve)
    maximum=(min(ceiling,int(hard_limit*GIB)) if hard_limit is not None else
             (ceiling if dynamic_mode(args) else min(ceiling,snapshot['available']-reserve,snapshot['cgroup_headroom'])))
    marker=os.environ.get('DEX_FRAX_RESOURCE_SCOPE')
    if marker:
        # An environment variable alone must not bypass containment.
        current=cgroup_ancestors()
        if not current or current[0].name!=marker or snapshot['cgroup_limit'] is None:
            raise RuntimeError('Resource scope was not established')
        if snapshot['cgroup_limit']>ceiling:raise RuntimeError('Resource scope exceeds host reserve')
        if hard_limit is not None and snapshot['cgroup_limit']>hard_limit*GIB:raise RuntimeError('Resource scope exceeds requested limit')
        return None
    startup_room=min(maximum,snapshot['cgroup_headroom']) if dynamic_mode(args) else min(maximum,snapshot['available']-reserve,snapshot['cgroup_headroom'])
    if startup_room<peak_bytes(1,args.worker_peak_gib) or snapshot['available']<reserve//2:raise RuntimeError('Insufficient memory for even one protected worker')
    if not shutil.which('systemd-run'):raise RuntimeError('Parallel Linux batch requires systemd-run user memory containment')
    unit='dex-frax-'+uuid.uuid4().hex+'.scope'
    command=['systemd-run','--user','--scope','--quiet','--unit='+unit,
             '-p','MemoryMax='+str(maximum),'-p','MemoryHigh='+str(int(maximum*(.98 if dynamic_mode(args) else .9))),
             '-p','MemorySwapMax=0',sys.executable,'-m','dex_frax',*sys.argv[1:]]
    env=dict(os.environ,DEX_FRAX_RESOURCE_SCOPE=unit)
    result=subprocess.run(command,env=env)
    if result.returncode:raise RuntimeError(f'Protected batch exited with code {result.returncode}; inspect batch.json and the systemd scope')
    return result.returncode
