"""Host-load admission; never kills work just to lower concurrency."""
import math
import os
from pathlib import Path
import sys
import time


def cpu_ticks(allowed,proc=Path('/proc')):
    result={}
    for line in (proc/'stat').read_text().splitlines():
        fields=line.split();name=fields[0]
        if name.startswith('cpu') and name[3:].isdigit() and int(name[3:]) in allowed:
            v=list(map(int,fields[1:9]));result[int(name[3:])]=(sum(v),v[3]+v[4])
    return result


def process_ticks(pid,proc=Path('/proc')):
    try:
        fields=(proc/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
        return int(fields[19]),int(fields[11])+int(fields[12])  # starttime, utime+stime
    except (OSError,ValueError,IndexError):return None


class CpuAdmission:
    """Subtract external CPU demand, not our own busy workers, from capacity.

    Thus a busy converter immediately refills a completed slot. Other workloads
    reduce the target; in-flight work drains naturally. No fixed reserved cores.
    """
    def __init__(self,allowed,ceiling,interval=1.,clock=time.monotonic,read_cpu=cpu_ticks,read_process=process_ticks):
        self.allowed=set(allowed);self.ceiling=ceiling;self.interval=interval
        self.clock=clock;self.read_cpu=read_cpu;self.read_process=read_process
        self.previous=None;self.previous_process={};self.last=None;self.external=None
        self.target=0;self.data=dict(ready=False,target_workers=0,reason='initial_cpu_sample')

    def sample(self,pids):
        now=self.clock()
        if self.last is not None and now-self.last<self.interval:return self.target
        try:current=self.read_cpu(self.allowed)
        except (OSError,ValueError):
            # Unsupported hosts still have an affinity/quota and memory ceiling.
            self.target=self.ceiling;self.data=dict(ready=True,target_workers=self.target,reason='cpu_sampling_unavailable');return self.target
        if not current:
            self.target=0;self.data=dict(ready=False,target_workers=0,reason='cpu_sample_empty');return 0
        own={pid:v for pid in set(pids) if pid and (v:=self.read_process(pid)) is not None}
        if self.previous is not None:
            total=busy=0
            for cpu,(ticks,idle) in current.items():
                if cpu not in self.previous:continue
                dt=ticks-self.previous[cpu][0];di=idle-self.previous[cpu][1]
                if dt>0:total+=dt;busy+=max(0,min(dt,dt-di))
            ours=sum(max(0,value[1]-self.previous_process[pid][1]) for pid,value in own.items()
                     if pid in self.previous_process and value[0]==self.previous_process[pid][0])
            if total>0:
                capacity=len(current);busy_cores=busy/total*capacity
                external=max(0.,(busy-ours)/total*capacity)
                # React promptly to added load, relax after it disappears.
                self.external=external if self.external is None else max(external,.5*self.external+.5*external)
                self.target=max(0,min(self.ceiling,int(math.floor(capacity-self.external+.5))))
                self.data=dict(ready=True,target_workers=self.target,reason='measured_cpu_demand',
                    busy_percent=100*busy/total,external_cpu_equivalents=self.external,
                    own_cpu_equivalents=max(0.,busy_cores-external),sample_interval_s=now-self.last)
        self.previous=current;self.previous_process=own;self.last=now
        return self.target

    def report(self):return dict(self.data)
