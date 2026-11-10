import argparse
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time
import pytest
from dex_frax import resources as r,managed_batch as batch
from dex_frax.cli import sha

@pytest.fixture(autouse=True)
def isolate_external_lease(monkeypatch):
    monkeypatch.delenv('DEX_FRAX_CPU_LEASE_FILE',raising=False)

def snapshot(available=60):return dict(total=64*r.GIB,available=available*r.GIB,cgroup_headroom=64*r.GIB,cgroup_limit=None)

def test_reserve_inflight_peak_before_launching_more():
    # 16 workers have allocated little so far but may each grow to 3 GiB.
    allowed,reason=r.can_admit(snapshot(60),10*r.GIB,100*r.GIB,80*r.GIB,[(3*r.GIB,.1*r.GIB)]*16,4*r.GIB)
    assert not allowed and reason=='memory_reserve'
    assert r.can_admit(snapshot(60),10*r.GIB,100*r.GIB,80*r.GIB,[(3*r.GIB,3*r.GIB)]*16,4*r.GIB)[0]

def test_cgroup_and_disk_both_limit_admission():
    s=snapshot();s['cgroup_headroom']=r.GIB
    assert not r.can_admit(s,10*r.GIB,100*r.GIB,80*r.GIB,[],2*r.GIB)[0]
    assert r.can_admit(snapshot(),10*r.GIB,79*r.GIB,80*r.GIB,[],r.GIB)==(False,'disk_reserve')

def test_physical_core_order_precedes_smt(tmp_path):
    for cpu,core in enumerate((0,0,1,1,2,2)):
        p=tmp_path/f'cpu{cpu}'/'topology';p.mkdir(parents=True)
        (p/'physical_package_id').write_text('0');(p/'core_id').write_text(str(core))
    assert r.cpu_order(range(6),tmp_path)==[0,2,4,1,3,5]
    assert r.cpu_order([1,2,3,5],tmp_path)==[1,2,5,3]

def test_single_compiler_reserve_allows_resident_workers_not_32_compilers():
    s=snapshot(60);s.update(cgroup_headroom=48*r.GIB,cgroup_limit=48*r.GIB)
    running=[(1.25*r.GIB,.4*r.GIB)]*31
    # Snapshot includes .4GiB already resident in each running worker.
    s['cgroup_headroom']-=int(31*.4*r.GIB)
    assert r.can_admit(s,10*r.GIB,100*r.GIB,80*r.GIB,running,1.25*r.GIB,shared_reserve=2.75*r.GIB)[0]
    assert not r.can_admit(s,10*r.GIB,100*r.GIB,80*r.GIB,running,4*r.GIB,shared_reserve=8*r.GIB)[0]

def test_global_lease_rejects_second_batch(tmp_path,monkeypatch):
    monkeypatch.setenv('DEX_FRAX_STATE_DIR',str(tmp_path))
    with r.batch_lock():
        with pytest.raises(RuntimeError,match='global'): 
            with r.batch_lock():pass

def test_external_lease_is_shared_across_state_directories(tmp_path,monkeypatch):
    monkeypatch.setenv('DEX_FRAX_CPU_LEASE_FILE',str(tmp_path/'standard_pipeline.lock'))
    monkeypatch.setenv('DEX_FRAX_STATE_DIR',str(tmp_path/'one'))
    with r.batch_lock():
        monkeypatch.setenv('DEX_FRAX_STATE_DIR',str(tmp_path/'two'))
        with pytest.raises(RuntimeError,match='global'):
            with r.batch_lock():pass

def test_killed_compiler_releases_os_lock(tmp_path):
    env=dict(os.environ,DEX_FRAX_COMPILE_LOCK_DIR=str(tmp_path))
    code='from dex_frax.resources import CompileLease; import time; lease=CompileLease(); print("held",flush=True); time.sleep(30)'
    p=subprocess.Popen([sys.executable,'-c',code],stdout=subprocess.PIPE,text=True,env=env)
    try:
        assert select.select([p.stdout],[],[],10)[0]
        assert p.stdout.readline().strip()=='held'
        blocked=subprocess.Popen([sys.executable,'-c','from dex_frax.resources import CompileLease; lease=CompileLease(); print("acquired",flush=True)'],stdout=subprocess.PIPE,text=True,env=env)
        try:
            assert not select.select([blocked.stdout],[],[],.3)[0]
            p.kill();p.wait(5)
            assert blocked.communicate(timeout=5)[0].strip()=='acquired'
        finally:
            if blocked.poll() is None:blocked.kill();blocked.wait()
    finally:
        if p.poll() is None:p.kill();p.wait()

def test_scope_marker_cannot_bypass_real_limit(monkeypatch):
    monkeypatch.setattr(r.sys,'platform','linux');monkeypatch.setattr(r,'workers',lambda _:32)
    monkeypatch.setattr(r,'memory',snapshot);monkeypatch.setattr(r,'cgroup_ancestors',lambda:[])
    monkeypatch.setenv('DEX_FRAX_RESOURCE_SCOPE','fake.scope')
    with pytest.raises(RuntimeError,match='not established'):r.contain_linux(argparse.Namespace(workers=32,memory_reserve_gib=10,worker_peak_gib=3))

class FakeSlot:
    calls=[];running=0;peak=0;plan={};instances=[]
    def __init__(self,options):self.pid=None;self.result=None;self.instances.append(self)
    def start(self,job,timeout):
        args=job['arguments'];folder=Path(args['output']);mode=args['mode'];name=Path(args['input']).stem
        self.calls.append((name,mode,args.get('reuse_initial_from')));type(self).running+=1;type(self).peak=max(self.peak,self.running)
        status=self.plan.get((name,mode),'review')
        if status in ('timeout','failed','input_error'):self.result=dict(status=status);return
        folder.mkdir();flags=[] if status=='accepted' else ['pose']
        (folder/'quality.json').write_text(json.dumps(dict(status=status,flags=flags)));files={}
        if status=='accepted':(folder/'training.npz').write_bytes(b'test');files['training.npz']=sha(folder/'training.npz')
        self.result=dict(status='complete',quality=status,flags=flags,result=dict(total_wall_s=.1,frax=None,acceptance_policy='common_pose_geometry_v1',fingerprint='test',files=files))
    def poll(self):
        result=self.result;self.result=None
        if result is not None:type(self).running-=1
        return result
    def close(self):
        if self.result is not None:type(self).running-=1;self.result=None

@pytest.fixture
def setup(tmp_path,monkeypatch):
    monkeypatch.setenv('DEX_FRAX_STATE_DIR',str(tmp_path/'cache'))
    monkeypatch.setattr(batch,'static_identity',lambda:dict(assets={},implementation={'test.py':'unchanged'}))
    from dex_frax import model_cache
    monkeypatch.setattr(model_cache,'prepare',lambda *_:{})
    monkeypatch.setattr(r,'workers',lambda _:32)
    monkeypatch.setattr(r.shutil,'disk_usage',lambda _:argparse.Namespace(free=200*r.GIB))
    FakeSlot.calls=[];FakeSlot.instances=[];FakeSlot.running=FakeSlot.peak=0;FakeSlot.plan={('easy','frax'):'accepted',('rescue','v8'):'accepted',('v8','v8'):'accepted',('invalid','frax'):'input_error'}
    cases=[]
    for name in ('easy','rescue','v8','review','invalid'):
        p=tmp_path/(name+'.bvh');p.write_text(f'{name}\nFrames: 181\nFrame Time: 0.033333\n')
        cases.append(dict(id=name,input=p.name,input_sha256=sha(p),source='lafan'))
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(cases=cases)))
    return argparse.Namespace(pipeline='two-stage',manifest=manifest,output=tmp_path/'out',workers=32,mode='auto',budget=30.,memory_reserve_gib=10.,disk_reserve_gib=80.,worker_peak_gib=3.,resource_wait=.01,iterations=8,initial_iterations=100,frax_timeout=None,v8_timeout=None)

def test_two_stage_cascade_independence_and_quality_do_not_change(setup):
    result=batch.run(setup,FakeSlot,lambda:snapshot(17))
    assert [x['status'] for x in result]==['accepted','accepted','accepted','review','input_error']
    assert FakeSlot.peak==2  # 32 requested slots, only two peak reservations fit.
    for name in ('easy','rescue','v8','review','invalid'):
        modes=[m for n,m,_ in FakeSlot.calls if n==name]
        assert modes==sorted(modes,key=['frax','v8'].index)
    assert all(reuse is None for _,_,reuse in FakeSlot.calls)
    assert len(json.loads((setup.output/'accepted.json').read_text())['motions'])==3


def test_v8_waits_for_entire_frax_pass(setup):
    class Slow(FakeSlot):
        slow_done=False;overlap=False
        def start(self,job,timeout):
            self.name=Path(job['input']).stem;self.mode=job['arguments']['mode'];self.polls=0
            if self.mode=='v8' and not type(self).slow_done:type(self).overlap=True
            super().start(job,timeout)
        def poll(self):
            if self.name=='review' and self.mode=='frax' and self.result is not None:
                self.polls+=1
                if self.polls<5:return None
                type(self).slow_done=True
            return super().poll()
    result=batch.run(setup,Slow,lambda:snapshot(60))
    assert not Slow.overlap and Slow.slow_done
    assert [r['status'] for r in result]==['accepted','accepted','accepted','review','input_error']


def test_budget_reload_fills_slots_without_waiting_for_running_jobs(setup,monkeypatch):
    setup.mode='frax';updated=False
    monkeypatch.setattr(FakeSlot,'poll',lambda self:None)
    def monitor():
        nonlocal updated
        if FakeSlot.running==2 and not updated:
            p=setup.output/'resource_policy.json';value=json.loads(p.read_text())
            value['stages']['frax']['base_gib']=.5
            p.write_text(json.dumps(value));updated=True
            time.sleep(2.05)
        if FakeSlot.running==5:(setup.output/'STOP').touch()
        return snapshot(17)
    records=batch.run(setup,FakeSlot,monitor)
    assert updated and FakeSlot.peak==5
    assert all(r['attempts'][0]['status']=='interrupted' for r in records)
    events=[json.loads(s) for s in (setup.output/'resource_policy_events.jsonl').read_text().splitlines()]
    assert sum(e['status']=='applied' for e in events)==2

def test_no_spawn_when_insufficient_memory(setup):
    records=batch.run(setup,FakeSlot,lambda:snapshot(12))
    assert FakeSlot.calls==[]
    assert all(x['status']=='resource_exhausted' for x in records)

def test_emergency_floor_aborts_own_workers(setup):
    calls=0
    def monitor():
        nonlocal calls
        calls+=1
        return snapshot(60 if calls<5 else 3)
    records=batch.run(setup,FakeSlot,monitor)
    assert FakeSlot.running==0
    assert all(x['status']=='resource_exhausted' for x in records)
    assert json.loads((setup.output/'accepted.json').read_text())['motions']==[]

def test_total_budget_cancels_without_adopting_partial_exports(setup,monkeypatch):
    setup.budget=.3
    monkeypatch.setattr(FakeSlot,'poll',lambda self:None)
    records=batch.run(setup,FakeSlot,lambda:snapshot(17))
    assert FakeSlot.running==0 and all(x['status']=='budget_exhausted' for x in records)
    assert not list(setup.output.rglob('training.npz'))

def test_worker_timeout_scales_with_full_motion(setup):
    assert batch.timeout_for(setup,'frax',15000)==7800.
    assert batch.timeout_for(setup,'v8',15000)==23100.
    setup.v8_timeout=42.;assert batch.timeout_for(setup,'v8',15000)==42.

def test_v8_uses_own_budget_without_lowering_jax_budget(setup,monkeypatch):
    setup.v8_worker_peak_gib=1.5
    peaks=[];original=r.peak_bytes
    def peak(frames,base,*args):
        peaks.append(base)
        return original(frames,base,*args)
    monkeypatch.setattr(r,'peak_bytes',peak)
    batch.run(setup,FakeSlot,lambda:snapshot(17))
    assert 1.5 in peaks and 3. in peaks

def test_resume_skips_complete_stages_and_retries_interrupted(setup):
    result=batch.run(setup,FakeSlot,lambda:snapshot(17))
    state=json.loads((setup.output/'batch.json').read_text())
    for record in state['records']:
        for attempt in record['attempts']:
            if attempt['status']!='complete':continue
            folder=setup.output/attempt['output']
            files={p.name:sha(p) for p in folder.iterdir() if p.is_file()}
            run=dict(status='complete',quality=attempt['quality'],flags=attempt['flags'],files=files,
                acceptance_policy='common_pose_geometry_v1',fingerprint='test')
            (folder/'run.json').write_text(json.dumps(run))
            (folder/'identity.json').write_text(json.dumps(dict(iterations=8,initial_iterations=100,max_frames=None,conversion_contract=batch.conversion_contract(setup))))
    target=next(x for x in state['records'] if x['case']=='v8')
    target['status']='interrupted';target['attempts'][-1]['status']='interrupted'
    state['status']='interrupted'
    (setup.output/'batch.json').write_text(json.dumps(state))
    FakeSlot.calls=[];setup.resume=True
    resumed=batch.run(setup,FakeSlot,lambda:snapshot(17))
    assert [(n,m) for n,m,_ in FakeSlot.calls]==[('v8','v8')]
    assert [r['status'] for r in resumed]==[r['status'] for r in result]
    assert list((setup.output/'v8').glob('v8.interrupted.*'))

def test_resume_refuses_changed_source_manifest(setup):
    batch.run(setup,FakeSlot,lambda:snapshot(17));setup.resume=True
    manifest=json.loads(setup.manifest.read_text());manifest['note']='changed'
    setup.manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='manifest'):batch.run(setup,FakeSlot,lambda:snapshot(17))

def test_quality_failure_runs_both_complete_stages(setup,monkeypatch):
    original=FakeSlot.start
    def start(self,job,timeout):
        original(self,job,timeout)
        if Path(job['input']).stem=='review':
            self.result['flags']=['too_short_for_training_window']
    monkeypatch.setattr(FakeSlot,'start',start)
    records=batch.run(setup,FakeSlot,lambda:snapshot(17))
    assert [(n,m) for n,m,_ in FakeSlot.calls if n=='review']==[('review','frax'),('review','v8')]
    record=next(r for r in records if r['case']=='review')
    assert record['status']=='review' and not record.get('escalation',{}).get('stop')
    assert (setup.output/'review/frax/quality.json').exists()


@pytest.mark.parametrize('force_v8',[False,True])
def test_collision_only_after_full_repair_has_explicit_v8_override(setup,monkeypatch,force_v8):
    original=FakeSlot.start;setup.v8_for_collision_only=force_v8
    def start(self,job,timeout):
        original(self,job,timeout)
        if Path(job['input']).stem=='review':
            self.result['flags']=['arm_self_collision_unresolved']
            self.result['result']['collision_repair_exhausted']=True
            p=Path(job['arguments']['output'])/'quality.json'
            p.write_text(json.dumps(dict(status='review',flags=self.result['flags'])))
    monkeypatch.setattr(FakeSlot,'start',start)
    result=batch.run(setup,FakeSlot,lambda:snapshot(17))
    modes=[m for n,m,_ in FakeSlot.calls if n=='review']
    assert modes==['frax','v8']
    assert next(r for r in result if r['case']=='review')['status']=='review'


def test_all_durations_start_with_frax_then_long_v8_first(setup):
    manifest=json.loads(setup.manifest.read_text())
    seconds={'easy':20,'rescue':59,'v8':60,'review':61,'invalid':10}
    for case in manifest['cases']:
        path=setup.manifest.parent/case['input'];path.write_text(f"Frames: {seconds[case['id']]}\nFrame Time: 1\n")
        case['input_sha256']=sha(path)
    setup.manifest.write_text(json.dumps(manifest))
    result=batch.run(setup,FakeSlot,lambda:snapshot(60))
    for name in ('v8','review'):assert [m for n,m,_ in FakeSlot.calls if n==name]==['frax','v8']
    assert all(m=='frax' for _,m,_ in FakeSlot.calls[:5])
    v8_calls=[n for n,m,_ in FakeSlot.calls if m=='v8']
    assert v8_calls.index('review')<v8_calls.index('v8')
    assert next(r for r in result if r['case']=='v8')['duration_s']==60.


def test_long_v8_cannot_consume_blocked_short_frax_budget(setup):
    manifest=json.loads(setup.manifest.read_text());manifest['cases']=manifest['cases'][:3]
    for case,seconds in zip(manifest['cases'],[1,2,100]):
        path=setup.manifest.parent/case['input'];path.write_text(f'Frames: {seconds*30}\nFrame Time: 0.033333\n')
        case['input_sha256']=sha(path)
    setup.manifest.write_text(json.dumps(manifest))
    setup.worker_resident_gib=1.;setup.worker_peak_gib=3.75;setup.v8_worker_peak_gib=.5;setup.frame_memory_kib=32
    class SlowFirst(FakeSlot):
        first_done=False
        def start(self,job,timeout):
            self.name=Path(job['input']).stem;self.polls=0
            if job['arguments']['mode']=='v8':assert self.first_done,'V8 bypassed a waiting short Frax'
            super().start(job,timeout)
        def poll(self):
            if self.name=='easy' and self.result is not None:
                self.polls+=1
                if self.polls<4:return None
                type(self).first_done=True
            return super().poll()
    batch.run(setup,SlowFirst,lambda:snapshot(14.5))
    assert SlowFirst.first_done
    assert [x[0] for x in FakeSlot.calls[:2]]==['easy','rescue']


def test_external_cpu_load_does_not_reduce_full_capacity_dispatch(setup,monkeypatch):
    from dex_frax import dynamic_resources
    setup.resource_mode='dynamic';setup.memory_target_percent=90.;setup.memory_reserve_gib=None
    class Cpu:
        def __init__(self,*a):self.calls=0;self.saw_inflight_at_zero=False
        def sample(self,pids):
            self.calls+=1
            if 2<=self.calls<=8:
                if Slow.running:self.saw_inflight_at_zero=True
                return 0
            return 2
        def report(self):return {'calls':self.calls,'preserved_inflight':self.saw_inflight_at_zero}
    class Slow(FakeSlot):
        def start(self,*a):self.polls=0;super().start(*a)
        def poll(self):
            self.polls+=1
            return None if self.polls<4 else super().poll()
    monkeypatch.setattr(dynamic_resources,'CpuAdmission',Cpu)
    result=batch.run(setup,Slow,lambda:snapshot(60))
    assert [r['status'] for r in result]==['accepted','accepted','accepted','review','input_error']
    state=json.loads((setup.output/'batch.json').read_text())
    assert state['resources']['cpu_admission']['preserved_inflight']
    assert Slow.peak==5  # All five available jobs launch despite a suggested 0/2-slot CPU target.
    assert state['resources']['cpu_admission']['target_workers']==32
    assert state['resources']['cpu_admission']['external_load_control'] is False
    assert not any(a['status']=='interrupted' for r in result for a in r['attempts'])


def test_dynamic_memory_wait_is_not_manual_sixty_second_failure(setup,monkeypatch):
    from dex_frax import dynamic_resources
    setup.resource_mode='dynamic';setup.memory_target_percent=90.;setup.memory_reserve_gib=None
    class Cpu:
        def __init__(self,*a):pass
        def sample(self,pids):return 32
        def report(self):return {}
    monkeypatch.setattr(dynamic_resources,'CpuAdmission',Cpu)
    calls=0
    def monitor():
        nonlocal calls
        calls+=1
        return snapshot(6.4 if calls<10 else 60)
    result=batch.run(setup,FakeSlot,monitor)
    assert calls>=10 and all(r['status']!='resource_exhausted' for r in result)


def test_refill_pss_sampling_is_linear_and_status_is_lightweight(setup,monkeypatch):
    """32 completions/refills must not perform 32 full PSS sweeps."""
    setup.mode='frax';setup.worker_peak_gib=.5
    cases=[]
    for index in range(32):
        name=f'clip{index:02d}';p=setup.manifest.parent/(name+'.bvh')
        p.write_text(f'{name}\nFrames: 31\nFrame Time: 0.033333\n')
        cases.append(dict(id=name,input=p.name,input_sha256=sha(p),source='lafan'))
        FakeSlot.plan[name,'frax']='accepted'
    setup.manifest.write_text(json.dumps(dict(cases=cases)))
    reads=[]
    monkeypatch.setattr(r,'process_memory',lambda pid:(reads.append(pid) or 0))
    writes=[];original=batch.atomic_json
    def write(path,value):
        writes.append((Path(path).name,'records' in value))
        original(path,value)
    monkeypatch.setattr(batch,'atomic_json',write)
    class Timed(FakeSlot):
        paused=False
        def poll(self):
            if not type(self).paused:
                type(self).paused=True;time.sleep(2.05)
            return super().poll()
    result=batch.run(setup,Timed,lambda:snapshot(60))
    assert all(x['status']=='accepted' for x in result)
    assert len(reads)<=64, 'PSS must be sampled once per loop, not per dispatch/report'
    assert sum(name=='batch.json' for name,_ in writes)==2  # initial + final
    assert sum(name=='batch.running.json' for name,_ in writes)>=3
    assert all(not records for name,records in writes if name=='batch.running.json')
    live=json.loads((setup.output/'batch.running.json').read_text())
    assert live['counts']=={'accepted':32} and live['active']==[]


def test_unified_batch_default_keeps_resource_controls_and_does_not_enqueue_v8(setup):
    del setup.pipeline
    class Unified(FakeSlot):
        def start(self,job,timeout):
            assert job['arguments']['mode']=='frax' and job['arguments']['unified_repair']
            super().start(job,timeout)
    result=batch.run(setup,Unified,lambda:snapshot(17))
    assert all(mode=='frax' for _,mode,_ in Unified.calls)
    assert [r['status'] for r in result]==['accepted','review','review','review','input_error']
    saved=json.loads((setup.output/'batch.json').read_text())
    assert saved['stage_order']==['frax'] and saved['pipeline']=='unified'
    assert saved['resources']['stage_barriers'] is False
