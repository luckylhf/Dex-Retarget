from types import SimpleNamespace
from dex_frax import resources as r,managed_batch as batch

class Idle:
    def __init__(self,pid,hot=True):self.pid=pid;self.telemetry={'cache_state':'jax_hot' if hot else 'pure'}
def snap(room):return dict(available=(room+6)*r.GIB,cgroup_headroom=room*r.GIB)

def test_reuse_hot_process_even_when_first_slot_is_empty():
    empty,cold,hot=Idle(None),Idle(2,False),Idle(3)
    assert r.select_idle([empty,cold,hot],{}, {cold:2*r.GIB,hot:r.GIB}) is hot
    assert r.select_idle([empty,cold,hot],{hot:{}},{cold:r.GIB}) is cold

def test_prediction_only_wait_does_not_destroy_warm_cache():
    hot=Idle(3);active=Idle(4);usage={hot:r.GIB,active:r.GIB}
    assert not r.can_admit(snap(8),6*r.GIB,100*r.GIB,80*r.GIB,[(10*r.GIB,r.GIB)],2*r.GIB,r.GIB)[0]
    assert r.reclaim_idle([hot,active],{active:{}},usage,snap(8),6*r.GIB,hot,2*r.GIB) is None

def test_real_pressure_reclaims_only_idle_and_prefers_cold():
    hot,cold,active=Idle(2),Idle(3,False),Idle(4)
    usage={hot:2*r.GIB,cold:r.GIB,active:4*r.GIB}
    assert r.reclaim_idle([hot,cold,active],{active:{}},usage,snap(-.1),6*r.GIB,hot,2*r.GIB) is cold
    assert r.reclaim_idle([active],{active:{}},usage,snap(-.1),6*r.GIB) is None

def test_other_idle_can_be_freed_but_sole_reusable_worker_is_not_churned():
    hot,other=Idle(2),Idle(3);usage={hot:r.GIB,other:r.GIB}
    assert r.reclaim_idle([hot,other],{},usage,snap(.2),6*r.GIB,hot,2*r.GIB) is other
    assert r.reclaim_idle([hot],{},usage,snap(.2),6*r.GIB,hot,2*r.GIB) is None

def test_cgroup_pressure_is_not_hidden_by_host_free_memory():
    hot=Idle(2);s=snap(20);s['cgroup_headroom']=-1
    assert r.reclaim_idle([hot],{},{hot:r.GIB},s,6*r.GIB) is hot

def test_healthy_worker_survives_64_jobs_and_close_clears_telemetry():
    class Process:
        pid=123
        def is_alive(self):return True
    sent=[];slot=batch.Slot({});slot.process=Process();slot.connection=SimpleNamespace(send=sent.append)
    slot.jobs=64;slot.telemetry={'cache_state':'jax_hot'}
    slot.start({'arguments':{'mode':'frax'}},10)
    assert slot.pid==123 and slot.jobs==64 and len(sent)==1
    # Avoid killing a fake process; a real close must clear process-local claims.
    slot.process=None;slot.connection=None;slot.close()
    assert slot.jobs==0 and slot.telemetry=={} and slot.phase_peaks=={}


def test_scheduler_retains_finished_worker_while_another_job_reserves_growth(tmp_path,monkeypatch):
    """Actual coordinator loop: prediction-only wait must reuse the same PID."""
    import argparse,json
    from pathlib import Path
    from dex_frax import model_cache
    monkeypatch.setenv('DEX_FRAX_STATE_DIR',str(tmp_path/'cache'))
    monkeypatch.delenv('DEX_FRAX_CPU_LEASE_FILE',raising=False)
    monkeypatch.setattr(batch,'static_identity',lambda:dict(assets={},implementation={}))
    monkeypatch.setattr(model_cache,'prepare',lambda *_:{})
    monkeypatch.setattr(r,'workers',lambda _:3)
    monkeypatch.setattr(r.shutil,'disk_usage',lambda _:argparse.Namespace(free=200*r.GIB))
    monkeypatch.setattr(r,'process_memory',lambda pid:0)
    cases=[]
    for i in range(3):
        p=tmp_path/f'c{i}.bvh';p.write_text(f'clip{i}\nFrames: 50\nFrame Time: 0.02\n')
        cases.append(dict(id=f'c{i}',input=str(p),input_sha256=batch.sha(p),source='soma'))
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(cases=cases)))
    class Slot:
        instances=[];starts=[];fast_done=False;slow_done=False;closed_while_waiting=False
        def __init__(self,_):self.pid=None;self.result=None;self.telemetry={};self.instances.append(self)
        def start(self,job,timeout):
            self.name=Path(job['input']).stem;self.polls=0
            if self.pid is None:self.pid=100+len(self.starts)
            self.starts.append((self.name,self.pid));folder=Path(job['arguments']['output']);folder.mkdir();self.telemetry={'cache_state':'jax_hot'}
            self.result=dict(status='complete',quality='review',flags=['pose'],result=dict(total_wall_s=.1,frax=None,acceptance_policy='common_pose_geometry_v1',fingerprint='test',files={}))
        def poll(self):
            if self.result is None:return None
            self.polls+=1
            if self.name=='c1' and self.polls<5:return None
            if self.name=='c0':type(self).fast_done=True
            if self.name=='c1':type(self).slow_done=True
            result=self.result;self.result=None;return result
        def close(self):
            if self.result is None and self.pid and self.fast_done and not self.slow_done:type(self).closed_while_waiting=True
            self.pid=None
    def monitor():
        room=5.5 if Slot.fast_done and not Slot.slow_done else 7
        return dict(total=64*r.GIB,available=int((10+room)*r.GIB),cgroup_headroom=64*r.GIB,cgroup_limit=None)
    args=argparse.Namespace(manifest=manifest,output=tmp_path/'out',workers=3,mode='frax',budget=20.,memory_reserve_gib=10.,disk_reserve_gib=80.,worker_peak_gib=3.,resource_wait=1.,iterations=8,initial_iterations=100,frax_timeout=None,hybrid_timeout=None,v8_timeout=None)
    records=batch.run(args,Slot,monitor)
    assert len(records)==3 and not Slot.closed_while_waiting
    assert Slot.starts[0][1]==Slot.starts[2][1]



def test_scheduler_resume_reader_keeps_export_identity_and_all_other_gates(tmp_path,monkeypatch):
    import copy,json
    import pytest
    from dex_frax import initial_reuse
    from test_initial_reuse import checkpoint
    old,fallback,robot,targets,config=checkpoint(tmp_path)
    previous=json.loads((tmp_path/'identity.json').read_text())
    previous['implementation'].update({'managed_batch.py':'old','resources.py':'same','ik.py':'same'})
    (tmp_path/'identity.json').write_text(json.dumps(previous))
    identity=copy.deepcopy(previous);identity['mode']='hybrid';identity['implementation']['managed_batch.py']='new'
    unchanged=copy.deepcopy(identity)
    monkeypatch.setattr(initial_reuse,'ReusedInitial',initial_reuse.ReusedInitial)
    batch.enable_scheduler_initial_reuse()
    reader=initial_reuse.ReusedInitial(tmp_path,identity,fallback,strict=True)
    reader(robot,targets,config)
    assert fallback.calls==0 and reader.report['initial_reuse']['used']
    assert set(reader.report['initial_reuse']['scheduler_compatibility'])=={'managed_batch.py'}
    assert identity==unchanged and json.loads((tmp_path/'identity.json').read_text())==previous
    for field in ['ik.py','assets','input_sha256','conversion_contract']:
        bad=copy.deepcopy(identity)
        if field=='ik.py':bad['implementation'][field]='changed'
        elif field=='input_sha256':bad[field]='changed'
        else:bad[field]={'changed':'value'}
        with pytest.raises(ValueError):initial_reuse.ReusedInitial(tmp_path,bad,fallback,strict=True)(robot,targets,config)
    (tmp_path/'motion.initial.npz').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='hash'):initial_reuse.ReusedInitial(tmp_path,identity,fallback,strict=True)(robot,targets,config)
