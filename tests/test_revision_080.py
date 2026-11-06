"""Regressions for long-window coverage, contracts and stage/resource isolation."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from dex_frax import graduated_collision as g
from dex_frax.contract import conversion_contract
from dex_frax.managed_batch import restore,sha_bytes
from dex_frax.memory_learning import MemoryLearning
from dex_frax.resources import GIB


def test_long_bad_span_gets_all_tiers_and_total_reference(monkeypatch):
    n=601;x=np.zeros((n,32));t=dict(fps=50.,root_r=np.tile(np.eye(3),(n,1,1)))
    report=dict(after=dict(collision_samples=n),windows=[dict(start=0,end=n,sides=['Left'],status='unresolved')])
    monkeypatch.setattr(g,'IncrementalRecovery',lambda *_:SimpleNamespace())
    monkeypatch.setattr(g,'scan',lambda *args:({'collision_samples':1,'minimum_distance_m':-.01},[],set()))
    calls=[]
    def failed(model,t,seed,cfg,a,b,sides,options):
        assert b-a<=120
        assert options['_reference'] is x
        if options['name']=='small':
            assert not seed.any()
            options['_scratch']['candidate_window']=seed[a:b]+.001
        else:assert np.all(seed[a:b]==.001)
        calls.append((options['name'],a,b,options['freeze_start'],options['freeze_end']))
        raise ValueError('local_search_failed')
    monkeypatch.setattr(g,'_repair_window',failed)
    out,r=g.recover(None,t,x,{},None,report)
    for tier in g.TIERS:
        covered=np.zeros(n,bool)
        for label,a,b,*_ in calls:
            if label==tier['name']:covered[a:b]=True
        assert covered.all()
    labels=[a[0] for a in calls]
    assert labels==sorted(labels,key=['small','medium','coupled'].index)
    assert all(not left and not right for _,_,_,left,right in calls)
    assert r['graduated_repair']['outcome']=='tiers_exhausted'
    np.testing.assert_array_equal(out,x)


def test_fast_budget_reports_deferred_not_exhausted(monkeypatch):
    n=601;x=np.zeros((n,32));t=dict(fps=50.,root_r=np.tile(np.eye(3),(n,1,1)))
    report=dict(after=dict(collision_samples=n),windows=[dict(start=0,end=n,sides=['Left'],status='unresolved')])
    monkeypatch.setattr(g,'IncrementalRecovery',lambda *_:SimpleNamespace())
    monkeypatch.setattr(g,'scan',lambda *a:({'collision_samples':1},[],set()))
    def fail(*a,**kw):raise ValueError('local_solver_failed')
    monkeypatch.setattr(g,'_repair_window',fail)
    _,r=g.recover(None,t,x,{},None,report,fast=True)
    assert r['graduated_repair']['outcome']=='deferred'
    assert not r['graduated_repair']['all_unresolved_tried_all_tiers']
    assert r['graduated_repair']['attempted_frames']<=600


@pytest.mark.parametrize('change',['cli','acceptance','block','iterations','legacy','policy'])
def test_resume_rejects_changed_conversion_contract(tmp_path,change):
    args=SimpleNamespace(mode='auto',iterations=8,initial_iterations=100,ik_block_frames=128)
    snapshot=dict(implementation={'cli.py':'old','acceptance.py':'old'})
    (tmp_path/'batch.json').write_text(json.dumps(dict(manifest_sha256=sha_bytes(b'source'),mode='auto',records=[])))
    (tmp_path/'batch_identity.json').write_text(json.dumps(snapshot))
    contract=conversion_contract(args)
    (tmp_path/'conversion_contract.json').write_text(json.dumps(contract))
    if change in ('cli','acceptance'):snapshot['implementation'][change+'.py']='new'
    elif change=='block':args.ik_block_frames=64
    elif change=='iterations':args.iterations=9
    elif change=='legacy':args.legacy_frax=True
    else:
        contract['quality_policy']='obsolete';(tmp_path/'conversion_contract.json').write_text(json.dumps(contract))
    with pytest.raises(ValueError,match='changed'):restore(args,tmp_path,b'source',[],snapshot)


def test_resource_values_do_not_change_conversion_contract():
    a=SimpleNamespace(iterations=8,initial_iterations=100,ik_block_frames=128,workers=8)
    before=conversion_contract(a);a.workers=32;a.memory_reserve_gib=10.
    assert conversion_contract(a)==before


def test_memory_remaining_stage_and_cache_separation(tmp_path):
    p=SimpleNamespace(peak=lambda mode,n:4*GIB,value=dict(stages={'v8':{'bytes_per_output_frame':0}}))
    learner=MemoryLearning(tmp_path,{})
    phases={s+'|pure':int((2 if s=='arms' else .5)*GIB) for s in
            ('preprocessing','initial','support','arms','body_contacts','self_collision','export_check','audit','export')}
    # End-of-job JAX import must not relabel the entire earlier pure-V8 job.
    slot=SimpleNamespace(telemetry={'cache_state':'jax_imported'})
    for _ in range(2):learner.complete(slot,dict(mode='v8',phase_peaks=phases),dict(status='complete'),{'source':'lafan'},1000,p)
    assert learner.predict('v8','lafan','pure',1000,p,stage='arms')>2*GIB
    assert learner.predict('v8','lafan','pure',1000,p,stage='audit')<GIB
    assert learner.predict('v8','lafan','jax_hot',1000,p,stage='audit')==4*GIB


def test_arm_subphases_cannot_release_remaining_arm_budget(monkeypatch):
    from dex_frax import resource_events as e
    events=[];monkeypatch.setattr(e,'_sink',events.append);monkeypatch.setattr(e,'process_memory',lambda pid:10)
    for name in ('loading','arms','arms_candidates','arms_temporal','arms_candidates','arms_frame_fit'):e.phase(name)
    assert all(v['stage']=='arms' for v in events[1:])
    assert 'arms' in e.remaining('v8',events[-1]['stage'])


def test_fitted_memory_envelope_never_undercuts_observed_samples():
    from dex_frax.memory_learning import envelope
    samples=[dict(frames=n,peak_bytes=200_000_000+n*100_000) for n in (100,700,1600)]
    for v in samples:assert envelope(samples,v['frames'],1)>=v['peak_bytes']
    assert envelope(samples,2000,1)>envelope(samples,1600,1)


def test_long_arm_fallback_stays_local(monkeypatch):
    from dex_frax import arm_tracking as arm,local_refine,adaptive_refine
    from dex_frax.v8 import arms
    n=600;x=np.zeros((n,32));ids=list(range(3,10));mask=np.zeros(n,bool);mask[255:265]=True
    robot=SimpleNamespace(sides={'Left':{'arm':ids}},joint_step=lambda *args:np.ones(29))
    monkeypatch.setattr(arm,'hard_frames',lambda *args:mask.copy())
    monkeypatch.setattr(arm,'geometry',lambda *args:dict(angles=np.zeros(7),soft=np.zeros(2)))
    monkeypatch.setattr(arms,'ArmObjective',lambda *args:None)
    monkeypatch.setattr(adaptive_refine,'window_targets',lambda t,a,b,n:dict(fps=50.))
    calls=[]
    def local(robot,t,x,cfg,progress,sides):
        calls.append(len(x));assert not cfg['arm_event_tracking'];return x,{'Left':{'candidate_solver_calls':0}}
    monkeypatch.setattr(arms,'plan_and_refine',local)
    monkeypatch.setattr(local_refine,'fit_window',lambda robot,t,x,cfg,a,b,**kw:(x,{'start':a,'end':b}))
    out,report=arm.recover_windows(robot,{'fps':50.},x,x,{},'Left')
    assert calls and max(calls)<=120 and sum(calls)<n/2
    assert report['unresolved_windows']==0 and not report['whole_clip_fallback']
    np.testing.assert_array_equal(out,x)


def test_memory_reuse_and_cold_profiles_do_not_mix(tmp_path):
    learner=MemoryLearning(tmp_path,{})
    p=SimpleNamespace(peak=lambda mode,n:4*GIB,value=dict(stages={'hybrid':{'bytes_per_output_frame':0}}))
    slot=SimpleNamespace(telemetry={'cache_state':'jax_hot'})
    for _ in range(3):learner.complete(slot,dict(mode='hybrid',reuse=True,phase_peaks={'arms|jax_hot':GIB}),
        dict(status='complete'),{'source':'lafan'},1000,p)
    assert learner.predict('hybrid','lafan','jax_hot',1000,p,reuse=True)<2*GIB
    assert learner.predict('hybrid','lafan','jax_hot',1000,p,reuse=False)==4*GIB
    profile=next(iter(learner.profiles.values()))
    assert len(profile['samples'])==1 and profile['completed_jobs']==3


def test_dense_short_window_guard_cannot_restore_whole_long_clip():
    from dex_frax.arm_tracking import original_window
    tiny=np.zeros(600,bool);tiny[200:245]=True
    assert not original_window(tiny)
    dense=np.zeros(300,bool);dense[50:152]=True
    assert original_window(dense)
    assert not original_window(np.ones(601,bool))
