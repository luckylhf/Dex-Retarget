from types import SimpleNamespace
import numpy as np
import pytest
from dex_frax import graduated_collision as g


def scene(monkeypatch,reason='collision_qp_primal infeasible'):
    x=np.zeros((12,32));t={'fps':50.,'root_r':np.tile(np.eye(3),(12,1,1))}
    report={'after':{'collision_samples':1},'windows':[dict(start=0,end=12,sides=['Left'],status='unresolved',reason=reason)]}
    monkeypatch.setattr(g,'IncrementalRecovery',lambda *_:SimpleNamespace())
    monkeypatch.setattr(g,'scan',lambda model,x,*args:({'collision_samples':int(not np.any(x)), 'minimum_distance_m':.004},[],set()))
    return x,t,report


def test_clean_trajectory_skips_recovery_model(monkeypatch):
    x=np.zeros((3,32));report={'after':{'collision_samples':0}}
    def fail(*a):raise AssertionError('no solver for clean motion')
    monkeypatch.setattr(g,'IncrementalRecovery',fail)
    out,result=g.recover(None,{},x,{},None,report)
    assert out is x and result is report


def test_tiers_only_escalate_failed_window_and_rollback_failed_attempts(monkeypatch):
    x,t,report=scene(monkeypatch);calls=[]
    def repair(model,t,source,weights,a,b,sides,options):
        np.testing.assert_array_equal(source,x)
        calls.append(options['joint_deg'])
        if options['name']=='small':raise ValueError('collision_qp_primal infeasible')
        candidate=source.copy();candidate[:,20]=.01
        return candidate,dict(status='repaired',start=a,end=b)
    monkeypatch.setattr(g,'_repair_window',repair)
    out,result=g.recover(None,t,x,{},None,report)
    assert calls==[10.,20.] and np.any(out!=x)
    np.testing.assert_array_equal(x,np.zeros_like(x))
    assert result['graduated_repair']['attempted_frames']==24
    assert result['graduated_repair']['attempts'][-1]['tier']=='medium'


def test_support_goes_to_coupled_tier_without_releasing_it(monkeypatch):
    x,t,report=scene(monkeypatch,'active_arm_support_requires_coupled_repair');calls=[]
    def repair(model,t,source,weights,a,b,sides,options):
        calls.append(options['name'])
        raise ValueError('repair_support_contact_changed')
    monkeypatch.setattr(g,'_repair_window',repair)
    out,result=g.recover(None,t,x,{},None,report)
    assert calls==['small','medium','coupled'];np.testing.assert_array_equal(out,x)
    assert result['after']['collision_samples']==1


def test_dense_interpolation_failure_never_commits(monkeypatch):
    x,t,report=scene(monkeypatch)
    monkeypatch.setattr(g,'_repair_window',lambda *a,**kw:(x+.01,dict(status='repaired')))
    monkeypatch.setattr(g,'scan',lambda *a:({'collision_samples':1},[],set()))
    out,result=g.recover(None,t,x,{},None,report)
    np.testing.assert_array_equal(out,x)
    assert all(a['reason']=='dense_interpolated_collision' for a in result['graduated_repair']['attempts'])


@pytest.fixture(scope='module')
def actual_scene():
    import importlib.util
    from pathlib import Path
    p=Path(__file__).parent/'upstream/test_self_collision.py'
    spec=importlib.util.spec_from_file_location('collision_geometry_fixture',p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.case.__wrapped__()


def test_actual_small_tier_clears_geometry_with_gesture_caps(actual_scene):
    from dex_frax.v8.self_collision import refine,CollisionModel,scan
    r,t,x,cfg=actual_scene
    original,report=refine(r,t,x,cfg['solver'],dict(cfg['self_collision'],max_iterations=0))
    assert report['after']['collision_samples']>0
    out,result=g.recover(r,t,original,cfg['solver'],cfg['self_collision'],report)
    assert result['after']['collision_samples']==0
    assert scan(CollisionModel(r),out,t['root_r'],8)[0]['collision_samples']==0
    assert np.rad2deg(abs(out[:,3:]-x[:,3:])).max()<=35.+1e-5
    assert np.all(abs(np.diff(out[:,3:],axis=0))<=r.joint_step(t['fps'],cfg['solver'])+2e-6)


def test_whole_arm_collision_clears_real_geometry_without_moving_waist(actual_scene):
    from dex_frax.arm_policy import POLICY,collision_options
    from dex_frax.v8.self_collision import CollisionModel,scan,_repair_window
    r,t,x,cfg=actual_scene;weights=dict(cfg['solver'],arm_repair_policy=POLICY)
    candidate=None
    for tier in (10.,20.,30.):
        try:
            candidate,_=_repair_window(g.IncrementalRecovery(r),t,x,weights,0,len(x),list(r.sides),
                                      options=dict(collision_options(tier,24),_reference=x))
            break
        except ValueError:pass
    assert candidate is not None
    assert scan(CollisionModel(r),candidate,t['root_r'],8)[0]['collision_samples']==0
    arm=np.asarray(sum([v['arm'] for v in r.sides.values()],[]))
    fixed=np.setdiff1d(np.arange(x.shape[1]),arm)
    np.testing.assert_array_equal(candidate[:,fixed],x[:,fixed])
    shoulders=sum([v['arm'][:3] for v in r.sides.values()],[])
    assert np.max(abs(candidate[:,shoulders]-x[:,shoulders]))<=np.deg2rad(tier)+2e-6
    assert np.all(candidate[:,3:]>=r.lower[3:]-2e-6)
    assert np.all(candidate[:,3:]<=r.upper[3:]+2e-6)
    assert np.all(abs(np.diff(candidate[:,3:],axis=0))<=r.joint_step(t['fps'],weights)+2e-6)


def test_actual_coupled_repair_preserves_fixed_material_contact(actual_scene):
    import copy
    from dex_frax.v8.self_collision import CollisionModel,scan,_repair_window
    r,t,x,cfg=actual_scene;t=copy.deepcopy(t);n=len(x)
    rep=lambda v:np.repeat(np.asarray(v)[None],n,axis=0)
    r.set(x[0],t['root_r'][0]);t.update(root_p=x[:,:3].copy(),torso_r=rep(r.rot(r.torso)),body_contact_height_tolerance_m=.035)
    for side,body in r.sides.items():
        direction=lambda a,b:rep((r.point(b)-r.point(a))/np.linalg.norm(r.point(b)-r.point(a)))
        t['sides'][side].update(thigh=direction(body['hip'],body['knee']),shin=direction(body['knee'],body['foot']),
            foot_target_p=rep(r.point(body['foot'])),foot_target_r=rep(r.rot(body['foot'])),contact=np.zeros(n),
            pivots=rep(body['sole']),anchors=np.zeros((n,3)),ground_contact_confidence=np.zeros(n))
    b=r.sides['Right']['forearm'];hull=r.surface_models[b]['hull'];local=hull[np.argmin(hull@r.rot(b)[2])].copy();point=r.point(b,local)
    t['body_contacts']=[dict(body=b,robot_hull=hull,mode=np.ones(n,int),source_geometry_confidence=1.,confidence=np.ones(n))]
    out,report=_repair_window(g.IncrementalRecovery(r),t,x,cfg['solver'],0,n,['Left'],options=g.TIERS[-1])
    assert scan(CollisionModel(r),out,t['root_r'],8)[0]['collision_samples']==0
    for i,row in enumerate(out):
        r.set(row,t['root_r'][i]);assert np.max(abs(r.point(b,local)-point))<=.000502
    assert report['coupled']


def test_wall_time_does_not_spend_normal_iteration_budget(actual_scene,monkeypatch):
    from dex_frax.v8 import self_collision as c
    r,t,x,cfg=actual_scene
    original,report=c.refine(r,t,x,cfg['solver'],dict(cfg['self_collision'],max_iterations=0))
    tick=iter(range(0,10**9,1000))
    monkeypatch.setattr(c.time,'perf_counter',lambda:next(tick))
    # Each clock read advances 1000 seconds. Normal work still receives all
    # configured iterations, rather than exhausting a wall-clock tier budget.
    out,result=g.recover(r,t,original,cfg['solver'],cfg['self_collision'],report)
    assert result['after']['collision_samples']==0
    assert c.scan(c.CollisionModel(r),out,t['root_r'],8)[0]['collision_samples']==0


def test_actual_leg_subset_keeps_root_waist_and_arms_exact(actual_scene):
    import copy
    from dex_frax.refine_routing import collision_groups,submodel
    from dex_frax.v8.self_collision import CollisionModel,_repair_window
    r,t,x,cfg=actual_scene;t=copy.deepcopy(t);n=len(x)
    rep=lambda v:np.repeat(np.asarray(v)[None],n,axis=0)
    r.set(x[0],t['root_r'][0]);t.update(root_p=x[:,:3].copy(),torso_r=rep(r.rot(r.torso)))
    for side,g in r.sides.items():
        direction=lambda a,b:rep((r.point(b)-r.point(a))/np.linalg.norm(r.point(b)-r.point(a)))
        t['sides'][side].update(thigh=direction(g['hip'],g['knee']),shin=direction(g['knee'],g['foot']),
            foot_target_p=rep(r.point(g['foot'])),foot_target_r=rep(r.rot(g['foot'])),contact=np.zeros(n),
            pivots=rep(g['sole']),anchors=np.zeros((n,3)),ground_contact_confidence=np.zeros(n))
    full=CollisionModel(r,dict(cfg['self_collision'],coverage='full'));model=submodel(full,collision_groups(full)['leg_collision'])
    ids=sum([g['leg'] for g in r.sides.values()],[]);progress={}
    out,report=_repair_window(model,t,x,cfg['solver'],0,n,list(r.sides),dict(coupled=True,_moving_ids=ids,
        _reference=x,_progress=progress,body_joint_deg=5.,joint_deg=0.,root_position_m=0.,
        whole_arm=True,gesture_caps_active=False,iterations=4))
    fixed=np.setdiff1d(np.arange(32),ids)
    assert np.array_equal(out[:,fixed],x[:,fixed])
    assert progress['window_qp']['ids']==ids and progress['window_qp']['iteration']>0
    assert np.all(abs(np.diff(out[:,ids],axis=0))*t['fps']<=r.speed_limits[np.asarray(ids)-3]+1e-4)


def test_branch_search_resumes_after_completed_knots_and_preserves_completed_side(actual_scene,monkeypatch):
    import copy
    from dex_frax.v8 import arms
    r,t,x,cfg=actual_scene;state={};weights=dict(cfg['solver'],arm_event_tracking=False,_branch_progress=state)
    factory=arms.arm_objective
    def interrupt(*args,**kwargs):
        if state.get('Left',{}).get('nodes'):raise TimeoutError('test_slice')
        return factory(*args,**kwargs)
    monkeypatch.setattr(arms,'arm_objective',interrupt)
    with pytest.raises(TimeoutError):arms.plan_and_refine(r,t,x,weights,_only_sides={'Left'})
    first=copy.deepcopy(state['Left']['nodes'][0])
    monkeypatch.setattr(arms,'arm_objective',factory)
    out,_=arms.plan_and_refine(r,t,x,weights,_only_sides={'Left'})
    assert state['Left']['nodes'][0]==first and state['Left']['frame_next']==len(x)
    assert 'complete' in state['Left']
    monkeypatch.setattr(arms,'arm_objective',lambda *a,**k:pytest.fail('completed side was searched again'))
    repeated,_=arms.plan_and_refine(r,t,x,weights,_only_sides={'Left'})
    np.testing.assert_array_equal(out,repeated)
