import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from dex_frax.shoulder_recovery import shoulder_bounds,shoulder_seeds,FrameRecovery,propose
from dex_frax.v8.robot import Robot
from dex_frax.v8.anatomy import configure
from dex_frax.v8.self_collision import CollisionModel


@pytest.fixture(scope='module')
def scene():
    import dex_frax.v8
    cfg=json.loads((Path(dex_frax.v8.__file__).parent/'config.json').read_text())
    r=Robot();configure(r,cfg['anatomy']);x=np.r_[0.,0.,1.,np.zeros(29)]
    x[3:]=np.clip(x[3:],r.lower[3:]+.01,r.upper[3:]-.01)
    r.set(x,np.eye(3));t={'root_r':np.eye(3)[None], 'sides':{}}
    for side,g in r.sides.items():
        unit=lambda v:v/np.linalg.norm(v)
        t['sides'][side]={k:np.array(v,copy=True)[None] for k,v in dict(
            upper=unit(r.point(g['elbow'])-r.point(g['shoulder'])),
            lower=unit(r.point(g['wrist'])-r.point(g['elbow'])),
            hand_forward=-r.rot(g['tcp'])[:,2],hand_width=r.rot(g['tcp'])[:,0],
            forearm_width=r.rot(g['forearm'])[:,0],elbow_axis=-r.rot(g['shoulder'])[:,1],
            signed_elbow_bend=g['elbow_geometric_zero']-x[g['arm'][3]]).items()}
    return r,x,t,cfg


def test_tiers_expand_about_fixed_reference_not_previous_candidate(scene):
    r,x,t,cfg=scene;ids=np.asarray(r.sides['Left']['arm']);reference=x.copy();candidate=x.copy()
    candidate[ids[:3]]+=np.deg2rad(9)
    for tier in (10.,20.,30.):
        seeds,lo,hi=shoulder_seeds(r,t,0,'Left',candidate,reference,tier)
        assert all(np.max(abs(q[:3]-reference[ids[:3]]))<=np.deg2rad(tier)+1e-8 for q in seeds)
        assert all(np.all(q>=lo) and np.all(q<=hi) for q in seeds)
        assert any(np.count_nonzero(abs(q[:3]-reference[ids[:3]])>.05)==3 for q in seeds)
        np.testing.assert_allclose(lo[3:],r.lower[ids[3:]]+1e-6)
        np.testing.assert_allclose(hi[3:],r.upper[ids[3:]]-1e-6)
    np.testing.assert_array_equal(x,reference)


def test_recovery_jacobian_matches_finite_difference(scene):
    r,x,t,cfg=scene;model=CollisionModel(r,cfg.get('self_collision'))
    weights=dict(cfg['solver'],arm_ground_penalty=0)
    ids=r.sides['Left']['arm'];q=x[ids].copy()+np.array([.05,.04,-.03,.02,.03,-.01,.02])
    obj=FrameRecovery(model,t,0,'Left',x,x,weights,None)
    _,J,_=obj.evaluate(q);numerical=[]
    for i in range(7):
        d=np.eye(7)[i]*1e-6
        numerical.append((obj.evaluate(q+d)[0]-obj.evaluate(q-d)[0])/2e-6)
    np.testing.assert_allclose(J,np.array(numerical).T,atol=2e-3,rtol=2e-3)


def test_supported_arm_never_enters_free_shoulder_recovery(monkeypatch):
    import dex_frax.shoulder_recovery as module
    monkeypatch.setattr(module,'active_arm_support',lambda *args:True)
    with pytest.raises(ValueError,match='active_arm_support'):
        propose(SimpleNamespace(r=object()),{},np.zeros((5,32)),{},0,5,{},float('inf'))


def test_seed_cannot_bypass_strict_window_validation(monkeypatch):
    import dex_frax.shoulder_recovery as module
    monkeypatch.setattr(module,'active_arm_support',lambda *args:False)
    robot=SimpleNamespace(sides={});model=SimpleNamespace(r=robot)
    calls=[]
    def reject(*args,**kwargs):
        calls.append((args,kwargs));raise ValueError('interpolated_collision')
    monkeypatch.setattr(module,'_repair_window',reject)
    xs=np.zeros((5,32))
    with pytest.raises(ValueError,match='interpolated_collision'):
        propose(model,{},xs,{},0,5,{'_reference':xs.copy(),'joint_deg':20},float('inf'))
    assert calls[0][0][-1]['_elastic_recovery']
    np.testing.assert_array_equal(xs,np.zeros((5,32)))


def test_float32_frozen_join_does_not_make_projection_infeasible():
    from dex_frax.v8.self_collision import _frozen_velocity_steps
    step=np.full(3,np.deg2rad(12.))
    actual=step+np.array([8.7e-8,8.7e-8,3e-6])
    corrected=_frozen_velocity_steps(step,actual,np.array([True,False,True]))
    assert corrected[0]==actual[0]  # Existing immutable export roundoff only.
    assert corrected[1]==step[1]   # A movable interval must keep the speed cap.
    assert corrected[2]==step[2]   # A real violation is not exempted.
    np.testing.assert_array_equal(step,np.full(3,np.deg2rad(12.)))


def test_overlap_seam_options_reach_shoulder_search(monkeypatch):
    import dex_frax.shoulder_recovery as module
    monkeypatch.setattr(module,'active_arm_support',lambda *args:False)
    visited=[];robot=SimpleNamespace(sides={'Left':{'arm':list(range(3,10))}})
    model=SimpleNamespace(r=robot,pairs=[(0,1)],arms={'Left':{0}},
        update=lambda row,rot:visited.append(int(row[0])),distances=lambda:np.array([.01]))
    monkeypatch.setattr(module,'_repair_window',lambda model,t,x,*args:(x,{}))
    xs=np.zeros((10,32));xs[:,0]=np.arange(10);t={'root_r':np.tile(np.eye(3),(10,1,1))}
    base={'_reference':xs.copy(),'joint_deg':10.}
    propose(model,t,xs,{},2,8,base,float('inf'))
    assert visited==[4,5]
    visited.clear()
    propose(model,t,xs,{},2,8,dict(base,freeze_start=False,freeze_end=False),float('inf'))
    assert visited==[2,3,4,5,6,7]


def test_successful_original_route_never_runs_continuous_seed(monkeypatch):
    import dex_frax.shoulder_recovery as module
    import dex_frax.trajectory_recovery as continuous
    monkeypatch.setattr(module,'active_arm_support',lambda *args:False)
    robot=SimpleNamespace(sides={},m=object());model=SimpleNamespace(r=robot)
    monkeypatch.setattr(module,'_repair_window',lambda *args:(args[2].copy(),{'original':True}))
    def forbidden(*args):raise AssertionError('successful route must stay fast')
    monkeypatch.setattr(continuous,'seed',forbidden)
    x=np.zeros((5,32));out,detail=propose(model,{},x,{},0,5,{'_reference':x,'joint_deg':10},float('inf'))
    np.testing.assert_array_equal(out,x)
    assert len(detail['validation_attempts'])==1


def test_frozen_numerical_cap_excess_only_allowed_within_actual_motor_limit():
    from dex_frax.v8.self_collision import _frozen_velocity_steps
    internal=np.array([.25,.25,.25]);actual=np.array([.30,.30,.40]);physical=np.full(3,.35)
    out=_frozen_velocity_steps(internal,actual,np.array([True,False,True]),physical,2e-6)
    np.testing.assert_array_equal(out,[.30,.25,.25])


def test_failed_temporal_seed_tries_other_directions_with_same_reference(monkeypatch):
    import dex_frax.shoulder_recovery as module
    import dex_frax.trajectory_recovery as continuous
    monkeypatch.setattr(module,'active_arm_support',lambda *args:False)
    robot=SimpleNamespace(sides={},m=object());model=SimpleNamespace(r=robot)
    calls=[];seen=[];x=np.zeros((5,32));reference=x.copy()
    def validate(*args):
        calls.append(1)
        if len(calls)<4:raise ValueError('nonlinear_collision_or_floor_clearance')
        return args[2].copy(),{}
    def seed(model,t,x,w,a,b,options,deadline):
        assert options['_reference'] is reference and options['joint_deg']==20.
        seen.append(options['_initial_shoulder_offset']);return x.copy(),{}
    monkeypatch.setattr(module,'_repair_window',validate)
    monkeypatch.setattr(continuous,'seed',seed)
    out,report=propose(model,{},x,{},0,5,{'_reference':reference,'joint_deg':20.},float('inf'))
    assert seen==[(0.,0.,0.),(1.,0.,0.),(-1.,0.,0.)]
    assert report['validation_attempts'][-1]['status']=='passed'
    np.testing.assert_array_equal(out,x)


def test_failed_hand_pose_budget_continues_other_validation_routes(monkeypatch):
    import dex_frax.shoulder_recovery as module
    monkeypatch.setattr(module,'active_arm_support',lambda *args:False)
    robot=SimpleNamespace(sides={});model=SimpleNamespace(r=robot);calls=[]
    def validate(*args):
        calls.append(args[-1])
        if len(calls)==1:raise ValueError('repair_hand_pose_budget')
        return args[2].copy(),{}
    monkeypatch.setattr(module,'_repair_window',validate)
    x=np.zeros((5,32));out,report=propose(model,{},x,{},0,5,{'_reference':x,'joint_deg':10.},float('inf'))
    assert len(calls)==2 and report['validation_attempts'][0]['reason']=='repair_hand_pose_budget'
    assert report['validation_attempts'][1]['status']=='passed'
    np.testing.assert_array_equal(out,x)
