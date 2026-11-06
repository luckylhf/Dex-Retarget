import copy
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from dex_frax.v8.arms import ArmObjective,plan_and_refine
from dex_frax.arm_tracking import acceptable,track,SearchEvents


@pytest.fixture(scope='module')
def scene():
    path=Path(__file__).parent/'upstream/test_support_fast.py'
    spec=importlib.util.spec_from_file_location('arm_tracking_fixture',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    r,t,x,cfg=m.sample.__wrapped__();n=40
    def expand(v):
        if isinstance(v,np.ndarray) and len(v)==len(x):return np.repeat(v[:1],n,axis=0)
        if isinstance(v,dict):return {k:expand(a) for k,a in v.items()}
        return copy.deepcopy(v)
    t=expand(t);x=np.repeat(x[:1],n,axis=0)
    return r,t,x,dict(cfg,arm_event_tracking=True)


def test_stable_pose_tracks_without_new_solver_and_keeps_distal_axes(scene):
    r,t,x,cfg=scene;side='Left';ids=r.sides[side]['arm'];q=x[0,ids]
    obj=ArmObjective(r,t,0,side,x[0],cfg)
    result,calls,kind=track(obj,[q],r.lower[ids]+1e-6,r.upper[ids]-1e-6)
    assert calls==0 and kind=='reuse';np.testing.assert_array_equal(result.x,q)
    assert acceptable(obj,result.x,q)


def test_straight_state_does_not_force_search_on_every_knot(scene):
    r,t,x,cfg=scene;t=copy.deepcopy(t);t['sides']['Left']['signed_elbow_bend'][:]=0.
    events=SearchEvents();reasons=[]
    # Test event selection with a valid constant state; repeated geometric
    # difficulty has a cooldown instead of unconditional near-straight search.
    for k in range(8):
        obj=ArmObjective(r,t,k,'Left',x[k],cfg)
        reasons.append(events.reason(obj,x[k,r.sides['Left']['arm']],0.,k,False))
    assert reasons[0]=='initial';assert reasons[1:4]==['continuation']*3


def test_event_tracking_preserves_pose_limits_speeds_and_body(scene):
    r,t,x,cfg=scene;out,report=plan_and_refine(r,t,x,cfg)
    arms=np.array(sum([g['arm'] for g in r.sides.values()],[]));outside=[i for i in range(32) if i not in arms]
    np.testing.assert_array_equal(out[:,outside],x[:,outside])
    for side,g in r.sides.items():
        ids=g['arm'];q=out[:,ids]
        assert report[side]['reused_frames']>0
        assert np.all(q>=r.lower[ids]-1e-8) and np.all(q<=r.upper[ids]+1e-8)
        assert np.all(abs(np.diff(q,axis=0))<=r.joint_step(t['fps'],cfg)[np.array(ids)-3]+1e-8)
        for i in range(len(x)):
            assert acceptable(ArmObjective(r,t,i,side,x[i],cfg),q[i],x[i,ids])


def test_proximal_preference_jacobian_and_distal_goals(scene):
    r,t,x,cfg=scene;ids=r.sides['Left']['arm'];q=x[0,ids].copy()+.01;prior=q-.02
    obj=ArmObjective(r,t,0,'Left',x[0],cfg,prior,.12);value,J=obj.evaluate(q)
    eps=1e-6
    numeric=np.column_stack([(obj.fun(q+np.eye(7)[j]*eps)-obj.fun(q-np.eye(7)[j]*eps))/(2*eps) for j in range(7)])
    np.testing.assert_allclose(J,numeric,atol=3e-5,rtol=1e-4)
    baseline=ArmObjective(r,t,0,'Left',x[0],dict(cfg,arm_event_tracking=False),prior,.12).fun(q)
    end=len(r.sides['Left']['ground_hulls']);at=slice(-end-7,-end)
    np.testing.assert_allclose(value[at][:3],baseline[at][:3]*.4)
    np.testing.assert_array_equal(value[at][3:],baseline[at][3:])


def test_local_arm_qp_preserves_material_contact_during_fit(scene):
    from dex_frax.local_refine import fit_window
    r,t,x,cfg=scene;t=copy.deepcopy(t);ids=np.array(r.sides['Left']['arm'])
    body=r.sides['Left']['forearm'];hull=r.surface_models[body]['hull'];n=len(x)
    r.set(x[0],t['root_r'][0]);local=hull[np.argmin(hull@r.rot(body)[2])].copy();point=r.point(body,local).copy()
    t['body_contacts']=[dict(body=body,robot_hull=hull,mode=np.ones(n,int))]
    desired=x[:,ids]+.03
    class Objective:
        def __init__(self,i):self.i=i
        def evaluate(self,row):
            J=np.zeros((7,32));J[:,ids]=np.eye(7)*10
            return (row[ids]-desired[self.i])*10,J
    out,report=fit_window(r,t,x,cfg,0,n,iterations=12,ids_override=ids,
        objective_factory=Objective,allow_support=True,local_checks=True)
    for i,row in enumerate(out):
        r.set(row,t['root_r'][i]);assert np.max(abs(r.point(body,local)-point))<=.0005
    assert report['status']=='repaired'
    assert np.max(abs(out[:,ids]-x[:,ids]))>1e-4
