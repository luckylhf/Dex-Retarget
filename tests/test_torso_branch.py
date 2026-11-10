import time
from types import SimpleNamespace
import numpy as np
import pytest
from dex_frax.torso_branch import suspect_frames, source_waist, structural_guard, recover
from dex_frax.v8.robot import Robot


def test_torso_trigger_requires_error_and_shoulder_limit():
    r=SimpleNamespace(sides={'Left':dict(arm=list(range(3,10)))},lower=np.full(32,-2.),upper=np.full(32,2.))
    x=np.zeros((3,32));x[0,3]=1.99;x[1,3]=1.99
    np.testing.assert_array_equal(suspect_frames(r,x,np.array([21.,1.,21.])),[True,False,False])


def test_source_torso_fit_uses_actual_model_preserving_root_legs_and_joins():
    r=Robot();n=50;x=np.zeros((n,32));x[:,2]=1.
    x[:,3:]=np.clip(x[:,3:],r.lower[3:]+.01,r.upper[3:]-.01)
    waist=np.array([3+r.names.index('waist_'+s+'_joint') for s in ('yaw','roll','pitch')])
    goal=x[0].copy();goal[waist]=[.2,.1,-.1];r.set(goal,np.eye(3))
    t=dict(fps=50.,root_r=np.tile(np.eye(3),(n,1,1)),torso_r=np.tile(r.rot(r.torso).copy(),(n,1,1)))
    out=source_waist(r,t,x,5,45,time.monotonic()+5.)
    fixed=np.setdiff1d(np.arange(32),waist)
    np.testing.assert_array_equal(out[:,fixed],x[:,fixed])
    np.testing.assert_array_equal(out[:7],x[:7]);np.testing.assert_array_equal(out[43:],x[43:])
    r.set(out[25],np.eye(3));np.testing.assert_allclose(r.rot(r.torso),t['torso_r'][25],atol=1e-6)
    with pytest.raises(TimeoutError):source_waist(r,t,x,5,45,time.monotonic()-1)


def test_recovery_clean_motion_does_not_search(monkeypatch):
    import dex_frax.torso_branch as module
    r=SimpleNamespace(sides={'Left':dict(arm=list(range(3,10)))},lower=np.full(32,-2.),upper=np.full(32,2.))
    x=np.zeros((30,32));monkeypatch.setattr(module,'torso_errors',lambda *args:np.zeros(30))
    out,report=recover(r,{'fps':50.},x,{})
    np.testing.assert_array_equal(x,out);assert report['kept']==0 and report['attempts']==[]


def test_structural_guard_does_not_trade_correct_waist_for_bad_frames(monkeypatch):
    import dex_frax.torso_branch as module
    r=SimpleNamespace(sides={'Left':dict(arm=list(range(3,10)),leg=list(range(10,16)))},
        lower=np.full(32,-3.),upper=np.full(32,3.),joint_step=lambda *args:np.ones(29),speed_limits=np.full(29,50.))
    before=np.zeros((10,32));after=before.copy();after[5,20]=.1
    monkeypatch.setattr(module,'torso_errors',lambda r,t,x:np.array([.1]*5+[20.]*5) if np.any(x) else np.array([0.]*5+[40.]*5))
    assert structural_guard(r,{'fps':50.,'sides':{}},before,after,{},0,10)=='torso_frame_regression'
    after=before.copy();after[5,10]=.1
    assert structural_guard(r,{'fps':50.,'sides':{}},before,after,{},0,10)=='root_or_legs_changed'
    after=before.copy();after[5,20]=1.1
    assert structural_guard(r,{'fps':50.,'sides':{}},before,after,{},0,10)=='speed_or_join'


def test_arm_search_deadline_aborts_inside_solver_residual():
    from dex_frax.v8.arms import ArmObjective
    r=SimpleNamespace(sides={'Left':dict(arm=list(range(7)))})
    obj=ArmObjective(r,{},0,'Left',np.zeros(7),{'_repair_deadline':time.monotonic()-1})
    with pytest.raises(TimeoutError,match='arm_branch_deadline'):obj.evaluate(np.zeros(7))


def test_rejected_branch_does_not_overwrite_original(monkeypatch):
    import json
    import dex_frax.torso_branch as module
    import dex_frax.v8.arms as arms
    r=SimpleNamespace(sides={'Left':dict(arm=list(range(3,10)))},lower=np.full(32,-2.),upper=np.full(32,2.))
    x=np.zeros((20,32));x[:,3]=1.99
    monkeypatch.setattr(module,'torso_errors',lambda *args:np.full(20,30.))
    monkeypatch.setattr(module,'source_waist',lambda r,t,x,*args:x+1.)
    monkeypatch.setattr(module,'inspect',lambda *args:{})
    def reject(*args,**kwargs):raise TimeoutError('arm_branch_deadline')
    monkeypatch.setattr(arms,'plan_and_refine',reject)
    out,report=recover(r,{'fps':50.,'sides':{}},x,{})
    np.testing.assert_array_equal(out,x)
    assert report['kept']==0 and report['attempts'][0]['reason']=='arm_branch_deadline'
    assert report['attempts'][0]['phase']=='branch_plan'
    json.dumps(report)  # Window indices must serialize in production metadata.


def test_subframe_floor_check_catches_clear_endpoints():
    from dex_frax.torso_branch import floor_samples
    class FloorRobot:
        surface_models={1:{'hull':np.zeros((1,3))}}
        def set(self,q,rotation):self.q=q
        def point(self,b):return np.array([0.,0.,(self.q[3]-.5)**2-.1])
        def rot(self,b):return np.eye(3)
    rows=np.zeros((2,32));rows[1,3]=1.
    values=floor_samples(FloorRobot(),{'root_r':np.tile(np.eye(3),(2,1,1))},rows)
    assert values[0,0]>0 and values[-1,0]>0
    assert values[4,0]==pytest.approx(-.1) and len(values)==9


def test_near_limit_flicker_keeps_sustained_torso_event():
    from dex_frax.torso_branch import sustained_suspects
    r=SimpleNamespace(sides={'Left':dict(arm=list(range(3,10)))},lower=np.full(32,-2.),upper=np.full(32,2.))
    x=np.zeros((10,32));x[:,3]=2.-np.deg2rad([4.9,5.1]*5)
    assert sustained_suspects(r,x,np.full(10,35.),50.).all()
    assert not sustained_suspects(r,x,np.array([35.,0.]*5),50.).any()
    x[:,3]=0.
    assert not sustained_suspects(r,x,np.full(10,35.),50.).any()
