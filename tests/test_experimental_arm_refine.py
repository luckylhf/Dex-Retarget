import numpy as np
import pytest
from types import SimpleNamespace
from dex_frax import experimental_arm_refine as e


def robot():
    return SimpleNamespace(sides={'Left':{'arm':list(range(3,10))}},lower=np.full(32,-2.),upper=np.full(32,2.),speed_limits=np.full(29,10.))


@pytest.mark.parametrize('tier',[10.,20.,30.])
def test_tiers_share_shoulder_cap_and_keep_distal_neighborhood(tier):
    r=robot();ids=np.arange(3,10);ref=np.zeros((3,7));lo,hi=e.limits(r,ids,ref,tier)
    assert np.allclose(np.rad2deg(hi[:,:3]),tier)
    assert np.allclose(np.rad2deg(lo[:,:3]),-tier)
    assert np.allclose(np.rad2deg(hi[:,3:]),10.)
    r.lower[8:10]=[-.1,-.12];r.upper[8:10]=[.08,.14]
    lo,hi=e.limits(r,ids,ref,tier)
    assert np.allclose(lo[:,5:],[-.1,-.12])
    assert np.allclose(hi[:,5:],[.08,.14])


@pytest.fixture
def guard(monkeypatch):
    r=robot();base=np.zeros((3,32));target=dict(fps=50.,root_r=np.tile(np.eye(3),(3,1,1)))
    monkeypatch.setattr(e,'features',lambda *args:np.zeros((3,6)))
    monkeypatch.setattr(e,'scan',lambda *args:(dict(collision_samples=0,minimum_distance_m=.004),None,None))
    monkeypatch.setattr(e,'_arm_bodies',lambda *args:[])
    r.set=lambda *a:None
    def check(candidate):return e.regression_guard(r,target,base,candidate,'Left',np.zeros((3,6)),None,.003)
    return base,check


@pytest.mark.parametrize('column',[0,1,2,12,20,31])
def test_any_non_arm_change_is_rejected(guard,column):
    base,check=guard;candidate=base.copy();candidate[1,column]=1e-8
    assert check(candidate)[1]=='non_arm_changed'


def test_joint_limit_and_speed_are_hard(guard):
    base,check=guard;candidate=base.copy();candidate[:,3]=2.01
    assert check(candidate)[1]=='joint_limits'
    candidate=base.copy();candidate[1,3]=.3
    assert check(candidate)[1]=='speed'


def test_palm_does_not_trade_away_for_arm_shape(guard,monkeypatch):
    base,check=guard
    errors=np.zeros((3,6));errors[:,2]=.3
    monkeypatch.setattr(e,'features',lambda *args:errors)
    assert check(base.copy())[1]=='palm_p95_regression'


def test_dense_collision_rejects_even_native_safe_candidate(guard,monkeypatch):
    base,check=guard
    monkeypatch.setattr(e,'scan',lambda *args:(dict(collision_samples=1,minimum_distance_m=-.001),None,None))
    assert check(base.copy())[1]=='dense_collision_regression'


def test_positive_clearance_reduction_without_collision_is_allowed(guard,monkeypatch):
    base,check=guard
    monkeypatch.setattr(e,'scan',lambda *args:(dict(collision_samples=0,minimum_distance_m=.002),None,None))
    assert check(base.copy())[0] is True


def test_safe_arm_candidate_can_pass_guard(guard):
    base,check=guard;candidate=base.copy();candidate[:,3]=.01
    assert check(candidate)[0]


def test_direction_samples_do_not_alias_reused_fk_buffers():
    from scipy.spatial.transform import Rotation
    rotations=Rotation.from_euler('z',np.array([0.,40.,80.])[:,None],degrees=True).as_matrix()
    matrix=np.eye(3)
    class R:
        sides={'Left':dict(shoulder='s',elbow='e',wrist='w',tcp='t',forearm='f')}
        def set(self,row,rot):matrix[:]=rotations[int(row[0])]
        def rot(self,body):return matrix
        def point(self,body):return np.array({'s':[0.,0.,0.],'e':[1.,0.,0.],'w':[1.,1.,0.]}[body])
    targets=dict(root_r=rotations,sides={'Left':dict(upper=np.tile([1.,0.,0.],(3,1)),lower=np.tile([0.,1.,0.],(3,1)),
        hand_forward=-rotations[:,:,2],hand_width=rotations[:,:,0],elbow_axis=-rotations[:,:,1],forearm_width=rotations[:,:,0])})
    values=e.features(R(),targets,np.arange(3)[:,None],'Left')
    assert np.max(values)<1e-5


def test_shape_and_elbow_envelopes_do_not_open_a_new_bad_direction():
    errors=np.array([[25.,22.,1.,2.,28.,10.],[15.,14.,1.,2.,40.,15.]])
    assert np.allclose(e.direction_bound(errors,0),[25.02,15.02])
    assert np.allclose(e.direction_bound(errors,4),[28.2,40.])
