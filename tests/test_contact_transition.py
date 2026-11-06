import copy
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from dex_frax.contact_transition import refine


def scene():
    hull = np.array([[x, y, -.05] for x in (-.1, .1) for y in (-.05, .05)])
    r = Rotation.from_euler('y', np.array([10.]*5+[0.]*7)[:, None], degrees=True).as_matrix()
    v = dict(foot_target_r=r, foot_target_p=np.zeros((12, 3)),
             anchors=np.tile([.2, .1, 0.], (12, 1)), pivots=np.tile([0., 0., -.05], (12, 1)),
             contact=np.ones(12), segments=[dict(start=0, end=5, phase='toe'), dict(start=5, end=12, phase='sole')])
    t = dict(fps=50., sides={'Right': v})
    return t, SimpleNamespace(sides={'Right': dict(hull=hull)})


def test_easing_keeps_anchor_floor_and_endpoints():
    t, r = scene(); original = copy.deepcopy(t)
    report = refine(t, r); v=t['sides']['Right']; old=original['sides']['Right']
    assert len(report['windows']) == 1
    np.testing.assert_array_equal(v['anchors'], old['anchors'])
    np.testing.assert_array_equal(v['contact'], old['contact'])
    np.testing.assert_array_equal(v['foot_target_r'][:5], old['foot_target_r'][:5])
    np.testing.assert_allclose(v['foot_target_r'][9:], old['foot_target_r'][9:], atol=1e-15)
    angle = Rotation.from_matrix(v['foot_target_r'][4].T @ v['foot_target_r'][5]).magnitude()
    assert angle < np.deg2rad(1.)
    points = v['foot_target_p'][5:10]+np.einsum('nij,nj->ni',v['foot_target_r'][5:10],v['pivots'][5:10])
    np.testing.assert_allclose(points[:, :2], v['anchors'][5:10, :2], atol=1e-15)
    floor = v['foot_target_p'][5:10, 2]+np.min(v['foot_target_r'][5:10,2,:] @ r.sides['Right']['hull'].T,axis=1)
    np.testing.assert_allclose(floor, 0., atol=1e-15)


def test_swing_gap_is_not_bridged():
    t, r = scene();t['sides']['Right']['segments'][0]['end'] = 4
    before=copy.deepcopy(t)
    assert refine(t,r)['windows'] == []
    np.testing.assert_array_equal(t['sides']['Right']['foot_target_r'], before['sides']['Right']['foot_target_r'])
