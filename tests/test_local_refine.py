import json
from pathlib import Path
import numpy as np
import pytest
from dex_frax import local_refine as local
from dex_frax.v8.robot import Robot
from dex_frax.v8.anatomy import configure
from dex_frax.v8 import convert


@pytest.fixture
def scene():
    robot = Robot()
    cfg = json.loads((Path(convert.__file__).parent/'config.json').read_text())
    configure(robot, cfg['anatomy'])
    x = np.tile(np.r_[0., 0., .9, np.zeros(29)], (11, 1))
    for g in robot.sides.values():
        x[:, g['arm'][3]] = -.4
    return robot, dict(fps=50., root_r=np.tile(np.eye(3), (11, 1, 1)), body_contacts=[]), x, cfg['solver']


def test_offline_fit_preserves_joins_and_limits(scene, monkeypatch):
    robot, t, x, cfg = scene
    target = x.copy()
    j = robot.sides['Left']['arm'][0]
    target[5, j] = 1.5  # An unreachable one-frame request must be shared in time.
    class Objective:
        def __init__(self, robot, t, i, cfg):
            self.i = i
        def evaluate(self, q):
            return 10*(q-target[self.i]), 10*np.eye(32)
    monkeypatch.setattr(local, 'Objective', Objective)
    monkeypatch.setattr(local, '_floor', lambda *args: 0.)
    out, report = local.fit_window(robot, t, x, cfg, 1, 10)
    assert report['cost'] < report['initial_cost']
    assert 0. < out[5, j] < target[5, j]
    np.testing.assert_array_equal(out[:3], x[:3])
    np.testing.assert_array_equal(out[8:], x[8:])
    fixed = [k for k in range(32) if k not in local.upper_ids(robot)]
    np.testing.assert_array_equal(out[:, fixed], x[:, fixed])
    assert np.max(abs(np.diff(out[:, 3:], axis=0))-robot.joint_step(50., cfg)) < 1e-7


def test_active_upper_support_refuses_fit(scene):
    robot, t, x, cfg = scene
    t['body_contacts'] = [dict(body=robot.sides['Right']['tcp'], mode=np.ones(len(x)))]
    with pytest.raises(ValueError, match='active_upper_body_support'):
        local.fit_window(robot, t, x, cfg, 0, len(x))


def test_easy_clip_bypasses_and_report_is_json(scene, monkeypatch):
    robot, t, x, cfg = scene
    monkeypatch.setattr(local, 'torso_errors', lambda *args: np.zeros(len(x)))
    out, report = local.refine(robot, t, x, cfg)
    np.testing.assert_array_equal(out, x)
    assert report['windows'] == []
    json.dumps(report)


def test_rejected_window_is_unchanged(scene, monkeypatch):
    robot, t, x, cfg = scene
    monkeypatch.setattr(local, 'torso_errors', lambda *args: np.full(len(x), 2.))
    def fail(*args):
        raise ValueError('test_infeasible')
    monkeypatch.setattr(local, 'fit_window', fail)
    out, report = local.refine(robot, t, x, cfg)
    np.testing.assert_array_equal(out, x)
    assert report['windows'][0]['status'] == 'unchanged'
    json.dumps(report)


def test_continuity_uses_actual_fps():
    report = local.continuity((np.arange(6.)**2)[:, None], 50.)
    assert report['acceleration_rad_s2']['max'] == 5000.
    assert report['jerk_rad_s3']['max'] == 0.


def test_long_windows_are_covered_with_overlap():
    result=local.split_windows([(21,251)],200,40)
    assert result==[(21,221),(181,251)]
    covered=np.zeros(300,dtype=int)
    for a,b in result:
        assert b-a<=200
        covered[a:b]+=1
    assert np.all(covered[21:251]>=1)
    assert np.all(covered[181:221]==2)


def test_clip_endpoint_can_improve_without_an_external_join(scene,monkeypatch):
    robot,t,x,cfg=scene;j=robot.sides['Left']['arm'][0]
    target=x.copy();target[:,j]=.1
    class Objective:
        def __init__(self,robot,t,i,cfg):self.i=i
        def evaluate(self,q):return 10*(q-target[self.i]),10*np.eye(32)
    monkeypatch.setattr(local,'Objective',Objective)
    monkeypatch.setattr(local,'_floor',lambda *args:0.)
    out,_=local.fit_window(robot,t,x,cfg,0,len(x))
    assert out[0,j]>.05 and out[-1,j]>.05
