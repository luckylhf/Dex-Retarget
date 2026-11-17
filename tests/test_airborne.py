import copy
import json
import pickle

import numpy as np
import pytest

from dex_frax.airborne import airborne_mask, floating_report, preserve_targets


def test_source_flight_breaks_false_floating_runs_but_not_real_floating():
    clearance = np.full(80, .3)
    speed = np.zeros(80)
    flight = np.zeros(80, bool)
    flight[20:60] = True
    actual = floating_report(clearance, speed, 50., {'source_airborne': flight})
    assert actual['candidate_frames'] == 80
    assert actual['source_airborne_exempt_frames'] == 40
    assert actual['unexplained_frames'] == 40
    assert actual['longest_seconds'] == .4  # Two separate 0.4 s runs.
    legacy = floating_report(clearance, speed, 50., {})
    assert legacy['longest_seconds'] == 1.6
    assert not legacy['source_airborne_available']
    flight[:] = False
    assert floating_report(clearance, speed, 50., {'source_airborne': flight})['longest_seconds'] == 1.6
    assert not floating_report(np.zeros(80), speed, 50., {})['candidate_frames']
    assert not floating_report(clearance, np.ones(80), 50., {})['candidate_frames']


@pytest.mark.parametrize('value', [[True], [0., np.nan], [0, 2], [[0, 1]]])
def test_invalid_flight_evidence_cannot_bypass_checks(value):
    with pytest.raises(ValueError, match='source_airborne'):
        airborne_mask({'source_airborne': value}, 2)


def target_scene():
    n = 5
    raw = np.tile([.2, .1, .5], (n, 1))
    raw_r = np.tile(np.eye(3), (n, 1, 1))
    foot = dict(raw_foot_p=raw, raw_foot_r=raw_r,
                foot_target_p=np.zeros((n, 3)), foot_target_r=raw_r.copy(),
                contact=np.ones(n), phase=np.ones(n, int),
                foot_flat_confidence=np.ones(n), foot_roll_level_weight=np.ones(n),
                ground_contact_confidence=np.ones(n),
                anchors=np.zeros((n, 3)), pivots=np.zeros((n, 3)),
                heel=np.zeros((n, 3)), toe=np.zeros((n, 3)),
                segments=[dict(start=0, end=5, phase='sole', label=1)])
    body = dict(mode=np.ones(n, int), confidence=np.ones(n))
    return dict(root_p=raw.copy(), root_r=raw_r.copy(), fps=50.,
                source_airborne=np.array([False, True, True, True, False]),
                source_expected_contact=np.ones(n, bool),
                sides={'Left': foot, 'Right': copy.deepcopy(foot)}, body_contacts=[body])


def test_flight_retains_source_feet_and_releases_ground_tasks_only():
    t = target_scene()
    before = copy.deepcopy(t)
    preserve_targets(t)
    for side, foot in t['sides'].items():
        np.testing.assert_array_equal(foot['foot_target_p'][1:4], foot['raw_foot_p'][1:4])
        np.testing.assert_array_equal(foot['foot_target_r'][1:4], foot['raw_foot_r'][1:4])
        np.testing.assert_array_equal(foot['contact'], [1, 0, 0, 0, 1])
        np.testing.assert_array_equal(foot['phase'], [1, 0, 0, 0, 1])
        assert [(s['start'], s['end']) for s in foot['segments']] == [(0, 1), (4, 5)]
        for key in ('foot_flat_confidence', 'foot_roll_level_weight', 'ground_contact_confidence'):
            np.testing.assert_array_equal(foot[key], [1, 0, 0, 0, 1])
        for key, value in foot.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value[[0, 4]], before['sides'][side][key][[0, 4]])
    np.testing.assert_array_equal(t['body_contacts'][0]['mode'], [1, 0, 0, 0, 1])
    np.testing.assert_array_equal(t['body_contacts'][0]['confidence'], [1, 0, 0, 0, 1])
    np.testing.assert_array_equal(t['source_expected_contact'], [1, 0, 0, 0, 1])
    for key in ('root_p', 'root_r', 'source_airborne'):
        np.testing.assert_array_equal(t[key], before[key])


def test_retiming_keeps_flight_free_of_interpolated_ground_targets():
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation
    from dex_frax.retiming import apply

    t = target_scene()
    preserve_targets(t)
    t['root_r'] = Rotation.from_euler('z', np.array([0., .01, .8, .82, .83])[:, None]).as_matrix()
    robot = SimpleNamespace(sides={side: dict(hull=np.zeros((4, 3))) for side in t['sides']})
    out, _, report = apply(t, robot, 7.8, 7.8)
    assert report['changed']
    flight = out['source_airborne']
    assert flight.sum() > t['source_airborne'].sum()
    for foot in out['sides'].values():
        assert not foot['contact'][flight].any()
        np.testing.assert_array_equal(foot['foot_target_p'][flight], foot['raw_foot_p'][flight])
        np.testing.assert_array_equal(foot['foot_target_r'][flight], foot['raw_foot_r'][flight])


@pytest.mark.parametrize('source_flight', [False, True, None])
def test_export_gate_distinguishes_source_flight_and_conversion_floating(tmp_path, source_flight):
    from test_incremental_audit import export_neutral
    from dex_frax.v8.audit import audit
    from dex_frax.quality_pose import quality

    p, data, meta, dg = export_neutral(tmp_path, n=30)
    # A stationary robot well above the plane, identical output for all cases.
    data['root_pos'][:, 2] += 1.
    dg['root_target'][:, 2] += 1.
    if source_flight is not None:
        dg['source_airborne'] = np.full(30, source_flight)
    meta['support_constraint_failures'] = []
    p.write_bytes(pickle.dumps(data))
    p.with_suffix('.json').write_text(json.dumps(meta))
    np.savez_compressed(p.with_suffix('.diagnostics.npz'), **dg)
    check = audit(p, write=False)
    result = quality(p, meta, check)
    assert ('stationary_floating_over_half_second' in result['flags']) == (source_flight is not True)
    assert check['stationary_floating_longest_seconds'] == (0. if source_flight else .6)
    assert result['floating_audit']['source_airborne_exempt_frames'] == (30 if source_flight else 0)
    if source_flight:
        data['root_pos'][:, 2] -= 3.
        p.write_bytes(pickle.dumps(data))
        penetrating = quality(p, meta, audit(p, write=False))
        assert 'body_penetration_below_minus_2cm' in penetrating['flags']


@pytest.mark.parametrize('policy', ['v1', 'v2'])
def test_preparation_preserves_flight_at_takeoff_and_landing(policy):
    from pathlib import Path
    from dex_frax.v8 import convert
    from dex_frax.v8.bvh import load
    from dex_frax.v8.source import adapt
    from dex_frax.v8.robot import Robot
    from dex_frax.v8.contacts import prepare
    from dex_frax.v8.foot_contact_v2 import DEFAULTS
    from dex_frax.contact_transition import refine

    path = Path(__file__).resolve().parents[1]/'examples/bvh/nobody/nobody.bvh'
    motion = load(path, fps=50.)
    motion.local_p = np.repeat(motion.local_p[:1], 80, axis=0)
    motion.local_r = np.repeat(motion.local_r[:1], 80, axis=0)
    motion.local_p[25:55, 0, 1] += .4
    cfg = json.loads(Path(convert.__file__).with_name('config.json').read_text())['contact']
    cfg.update(policy=policy, v2=dict(DEFAULTS))
    robot = Robot()
    targets = prepare(adapt(motion, 'mocap'), robot, cfg)
    refine(targets, robot)
    flight = targets['source_airborne']
    assert flight[25:55].all() and not flight[:20].any() and not flight[60:].any()
    for foot in targets['sides'].values():
        assert foot['contact'][:20].any() and foot['contact'][60:].any()
        assert not foot['contact'][flight].any()
        np.testing.assert_array_equal(foot['foot_target_p'][flight], foot['raw_foot_p'][flight])
        np.testing.assert_array_equal(foot['foot_target_r'][flight], foot['raw_foot_r'][flight])
    assert not targets['source_expected_contact'][flight].any()
    assert all(not contact['mode'][flight].any() for contact in targets['body_contacts'])
