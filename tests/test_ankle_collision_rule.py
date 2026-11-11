"""Real Dex geometry: bounded same-leg exclusion, retained other collisions."""
import itertools

import mujoco
import numpy as np
import pytest

from dex_frax.v8.robot import Robot
from dex_frax.v8.self_collision import CollisionModel, scan
from dex_frax.refine_routing import collision_groups, movable_submodel, submodel
from dex_frax.frax_foot_constraints import FootGeometry


@pytest.fixture(scope='module')
def model():
    return CollisionModel(Robot(), {'coverage': 'full'})


def pose():
    return np.r_[0., 0., 1., np.zeros(29)]


def ankle_columns(model, side):
    return model.r.sides[side]['leg'][4:6]


def raw_distances(model):
    return np.array([mujoco.mj_geomDistance(model.m, model.configuration.data,
                                          int(a), int(b), model.cfg['detection_m'], None)
                     for a, b in model.pairs])


def test_both_sides_include_all_limit_corners_and_float32_endpoints(model):
    for side in model.limit.ankle_rule.sides:
        assert len(side['indices']) > 0
        columns = ankle_columns(model, side['side'])
        limits = side['limits']
        for angles in itertools.product(*[(lo, 0., hi) for lo, hi in limits]):
            for dtype in (np.float64, np.float32):
                x = pose(); x[columns] = np.array(angles, dtype=dtype)
                model.update(x, np.eye(3))
                assert model.limit.excluded_pairs(model.configuration)[side['indices']].all()
                np.testing.assert_array_equal(model.distances()[side['indices']], model.cfg['detection_m'])


def test_each_axis_outside_either_bound_restores_raw_detection_per_side(model):
    for side_index, side in enumerate(model.limit.ankle_rule.sides):
        columns = ankle_columns(model, side['side'])
        other = model.limit.ankle_rule.sides[1-side_index]
        for axis, endpoint in itertools.product(range(2), range(2)):
            x = pose()
            x[columns[axis]] = side['limits'][axis, endpoint] + (-1 if endpoint == 0 else 1)*1e-5
            model.update(x, np.eye(3))
            excluded = model.limit.excluded_pairs(model.configuration)
            assert not excluded[side['indices']].any()
            assert excluded[other['indices']].all()
            np.testing.assert_allclose(model.distances()[side['indices']], raw_distances(model)[side['indices']], atol=1e-12)


def test_nonfinite_angles_do_not_receive_exclusion(model):
    for side in model.limit.ankle_rule.sides:
        for value in (np.nan, np.inf, -np.inf):
            qpos = model.configuration.data.qpos.copy()
            qpos[side['qadr'][0]] = value
            assert not model.limit.ankle_rule.excluded(qpos)[side['indices']].any()


def test_known_false_proxy_overlap_is_removed_from_scan_and_solver_rows(model):
    x = pose()
    x[ankle_columns(model, 'Left')] = [-.586134672164917, .0007436046143993735]
    model.update(x, np.eye(3))
    pair = next(i for i, names in enumerate(model.names)
                if set(names) == {'retarget_foot_l_toe', 'retarget_shin_l_shaft'})
    assert raw_distances(model)[pair] < -model.cfg['penetration_tolerance_m']
    assert model.distances()[pair] == model.cfg['detection_m']
    rows = model.limit.compute_qp_inequalities(model.configuration, 1.)
    assert np.isposinf(rows.h[pair])
    np.testing.assert_array_equal(rows.G[pair], 0.)
    _, events, _ = scan(model, np.tile(x, (2, 1)), np.tile(np.eye(3), (2, 1, 1)), 8)
    assert all(pair not in event['pairs'] for event in events)
    x[ankle_columns(model, 'Left')[0]] = model.limit.ankle_rule.sides[0]['limits'][0, 0] - .01
    model.update(x, np.eye(3))
    raw = raw_distances(model)[pair]
    assert raw < 0.
    assert model.distances()[pair] == raw
    assert np.isfinite(model.limit.compute_qp_inequalities(model.configuration, 1.).h[pair])


def test_cross_leg_and_other_distances_remain_raw_and_scan_keeps_crossing(model):
    x = pose()
    x[3:15] = [-1.180467963218689, -.32467925548553467, -.055863335728645325,
               .8735106587409973, -.20397108793258667, .0928465947508812,
               -1.0703601837158203, .1650853008031845, .19595235586166382,
               1.1469261646270752, -.33886197209358215, .002749197417870164]
    model.update(x, np.eye(3))
    excluded = model.limit.excluded_pairs(model.configuration)
    measured = model.distances(); raw = raw_distances(model)
    np.testing.assert_allclose(measured[~excluded], raw[~excluded], atol=1e-12)
    pair = next(i for i, names in enumerate(model.names)
                if set(names) == {'ankle_roll_l_link_visual', 'ankle_roll_r_link_visual'})
    assert not excluded[pair]
    assert measured[pair] < -model.cfg['penetration_tolerance_m']
    report, events, _ = scan(model, np.tile(x, (2, 1)), np.tile(np.eye(3), (2, 1, 1)), 8)
    assert any(pair in event['pairs'] for event in events)
    assert report['conditional_exclusions']['policy'] == 'same_leg_foot_shin_within_ankle_limits_v1'


def test_submodels_and_initial_foot_qp_apply_the_same_rule(model):
    indices = collision_groups(model)['leg_collision']
    routed = [submodel(model, indices),
              movable_submodel(model, sum([g['leg'] for g in model.r.sides.values()], []))]
    for part in routed:
        part.update(pose(), np.eye(3))
        excluded = part.limit.excluded_pairs(part.configuration)
        assert excluded.any()
        rows = part.limit.compute_qp_inequalities(part.configuration, 1.)
        assert np.isposinf(rows.h[excluded]).all()
        np.testing.assert_array_equal(rows.G[excluded], 0.)
    initial = FootGeometry(object(), model.r, 50.)
    x = pose(); x[ankle_columns(model, 'Left')] = [-.586134672164917, .0007436046143993735]
    G, h = initial.measure(x, np.eye(3))
    excluded = initial.geometry.limit.excluded_pairs(initial.geometry.configuration)
    assert excluded.any()
    np.testing.assert_array_equal(G[excluded], 0.)
    assert (h[excluded] > 0.).all()
