import json
import os
from pathlib import Path
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from dex_frax.kinematics import DexKinematics
from dex_frax.v8.robot import Robot
from dex_frax.v8.anatomy import configure
from dex_frax.v8 import convert
from dex_frax.ik import FrameBounds, make_residual, targets


@pytest.fixture(scope="module")
def robot():
    r = Robot()
    cfg = json.loads((Path(convert.__file__).parent / "config.json").read_text())
    configure(r, cfg["anatomy"])
    return r, cfg


def test_fk_random_poses(robot):
    r, _ = robot
    assert DexKinematics(r).verify(r, 24)["max_position_m"] < 1e-10


def test_qp_boxes_use_previous_frame_not_inner_step(robot):
    r, _ = robot
    model = DexKinematics(r)
    step = np.full(29, .02)
    con = FrameBounds(model, r.lower[3:], r.upper[3:], step, 50.)
    prev = np.zeros(32)
    q = prev.copy(); q[3:] = .019
    G, h = con.rows(q, {"previous": prev, "first": False})
    v = np.r_[np.zeros(3), np.full(29, .051)]
    # Only 0.001 rad remains, rather than another whole 0.02 rad.
    assert np.any(G @ v > h)


def test_residual_cost_and_gradient_against_v8(robot):
    path = os.environ.get("DEX_FRAX_TEST_BVH")
    if not path:
        pytest.skip("Set DEX_FRAX_TEST_BVH to the explicitly authorized mocap fixture")
    from dex_frax.v8.bvh import load
    from dex_frax.v8.source import adapt
    from dex_frax.v8.contacts import prepare
    from dex_frax.v8.solver import Objective
    r, cfg = robot
    motion = load(path, fps=50.)
    t = prepare(adapt(motion, "mocap"), r, cfg["contact"])
    model = DexKinematics(r)
    residual = make_residual(model, r, t, cfg["solver"])
    refs = targets(t)
    value_grad = jax.jit(jax.value_and_grad(lambda x, ref: .5 * jnp.sum(residual(x, ref)**2)))
    for i in (0, 50, min(900, len(t["hips"])-1)):
        x = np.r_[t["root_p"][i], np.zeros(29)]
        for g in r.sides.values():
            x[g["leg"][3]] = .3
            x[g["arm"][3]] = -.4
        # Nonzero temporal terms check the weighting and source root increment.
        prev = x + .001
        pred = x + .002
        ref = jax.tree.map(lambda a: a[i], refs)
        root_goal = prev[:3] + t["root_p"][i] - t["root_p"][max(i-1, 0)]
        ref.update(first=float(i == 0), previous=prev, predicted=pred, root_velocity_goal=root_goal)
        expected = Objective(r, t, i, cfg["solver"], prev if i else None, pred if i else None)
        e, J = expected.evaluate(x)
        value, gradient = value_grad(x, ref)
        np.testing.assert_allclose(value, .5 * e @ e, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(gradient, J.T @ e, rtol=1e-8, atol=1e-8)
        if i == 0:
            preference = 3.
            preferred = np.deg2rad(1.)
            enhanced = make_residual(model, r, t, {**cfg['solver'],
                'knee_extension_preference': preference, 'knee_preferred_flexion_rad': preferred})
            knees = np.array([g['leg'][3] for g in r.sides.values()])
            x[knees] = [-.05, .3]
            base_value, base_grad = value_grad(x, ref)
            new_value, new_grad = jax.value_and_grad(lambda q: .5*jnp.sum(enhanced(q, ref)**2))(x)
            penalty = np.minimum(x[knees]-preferred, 0.)
            np.testing.assert_allclose(new_value-base_value, .5*preference**2*np.sum(penalty**2), atol=1e-9)
            expected_grad = np.zeros(32);expected_grad[knees] = preference**2*penalty
            np.testing.assert_allclose(new_grad-base_grad, expected_grad, atol=1e-9)


def test_quaternion_training_roundtrip(tmp_path):
    import pickle
    from dex_frax.cli import export_training
    from scipy.spatial.transform import Rotation
    p = dict(dof_pos=np.zeros((3, 29), np.float32), root_pos=np.ones((3, 3), np.float32),
             root_rot=Rotation.from_euler("z", np.array([[.1], [.2], [.3]])).as_quat().astype(np.float32), joint_names=[str(i) for i in range(29)])
    src = tmp_path / "motion.pkl"; src.write_bytes(pickle.dumps(p))
    dest = tmp_path / "training.npz"
    export_training(src, dest, {"assets": {"mjcf/dex_29.xml": "test"}, "fingerprint": "test"})
    with np.load(dest, allow_pickle=False) as a:
        np.testing.assert_array_equal(a["body_quat_w"][:, 0, [1, 2, 3, 0]], p["root_rot"])
        np.testing.assert_array_equal(a["joint_pos"], p["dof_pos"])
