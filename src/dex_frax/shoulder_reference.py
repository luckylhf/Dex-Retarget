"""Deterministic, full-clip shoulder coordinates before the existing arm IK.

The two Euler representations describe the same target orientation. They are
resolved once with full temporal context, then projected to the original joint
and speed bounds. This is a soft IK reference, never an accepted trajectory.
"""
import time
import warnings
import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation
from osqp import OSQP
from .v8.math3d import frame_yz, unit

POLICY = 'source_shoulder_frame_continuous_coordinates_v1'


def continuous_coordinates(rotations, lower, upper, step):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Gimbal lock detected')
        first = Rotation.from_matrix(rotations).as_euler('YXZ')
    second = first.copy()
    second[:, [0, 2]] += np.pi
    second[:, 1] = np.pi-second[:, 1]
    raw = np.stack([first, second], axis=1)
    # Choose the equivalent turn closest to the finite physical interval.
    center = (lower+upper)/2
    raw += 2*np.pi*np.rint((center-raw)/(2*np.pi))
    bounded = np.clip(raw, lower+1e-6, upper-1e-6)
    target = rotations[:, None]
    actual = Rotation.from_euler('YXZ', bounded.reshape(-1, 3)).as_matrix().reshape(-1, 2, 3, 3)
    emission = np.sum((actual-target)**2, axis=(2, 3))
    emission += .0025*np.sum(((bounded-center)/(upper-lower))**2, axis=2)
    costs = emission[0].copy(); previous = np.zeros((len(raw), 2), int)
    for i in range(1, len(raw)):
        change = (bounded[i][None, :, :]-bounded[i-1][:, None, :])/step
        transition = .01*np.sum(change**2, axis=2)
        candidates = costs[:, None]+transition
        previous[i] = np.argmin(candidates, axis=0)
        costs = emission[i]+candidates[previous[i], np.arange(2)]
    selected = np.zeros(len(raw), int); selected[-1] = np.argmin(costs)
    for i in range(len(raw)-1, 0, -1): selected[i-1] = previous[i, selected[i]]
    reference = bounded[np.arange(len(raw)), selected]
    # A joint-space projection smooths branch coordinates under the actual
    # speed bounds; source geometric tasks remain authoritative in Frax.
    n = len(reference)
    identity = sparse.eye(n, format='csc')
    d1 = sparse.diags([-np.ones(n-1), np.ones(n-1)], [0, 1], shape=(n-1, n), format='csc')
    d2 = sparse.diags([np.ones(n-2), -2*np.ones(n-2), np.ones(n-2)], [0, 1, 2], shape=(n-2, n), format='csc') if n > 2 else sparse.csc_matrix((0, n))
    P = sparse.triu(identity+.05*d1.T@d1+.25*d2.T@d2, format='csc')
    A = sparse.vstack([identity, d1], format='csc')
    output = reference.copy()
    for j in range(3):
        solver = OSQP()
        solver.setup(P=P, q=-reference[:, j], A=A,
                     l=np.r_[np.full(n, lower[j]+1e-6), np.full(n-1, -step[j])],
                     u=np.r_[np.full(n, upper[j]-1e-6), np.full(n-1, step[j])],
                     verbose=False, eps_abs=1e-7, eps_rel=1e-7, max_iter=20000, polishing=True)
        result = solver.solve()
        if result.info.status_val not in (1, 2):
            raise ValueError('shoulder_reference_projection_'+result.info.status)
        output[:, j] = result.x
    if np.any(abs(np.diff(output, axis=0)) > step+2e-6):
        raise ValueError('shoulder_reference_speed_projection_failed')
    return output, dict(coordinate_switches=int(np.count_nonzero(np.diff(selected))),
                        maximum_projection_change_rad=float(abs(output-reference).max()))


def prepare(robot, targets, weights):
    begin = time.perf_counter()
    saved = robot.d.qpos.copy()
    rows = np.asarray(targets['priority_pose']); n = len(rows)
    if n < 2: return targets, dict(policy=POLICY, used=False, reason='too_short')
    references = []; details = {}
    try:
        zero = np.zeros(32)
        robot.set(zero, np.eye(3))
        bases = {side: frame_yz(np.array([0., 1., 0.]),
                   -unit(robot.rot(g['shoulder']).T@(robot.point(g['elbow'])-robot.point(g['shoulder']))))
                 for side, g in robot.sides.items()}
        all_arms = sum([g['arm'] for g in robot.sides.values()], [])
        rest = {side: [] for side in robot.sides}
        for i, row in enumerate(rows):
            q = row.copy(); q[all_arms] = 0.
            robot.set(q, targets['root_r'][i])
            for side, g in robot.sides.items(): rest[side].append(robot.rot(g['shoulder']).copy())
        for side, g in robot.sides.items():
            values = targets['sides'][side]
            desired = frame_yz(-values['elbow_axis'], -values['upper'])@bases[side].T
            local = np.swapaxes(np.asarray(rest[side]), 1, 2)@desired
            ids = np.asarray(g['arm'][:3])
            ref, detail = continuous_coordinates(local, robot.lower[ids], robot.upper[ids],
                                  robot.joint_step(targets['fps'], weights)[ids-3])
            references.append(ref); details[side] = detail
    finally:
        robot.d.qpos[:] = saved
        import mujoco
        mujoco.mj_forward(robot.m, robot.d)
        robot._fk_qpos = None
    out = dict(targets, shoulder_reference=np.concatenate(references, axis=1))
    return out, dict(policy=POLICY, used=True, preparation_s=time.perf_counter()-begin,
                     frames=n, sides=details, scope='soft shoulder reference; original collision and speed constraints decide final IK')
