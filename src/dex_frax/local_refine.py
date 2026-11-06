"""Selective offline pose fitting with fixed root/legs and window boundaries.

Only difficult upper-body windows enter the sparse pose/velocity solve. The
initial frax trajectory stays the reference elsewhere; there is no full-clip
arm branch search or output-only angle filter.
"""
import time
import numpy as np
from scipy.sparse import block_diag, diags, eye, kron, vstack
from scipy.spatial.transform import Rotation
from .v8.solver import Objective
from .v8.trajectory import _qp
from .v8.self_collision import windows


def upper_ids(robot):
    legs = {j for g in robot.sides.values() for j in g['leg']}
    return np.array([j for j in range(3, 32) if j not in legs])


def torso_errors(robot, t, xs):
    errors = []
    for i, x in enumerate(xs):
        robot.set(x, t['root_r'][i])
        errors.append(Rotation.from_matrix(robot.rot(robot.torso).T @ t['torso_r'][i]).magnitude())
    return np.rad2deg(errors)


def continuity(q, fps):
    """Measured finite differences, not invented hardware acceleration limits."""
    q = np.asarray(q, dtype=float)
    out = {}
    for order, name in ((1, 'velocity_rad_s'), (2, 'acceleration_rad_s2'), (3, 'jerk_rad_s3')):
        values = abs(np.diff(q, n=order, axis=0)) * fps**order
        out[name] = {k: float(np.percentile(values, p)) if values.size else 0.
                     for k, p in (('p95', 95), ('p99', 99), ('max', 100))}
    return out


def _floor(robot, t, xs, a, b):
    minimum = np.inf
    for i in range(a, b):
        robot.set(xs[i], t['root_r'][i])
        for body, surface in robot.surface_models.items():
            z = robot.point(body)[2] + np.min(np.asarray(surface['hull']) @ robot.rot(body)[2])
            minimum = min(minimum, z)
    return float(minimum)


def fit_window(robot, t, xs, cfg, a, b, iterations=12,ids_override=None,objective_factory=None,local_checks=False,allow_support=False,seed=None):
    ids = upper_ids(robot) if ids_override is None else np.asarray(ids_override)
    # Inferred active upper-body support requires a contact-aware whole-body
    # solve. Keep this fast path conservative and leave the final gate active.
    roots = set(robot.m.jnt_bodyid[robot.jids[ids-3]].tolist())
    def moves(body):
        while body:
            if body in roots:
                return True
            body = int(robot.m.body_parentid[body])
        return False
    for c in t.get('body_contacts', []):
        if not allow_support and moves(c['body']) and np.any(np.asarray(c['mode'])[a:b]):
            raise ValueError('active_upper_body_support')
    n, d = b-a, len(ids)
    if n < 5:
        raise ValueError('window_too_short')
    q = (xs if seed is None else seed)[a:b, ids].copy()
    # Offline second differences replace the causal temporal residual only
    # within this fit. Root/legs and all source targets remain fixed.
    objectives = [Objective(robot, t, i, cfg) if objective_factory is None else objective_factory(i) for i in range(a, b)]
    f = t['fps'] / 60.
    D1 = kron(diags([-np.ones(n-1), np.ones(n-1)], [0, 1], shape=(n-1, n)), eye(d), format='csc')
    D2 = kron(diags([np.ones(n-2), -2*np.ones(n-2), np.ones(n-2)], [0, 1, 2], shape=(n-2, n)), eye(d), format='csc') * (6. * f*f)
    temporal = vstack([D1 * (.18*f), D2], format='csc')
    step = robot.joint_step(t['fps'], cfg)[ids-3]
    velocity = diags(np.tile(1./step, n-1)) @ D1
    A = vstack([eye(n*d), velocity], format='csc')
    lo = np.broadcast_to(robot.lower[ids]+1e-7, q.shape).copy()
    hi = np.broadcast_to(robot.upper[ids]-1e-7, q.shape).copy()
    # Two immutable samples at either end preserve both joins and their
    # outside second differences. Clip endpoints follow the same convention.
    if a > 0: lo[:2] = hi[:2] = xs[a:a+2,ids]
    if b < len(xs): lo[-2:] = hi[-2:] = xs[b-2:b,ids]
    low = np.r_[lo.ravel(), -np.ones((n-1)*d)]
    high = np.r_[hi.ravel(), np.ones((n-1)*d)]

    def evaluate(z, jacobian=False):
        qs = z.reshape(n, d)
        res, jac = [], []
        for k, obj in enumerate(objectives):
            x = xs[a+k].copy(); x[ids] = qs[k]
            r, J = obj.evaluate(x)
            res.append(r)
            if jacobian:
                jac.append(J[:, ids])
        r = np.r_[np.concatenate(res), temporal @ z]
        return (r, vstack([block_diag(jac, format='csc'), temporal], format='csc')) if jacobian else r

    geometry=None
    if allow_support:
        from .local_geometry import Geometry
        geometry=Geometry(robot,t,xs,a,b,ids,moves)
    if geometry and geometry.evaluate(q.ravel())>1e-8:
        # Start from the feasible boundary reference when a branch proposal
        # violates floor/support; hard constraints cannot certify that seed.
        q=xs[a:b,ids].copy()
    if a>0:q[:2]=xs[a:a+2,ids]
    if b<len(xs):q[-2:]=xs[b-2:b,ids]
    z = q.ravel(); initial = evaluate(z); cost0 = cost = float(initial @ initial)+(geometry.evaluate(z) if geometry else 0.)
    accepted = 0
    for _ in range(iterations):
        res, J = evaluate(z, True)
        at = A @ z; dl, du = low-at, high-at
        dl[:z.size] = np.maximum(dl[:z.size], -.3)
        du[:z.size] = np.minimum(du[:z.size], .3)
        constraints=A
        if geometry:
            G,gl,gh=geometry.evaluate(z,linear=True)
            constraints=vstack([A,G],format='csc');dl=np.r_[dl,gl];du=np.r_[du,gh]
        delta, _ = _qp(J.T @ J + eye(z.size, format='csc')*.01,
                       J.T @ res, constraints, dl, du, np.zeros(z.size), scaling=10)
        moved = False
        for alpha in (1., .5, .25, .125):
            candidate = z + alpha*delta
            r = evaluate(candidate); new_cost = float(r @ r)+(geometry.evaluate(candidate) if geometry else 0.)
            if new_cost < cost-1e-7:
                improvement = cost-new_cost
                z, cost = candidate, new_cost
                accepted += 1; moved = True
                break
        if not moved or improvement < 1e-5*max(1., cost):
            break
    out = xs.copy()
    out[a:b, ids] = z.reshape(n, d)
    # Equality solver tolerance must never modify the fixed boundary samples.
    if a > 0: out[a:a+2] = xs[a:a+2]
    if b < len(xs): out[b-2:b] = xs[b-2:b]
    checked=out[max(0,a-1):min(len(out),b+1),ids] if local_checks else out[:,ids]
    if (np.max(abs(np.diff(checked, axis=0)) - step) > 1e-7 or
            np.any(out[:, ids] < robot.lower[ids]-1e-7) or
            np.any(out[:, ids] > robot.upper[ids]+1e-7)):
        raise ValueError('window_join_or_joint_bounds')
    old_floor, new_floor = _floor(robot, t, xs, a, b), _floor(robot, t, out, a, b)
    if new_floor < min(-.001, old_floor-1e-6):
        raise ValueError('window_ground_regression')
    if allow_support:
        for c in t.get('body_contacts',[]):
            if not moves(c['body']):continue
            body=c['body'];hull=np.asarray(c['robot_hull'])
            for i in range(a,b):
                mode=int(c['mode'][i])
                if not mode:continue
                robot.set(xs[i],t['root_r'][i]);local=hull[np.argmin(hull@robot.rot(body)[2])];before=robot.point(body,local).copy()
                robot.set(out[i],t['root_r'][i])
                if mode!=1:local=hull[np.argmin(hull@robot.rot(body)[2])]
                delta=robot.point(body,local)-before
                if np.max(abs(delta if mode==1 else delta[2:]))>.0005:raise ValueError('local_arm_support_changed')
    return out, dict(start=a, end=b, status='repaired', iterations=accepted,
                     initial_cost=cost0, cost=cost, minimum_floor_m=new_floor)


def split_windows(ranges, maximum, overlap):
    """Cover long difficulties with bounded overlapping solves, never gaps."""
    if maximum < 5 or not 2 <= overlap < maximum-2:
        raise ValueError('Invalid window size/overlap')
    out=[]
    for a,b in ranges:
        while b-a > maximum:
            out.append((a,a+maximum));a += maximum-overlap
        out.append((a,b))
    return out


def refine(robot, t, xs, cfg):
    from .adaptive_refine import policy
    budget=policy()
    start = time.perf_counter()
    ids = upper_ids(robot)
    before = torso_errors(robot, t, xs)
    trigger = before > 1.
    acceleration = np.max(abs(np.diff(xs[:, ids], n=2, axis=0))*t['fps']**2, axis=1)
    trigger[1:-1] |= acceleration > 400.
    ranges = windows(set(np.flatnonzero(trigger).tolist()), len(xs), round(.4*t['fps']))
    ranges = split_windows(ranges,round(4.*t['fps']),max(3,round(.8*t['fps'])))
    out = xs.copy(); reports = []
    for index,(a, b) in enumerate(ranges):
        try:
            if index>=budget.get('maximum_pose_windows',4):
                raise ValueError('fast_pose_work_budget')
            out, report = fit_window(robot, t, out, cfg, a, b,budget.get('pose_iterations',6))
        except ValueError as exc:
            report = dict(start=a, end=b, status='unchanged', reason=str(exc))
        reports.append(report)
    return out, dict(policy='overlapping_upper_body_sqp_v2', windows=reports,
                     work_budget= dict(maximum_windows=budget.get('maximum_pose_windows',4),
                                       iterations_per_window=budget.get('pose_iterations',6),hard_wall_time_limit=False),
                     torso_trigger_deg=1., acceleration_trigger_rad_s2=400.,
                     acceleration_threshold_is_hardware_limit=False,
                     torso_max_before_deg=float(before.max()),
                     torso_max_after_deg=float(torso_errors(robot, t, out).max()),
                     continuity_before=continuity(xs[:, 3:], t['fps']),
                     continuity_after=continuity(out[:, 3:], t['fps']),
                     elapsed_s=time.perf_counter()-start)
