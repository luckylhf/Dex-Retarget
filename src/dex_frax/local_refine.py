"""Constrained window fitting with task-selected joints and checked seams.

Only difficult upper-body windows enter the sparse pose/velocity solve. The
initial frax trajectory stays the reference elsewhere; there is no full-clip
arm branch search or output-only angle filter.
"""
import time
import json
from pathlib import Path
import numpy as np
from scipy.sparse import block_diag, csc_matrix, diags, eye, hstack, kron, vstack
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


def _qp_with_collision_slack(P, gradient, constraints, lower, upper, soft_rows):
    n, count = len(gradient), len(soft_rows)
    columns = csc_matrix((-np.ones(count), (soft_rows, np.arange(count))),
                         shape=(constraints.shape[0], count))
    # Slack only changes the selected collision inequalities. Joint, speed,
    # contact and floor rows keep their original bounds.
    constraints = vstack([hstack([constraints, columns]),
                          hstack([csc_matrix((count, n)), eye(count)])], format='csc')
    objective = block_diag([P, .01 * eye(count)], format='csc')
    result, report = _qp(objective, np.r_[gradient, np.full(count, 8.)], constraints,
                         np.r_[lower, np.zeros(count)], np.r_[upper, np.full(count, np.inf)],
                         np.r_[np.zeros(n), np.maximum(-upper[soft_rows], 0.)],
                         scaling=10, require_optimal=False, feasible_tolerance=1e-6)
    report.update(owned_elastic_rows=count, owned_collision_slack_max_m=float(result[n:].max()))
    return result[:n], report


def fit_window(robot, t, xs, cfg, a, b, iterations=12,ids_override=None,objective_factory=None,local_checks=False,allow_support=False,seed=None):
    ids = upper_ids(robot) if ids_override is None else np.asarray(ids_override)
    # Inferred active upper-body support requires a contact-aware whole-body
    # solve. Keep this fast path conservative and leave the final gate active.
    roots = set(robot.m.jnt_bodyid[robot.jids[ids[ids>=3]-3]].tolist())
    def moves(body):
        if np.any(ids<3):return True
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
    D2 = kron(diags([np.ones(n-2), -2*np.ones(n-2), np.ones(n-2)], [0, 1, 2], shape=(n-2, n)), eye(d), format='csc') * (6. * f*f*float(cfg.get('_temporal_repair_gain',1.)))
    temporal = vstack([D1 * (.18*f), D2], format='csc')
    step = np.r_[np.full(3,np.inf),robot.joint_step(t['fps'], cfg)][ids]
    velocity = diags(np.tile(1./step, n-1)) @ D1
    A = vstack([eye(n*d), velocity], format='csc')
    lo = np.broadcast_to(robot.lower[ids]+1e-7, q.shape).copy()
    hi = np.broadcast_to(robot.upper[ids]-1e-7, q.shape).copy()
    if 'repair_lower_bounds' in t:lo=np.maximum(lo,np.asarray(t['repair_lower_bounds'])[a:b,ids])
    for g in robot.sides.values():
        matches=np.flatnonzero(ids==g['leg'][3])
        if len(matches):
            k=int(matches[0]);lo[:,k]=np.maximum(lo[:,k],np.where(xs[a:b,g['leg'][3]]>=0.,0.,-np.inf))
    # Two immutable samples at either end preserve both joins and their
    # outside second differences. Clip endpoints follow the same convention.
    fixed_frames=[]
    restore=bool(cfg.get('_restore_geometry',False))
    initial_leg=restore and bool(cfg.get('_initial_leg_repair'))
    if restore and not initial_leg:
        from .adaptive_refine import inspect,window_targets
        boundary=inspect(robot,window_targets(t,a,b,len(xs)),xs[a:b])
        passing=(boundary['contact']<=.001)&(boundary['floor']>=-.001)
        from .v8.self_collision import CollisionModel
        check=CollisionModel(robot,{'coverage':'full'})
        for j in set([0,1,n-2,n-1]):
            check.update(xs[a+j],t['root_r'][a+j]);passing[j]&=check.distances().min()>=-check.cfg['penetration_tolerance_m']
    else:passing=np.ones(n,bool)
    if a>0:fixed_frames.extend(j for j in (0,1) if passing[j])
    if b<len(xs):fixed_frames.extend(j for j in (n-2,n-1) if passing[j])
    if fixed_frames:lo[fixed_frames]=hi[fixed_frames]=xs[a:b,ids][fixed_frames]
    if cfg.get('_safety_joint_deg'):
        cap=np.where(ids<3,np.inf,np.deg2rad(cfg['_safety_joint_deg']))
        reference=np.asarray(cfg.get('_safety_reference',xs))[a:b,ids]
        lo=np.maximum(lo,reference-cap);hi=np.minimum(hi,reference+cap)
    physical_step=np.r_[np.full(3,np.inf),robot.speed_limits/t['fps']][ids]
    for j,neighbor in ((0,a-1),(n-1,b)):
        if 0<=neighbor<len(xs):
            lo[j]=np.maximum(lo[j],xs[neighbor,ids]-physical_step)
            hi[j]=np.minimum(hi[j],xs[neighbor,ids]+physical_step)
    velocity_tolerance=1e-4/t['fps']
    speed_bounds=np.ones((n-1,d))
    # Numerical step caps stabilize new solves; they are not motor limits.
    # Preserve immutable already-valid intervals, including outputs from the
    # arm fitter that legitimately exceed the numerical cap. Movable rows
    # retain that cap, and actual motor violations are never exempted.
    for edge,enabled in ((0,a>0),(-1,b<len(xs))):
        if enabled and ((edge==0 and {0,1}.issubset(fixed_frames)) or (edge==-1 and {n-2,n-1}.issubset(fixed_frames))):
            actual=abs(np.diff(xs[a:b,ids],axis=0))[edge]
            rounding=(actual>step)&(actual<=physical_step+velocity_tolerance)
            speed_bounds[edge,rounding]=actual[rounding]/step[rounding]
    low = np.r_[lo.ravel(), -speed_bounds.ravel()]
    high = np.r_[hi.ravel(), speed_bounds.ravel()]
    temporal_corridor=None
    if cfg.get('_temporal_repair_gain',0.):
        from .temporal_constraints import corridor
        bands,bounds,temporal_corridor=corridor(robot,t,xs,a,b,ids)
        A=vstack([A,bands],format='csc');low=np.r_[low,-bounds];high=np.r_[high,bounds]

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
        geometry=Geometry(robot,t,xs,a,b,ids,moves,restore=restore,fixed_frames=fixed_frames,
                          initial_leg_repair=initial_leg,collision_cfg=cfg.get('_initial_leg_collision_cfg'))
    if geometry and not restore and geometry.evaluate(q.ravel())>1e-8:
        # Start from the feasible boundary reference when a branch proposal
        # violates floor/support; hard constraints cannot certify that seed.
        q=xs[a:b,ids].copy()
    if fixed_frames:q[fixed_frames]=xs[a:b,ids][fixed_frames]
    z = q.ravel()
    seed_projection = None
    if seed is not None:
        # A different IK branch can be far from the immutable joins. The
        # per-iteration 0.3-rad trust region is not a feasibility solver:
        # first find a globally speed/bounds-feasible seed in this window.
        at = A @ z
        if np.any(at < low-1e-8) or np.any(at > high+1e-8):
            z, seed_projection = _qp(eye(z.size,format='csc'),-z,A,low,high,
                xs[a:b,ids].ravel(),scaling=10)
    if np.any(low>high+1e-10):raise ValueError('local_joint_corridor_empty; expand fixed-reference tier or context')
    progress=cfg.get('_local_fit_progress');saved=(progress or {}).get('sqp',{});start_iteration=0
    if saved.get('shape')==[n,d] and saved.get('ids')==ids.tolist():
        resumed=np.asarray(saved['q'],float)
        if resumed.shape==(n,d) and np.isfinite(resumed).all():z=resumed.ravel();start_iteration=int(saved.get('iteration',0))
    def save_iteration(k):
        if progress is not None:progress['sqp']=dict(shape=[n,d],ids=ids.tolist(),q=z.reshape(n,d).tolist(),iteration=k)
    initial = evaluate(z); cost0 = cost = float(initial @ initial)+(geometry.evaluate(z) if geometry else 0.)
    accepted = 0;qp_reports=[]
    already_clear=bool(restore and geometry and geometry.last.get('safety_goal_met',False))
    for iteration in range(start_iteration,start_iteration if already_clear else iterations):
        save_iteration(iteration)
        if cfg.get('_repair_deadline') is not None and time.monotonic()>=cfg['_repair_deadline']:
            if initial_leg and accepted:break
            raise TimeoutError('local_geometry_deadline')
        res, J = evaluate(z, True)
        at = A @ z; dl, du = low-at, high-at
        dl[:z.size] = np.maximum(dl[:z.size], -.3)
        du[:z.size] = np.minimum(du[:z.size], .3)
        constraints=A
        if geometry:
            G,gl,gh=geometry.evaluate(z,linear=True)
            constraints=vstack([A,G],format='csc');dl=np.r_[dl,gl];du=np.r_[du,gh]
        H=J.T@J+eye(z.size,format='csc')*.01;gradient=J.T@res
        objective_scale=max(1.,float(abs(H.diagonal()).max()))
        try:
            if initial_leg and geometry and geometry.elastic_rows:
                delta, qp_report = _qp_with_collision_slack(H/objective_scale,gradient/objective_scale,
                    constraints,dl,du,np.asarray(geometry.elastic_rows)+A.shape[0])
            else:
                delta, qp_report = _qp(H/objective_scale,
                    gradient/objective_scale, constraints, dl, du, np.zeros(z.size), scaling=10,
                    require_optimal=not (restore or bool(cfg.get('_temporal_repair_gain',0.))),
                    feasible_tolerance=1e-6 if restore else 1e-7)
        except (ValueError,RuntimeError):
            if initial_leg and accepted:break
            raise
        qp_report['objective_scale']=objective_scale
        qp_reports.append(qp_report)
        moved = False
        for alpha in (1., .5, .25, .125, .0625, .03125, .015625):
            candidate = z + alpha*delta
            r = evaluate(candidate); geometric_cost=geometry.evaluate(candidate) if geometry else 0.
            if geometry and cfg.get('_temporal_repair_gain',0.) and not restore and geometric_cost>1e-6:continue
            if initial_leg and geometry and (geometry.last['floor_regression_m']>2e-6 or
                                              geometry.last['contact_regression_m']>1e-7):
                continue
            new_cost = float(r @ r)+geometric_cost
            old_linear=max(float(np.max(low-A@z)),float(np.max(A@z-high)),0.)
            new_linear=max(float(np.max(low-A@candidate)),float(np.max(A@candidate-high)),0.)
            restoring=(bool(cfg.get('_temporal_repair_gain',0.)) and old_linear>1e-7 and new_linear<old_linear-1e-8)
            if new_cost < cost-1e-7 or restoring:
                improvement = cost-new_cost
                z, cost = candidate, new_cost
                accepted += 1; moved = True
                break
        save_iteration(iteration+1)
        if restore and moved and geometry.last.get('safety_goal_met',False):break
        linear_feasible=(not cfg.get('_temporal_repair_gain',0.) or (np.all(A@z>=low-1e-7) and np.all(A@z<=high+1e-7)))
        if not moved or (linear_feasible and improvement < 1e-5*max(1., cost)):
            break
    out = xs.copy()
    out[a:b, ids] = z.reshape(n, d)
    # Equality solver tolerance must never modify the fixed boundary samples.
    if fixed_frames:out[np.asarray(fixed_frames)+a]=xs[np.asarray(fixed_frames)+a]
    checked=out[max(0,a-1):min(len(out),b+1),ids] if local_checks else out[:,ids]
    joint_tolerance=np.deg2rad(json.loads(Path(__file__).with_name('quality_pose.json').read_text())['joint_limit_tolerance_deg'])
    if (np.max(abs(np.diff(checked, axis=0)) - physical_step) > velocity_tolerance or
            np.any(out[:, ids] < robot.lower[ids]-joint_tolerance) or
            np.any(out[:, ids] > robot.upper[ids]+joint_tolerance)):
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
                from .support_quality import preserve_contact
                if not mode or not preserve_contact(c,i):continue
                robot.set(xs[i],t['root_r'][i]);local=hull[np.argmin(hull@robot.rot(body)[2])];before=robot.point(body,local).copy()
                robot.set(out[i],t['root_r'][i])
                if mode!=1:local=hull[np.argmin(hull@robot.rot(body)[2])]
                delta=robot.point(body,local)-before
                if np.max(abs(delta if mode==1 else delta[2:]))>.0005:raise ValueError('local_arm_support_changed')
    return out, dict(start=a, end=b, status='repaired', iterations=accepted,
                     initial_cost=cost0, cost=cost, minimum_floor_m=new_floor,seed_projection=seed_projection,
                     qp_diagnostics=qp_reports,temporal_corridor=temporal_corridor,
                     classification='nonlinear_checked_progress' if accepted else 'no_feasible_descent',
                     boundary_fixed_frames=sorted(fixed_frames),resumed_from_iteration=start_iteration,geometry=(geometry.last if geometry else None))


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
