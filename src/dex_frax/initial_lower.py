"""Route small lower-body collision sets to guarded local repair before arm IK."""
import time

import numpy as np
from osqp import OSQPException
from scipy.spatial.transform import Rotation

from .adaptive_refine import inspect, window_targets
from .refine_queue import jobs_from_masks
from .refine_routing import collision_groups, submodel
from .temporal_constraints import diagnostics, regressions
from .temporal_quality import assess
from .unified_repair import dense_geometry
from .v8.self_collision import CollisionModel, scan

MAX_WINDOWS = 4
MAX_AFFECTED_SECONDS = .8
MAX_EPISODE_SECONDS = .4
MAX_PENETRATION_M = .006
WINDOW_SECONDS = 30.


def classify(mask, minimum_distance_m, fps):
    """Use the same window grid as the refine queue."""
    jobs = jobs_from_masks([('leg_collision', None, mask)], len(mask))
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    longest = int(max(edges[1::2] - edges[::2], default=0))
    count = int(mask.sum())
    eligible = (len(jobs) <= MAX_WINDOWS and count <= round(MAX_AFFECTED_SECONDS * fps)
                and longest <= round(MAX_EPISODE_SECONDS * fps)
                and minimum_distance_m >= -MAX_PENETRATION_M)
    return eligible, jobs, dict(affected_frames=count, longest_episode_frames=longest,
                                windows=len(jobs), minimum_distance_m=float(minimum_distance_m))


def measure_route(model, rows, root_r, fps, subdivisions):
    report, _, frames = scan(model, rows, root_r, subdivisions)
    mask = np.zeros(len(rows), dtype=bool)
    mask[list(frames)] = True
    eligible, jobs, metrics = classify(mask, report['minimum_distance_m'], fps)
    metrics['native_collision_frames'] = report['native_colliding_frames']
    return eligible, jobs, metrics


def route(robot, targets, rows, cfg, collision_cfg=None):
    """Return a checked lower candidate, or None for full-sequence foot IK."""
    from .refine_pipeline import propose

    start = time.monotonic()
    model = CollisionModel(robot, dict(collision_cfg or {}, coverage='full'))
    owned = collision_groups(model)['leg_collision']
    lower = submodel(model, owned)
    rounded = rows.astype(np.float32).astype(float)
    rotation = Rotation.from_quat(Rotation.from_matrix(targets['root_r']).as_quat().astype(np.float32)).as_matrix()
    t = dict(targets, root_r=rotation)
    fps = t['fps']
    use_local, _, native = measure_route(lower, rounded, rotation, fps, 1)
    report = dict(selected='full_feet', native=native, trials=[])
    if not use_local:
        report.update(reason='native_lower_collision_load', elapsed_s=time.monotonic() - start)
        return None, report

    use_local, jobs, swept = measure_route(lower, rounded, rotation, fps, 8)
    report['swept'] = swept
    if not use_local:
        report.update(reason='swept_lower_collision_load', elapsed_s=time.monotonic() - start)
        return None, report

    current = rounded
    source = diagnostics(t)
    temporal = assess(current[:, 3:], fps, robot.names, source) if jobs else None
    tolerance = lower.cfg['penetration_tolerance_m']
    weights = dict(cfg, _initial_leg_repair=True, _temporal_repair_gain=2.,
                   _initial_leg_collision_cfg=collision_cfg)
    for job in jobs:
        a, b = job['start'], job['end']
        lo, hi = max(0, a - 1), min(len(rows), b + 1)
        window = window_targets(t, lo, hi, len(rows))
        old_d, old_floor = dense_geometry(robot, model, window, current[lo:hi])
        core = slice(8 * (job['core_start'] - lo), 8 * (job['core_end'] - lo))
        if not np.any(old_d[core, owned] < -tolerance):
            continue
        detail = dict(core_start=job['core_start'], core_end=job['core_end'])
        try:
            candidate, solver = propose(robot, t, current, rounded, weights, model, job, 'legs10',
                                        time.monotonic() + WINDOW_SECONDS)
            candidate = candidate.astype(np.float32).astype(float)
            new_d, new_floor = dense_geometry(robot, model, window, candidate[lo:hi])
            old_contact = inspect(robot, window, current[lo:hi])['contact']
            new_contact = inspect(robot, window, candidate[lo:hi])['contact']
            updated = assess(candidate[:, 3:], fps, robot.names, source)
            detail.update(minimum_before_m=float(old_d[core, owned].min()),
                          minimum_after_m=float(new_d[core, owned].min()),
                          solver=solver.get('classification'))
            if np.any(new_contact > np.maximum(.001, old_contact) + 1e-7):
                reason = 'foot_anchor_regression'
            elif np.any(new_floor < np.minimum(old_floor, -.001) - 2e-6):
                reason = 'dense_floor_regression'
            elif np.any(new_d < np.minimum(old_d, 0.) - 2e-6):
                reason = 'new_or_deeper_collision_sample'
            elif np.any((new_d[:, owned] < -tolerance) & (old_d[:, owned] >= -tolerance)):
                reason = 'new_lower_collision_sample'
            elif np.any(new_d[core, owned] < -tolerance):
                reason = 'lower_collision_remains'
            elif regressions(temporal, updated):
                reason = 'new_jitter'
            else:
                reason = 'kept'
                current, temporal = candidate, updated
        except (ValueError, RuntimeError, TimeoutError, OSQPException) as exc:
            reason = str(exc)
        detail['reason'] = reason
        report['trials'].append(detail)
        if reason != 'kept':
            report.update(reason='local_lower_not_cleared', elapsed_s=time.monotonic() - start)
            return None, report

    report.update(selected='local_feet', reason='small_lower_collision_set', elapsed_s=time.monotonic() - start)
    return current, report
