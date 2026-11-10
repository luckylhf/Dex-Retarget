"""Bounded recovery of failed V8 collision windows, without releasing support.

Normal successful V8 repairs remain byte-for-byte unchanged. Deeply penetrating
windows may need several feasible displacement steps before the final clearance
can be imposed. Intermediate recovery is relaxed; final checks are not.
"""
import time
import numpy as np
from .v8.self_collision import (CollisionModel, refine as baseline_refine, scan,
                                _repair_window)


class RecoveryModel(CollisionModel):
    def linear(self, columns):
        C, upper = super().linear(columns)
        # Limit demanded recovery per SQP step, not the final clearance.
        return C, np.maximum(upper, -0.008)


def refine(robot, t, xs, solver_cfg, config=None):
    start = time.perf_counter()
    out, report = baseline_refine(robot, t, xs, solver_cfg, config)
    failed = [r for r in report.get('windows', []) if r['status'] == 'unresolved'
              and r.get('reason') == 'collision_qp_primal infeasible']
    if not failed:
        return out, report
    # Extra attempts are fixed work, not an unbounded fallback or full-clip solve.
    from .adaptive_refine import policy
    cfg = policy(); max_windows = cfg.get('maximum_collision_recovery_windows', 2)
    max_frames = cfg.get('maximum_collision_recovery_frames', 100)
    used_windows = used_frames = 0; attempts = []
    model = RecoveryModel(robot, config)
    for old in failed:
        a, b = old['start'], old['end']
        detail = dict(start=a, end=b, sides=old['sides'], status='unresolved')
        try:
            if used_windows >= max_windows or used_frames+b-a > max_frames:
                raise ValueError('fast_collision_recovery_budget')
            used_windows += 1; used_frames += b-a
            from .arm_policy import enabled,collision_options
            options=dict(collision_options(10.),_reference=xs) if enabled(solver_cfg) else None
            if options is None:
                candidate, detail = _repair_window(model, t, out, solver_cfg, a, b, old['sides'])
            else:
                candidate, detail = _repair_window(model, t, out, solver_cfg, a, b, old['sides'],options=options)
            out = candidate
        except ValueError as exc:
            detail['reason'] = str(exc)
        attempts.append(detail)
    report['recovery'] = dict(policy='bounded_incremental_free_arm_v1', attempts=attempts,
        used_windows=used_windows, used_frames=used_frames, maximum_windows=max_windows,
        maximum_frames=max_frames, intermediate_recovery_m=.008,
        final_acceptance_unchanged=True, support_frames_immutable=True)
    report['after'] = scan(model, out, t['root_r'])[0]
    report['elapsed_s'] = time.perf_counter()-start
    return out, report
