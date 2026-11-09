"""Check a fast trajectory and dispatch existing V8 solvers only when needed.

Local candidates keep original joins and are checked over the entire clip.
Failed local geometry repairs escalate only when final acceptance is at risk.
Neither a solver return nor a repaired-window label implies final acceptance.
"""
import copy
import json
import time
from pathlib import Path
import numpy as np
from .v8.self_collision import windows
from .v8.support_fast import constrain_support
from .v8.arms import plan_and_refine
from .v8.contact_solver import refine as refine_body


def policy():
    return json.loads(Path(__file__).with_name('adaptive_policy.json').read_text())


def sustained(mask, count):
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros_like(mask)
    edges = np.diff(np.r_[False, mask, False].astype(int))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if b-a >= count:
            out[a:b] = True
    return out


def angular(a, b):
    return float(np.rad2deg(np.arccos(np.clip(np.dot(a, b), -1., 1.))))


def inspect(robot, t, xs):
    n = len(xs)
    contact, floor, body_floor = np.zeros(n), np.full(n, np.inf), np.full(n, np.inf)
    foot_floor = np.full(n, np.inf)
    errors = {s: {k: np.zeros(n) for k in ('upper', 'lower', 'hand_forward', 'hand_width', 'elbow_axis')}
              for s in robot.sides}
    feet = {g['foot'] for g in robot.sides.values()}
    for i, x in enumerate(xs):
        robot.set(x, t['root_r'][i])
        for b, surface in robot.surface_models.items():
            height = robot.point(b)[2]+np.min(np.asarray(surface['hull']) @ robot.rot(b)[2])
            floor[i] = min(floor[i], height)
            if b not in feet:
                body_floor[i] = min(body_floor[i], height)
            else:
                foot_floor[i] = min(foot_floor[i], height)
        for side, g in robot.sides.items():
            v = t['sides'][side]
            if v['contact'][i] > 0:
                error = robot.point(g['foot'], v['pivots'][i])-v['anchors'][i]
                error[2] = robot.point(g['foot'])[2]+np.min(g['hull'] @ robot.rot(g['foot'])[2])
                contact[i] = max(contact[i], np.linalg.norm(error))
            for a, b, key in (('shoulder', 'elbow', 'upper'), ('elbow', 'wrist', 'lower')):
                d = robot.point(g[b])-robot.point(g[a]);d /= max(np.linalg.norm(d), 1e-12)
                errors[side][key][i] = angular(d, v[key][i])
            for b, axis, key in ((g['tcp'], [0.,0.,-1.], 'hand_forward'),
                                 (g['tcp'], [1.,0.,0.], 'hand_width'),
                                 (g['shoulder'], [0.,-1.,0.], 'elbow_axis')):
                errors[side][key][i] = angular(robot.rot(b) @ axis, v[key][i])
    return dict(contact=contact, floor=floor, foot_floor=foot_floor, body_floor=body_floor, errors=errors)


def pose_mask(errors, fps, cfg):
    mask = np.zeros(len(next(iter(next(iter(errors.values())).values()))), dtype=bool)
    count = max(2, round(cfg['pose_duration_s']*fps))
    for values in errors.values():
        for key, value in values.items():
            threshold = cfg['elbow_axis_deg'] if key == 'elbow_axis' else (
                cfg['arm_direction_deg'] if key in ('upper', 'lower') else cfg['hand_axis_deg'])
            mask |= sustained(value > threshold, count)
    return mask


def window_targets(t, a, b, n):
    """Slice already prepared full-clip targets; never re-estimate contacts."""
    def visit(value, key=None):
        if isinstance(value, np.ndarray):
            if key in ('robot_hull','hull','world_origin','local_axis'):
                return value.copy()
            return value[a:b].copy() if value.ndim and len(value) == n else value.copy()
        if isinstance(value, dict):
            return {k: visit(v,k) for k, v in value.items()}
        if isinstance(value, list):
            return [visit(v) for v in value]
        return value
    out = visit(t)
    for v in out['sides'].values():
        v['segments'] = [{**s, 'start': max(s['start'], a)-a, 'end': min(s['end'], b)-a}
                         for s in v.get('segments', []) if s['start'] < b and s['end'] > a]
    return out


class MinimumChangeObjective:
    """Clear geometry defects without re-optimizing the source pose objective."""
    def __init__(self, reference):
        self.reference = np.asarray(reference).copy()
        self.jacobian = np.eye(len(reference))

    def evaluate(self, x):
        return x-self.reference, self.jacobian


def _stage(robot, t, xs, weights, stage):
    # V8 functions attach diagnostics to Robot. Window-local arrays must not
    # masquerade as full-clip diagnostics in the exporter.
    attributes = ('support_fast_report', 'support_branch_relaxations', 'body_contact_qp_summary')
    saved = {k: copy.deepcopy(getattr(robot, k)) for k in attributes if hasattr(robot, k)}
    saved_lower = robot.lower.copy()
    try:
        # Keep an already nonnegative knee trajectory on its flexion branch.
        # This is a temporary repair corridor, not a changed physical limit.
        for g in robot.sides.values():
            j = g['leg'][3]
            if np.all(xs[:,j] >= 0.): robot.lower[j] = max(robot.lower[j],0.)
        fn = {'support': constrain_support, 'arms': plan_and_refine, 'body': refine_body}[stage]
        target = t
        if stage == 'body':
            target = dict(t)
            corridor = np.broadcast_to(robot.lower, xs.shape).copy()
            for g in robot.sides.values():
                j = g['leg'][3]
                corridor[xs[:,j] >= 0.,j] = max(robot.lower[j],0.)
            target['repair_lower_bounds'] = corridor
        if stage == 'body':
            def objective_factory(robot, target, i, weights, previous, predicted):
                return MinimumChangeObjective(xs[i])
            result, report = fn(robot, target, xs, weights, objective_factory=objective_factory)
        else:
            result, report = fn(robot, target, xs, weights)
        return result, report
    finally:
        robot.lower[:] = saved_lower
        for k in attributes:
            if k in saved:
                setattr(robot, k, saved[k])
            elif hasattr(robot, k):
                delattr(robot, k)


def check_candidate(robot, t, before, after, weights, cfg, original_metrics):
    if not np.isfinite(after).all():
        raise ValueError('nonfinite_candidate')
    if np.any(after[:,3:] < robot.lower[3:]-1e-7) or np.any(after[:,3:] > robot.upper[3:]+1e-7):
        raise ValueError('joint_limit_regression')
    if np.max(abs(np.diff(after[:,3:],axis=0))-robot.joint_step(t['fps'],weights)) > 1e-7:
        raise ValueError('join_or_velocity_failure')
    knees = [g['leg'][3] for g in robot.sides.values()]
    if np.any((before[:,knees] >= 0.) & (after[:,knees] < -1e-7)):
        raise ValueError('new_knee_hyperextension')
    metrics = inspect(robot,t,after)
    limit = cfg['geometry_acceptance_m']
    if np.any(metrics['contact'] > np.maximum(limit, original_metrics['contact'])+1e-7):
        raise ValueError('contact_regression')
    if np.any(metrics['floor'] < np.minimum(-limit, original_metrics['floor'])-1e-7):
        raise ValueError('floor_regression')
    return metrics


def summary(metrics, t, cfg):
    return dict(max_contact_mm=float(np.max(metrics['contact'])*1000),
                minimum_floor_mm=float(np.min(metrics['floor'])*1000),
                sustained_arm_bad_frames=int(pose_mask(metrics['errors'],t['fps'],cfg).sum()))


def bounded_window(a, b, score, budget, padding):
    """Spend the existing frame budget on defects, shrinking padding first.

    When all defects cannot fit, choose the contiguous region with the greatest
    residual mass. Two immutable samples are reserved at internal joins.
    """
    if b-a <= budget:
        return a, b
    if budget < 5:
        return a, b  # Caller records budget exhaustion without dispatch.
    active = np.flatnonzero(np.asarray(score)[a:b] > 0)+a
    if not len(active):
        return a, b
    lo, hi = int(active[0]), int(active[-1])+1
    if hi-lo+4 <= budget:
        extra = min(padding, (budget-(hi-lo))//2)
        return max(a,lo-extra), min(b,hi+extra)
    # Scoring only the movable interior avoids spending budget on frozen ends.
    width = budget-4
    values = np.asarray(score)[a+2:b-2]
    sums = np.convolve(values, np.ones(width), mode='valid')
    start = a+int(np.argmax(sums))
    return start, start+budget


def refine(robot, t, xs, weights, config=None):
    cfg = policy() if config is None else config
    start = time.perf_counter(); out = xs.copy(); records = []
    initial = inspect(robot,t,out); current = initial
    padding = round(cfg['window_padding_s']*t['fps'])
    frame_budget=max(cfg.get('minimum_v8_frame_budget',100),round(len(xs)*cfg.get('maximum_v8_frame_fraction',.25)))
    used_frames=0;used_windows=0
    # Preserve the fast path for clean clips; each stage sees the prior result.
    stages = ('body', 'support', 'arms') if np.any(current['body_floor'] < -cfg['geometry_acceptance_m']) else ('support', 'arms', 'body')
    for stage in stages:
        if stage == 'support':
            foot_floor = current.get('foot_floor', current['floor'])
            score = np.maximum(current['contact']-cfg['contact_repair_m'],0.) + np.maximum(-foot_floor-cfg['contact_repair_m'],0.)
            mask = score > 0
        elif stage == 'arms':
            mask = pose_mask(current['errors'],t['fps'],cfg)
            score = mask.astype(float)
        else:
            score = np.maximum(-current['body_floor']-cfg['contact_repair_m'],0.)
            mask = current['body_floor'] < -cfg['geometry_acceptance_m']
            for c in t.get('body_contacts', []):
                mask |= np.asarray(c['mode']) > 0
        # Inferred support alone must not merge distant clearance failures
        # into one large solve. Spend budget on the deepest actual defects.
        routing_mask = score > 0 if stage == 'body' and np.any(score > 0) else mask
        ranges = windows(np.flatnonzero(routing_mask).tolist(),len(out),padding)
        if stage == 'body':
            ranges.sort(key=lambda ab: (-float(np.max(score[ab[0]:ab[1]])),
                                       -float(np.sum(score[ab[0]:ab[1]]))))
        for original_a,original_b in ranges:
            cap = min(round(cfg['maximum_local_window_s']*t['fps']),frame_budget-used_frames)
            a,b = bounded_window(original_a,original_b,score,cap,padding)
            begin = time.perf_counter(); baseline = out.copy()
            try:
                if b-a > round(cfg['maximum_local_window_s']*t['fps']):
                    raise ValueError('window_exceeds_local_budget')
                # Arm-only branch planning cannot release inferred body support.
                if stage == 'arms' and any(np.any(c['mode'][a:b]) for c in t.get('body_contacts', [])):
                    raise ValueError('active_body_contact')
                if used_windows >= cfg.get('maximum_v8_windows',3) or used_frames+b-a > frame_budget:
                    raise ValueError('fast_v8_work_budget')
                used_windows+=1;used_frames+=b-a
                target = window_targets(t,a,b,len(out))
                fitted, detail = _stage(robot,target,out[a:b],weights,stage)
                candidate = out.copy();candidate[a:b] = fitted
                if a > 0: candidate[a:a+2] = out[a:a+2]
                if b < len(out): candidate[b-2:b] = out[b-2:b]
                checked = check_candidate(robot,t,out,candidate,weights,cfg,current)
                if stage == 'support' and np.max(checked['contact'][a:b]) >= np.max(current['contact'][a:b])-1e-7 and np.min(checked['floor'][a:b]) <= np.min(current['floor'][a:b])+1e-7:
                    raise ValueError('no_geometry_improvement')
                out,current = candidate,checked
                record = dict(stage=stage,start=a,end=b,status='accepted',solver_report=detail)
            except ValueError as exc:
                out = baseline
                record = dict(stage=stage,start=a,end=b,status='unchanged',reason=str(exc))
            record['original_range'] = [original_a,original_b]
            record['elapsed_s'] = time.perf_counter()-begin; records.append(record)
    # An unsuccessful local fix must not silently give up on a failing clip.
    # These full-clip stages keep both endpoints' temporal context intact.
    needed = []
    if np.any(current['contact'] > cfg['geometry_acceptance_m']) or np.any(current['floor'] < -cfg['geometry_acceptance_m']):
        needed.append('support')
    if pose_mask(current['errors'],t['fps'],cfg).any(): needed.append('arms')
    if np.any(current['body_floor'] < -cfg['geometry_acceptance_m']) or any(r['stage']=='body' and r['status']=='unchanged' for r in records): needed.append('body')
    for stage in needed:
        begin=time.perf_counter()
        try:
            if not cfg.get('allow_full_clip_fallback',False):
                raise ValueError('fast_mode_full_clip_disabled_use_v8')
            if stage == 'arms' and any(np.any(c['mode']) for c in t.get('body_contacts', [])):
                raise ValueError('active_body_contact_requires_coupled_pose_search')
            candidate, detail = _stage(robot,t,out,weights,stage)
            checked = check_candidate(robot,t,out,candidate,weights,cfg,current)
            out,current = candidate,checked
            record=dict(stage=stage,scope='full_clip',status='accepted',solver_report=detail)
        except ValueError as exc:
            record=dict(stage=stage,scope='full_clip',status='unchanged',reason=str(exc))
        record['elapsed_s']=time.perf_counter()-begin;records.append(record)
    return out,dict(policy=cfg,repairs=records,before=summary(initial,t,cfg),after=summary(current,t,cfg),
                    work_budget=dict(frame_budget=frame_budget,used_frames=used_frames,used_windows=used_windows,
                                     hard_wall_time_limit=False),
                    elapsed_s=time.perf_counter()-start)
