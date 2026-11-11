"""Bounded hand/leg cooperation through the existing window SQP.

This module proposes candidates only. The normal dense preservation guard and
independent float32 export audit retain sole authority to adopt a trajectory.
Other repair routes, target preparation and acceptance thresholds are unchanged.
"""
import time
import numpy as np

from .refine_routing import descendant_bodies, movable_submodel
from .v8.self_collision import _repair_window

POLICY = 'hand_leg_cooperation_v1'
RECIPES = ('hand_leg5', 'hand_leg10')


def pair_sides(model):
    """Map existing full-model pair IDs to their arm and leg descendants."""
    r = model.r
    arms = {s: descendant_bodies(r, g['arm']) for s, g in r.sides.items()}
    legs = {s: descendant_bodies(r, g['leg']) for s, g in r.sides.items()}
    result = {}
    for k, (ga, gb) in enumerate(model.pairs):
        a, b = int(model.m.geom_bodyid[ga]), int(model.m.geom_bodyid[gb])
        arm = [s for s, bodies in arms.items() if a in bodies or b in bodies]
        leg = [s for s, bodies in legs.items() if a in bodies or b in bodies]
        if arm and leg:
            result[k] = dict(arms=arm, legs=leg)
    return result


def attach(jobs, samples, model):
    """Add joint proposals only to windows with measured hand/leg penetration.

    Keep legacy branches as fallbacks. Leg jobs already run before arm jobs,
    so a blocked leg window can now ask the intersecting hand to cooperate.
    """
    mapping = pair_sides(model)
    events = [(i // 8, set(pairs).intersection(mapping))
              for i, (_, _, pairs) in enumerate(samples) if pairs]
    from .refine_queue import RECIPES as legacy
    result = []
    for raw in jobs:
        job = dict(raw)
        if job['module'] in ('leg_collision', 'collision'):
            ids = sorted({k for frame, pairs in events
                          if job['start'] - 1 <= frame < job['end'] for k in pairs})
            if ids:
                job['hand_leg_pair_ids'] = ids
                previous=list(legacy[job['module']])
                # Cheap, already successful leg repairs keep their fast path.
                # Cooperation precedes the expensive elastic leg restorations.
                job['candidate_order'] = previous[:1]+list(RECIPES)+previous[1:] if job['module']=='leg_collision' else list(RECIPES)+previous
        result.append(job)
    return result


def columns(model, pair_ids, include_legs):
    mapping = pair_sides(model)
    if not pair_ids or any(k not in mapping for k in pair_ids):
        raise ValueError('hand_leg_requires_measured_pair_ids')
    arms = sorted({s for k in pair_ids for s in mapping[k]['arms']})
    legs = sorted({s for k in pair_ids for s in mapping[k]['legs']})
    ids = sum([model.r.sides[s]['arm'] for s in arms], [])
    if include_legs:
        ids = [2] + sum([model.r.sides[s]['leg'] for s in legs], []) + ids
    return ids, arms, legs


def owned_pairs(model, pair_ids, include_legs):
    """Clear the mixed issue; every other pair keeps its measured baseline.

    A hand can unblock a leg repair without first clearing unrelated existing
    leg penetrations. Those issues remain in the measured queue and final audit.
    """
    columns(model, pair_ids, include_legs)  # Validate physical ownership.
    return sorted(set(pair_ids))


def geometric_seed(model,t,rows,reference,a,b,arms,pairs,degrees,job,seams):
    """One coherent alternative start, never an independently adopted result."""
    from .shoulder_recovery import shoulder_seeds
    from .trajectory_escape import envelope
    r=model.r;best_frame=a;depth=0.
    for i in range(a,b):
        model.update(rows[i],t['root_r'][i]);value=float(model.distances()[pairs].min())
        if value<depth:depth=value;best_frame=i
    seed=rows.copy()
    ramp=envelope(b-a,max(0,job.get('core_start',a)-a),min(b-a-1,job.get('core_end',b)-a-1),
                  seams.get('freeze_start',True),seams.get('freeze_end',True))
    for side in arms:
        ids=r.sides[side]['arm']
        starts,_,_=shoulder_seeds(r,t,best_frame,side,rows[best_frame],reference[best_frame],degrees)
        def rank(q):
            x=rows[best_frame].copy();x[ids]=q;model.update(x,t['root_r'][best_frame])
            d=model.distances()[pairs]
            return float(np.maximum(.0001-d,0.).sum())
        chosen=min(starts,key=rank);offset=chosen-rows[best_frame,ids]
        lower=r.lower[ids]+1e-6;upper=r.upper[ids]-1e-6
        low=np.broadcast_to(lower,(b-a,len(ids))).copy();high=np.broadcast_to(upper,low.shape).copy()
        low[:,:3]=np.maximum(low[:,:3],reference[a:b,ids[:3]]-np.deg2rad(degrees))
        high[:,:3]=np.minimum(high[:,:3],reference[a:b,ids[:3]]+np.deg2rad(degrees))
        seed[a:b,ids]=np.clip(rows[a:b,ids]+ramp[:,None]*offset,low,high)
    return seed


def propose(model, t, rows, reference, weights, job, recipe, deadline, progress):
    """Try arm-only first; open only implicated legs if it cannot clear them."""
    if recipe not in RECIPES:
        raise ValueError('invalid_hand_leg_recipe')
    from .leg_policy import plan as leg_plan
    from .unified_repair import collision_seams
    a, b = job['start'], job['end']
    pairs = job['hand_leg_pair_ids']
    tier = float(recipe.removeprefix('hand_leg'))
    arm_tier = 10. if tier == 5. else 20.
    routes = progress.setdefault('routes', {})
    attempts = []
    for route in ('arms_only', 'arms_seed', 'cooperate'):
        state = routes.setdefault(route, {})
        if state.get('failed'):
            attempts.append(state['failed'])
            continue
        remaining = deadline - time.monotonic()
        if remaining <= .01:
            raise TimeoutError('hand_leg_candidate_deadline')
        coupled = route == 'cooperate'
        ids, arms, legs = columns(model, pairs, coupled)
        owned = owned_pairs(model, pairs, coupled)
        # A global root-height shift cannot change self-collision distances.
        # Include every pair affected by a movable hinge, not unrelated pairs
        # whose zero Jacobian only enlarges the sparse QP.
        local_model = movable_submodel(model, [k for k in ids if k >= 3])
        physical_owned = {tuple(model.pairs[k]) for k in owned}
        local_owned = [k for k,pair in enumerate(local_model.pairs) if tuple(pair) in physical_owned]
        prior = np.full((b-a, len(ids)), .12)
        if coupled:
            # Legs can help, but moving the hand is tried first, not merely
            # encouraged by an arbitrary exchange rate between pose losses.
            leg_ids = [2] + sum([model.r.sides[s]['leg'] for s in legs], [])
            all_legs = [2] + sum([g['leg'] for g in model.r.sides.values()], [])
            leg_prior, _ = leg_plan(model, t, rows, a, b, all_legs)
            prior[:, :len(leg_ids)] = leg_prior[:, [all_legs.index(k) for k in leg_ids]] * 10.
        local_weights = dict(weights)
        local_weights['leg_direction'] = max(float(weights['leg_direction']), float(weights['arm_direction']),
                                             2.*float(weights.get('foot_orientation', 0.)))
        # One time slice covers both routes; the first cannot consume it all.
        seconds = remaining if coupled else min(remaining, max(.05, remaining*.35))
        seams=collision_seams(local_model, t, rows, a, b)
        initial=geometric_seed(model,t,rows,reference,a,b,arms,pairs,arm_tier,job,seams) if route=='arms_seed' else rows
        options = dict(coupled=coupled, _moving_ids=ids, _joint_prior=prior,
            _reference=reference, _geometry_reference=rows, _palm_reference=reference,
            _source_palm_constraints=True, _progress=state,
            _passive_collision_indices=sorted(set(range(len(local_model.pairs)))-set(local_owned)),
            body_joint_deg=tier, joint_deg=arm_tier, free_distal=True,
            root_position_m=float('inf'), whole_arm=True, gesture_caps_active=False,
            _elastic_recovery=coupled, _project_seed=route=='arms_seed', iterations=24, qp_iterations=4000,
            watchdog_seconds=seconds, **seams)
        try:
            candidate, detail = _repair_window(local_model, t, initial, local_weights, a, b,
                                               list(model.r.sides) if coupled else arms, options)
        except (ValueError, RuntimeError) as exc:
            from .refine_queue import time_sliced
            record = dict(route=route, reason=str(exc), diagnostics=getattr(exc, 'diagnostics', None))
            attempts.append(record)
            if not time_sliced(exc):
                state['failed'] = record
            # A completed first route is never retried during a continuation;
            # an interrupted route retains its own finite SQP progress.
            if coupled:
                exc.diagnostics = dict(policy=POLICY, attempts=attempts)
                raise
            continue
        return candidate, dict(policy=POLICY, route=route, tier_deg=tier, arm_tier_deg=arm_tier,
            moving_ids=ids, arm_sides=arms, leg_sides=legs if coupled else [],
            owned_pair_ids=owned, attempts=attempts, validation=detail,
            reference='fixed avoidance reference; never rebased',
            boundary_policy='existing checked joins; unchanged in phase one')
    error = ValueError('hand_leg_routes_exhausted')
    error.diagnostics = dict(policy=POLICY, attempts=attempts)
    raise error
