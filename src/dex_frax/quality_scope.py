"""State what an accepted export does, and does not, certify."""
import hashlib
import json
from pathlib import Path
from functools import lru_cache

@lru_cache(maxsize=1)
def audit_implementation():
    root=Path(__file__).parent
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ("acceptance.py","quality_pose.py","quality_v8.py","v8/audit.py","v8/self_collision.py","foot_collision.py","ankle_collision.py","support_quality.py","temporal_quality.py")}


def solver_history(report):
    report=report or {};priority=report.get('body_priority') or {}
    stages={}
    if priority:
        stages['raw_frax']=report.get('raw_frax_report') or {}
        if priority.get('pelvis_feasibility_solver') is not None:stages['pelvis_feasibility']=priority['pelvis_feasibility_solver']
        stages.update(priority.get('ordered_limb_solvers') or {})
    if not stages:stages={'initial':report}
    details={};frames=set();frame_coverage=True
    for name,s in stages.items():
        count=int(s.get('qp_nonconverged_steps',0));indices=s.get('qp_nonconverged_output_frames')
        if count and indices is None:frame_coverage=False
        if indices is not None:frames.update(indices)
        total=s.get('qp_selected_iteration_count')
        details[name]=dict(nonconverged_iterations=count,affected_output_frames=indices,
            selected_trajectory_iterations=total,iteration_failure_fraction=count/total if total else None)
    count=int(report.get('qp_nonconverged_steps',0))
    # Old checkpoints have no per-frame residual evidence. Do not reinterpret a
    # boolean failure as either a tiny numerical residual or unsafe geometry.
    return dict(qp_nonconverged_steps=count,count_semantics='sum of nonconverged selected-trajectory QP iterations across stages, not failure probability',
        stages=details,unique_affected_output_frames=sorted(frames) if frame_coverage else None,
        cleared_by_repair=False,final_output_failure_is_separate=True,
        residual_severity='not_recorded; cannot infer a 1e-9 residual from convergence status',
        policy='retain conservative history block until independent replacement evidence is available')


def describe(meta,check,result):
    limits=result.get('criteria',{})
    metrics={
        'leg_direction':dict(threshold_deg=limits.get('leg_direction_p95_deg'),values={s:{k:v.get(k) for k in ('thigh_deg','shin_deg')} for s,v in check['sides'].items()}),
        'forearm_axial_direction':dict(threshold_deg=limits.get('forearm_width_p95_deg'),values={s:v.get('forearm_width_deg') for s,v in check['sides'].items()}),
        'root_xy':dict(threshold_m=limits.get('root_xy_max_m'),values_cm=check.get('root_translation_error_cm',{}).get('xy')),
        'root_z':dict(threshold_m=limits.get('root_z_max_m'),values_cm=check.get('root_translation_error_cm',{}).get('z'))}
    ungated=[k for k,v in metrics.items() if v.get('threshold_deg',v.get('threshold_m')) is None]
    execution=meta.get('support_solver_execution')
    if execution is None:execution=dict(status='unknown_legacy',executed=None,scope='legacy_support_postprocess')
    collision=check.get('full_self_collision') or {}
    contract=dict(version='export_audit_scope_v2',criteria=limits,implementation=audit_implementation(),model_sha256=meta.get('model_sha256'),foot_geometry_policy=collision.get('foot_coverage'),
        collision_pair_set_sha256=collision.get('pair_set_sha256'),collision_scope=collision.get('scope'),
        collision_pair_count=collision.get('pair_count'),interpolation_steps=collision.get('interpolation_steps'),
        conditional_collision_exclusions=collision.get('conditional_exclusions'),
        temporal_policy=(check.get('temporal_quality') or {}).get('policy'),
        support_acceptance_m=result.get('support_acceptance_m'),adaptive_pose_policy=(result.get('adaptive_pose_gate') or {}).get('policy'))
    contract['fingerprint']=hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(
        acceptance_scope=dict(statement='accepted means passed configured export checks, not complete source fidelity or dynamic feasibility',
            unthresholded_similarity=ungated,hardware_acceleration_and_jerk_certified=False,dynamic_feasibility_certified=False),
        metric_semantics=dict(
            adjacent_joint_speed='recomputed from exported adjacent frames against model limits; shared limit is intentional, postprocessing may violate it',
            root_orientation='prescribed-source orientation preservation and export-integrity check; not an independent IK-fitting score',
            signed_elbow_goal_error='fit to the same clipped anatomical target as the generator; diagnostic, not a separate final acceptance gate',
            acceleration_and_jerk='descriptive per-joint statistics plus source-relative artifact screening; no hardware acceleration bounds'),
        optional_similarity_checks=metrics,support_solver_execution=execution,
        independent_support_audit=dict(executed=True,scope='exported full-foot contact/floor and confidence-qualified body supports'),
        audit_contract=contract)
