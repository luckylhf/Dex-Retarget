"""Frax backbone with checked, transactional repairs on original full-clip targets.

No independent full V8 solve. Existing V8 routines propose local candidates;
the common export auditor, dense geometry and preservation guards decide.
"""
import copy
import json
import pickle
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .adaptive_refine import inspect, policy, pose_mask, window_targets, _stage
from .local_refine import torso_errors, fit_window, split_windows
from .similarity import POSE_FLAGS, palm_envelope, score as similarity_score
from .v8.self_collision import CollisionModel, scan, windows, _repair_window

POLICY = dict(name='frax_joint_shoulder_recovery_v4', maximum_windows=6,
              maximum_window_frames=251, overlap_frames=20, budget_s=120.,
              shoulder_tiers_deg=[10.,20.,30.], distal_offset_deg=10.,
              branch_references=['current_best','original_frax'])


def arm_window_pending(metrics, side, a, b, fps):
    """Recompute the original gates on current results, not candidate status.

Sustained runs are measured with full-clip context, so a window boundary
cannot erase a failure. Global P95 failures only retry relevant windows.
"""
    cfg=policy();errors=metrics['errors'][side]
    if pose_mask({side:errors},fps,cfg)[a:b].any():return True
    for key,values in errors.items():
        if key=='elbow_axis':continue
        limit=cfg['arm_direction_deg'] if key in ('upper','lower') else cfg['hand_axis_deg']
        if np.percentile(values,95)>limit and np.any(values[a:b]>limit):return True
    return False


def palm_guard(before, after):
    """Preserve both palm axes, including isolated peaks hidden by P95."""
    for side,errors in before.items():
        for key in ('hand_forward','hand_width'):
            old=errors[key];new=after[side][key]
            if np.any(new>np.maximum(old+.2,3.)+1e-3):return 'palm_frame_regression:'+side+':'+key
            if np.percentile(new,95)>np.percentile(old,95)+.25:return 'palm_p95_regression:'+side+':'+key
            if np.max(new)>np.max(old)+.25:return 'palm_peak_regression:'+side+':'+key
    return None


def propose_arm_branch(robot, local, current, original, weights, side, reference):
    """Search existing V8 seeds for this side only; don't refit the other arm."""
    from .v8.arms import plan_and_refine
    seed=current.copy();ids=robot.sides[side]['arm']
    if reference=='original_frax':seed[:,ids]=original[:,ids]
    elif reference!='current_best':raise ValueError('Unknown branch reference')
    return plan_and_refine(robot,local,seed,dict(weights,arm_event_tracking=False),_only_sides={side})


def quality_guard(before, after, soft_similarity=False):
    """Flags alone cannot detect a worsening already-failing metric."""
    added=set(after['flags'])-set(before['flags'])
    if soft_similarity:added-=POSE_FLAGS
    if added:return 'new_quality_flags:'+','.join(sorted(added))
    if not soft_similarity and after['adaptive_pose_gate']['violating_frames']>before['adaptive_pose_gate']['violating_frames']:
        return 'more_sustained_pose_frames'
    for metric in ('torso_orientation_deg','torso_up_deg','root_orientation_deg'):
        for stat in ('p95','max'):
            if after['source_pose_audit'][metric][stat]>before['source_pose_audit'][metric][stat]+.02:
                return 'source_orientation_regression'
    for key in ('collision_samples','native_colliding_frames','interpolated_colliding_intervals'):
        if after['self_collision'][key]>before['self_collision'][key]:return 'collision_count_regression'
    if after['self_collision']['minimum_distance_m']<min(before['self_collision']['minimum_distance_m'],0.)-2e-6:
        return 'collision_depth_regression'
    return None


def plan(quality, metrics, torso, fps):
    """Route independent measured defects; passing clips have no repair jobs."""
    if not quality['flags']:return []
    cfg=policy();jobs=[]
    collision=quality['self_collision']
    if collision['collision_samples']:
        jobs.append(('collision',None,None))
    support=(metrics['contact']>cfg['geometry_acceptance_m']) | (metrics['foot_floor'] < -cfg['geometry_acceptance_m'])
    body=metrics['body_floor'] < -cfg['geometry_acceptance_m']
    dense=quality.get('interpolated_floor') or {}
    intervals=dense.get('violating_intervals',[])
    if intervals:
        mask=np.zeros(len(torso),bool)
        for i in intervals:
            mask[max(0,i):min(len(mask),i+2)]=True
        jobs.append(('dense_floor',None,mask))
    if support.any():jobs.append(('support',None,support))
    if body.any():jobs.append(('body',None,body))
    # A measured local torso defect may exist below the global P95 gate.
    if (torso>cfg['arm_direction_deg']).any():jobs.append(('torso',None,torso>cfg['arm_direction_deg']))
    for side,errors in metrics['errors'].items():
        mask=pose_mask({side:errors},fps,cfg)
        for key,values in errors.items():
            limit=cfg['arm_direction_deg'] if key in ('upper','lower') else cfg['hand_axis_deg']
            if key!='elbow_axis' and np.percentile(values,95)>limit:mask|=values>limit
        if mask.any():jobs.append(('arms',side,mask))
    return jobs


def dense_geometry(robot, model, target, rows):
    """Per-pair, per-sample comparisons prevent moving a collision elsewhere."""
    rotations=target['root_r']; delta=Rotation.from_matrix(rotations[:-1].transpose(0,2,1)@rotations[1:]).as_rotvec()
    distances=[];floors=[]
    for i in range(len(rows)):
        for k in range(8 if i+1<len(rows) else 1):
            u=k/8
            row=rows[i] if not k else rows[i]*(1-u)+rows[i+1]*u
            rot=rotations[i] if not k else rotations[i]@Rotation.from_rotvec(delta[i]*u).as_matrix()
            model.update(row,rot);distances.append(model.distances().copy())
            floors.append([robot.point(b)[2]+np.min(shape['hull']@robot.rot(b)[2]) for b,shape in robot.surface_models.items()])
    return np.asarray(distances),np.asarray(floors)


def collision_seams(model, targets, rows, a, b):
    """A colliding internal cut cannot be an immutable success condition.

    Unfrozen boundary frames still obey adjacent speed bounds against the
    untouched outside trajectory and dense seam nonregression at adoption.
    """
    def clear(lo,hi):
        return not scan(model,rows[lo:hi],targets['root_r'][lo:hi],8)[0]['collision_samples']
    return dict(freeze_start=bool(a>0 and clear(a,min(b,a+2))),
                freeze_end=bool(b<len(rows) and clear(max(a,b-2),b)))


def preservation_guard(robot, targets, before, candidate, weights, metrics, original, stage, side, a, b, soft_similarity=False, original_metrics=None, freeze_start=True, freeze_end=True):
    if candidate.shape!=before.shape:return 'shape_changed'
    # Local modules may never change root rotations/targets; they receive copies.
    if not np.array_equal(candidate[:a],before[:a]) or not np.array_equal(candidate[b:],before[b:]):return 'outside_window_changed'
    if a>0 and (stage!='collision' or freeze_start) and not np.array_equal(candidate[a:a+2],before[a:a+2]):return 'left_join_changed'
    if b<len(before) and (stage!='collision' or freeze_end) and not np.array_equal(candidate[b-2:b],before[b-2:b]):return 'right_join_changed'
    if stage in ('arms','branch','collision'):
        sides=[side] if side else list(robot.sides)
        ids=np.asarray(sum([robot.sides[s]['arm'] for s in sides],[]))
        fixed=np.setdiff1d(np.arange(before.shape[1]),ids)
        if not np.array_equal(candidate[:,fixed],before[:,fixed]):return 'protected_body_changed'
        # Outside rows are proven unchanged above. A legitimate earlier branch
        # elsewhere is not part of this window's fixed-reference avoidance cap.
        for s in sides:
            arm=robot.sides[s]['arm']
            if not soft_similarity:
                cap=np.deg2rad([30.]*3+[10.]*4)
                if np.any(abs(candidate[a:b,arm]-original[a:b,arm])>cap+2e-6):return 'original_frax_angle_cap'
            elif stage=='collision':
                if np.any(abs(candidate[a:b,arm[:3]]-original[a:b,arm[:3]])>np.deg2rad(30.)+2e-6):return 'original_frax_shoulder_cap'
    if not np.isfinite(candidate).all():return 'nonfinite_candidate'
    # Match the final export gate's numerical tolerances, not the stricter
    # double-precision internal solver tolerances, after float32 serialization.
    limits=json.loads(Path(__file__).with_name('quality_pose.json').read_text())
    tolerance=np.deg2rad(limits['joint_limit_tolerance_deg'])
    if np.any(candidate[:,3:]<robot.lower[3:]-tolerance) or np.any(candidate[:,3:]>robot.upper[3:]+tolerance):return 'joint_limit_regression'
    if np.any(abs(np.diff(candidate[:,3:],axis=0))*targets['fps']>robot.speed_limits+1e-4):return 'join_or_velocity_failure'
    knees=[g['leg'][3] for g in robot.sides.values()]
    if np.any((before[:,knees]>=0.)&(candidate[:,knees]<-1e-7)):return 'new_knee_hyperextension'
    checked=inspect(robot,targets,candidate)
    if soft_similarity:
        reason=palm_envelope((original_metrics or metrics)['errors'],checked['errors'])
        if reason:return reason
    elif stage=='branch':
        reason=palm_guard(metrics['errors'],checked['errors'])
        if reason:return reason
    limit=policy()['geometry_acceptance_m']
    if np.any(checked['contact']>np.maximum(limit,metrics['contact'])+1e-7):return 'contact_regression'
    if np.any(checked['floor']<np.minimum(-limit,metrics['floor'])-1e-7):return 'floor_regression'
    for contact in targets.get('body_contacts',[]):
        body=contact['body'];hull=np.asarray(contact['robot_hull'])
        for i in np.flatnonzero(np.asarray(contact['mode'])[a:b]>0)+a:
            robot.set(before[i],targets['root_r'][i]);point=hull[np.argmin(hull@robot.rot(body)[2])]
            old=robot.point(body,point).copy();robot.set(candidate[i],targets['root_r'][i])
            if contact['mode'][i]!=1:point=hull[np.argmin(hull@robot.rot(body)[2])]
            delta=robot.point(body,point)-old
            if np.max(abs(delta if contact['mode'][i]==1 else delta[2:]))>.0005:return 'body_support_regression'
    # Compare on the same window with the original full-sequence calibration.
    local=window_targets(targets,a,b,len(before))
    if np.any(torso_errors(robot,local,candidate[a:b])>torso_errors(robot,local,before[a:b])+.02):return 'torso_frame_regression'
    if soft_similarity:return None
    for s in robot.sides:
        for key,values in metrics['errors'][s].items():
            slack=.03 if key in ('upper','lower') else (.2 if key=='elbow_axis' else .25)
            if np.percentile(checked['errors'][s][key],95)>np.percentile(values,95)+slack:return 'pose_tail_regression:'+s+':'+key
    return None


def repair_output(robot, targets, xs, meta, payload, path, solve, similarity_mode='soft', collision_reference=None):
    """Write one final export; rejected candidates cannot overwrite the baseline."""
    if similarity_mode not in ('soft','conservative'):raise ValueError('Unknown repair similarity policy')
    from .acceptance import evaluate
    from .cli import atomic_json,sha
    from .experimental_arm_refine import refine as arm_refine
    from .resource_events import phase
    phase('local_repair',compile_possible=False)
    begin=time.monotonic();deadline=begin+POLICY['budget_s'];path=Path(path)
    quality=evaluate(path,meta,solve,'frax');before_quality=copy.deepcopy(quality)
    baseline_hash=sha(path)
    baseline_folder=path.parent/'frax_baseline';baseline_folder.mkdir()
    for item in path.parent.glob('motion.*'):
        if item.is_file():shutil.copyfile(item,baseline_folder/item.name)
    atomic_json(baseline_folder/'quality.json',quality)
    def finish(report,quality):
        description=copy.deepcopy(meta)
        description['unified_repair']=dict(report_path='repair.json',kept_candidates=report['kept_candidates'],similarity_mode=similarity_mode)
        report['similarity_mode']=similarity_mode
        if similarity_mode=='soft':
            report['soft_similarity_policy']=dict(policy='whole_arm_resemblance_v2',palm_axis_limit_deg=25.,palm_comfort_deg=5.,
                resemblance_tiers_deg=[],collision_shoulder_tiers_deg=[10.,20.,30.],distal_offset_cap_deg=None,
                elbow_axis_gate_deg=40.,arm_direction_gate_deg=20.,safety_acceptance_unchanged=True,elbow_wrist_direction_objectives_retained=True)
        description['stage_times_s']['unified_repair']=report['elapsed_s']
        description['elapsed_s']+=report['elapsed_s'];description['output_sha256']=sha(path)
        description['self_collision']['export_check']=quality['self_collision']
        description['solver_velocity_check']=quality['step_velocity']
        atomic_json(path.with_suffix('.json'),description);atomic_json(path.parent/'repair.json',report)
        # Reuse the exact export's completed common audit; do not scan the same
        # long motion again in CLI. This transient value is never in metadata.
        description['_unified_quality']=quality
        return description
    if not quality['flags']:
        return finish(dict(policy=POLICY,attempts=[],kept_candidates=0,baseline_sha256=baseline_hash,
            final_sha256=baseline_hash,before=before_quality,after=quality,elapsed_s=time.monotonic()-begin,
            target_calibration='full_sequence_once_then_slice',full_v8_restarted=False),quality)
    # All acceptance uses the exact float32 export, including its root rotation.
    targets=copy.deepcopy(targets);targets['root_r']=Rotation.from_quat(payload['root_rot']).as_matrix()
    original=np.c_[payload['root_pos'],payload['dof_pos']].astype(float);out=original.copy()
    collision_reference=original if collision_reference is None else np.asarray(collision_reference,dtype=float).copy()
    weights=meta['configuration']['solver'];collision_cfg=meta['configuration']['self_collision']
    model=CollisionModel(robot,collision_cfg);metrics=inspect(robot,targets,out);original_metrics=copy.deepcopy(metrics)
    jobs=plan(quality,metrics,torso_errors(robot,targets,out),targets['fps'])
    attempts=[];used=0;kept=0
    with tempfile.TemporaryDirectory(prefix='dex-repair-') as scratch:
        trial=Path(scratch)/'motion.pkl'
        shutil.copyfile(path.with_suffix('.diagnostics.npz'),trial.with_suffix('.diagnostics.npz'))
        def audit_candidate(candidate):
            data=copy.deepcopy(payload);data['root_pos']=candidate[:,:3].astype(np.float32);data['dof_pos']=candidate[:,3:].astype(np.float32)
            trial.write_bytes(pickle.dumps(data,protocol=4))
            description=copy.deepcopy(meta);description['output_sha256']=sha(trial)
            atomic_json(trial.with_suffix('.json'),description)
            return evaluate(trial,description,solve,'frax'),data
        def consider(candidate,stage,side,a,b,detail):
            nonlocal out,metrics,quality,kept
            candidate=candidate.astype(np.float32).astype(float)
            soft_candidate=similarity_mode=='soft' and stage in ('arms','branch','collision')
            reason=preservation_guard(robot,targets,out,candidate,weights,metrics,collision_reference if stage=='collision' and soft_candidate else original,stage,side,a,b,soft_candidate,original_metrics,detail.get('freeze_start',True),detail.get('freeze_end',True))
            if reason:return False,reason
            lo=max(0,a-1);hi=min(len(out),b+1);local=window_targets(targets,lo,hi,len(out))
            old_d,old_f=dense_geometry(robot,model,local,out[lo:hi]);new_d,new_f=dense_geometry(robot,model,local,candidate[lo:hi])
            if np.any(new_d<np.minimum(old_d,0.)-2e-6):return False,'new_or_deeper_collision_sample'
            if np.any(new_f<np.minimum(old_f,-.001)-2e-6):return False,'dense_floor_regression'
            proposed,data=audit_candidate(candidate)
            reason=quality_guard(quality,proposed,soft_candidate)
            if reason:return False,reason
            newer=inspect(robot,targets,candidate)
            if stage in ('arms','branch'):
                old=sum(np.mean(metrics['errors'][side][k]) for k in ('upper','lower'))
                new=sum(np.mean(newer['errors'][side][k]) for k in ('upper','lower'))
                if soft_candidate:
                    old_score=similarity_score(metrics['errors'][side]);new_score=similarity_score(newer['errors'][side])
                    detail['candidate_similarity_score']=dict(before=old_score,after=new_score)
                    improved=new_score<old_score-1e-6 or (bool(quality['flags']) and not proposed['flags'])
                else:improved=new<old-1e-4
            elif stage=='torso':improved=np.mean(torso_errors(robot,targets,candidate))<np.mean(torso_errors(robot,targets,out))-1e-4
            elif stage=='dense_floor':
                old=quality.get('interpolated_floor') or {};new=proposed.get('interpolated_floor') or {}
                improved=new.get('samples_below_minus_1mm',0)<old.get('samples_below_minus_1mm',0) or new.get('minimum_body_visual_cm',0)>old.get('minimum_body_visual_cm',0)+1e-5
            elif stage=='collision':improved=proposed['self_collision']['collision_samples']<quality['self_collision']['collision_samples'] or proposed['self_collision']['minimum_distance_m']>quality['self_collision']['minimum_distance_m']+1e-6
            else:improved=np.max(newer['contact'])<np.max(metrics['contact'])-1e-7 or np.min(newer['floor'])>np.min(metrics['floor'])+1e-7
            if not improved:return False,'no_target_metric_improvement'
            out=candidate;metrics=newer;quality=proposed;kept+=1
            path.write_bytes(trial.read_bytes())
            np.savez_compressed(path.with_suffix('.npz'),**{k:v for k,v in data.items() if v is not None})
            for suffix in ('.check.json','.pose_errors.npz'):
                if trial.with_suffix(suffix).exists():shutil.copyfile(trial.with_suffix(suffix),path.with_suffix(suffix))
            return True,'improved_and_checked'
        for job_index,(stage,side,mask) in enumerate(jobs):
            if not quality['flags']:break
            if stage=='collision':
                report,_,bad=scan(model,out,targets['root_r'],8)
                mask=np.zeros(len(out),bool);mask[list(bad)]=True
            if not np.any(mask):continue
            ranges=windows(np.flatnonzero(mask).tolist(),len(out),6 if stage=='dense_floor' else 20)
            ranges=split_windows(ranges,31 if stage=='dense_floor' else POLICY['maximum_window_frames'],8 if stage=='dense_floor' else POLICY['overlap_frames'])
            ranges.sort(key=lambda ab:-int(np.sum(mask[ab[0]:ab[1]])))
            for a,b in ranges:
                if b-a<5:continue
                if used>=POLICY['maximum_windows'] or time.monotonic()>=deadline:
                    attempts.append(dict(module=stage,side=side,start=a,end=b,status='deferred',reason='repair_budget'));continue
                used+=1;started=time.monotonic();local=window_targets(targets,a,b,len(out));entry=dict(module=stage,side=side,start=a,end=b,status='rejected')
                try:
                    if stage=='arms':
                        candidate=out.copy()
                        fitted,detail=arm_refine(robot,local,out[a:b],weights,collision_cfg,side,
                            max(.01,min(40.,deadline-time.monotonic())),6,freeze_start=a>0,freeze_end=b<len(out),soft_similarity=similarity_mode=='soft')
                        candidate[a:b]=fitted
                    elif stage=='torso':candidate,detail=fit_window(robot,targets,out,weights,a,b,6)
                    elif stage=='dense_floor':
                        from .dense_floor_refine import refine as refine_dense_floor
                        candidate,detail=refine_dense_floor(robot,targets,out,weights,a,b,deadline)
                    elif stage=='collision':
                        seams=collision_seams(model,targets,out,a,b)
                        detail={'tiers':[],**seams};success=False
                        for tier in POLICY['shoulder_tiers_deg']:
                            if time.monotonic()>=deadline:break
                            options=dict(joint_deg=tier,distal_joint_deg=10.,hand_rotation_deg=tier,
                                hand_position_m={10.:.03,20.:.08,30.:.15}[tier],tracking_weight=1.,iterations=6,
                                qp_iterations=12000,coupled=False,_reference=original,freeze_start=True,freeze_end=True)
                            if similarity_mode=='soft':
                                from .arm_policy import collision_options
                                options=dict(collision_options(tier,6),_reference=collision_reference,freeze_start=True,freeze_end=True)
                            options.update(seams,internal_overlap=not(seams['freeze_start'] and seams['freeze_end']),_palm_reference=original)
                            recovery=None
                            try:
                                candidate,_=_repair_window(model,targets,out,weights,a,b,list(robot.sides),options)
                                good,reason=consider(candidate,stage,None,a,b,seams)
                            except (ValueError,RuntimeError) as exc:
                                good,reason=False,str(exc)
                            # Infeasible local geometry needs a new nonlinear
                            # shoulder branch, not another identical first QP.
                            if not good and similarity_mode=='soft' and time.monotonic()<deadline:
                                from .shoulder_recovery import propose
                                try:
                                    candidate,recovery=propose(model,targets,out,weights,a,b,options,
                                        min(deadline,time.monotonic()+{10.:12.,20.:24.,30.:60.}[tier]))
                                    good,reason=consider(candidate,stage,None,a,b,dict(recovery,**seams))
                                except (ValueError,RuntimeError) as exc:
                                    recovery=getattr(exc,'recovery_diagnostics',None)
                                    reason=str(exc)
                            detail['tiers'].append(dict(degrees=tier,kept=good,reason=reason,recovery=recovery))
                            success|=good
                            if not quality['self_collision']['collision_samples']:break
                        entry.update(status='kept' if success else 'rejected',detail=detail)
                        if success:
                            # Avoidance may introduce new resemblance defects;
                            # route them using current measurements, not the
                            # stale pre-collision list of arm windows.
                            jobs[job_index+1:]=[job for job in plan(quality,metrics,
                                torso_errors(robot,targets,out),targets['fps']) if job[0]!='collision']
                        attempts.append(dict(entry,elapsed_s=time.monotonic()-started));continue
                    else:
                        fitted,detail=_stage(robot,local,out[a:b].copy(),weights,stage)
                        candidate=out.copy();candidate[a:b]=fitted
                    if a>0:candidate[a:a+2]=out[a:a+2]
                    if b<len(out):candidate[b-2:b]=out[b-2:b]
                    good,reason=consider(candidate,stage,side,a,b,detail)
                    entry.update(status='kept' if good else 'rejected',reason=reason,detail=detail)
                except (ValueError,RuntimeError) as exc:entry['reason']=str(exc)
                attempts.append(dict(entry,elapsed_s=time.monotonic()-started))
                # A different existing V8 branch is a local candidate only;
                # never replace legs/waist with a freshly solved full motion.
                if stage=='arms':
                    # A small accepted improvement is not a pass. Try all
                    # bounded references while this side/window still fails.
                    for reference in POLICY['branch_references']:
                        if not arm_window_pending(metrics,side,a,b,targets['fps']):break
                        branch=dict(module='branch',side=side,start=a,end=b,reference=reference,status='rejected')
                        if time.monotonic()>=deadline:
                            attempts.append(dict(branch,status='deferred',reason='repair_budget'));break
                        started=time.monotonic()
                        try:
                            fitted,detail=propose_arm_branch(robot,local,out[a:b],original[a:b],weights,side,reference)
                            candidate=out.copy();ids=robot.sides[side]['arm'];candidate[a:b,ids]=fitted[:,ids]
                            candidate,join_fit=fit_window(robot,targets,out,weights,a,b,6,ids_override=ids,seed=candidate)
                            detail=dict(branch=detail,join_fit=join_fit)
                            if a>0:candidate[a:a+2]=out[a:a+2]
                            if b<len(out):candidate[b-2:b]=out[b-2:b]
                            good,reason=consider(candidate,'branch',side,a,b,detail)
                            branch.update(status='kept' if good else 'rejected',reason=reason,detail=detail)
                        except (ValueError,RuntimeError) as exc:branch['reason']=str(exc)
                        branch['window_still_pending']=arm_window_pending(metrics,side,a,b,targets['fps'])
                        attempts.append(dict(branch,elapsed_s=time.monotonic()-started))
                    pending=arm_window_pending(metrics,side,a,b,targets['fps'])
                    attempts.append(dict(module='arm_search_summary',side=side,start=a,end=b,
                        status='unresolved' if pending else 'passed',
                        reason=('repair_budget' if time.monotonic()>=deadline else 'branch_candidates_exhausted') if pending else 'arm_window_passed'))
    report=dict(policy=POLICY,attempts=attempts,kept_candidates=kept,baseline_sha256=baseline_hash,
        final_sha256=sha(path),before=before_quality,after=quality,elapsed_s=time.monotonic()-begin,
        target_calibration='full_sequence_once_then_slice',full_v8_restarted=False,
        budget_scope='cooperative between bounded window solves; enclosing worker enforces total timeout')
    return finish(report,quality)
