"""Frax export -> persistent measured-window repairs -> independent full audit.

No per-clip window cap. Each finite candidate family gets a fair visit. Safe
adoptions persist immediately; a user budget leaves a resumable pending queue.
"""
import copy
import json
import pickle
import shutil
import tempfile
import time
from pathlib import Path
import numpy as np
from osqp import OSQPException
from scipy.spatial.transform import Rotation
from .adaptive_refine import policy,pose_mask,window_targets,_stage
from .audit_cache import fork
from .local_refine import fit_window,torso_errors,upper_ids
from .refine_queue import WorkQueue,jobs_from_masks,time_sliced
from .refine_routing import collision_groups,collision_masks,submodel,movable_submodel,dependencies,alignment_reference
from .unified_repair import quality_guard,dense_geometry,collision_seams,arm_window_pending
from .similarity import palm_envelope,score
from .v8.self_collision import CollisionModel,scan,_repair_window

POLICY=dict(name='ordered_support_waist_legs_arms_v4',core_frames=101,padding_frames=20,
            candidate_seconds=60.,collision_tiers_deg=[10.,20.,30.],
            maximum_windows=None,whole_clip_repair_seconds=None,
            context_padding_frames=[20,50,100],neighbor_reactivations=None,final_fresh_audit=True)


def measurements(cache,robot):
    frames=cache['frames'];n=len(frames['body_heights']);floor=cache['floor'][::8]
    bodies=list(robot.surface_models);feet={g['foot'] for g in robot.sides.values()}
    contact=np.zeros(n)
    for values in frames['contact_xyz'].values():
        contact=np.maximum(contact,np.array([0 if v is None else np.linalg.norm(v)/100 for v in values]))
    return dict(contact=contact,floor=floor.min(axis=1),
        foot_floor=floor[:,[i for i,b in enumerate(bodies) if b in feet]].min(axis=1),
        body_floor=floor[:,[i for i,b in enumerate(bodies) if b not in feet]].min(axis=1),
        errors=cache['metrics']['errors'],torso=cache['metrics']['torso'])


def masks(quality,cache,robot,groups=None,torso_reference=None):
    if not quality['flags']:return []
    m=measurements(cache,robot);n=len(m['torso']);result=[];cfg=policy()
    def add(stage,indices,side=None):
        mask=np.zeros(n,bool);mask[np.asarray(indices,int)]=True
        if mask.any():result.append((stage,side,mask))
    add('support',np.flatnonzero((m['contact']>cfg['geometry_acceptance_m'])|(m['foot_floor']<-.001)))
    add('body',(quality.get('body_support') or {}).get('violating_frames',[]))
    add('body',np.flatnonzero(m['body_floor']<-.001))
    dense=[]
    for i in (quality.get('interpolated_floor') or {}).get('violating_intervals',[]):dense.extend([i,min(n-1,i+1)])
    add('dense_floor',dense)
    torso_limit=20. if torso_reference is None else np.maximum(20.,np.asarray(torso_reference)+.02)
    add('torso',np.flatnonzero(m['torso']>torso_limit))
    if groups is None:
        count=len(cache['full_collision']['key'][1]);groups={'collision':list(range(count))}
    for stage,mask in collision_masks(cache['full_collision']['samples'],groups,n).items():
        add(stage,np.flatnonzero(mask))
    for joint,data in (quality.get('temporal_quality') or {}).get('joints',{}).items():
        add('jitter',data.get('violating_frames',data.get('added_reversal_frames',[])),joint)
    for side,errors in m['errors'].items():
        mask=pose_mask({side:errors},50.,cfg)
        for key,values in errors.items():
            limit=20. if key in ('upper','lower') else 25.
            if key!='elbow_axis' and np.percentile(values,95)>limit:mask|=values>limit
        add('arms',np.flatnonzero(mask),side)
    # Merge same-stage masks before generating stable grid windows.
    merged={}
    for stage,side,mask in result:merged[(stage,side)]=merged.get((stage,side),np.zeros(n,bool))|mask
    return [(stage,side,mask) for (stage,side),mask in merged.items()]


def guard(robot,t,before,after,original,old_m,new_m,old_q,new_q,old_cache,new_cache,model,job):
    a,b=job['start'],job['end'];stage=job['module'];n=len(before)
    if after.shape!=before.shape or not np.isfinite(after).all():return 'shape_or_nonfinite'
    if not np.array_equal(before[:a],after[:a]) or not np.array_equal(before[b:],after[b:]):return 'outside_window_changed'
    lo=max(0,a-1);hi=min(n,b+1)
    tolerance=np.deg2rad(json.loads(Path(__file__).with_name('quality_pose.json').read_text())['joint_limit_tolerance_deg'])
    if np.any(after[a:b,3:]<robot.lower[3:]-tolerance) or np.any(after[a:b,3:]>robot.upper[3:]+tolerance):return 'effective_joint_limits'
    if np.any(abs(np.diff(after[lo:hi,3:],axis=0))*t['fps']>robot.speed_limits+1e-4):return 'speed_or_seam'
    knees=[g['leg'][3] for g in robot.sides.values()]
    if np.any((before[a:b,knees]>=0)&(after[a:b,knees]<-1e-7)):return 'new_knee_hyperextension'
    if stage in ('arms','collision'):
        sides=[job['side']] if job['side'] else list(robot.sides)
        fixed=np.setdiff1d(np.arange(32),sum([robot.sides[s]['arm'] for s in sides],[]))
        if not np.array_equal(before[:,fixed],after[:,fixed]):return 'protected_body_changed'
    if not np.array_equal(before[:,:2],after[:,:2]):return 'pelvis_horizontal_changed'
    if stage=='leg_collision':
        fixed=np.setdiff1d(np.arange(32),[2]+sum([g['leg'] for g in robot.sides.values()],[]))
        if not np.array_equal(before[:,fixed],after[:,fixed]):return 'leg_repair_changed_upper_body'
    if stage in ('collision','torso'):
        for g in robot.sides.values():
            if stage=='collision' and np.any(abs(after[a:b,g['arm'][:3]]-original[a:b,g['arm'][:3]])>np.deg2rad(30.)+2e-6):return 'fixed_reference_shoulder_cap'
    if np.any(new_m['torso'][a:b]>old_m['torso'][a:b]+.02):return 'torso_frame_regression'
    reason=palm_envelope({s:{k:v[a:b] for k,v in e.items()} for s,e in old_m['errors'].items()},
                         {s:{k:v[a:b] for k,v in e.items()} for s,e in new_m['errors'].items()})
    if reason:return reason
    if np.any(new_m['contact'][a:b]>np.maximum(.001,old_m['contact'][a:b])+1e-7):return 'foot_anchor_regression'
    local=window_targets(t,lo,hi,n)
    old_d,old_f=dense_geometry(robot,model,local,before[lo:hi]);new_d,new_f=dense_geometry(robot,model,local,after[lo:hi])
    if np.any(new_d<np.minimum(old_d,0.)-2e-6):return 'new_or_deeper_collision_sample'
    if stage in ('collision','leg_collision'):
        owned=collision_groups(model)[stage]
        old_m['owned_collision_deficit']=float(np.maximum(-old_d[:,owned],0.).sum())
        new_m['owned_collision_deficit']=float(np.maximum(-new_d[:,owned],0.).sum())
    if stage=='leg_collision':
        from .leg_similarity import tradeoff
        def leg_angles(c):return np.column_stack([c['frames']['errors'][s][k][a:b] for s in ('Left','Right') for k in ('thigh_deg','shin_deg')])
        result=tradeoff(leg_angles(old_cache),leg_angles(new_cache),old_m['owned_collision_deficit'],new_m['owned_collision_deficit'])
        new_m['leg_shape_tradeoff']=result
        if result['dominated']:return 'leg_shape_regression_without_collision_gain'
    if np.any(new_f<np.minimum(old_f,-.001)-2e-6):return 'dense_floor_regression'
    for name,values in old_cache['frames']['body_errors'].items():
        for key,old in values.items():
            for i in range(a,b):
                new=new_cache['frames']['body_errors'][name][key][i]
                # All inferred contacts keep existing treatment; trusted supports
                # also get independent final gates. Let existing errors improve.
                if old[i] is not None and new is not None and new>max(old[i],.1)+.05:return 'body_support_regression'
    before_j=set((old_q.get('temporal_quality') or {}).get('violating_frames',[]))
    after_j=set((new_q.get('temporal_quality') or {}).get('violating_frames',[]))
    if after_j-before_j:return 'new_jitter_frames'
    return quality_guard(old_q,new_q,True)


def improved(stage,side,a,b,old,new,old_m,new_m):
    if old['flags'] and not new['flags']:return True
    if stage in ('collision','leg_collision'):
        x=old['full_self_collision'];y=new['full_self_collision']
        return (y['collision_samples']<x['collision_samples'] or y['minimum_distance_m']>x['minimum_distance_m']+1e-6
            or new_m.get('owned_collision_deficit',float('inf'))<old_m.get('owned_collision_deficit',0.)-1e-7)
    if stage=='torso':return np.sum(np.maximum(new_m['torso'][a:b]-20.,0.))<np.sum(np.maximum(old_m['torso'][a:b]-20.,0.))-.01
    if stage=='arms':return score(new_m['errors'][side])<score(old_m['errors'][side])-1e-5
    if stage=='jitter':return len(new['temporal_quality']['violating_frames'])<len(old['temporal_quality']['violating_frames'])
    if stage=='body' and len(new['body_support']['violating_frames'])<len(old['body_support']['violating_frames']):return True
    if stage=='dense_floor':return new['interpolated_floor']['samples_below_minus_1mm']<old['interpolated_floor']['samples_below_minus_1mm'] or new['interpolated_floor']['minimum_body_visual_cm']>old['interpolated_floor']['minimum_body_visual_cm']+1e-5
    return (np.maximum(new_m['contact'][a:b]-.001,0.).sum()<np.maximum(old_m['contact'][a:b]-.001,0.).sum()-1e-7
        or np.maximum(-new_m['floor'][a:b]-.001,0.).sum()<np.maximum(-old_m['floor'][a:b]-.001,0.).sum()-1e-7)


def propose(robot,t,out,original,weights,model,job,recipe,deadline,progress=None,unfixed_waist=None):
    a,b=job['start'],job['end'];stage=job['module'];side=job['side'];n=len(out)
    local=window_targets(t,a,b,n);cfg=dict(weights,_repair_deadline=deadline,_pelvis_height_only=True);candidate=out.copy();detail={}
    progress={} if progress is None else progress
    cfg['_branch_progress']=progress.setdefault('branches',{})
    cfg['_support_progress']=progress.setdefault('support',{})
    from .torso_branch import TimedObjective
    from .v8.solver import Objective
    def fit(seed=None,ids=None,iterations=6):
        return fit_window(robot,t,out,cfg,a,b,iterations,ids_override=ids,seed=seed,local_checks=True,
            objective_factory=lambda i:TimedObjective(Objective(robot,t,i,cfg),deadline))
    if stage=='collision':
        from .arm_policy import collision_options
        tier=float(''.join(c for c in recipe if c.isdigit()))
        options=dict(collision_options(tier,6),_reference=original,_palm_reference=out,_progress=progress,**collision_seams(model,t,out,a,b))
        options['internal_overlap']=not(options['freeze_start'] and options['freeze_end'])
        if recipe.startswith('sweep'):
            from .trajectory_escape import propose as escape
            candidate,detail=escape(model,t,out,cfg,a,b,options,deadline)
        elif recipe.startswith('continuous'):
            from .trajectory_recovery import seed as continuous_seed
            from .palm_constraints import SourcePalms
            axis=recipe.split('_')[1];offset=dict(zero=(0.,0.,0.),xpos=(1.,0.,0.),xneg=(-1.,0.,0.),ypos=(0.,1.,0.),yneg=(0.,-1.,0.),zpos=(0.,0.,1.),zneg=(0.,0.,-1.))[axis]
            seed,seed_detail=continuous_seed(model,t,out,cfg,a,b,dict(options,_initial_shoulder_offset=offset),deadline)
            candidate,validation=_repair_window(model,t,seed,cfg,a,b,list(robot.sides),dict(options,_elastic_recovery=True,_project_seed=True,_source_palm_constraints=True,iterations=64,watchdog_seconds=max(.01,deadline-time.monotonic())))
            detail=dict(continuous_seed=seed_detail,validation=validation,offset=offset)
        elif recipe.startswith('direct'):candidate,detail=_repair_window(model,t,out,cfg,a,b,list(robot.sides),options)
        else:
            from .shoulder_recovery import propose as recover
            candidate,detail=recover(model,t,out,cfg,a,b,options,deadline)
        detail.update(tier_deg=tier,**{k:options[k] for k in ('freeze_start','freeze_end')})
    elif stage=='leg_collision' or (stage=='support' and recipe.startswith('geometry')):
        if stage=='support':
            model=movable_submodel(model,sum([g['leg'] for g in robot.sides.values()],[]))
        cfg['leg_direction']=max(float(cfg['leg_direction']),float(cfg['arm_direction']),2.*float(cfg.get('foot_orientation',0.)))
        ids=[2]+sum([g['leg'] for g in robot.sides.values()],[])
        from .leg_policy import plan
        joint_prior,preference_report=plan(model,t,out,a,b,ids)
        options=dict(_joint_prior=joint_prior,coupled=True,_moving_ids=ids,_reference=original,_progress=progress,
            _passive_collision_indices=collision_groups(model)['collision'],_geometry_reference=out,
            body_joint_deg=float(''.join(c for c in recipe if c.isdigit())),_elastic_recovery=recipe.endswith('_elastic'),joint_deg=0.,root_position_m=float('inf'),whole_arm=True,
            gesture_caps_active=False,iterations=24,
            # Dense foot constraints need more QP iterations; the existing
            # wall-time slice and saved trajectory still bound and resume work.
            qp_iterations=80000 if recipe.endswith('_elastic') else 20000,
            watchdog_seconds=max(.01,deadline-time.monotonic()),
            **collision_seams(model,t,out,a,b))
        candidate,detail=_repair_window(model,t,out,cfg,a,b,list(robot.sides),options)
        from .leg_similarity import errors
        detail['joint_preferences']=preference_report
        detail['pelvis_height_change_max_m']=float(np.max(abs(candidate[a:b,2]-out[a:b,2])))
        detail['source_leg_shape_deg']={label:dict(mean=float(v.mean()),p95=float(np.percentile(v,95)),maximum=float(v.max())) for label,v in [('before',errors(robot,t,out,a,b)),('after',errors(robot,t,candidate,a,b))]}
    elif stage=='arms':
        if recipe=='fit':
            from .experimental_arm_refine import refine
            candidate[a:b],detail=refine(robot,local,out[a:b],cfg,model.cfg,side,max(.01,deadline-time.monotonic()),6,
                freeze_start=a>0,freeze_end=b<n,soft_similarity=True,movable_collision_only=True)
        else:
            from .unified_repair import propose_arm_branch
            seed=out.copy();ids=robot.sides[side]['arm']
            route={'guided_current':'current_best','guided_original':'original_frax'}.get(recipe,recipe)
            result,detail=propose_arm_branch(robot,local,out[a:b],original[a:b],cfg,side,route)
            if recipe.startswith('guided_'):
                from .experimental_arm_refine import refine
                candidate[a:b],join=refine(robot,local,out[a:b],cfg,model.cfg,side,max(.01,deadline-time.monotonic()),12,
                    freeze_start=a>0,freeze_end=b<n,soft_similarity=True,movable_collision_only=True,branch_hint=result)
            else:
                seed[a:b,ids]=result[:,ids];candidate,join=fit(seed,ids)
            detail=dict(branch=detail,join=join,guided_from_safe_baseline=recipe.startswith('guided_'))
    elif stage=='torso':
        if recipe=='fit':candidate,detail=fit()
        else:
            from .torso_branch import source_waist
            from .v8.arms import plan_and_refine
            guided=source_waist(robot,t,out,a,b,deadline)
            seed=guided.copy();branch={'executed':False}
            if recipe=='source_branch_collision' and 'aligned' not in progress:seed[a:b],branch=plan_and_refine(robot,local,guided[a:b],cfg)
            arm_ids=np.asarray(sum([g['arm'] for g in robot.sides.values()],[]))
            # Once waist FK matches the source, join only the arms; optimizing
            # waist again can re-enter the very branch this candidate fixes.
            if 'aligned' in progress:
                candidate=out.copy();candidate[a:b]=np.asarray(progress['aligned']);detail={'resumed_alignment':True}
            else:
                candidate,detail=fit_window(robot,t,guided,cfg,a,b,6,ids_override=arm_ids,seed=seed,local_checks=True,
                    objective_factory=lambda i:TimedObjective(Objective(robot,t,i,cfg),deadline))
                progress['aligned']=candidate[a:b].tolist()
            if unfixed_waist is None:unfixed_waist=torso_errors(robot,t,out)>20.
            aligned=alignment_reference(original,candidate,unfixed_waist,a,b)
            detail=dict(fit=detail,branch=branch,_aligned_reference=aligned[a:b].tolist())
            if recipe.endswith('_collision'):
                from .arm_policy import collision_options
                from .shoulder_recovery import propose as recover
                detail['collision_tiers']=[]
                for tier in (10.,20.,30.):
                    if not scan(model,candidate[a:b],local['root_r'],8)[0]['collision_samples']:break
                    if time.monotonic()>=deadline:raise TimeoutError('candidate_time_slice')
                    options=dict(collision_options(tier,6),_reference=aligned,_palm_reference=out,_progress=progress.setdefault('tier'+str(tier),{}),**collision_seams(model,t,candidate,a,b))
                    try:candidate,recovery=recover(model,t,candidate,cfg,a,b,options,deadline)
                    except (ValueError,RuntimeError) as exc:
                        if time_sliced(exc):raise
                        detail['collision_tiers'].append(dict(degrees=tier,reason=str(exc)));continue
                    detail['collision_tiers'].append(dict(degrees=tier,detail=recovery))
                # The provisional torso pose was never written. Only the whole
                # transaction can pass the external guard.
    elif stage=='dense_floor':
        from .dense_floor_refine import refine
        candidate,detail=refine(robot,t,out,cfg,a,b,deadline)
    elif stage=='jitter':
        cfg['joint_acceleration']=float(cfg.get('joint_acceleration',.18))*(2 if recipe=='temporal' else 4)
        if side not in robot.names:raise ValueError('Jitter task requires its measured joint')
        j=3+robot.names.index(side)
        # Correct the measured body region; leg/waist jitter cannot be routed
        # into a shoulder-only solver. Supports remain hard local constraints.
        ids=next((np.asarray(g[k]) for g in robot.sides.values() for k in ('leg','arm') if j in g[k]),np.array([j]))
        candidate,detail=fit_window(robot,t,out,cfg,a,b,6 if recipe=='temporal' else 12,ids_override=ids,
            local_checks=True,allow_support=True,objective_factory=lambda i:TimedObjective(Objective(robot,t,i,cfg),deadline))
    else:
        cfg['max_iterations']=30 if recipe=='solve' else 60
        cfg['_repair_trusted_support']=stage=='body'
        candidate[a:b],detail=_stage(robot,local,out[a:b].copy(),cfg,stage)
    if stage!='collision':
        if a:candidate[a:a+2]=out[a:a+2]
        if b<n:candidate[b-2:b]=out[b-2:b]
    return candidate,detail


def repair_output(robot,targets,xs,meta,payload,path,solve,repair_budget=None,continue_from=None,resume_recheck=False):
    from .acceptance import evaluate
    from .cli import atomic_json,sha
    from .resource_events import phase
    phase('local_repair',compile_possible=False)
    path=Path(path);started=time.monotonic();cache={}
    resume_state=None
    if continue_from is not None and (Path(continue_from)/'refine_state.json').exists():
        previous=Path(continue_from)
        resume_state=json.loads((previous/'refine_state.json').read_text())
        if resume_recheck:
            resume_state['states']={}
            atomic_json(path.parent/'recheck_provenance.json',dict(previous=str(previous),previous_identity_sha256=sha(previous/'identity.json'),previous_motion_sha256=sha(previous/'motion.pkl'),policy='preserve motion and fixed reference; fresh audit and queue'))
        path.write_bytes((previous/'motion.pkl').read_bytes())
        saved=pickle.loads(path.read_bytes())
        np.savez_compressed(path.with_suffix('.npz'),**{k:v for k,v in saved.items() if v is not None})
        meta['output_sha256']=sha(path);atomic_json(path.with_suffix('.json'),meta)
    quality=evaluate(path,meta,solve,'frax',cache=cache);before=copy.deepcopy(quality)
    # Initial full audit is not charged against the optional repair allowance.
    deadline=float('inf') if repair_budget is None else time.monotonic()+repair_budget
    t=copy.deepcopy(targets);t['root_r']=Rotation.from_quat(payload['root_rot']).as_matrix()
    original=np.c_[payload['root_pos'],payload['dof_pos']].astype(float);out=original.copy()
    baseline=path.parent/'frax_baseline';baseline.mkdir(exist_ok=False)
    for item in path.parent.glob('motion.*'):
        if item.is_file():shutil.copyfile(item,baseline/item.name)
    atomic_json(baseline/'quality.json',quality)
    if resume_state is not None:
        shutil.rmtree(baseline);shutil.copytree(previous/'frax_baseline',baseline)
        original_data=pickle.loads((baseline/'motion.pkl').read_bytes())
        original=np.c_[original_data['root_pos'],original_data['dof_pos']].astype(float)
        out=np.c_[saved['root_pos'],saved['dof_pos']].astype(float)
        before=json.loads((baseline/'quality.json').read_text())
    baseline_sha=sha(baseline/'motion.pkl');queue=WorkQueue(None if resume_state is None else resume_state['states'],retry_timed=resume_state is not None)
    attempts=[] if resume_state is None else resume_state['attempts'];kept=0 if resume_state is None else resume_state['kept_candidates']
    model=CollisionModel(robot,dict(meta['configuration'].get('self_collision') or {},coverage='full'))
    groups=collision_groups(model);models={key:submodel(model,ids) for key,ids in groups.items() if ids}
    models['leg_collision']=movable_submodel(model,sum([g['leg'] for g in robot.sides.values()],[]))
    reference=original.copy();unfixed_waist=cache['metrics']['torso']>20.
    if resume_state is not None:
        reference=np.load(previous/'avoidance_reference.npy',allow_pickle=False)
        unfixed_waist=np.asarray(resume_state['unfixed_waist'],bool)
        if unfixed_waist.shape!=(len(out),):raise ValueError('Invalid waist reference mask')
        if reference.shape!=original.shape or not np.isfinite(reference).all():raise ValueError('Invalid avoidance reference')
    reference_path=path.parent/'avoidance_reference.npy'
    np.save(reference_path,reference,allow_pickle=False)
    weights=meta['configuration']['solver'];current=measurements(cache,robot)
    torso_reference=(meta.get('frax',{}).get('body_priority') or {}).get('torso_reference_error_deg')
    if torso_reference is not None and np.shape(torso_reference)!=(len(out),):raise ValueError('Body reference certificate length mismatch')
    def dependency_torso():
        return current['torso'] if torso_reference is None else current['torso']-np.maximum(0.,np.asarray(torso_reference)+.02-20.)
    def jobs():
        pending=jobs_from_masks(masks(quality,cache,robot,groups,torso_reference),len(out),POLICY['core_frames'],POLICY['padding_frames'])
        for job in pending:
            if job['module']=='arms':
                errors=current['errors'][job['side']];a,b=job['core_start'],job['core_end']
                # Large direction errors deserve a branch trial before repeated
                # small local steps. This orders search; it changes no gate.
                if max(np.percentile(errors[k][a:b],95) for k in ('upper','lower'))>2*policy()['arm_direction_deg']:
                    job['candidate_order']=['current_best','guided_current','original_frax','guided_original','fit']
        return queue.configure(pending,out)
    def eligible(pending):
        cm=collision_masks(cache['full_collision']['samples'],groups,len(out))
        return dependencies(queue.actionable(pending),dependency_torso(),cm['collision'],cm.get('body_collision'),pending=pending)
    def pending_report():
        pending=jobs();_,blocked=eligible(pending)
        cm=collision_masks(cache['full_collision']['samples'],groups,len(out))
        _,local_blocked=dependencies([j for j in pending if j['module']!='torso'],dependency_torso(),cm['collision'],cm.get('body_collision'),pending=pending)
        block={queue.key(j):j['reason'] for j in blocked+local_blocked}
        return [dict(j,dependency=block.get(queue.key(j))) for j in queue.pending(pending)]
    def checkpoint(status):
        atomic_json(path.parent/'refine_state.json',dict(schema=3,policy=POLICY,status=status,baseline_sha256=baseline_sha,
            output_sha256=sha(path),states=queue.states,attempts=attempts,kept_candidates=kept,pending=pending_report(),
            fixed_reference='avoidance_reference.npy',avoidance_reference_sha256=sha(reference_path),unfixed_waist=unfixed_waist.tolist(),prepared_targets_sha256=meta['prepared_targets_sha256'],
            elapsed_s=time.monotonic()-started))
    checkpoint('running')
    with tempfile.TemporaryDirectory(prefix='dex-window-refine-') as temporary:
        trial=Path(temporary)/'motion.pkl'
        shutil.copyfile(path.with_suffix('.diagnostics.npz'),trial.with_suffix('.diagnostics.npz'))
        while quality['flags']:
            pending=jobs();ready,blocked=eligible(pending);job=queue.next(ready)
            if job is None or time.monotonic()>=deadline:break
            recipe=queue.recipe(job);entry=dict(job,recipe=recipe,status='rejected');begin=time.monotonic()
            try:
                local_deadline=min(deadline,time.monotonic()+POLICY['candidate_seconds'])
                local_model=models.get(job['module'],models.get('collision',model)) if job['module'] in ('collision','leg_collision','torso') else model
                candidate,detail=propose(robot,t,out,reference,weights,local_model,job,recipe,local_deadline,queue.progress(job),unfixed_waist)
                aligned=detail.pop('_aligned_reference',None) if isinstance(detail,dict) else None
                entry['detail']=detail
                candidate=candidate.astype(np.float32).astype(float)
                data=dict(payload,root_pos=candidate[:,:3].astype(np.float32),dof_pos=candidate[:,3:].astype(np.float32))
                trial.write_bytes(pickle.dumps(data,protocol=4));description=copy.deepcopy(meta);description['output_sha256']=sha(trial)
                atomic_json(trial.with_suffix('.json'),description)
                trial_cache=fork(cache)
                proposed=evaluate(trial,description,solve,'frax',cache=trial_cache,changed=(job['start'],job['end']))
                newer=measurements(trial_cache,robot)
                reason=guard(robot,t,out,candidate,reference,current,newer,quality,proposed,cache,trial_cache,model,job)
                if 'leg_shape_tradeoff' in newer:entry['leg_shape_tradeoff']=newer['leg_shape_tradeoff']
                if reason is None and not improved(job['module'],job['side'],job['start'],job['end'],quality,proposed,current,newer):reason='no_target_metric_improvement'
                if reason is None:
                    out=candidate;quality=proposed;cache=trial_cache;current=newer;kept+=1;entry['status']='kept'
                    if aligned is not None:
                        reference[job['start']:job['end']]=np.asarray(aligned)
                        unfixed_waist &= newer['torso']>20.
                        np.save(reference_path,reference,allow_pickle=False)
                    # Commit the playable output before recording its checkpoint.
                    temp=path.with_suffix('.pkl.tmp');temp.write_bytes(trial.read_bytes());temp.replace(path)
                    np.savez_compressed(path.with_suffix('.npz'),**{k:v for k,v in data.items() if v is not None})
                    for suffix in ('.check.json','.pose_errors.npz'):shutil.copyfile(trial.with_suffix(suffix),path.with_suffix(suffix))
                    entry['reason']='improved_and_checked'
                else:entry['reason']=reason
                entry['audit_work']=trial_cache['work']
            except (ValueError,RuntimeError,TimeoutError,OSQPException) as exc:
                entry['reason']=str(exc);entry['error_type']=type(exc).__name__
                if getattr(exc,'diagnostics',None) is not None:entry['solver_diagnostics']=exc.diagnostics
                entry['status']='time_slice' if time_sliced(exc) else 'rejected'
            queue.complete(job,entry['status']=='kept',timed=entry['status']=='time_slice',recipe=recipe,result_rows=out if entry['status']=='kept' else None)
            entry['elapsed_s']=time.monotonic()-begin;attempts.append(entry)
            print(f"refine: {job['module']} {job['start']}:{job['end']} {recipe}: {entry['status']} ({entry.get('reason','')})",flush=True)
            checkpoint('running')
    # No cached final certificate: independently audit the exact persisted float32 output.
    description=copy.deepcopy(meta);description['output_sha256']=sha(path)
    atomic_json(path.with_suffix('.json'),description)
    final=evaluate(path,description,solve,'frax')
    # Audit_s is diagnostic; all actual flags and measured fields must agree.
    left={k:v for k,v in quality.items() if k!='audit_s'};right={k:v for k,v in final.items() if k!='audit_s'}
    if left!=right:raise RuntimeError('Incremental/full audit disagreement; no training export permitted')
    quality=final
    status='passed' if not final['flags'] else ('pending_budget' if time.monotonic()>=deadline and queue.next(eligible(jobs())[0]) is not None else 'search_exhausted_review')
    if status=='search_exhausted_review' and any(v['reason']=='time_sliced_candidates' for v in queue.pending(jobs())):status='time_slice_review'
    checkpoint(status)
    report=dict(policy=POLICY,status=status,attempts=attempts,kept_candidates=kept,before=before,after=quality,
        pending=pending_report(),collision_groups={k:len(v) for k,v in groups.items()},baseline_sha256=baseline_sha,final_sha256=sha(path),
        elapsed_s=time.monotonic()-started,target_calibration='full_sequence_once_then_slice',full_v8_restarted=False,
        final_fresh_audit_equal=True,similarity_mode='soft',budget_scope='optional whole repair wall time; finite candidate time slices, no clip window-count cap')
    atomic_json(path.parent/'repair.json',report)
    description['unified_repair']=dict(report_path='repair.json',kept_candidates=kept,similarity_mode='soft',status=status)
    description['stage_times_s']['unified_repair']=report['elapsed_s'];description['elapsed_s']+=report['elapsed_s']
    description['self_collision']['export_check']=quality['self_collision'];description['solver_velocity_check']=quality['step_velocity']
    atomic_json(path.with_suffix('.json'),description);description['_unified_quality']=quality
    return description
