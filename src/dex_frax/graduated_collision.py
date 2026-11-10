"""Tier rounds over bounded overlapping collision windows, with fixed references.

Larger search bounds never weaken physical checks or source-pose objectives.
Only the final whole-clip gate can accept a motion for training.
"""
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .v8.self_collision import CollisionModel,_repair_window,scan

TIERS=(
    dict(name='small',joint_deg=10.,hand_rotation_deg=10.,hand_position_m=.03,tracking_weight=1.,iterations=10,qp_iterations=20000,coupled=False),
    dict(name='medium',joint_deg=20.,hand_rotation_deg=20.,hand_position_m=.08,tracking_weight=1.,iterations=16,qp_iterations=20000,coupled=False),
    dict(name='coupled',joint_deg=35.,hand_rotation_deg=30.,hand_position_m=.15,tracking_weight=1.,iterations=24,qp_iterations=20000,coupled=True,body_joint_deg=5.,root_position_m=.02),
)

class IncrementalRecovery(CollisionModel):
    def linear(self,columns):
        C,upper=super().linear(columns)
        return C,np.maximum(upper,-.008)


def split_windows(a,b,limit=120,overlap=20):
    """Cover every frame; small tails remain inside an overlapping full window."""
    if limit<=overlap or overlap<2:raise ValueError('Invalid collision window overlap')
    if b-a<=limit:return [(a,b)]
    starts=list(range(a,b-limit+1,limit-overlap))
    if starts[-1]!=b-limit:starts.append(b-limit)
    return [(i,i+limit) for i in starts]



def seam_floor(robot,t,xs,a,b):
    if robot is None:return 0.  # Mock geometry in policy-only tests.
    minimum=np.inf
    for i in range(a,b):
        for k in range(8 if i+1<b else 1):
            alpha=k/8;rotation=t['root_r'][i];row=xs[i]
            if k:
                delta=Rotation.from_matrix(rotation.T@t['root_r'][i+1]).as_rotvec()
                rotation=rotation@Rotation.from_rotvec(delta*alpha).as_matrix()
                row=xs[i]*(1-alpha)+xs[i+1]*alpha
            robot.set(row,rotation)
            for body,shape in robot.surface_models.items():
                minimum=min(minimum,float((shape['hull']@robot.rot(body)[2]+robot.point(body)[2]).min()))
    return minimum

def recover(robot,t,xs,solver_cfg,config,report,fast=False,reference=None):
    if not report.get('after',{}).get('collision_samples',0):return xs,report
    reference=xs if reference is None else reference
    begin=time.perf_counter();out=xs.copy();attempts=[];attempted_frames=0
    model=IncrementalRecovery(robot,config);n=len(xs);fps=float(t['fps'])
    limit=max(20,min(120,round(2.4*fps)));overlap=min(limit//3,max(2,round(.4*fps)))
    jobs=[]
    for old in report.get('windows',[]):
        if old.get('status')!='unresolved':continue
        first,last=old['start'],old['end']
        # Include context, without making a colliding internal cut immutable.
        a=max(0,first-overlap);b=min(n,last+overlap)
        for lo,hi in split_windows(a,b,limit,overlap):
            if any(j['start']==lo and j['end']==hi for j in jobs):continue
            left=lo==a and lo+2<=first and not scan(model,xs[lo:lo+2],t['root_r'][lo:lo+2],8)[0]['collision_samples']
            right=hi==b and hi-2>=last and not scan(model,xs[hi-2:hi],t['root_r'][hi-2:hi],8)[0]['collision_samples']
            jobs.append(dict(start=lo,end=hi,sides=old['sides'],tried=[],done=False,
                freeze_start=left,freeze_end=right))
    budget=600 if fast else None
    from .arm_policy import enabled,collision_options,active_arm_support
    for job in jobs:
        job['whole_arm']=enabled(solver_cfg) and not active_arm_support(robot,t,job['start'],job['end'])
    for tier_index in range(3):
        pending=[]
        for job in jobs:
            a,b=job['start'],job['end'];before,_,_=scan(model,out[a:b],t['root_r'][a:b],8)
            job['done']=not before['collision_samples']
            if not job['done']:pending.append((before.get('minimum_distance_m',0.),job))
        # Every region receives this tier before any region gets the next tier.
        for _,job in sorted(pending,key=lambda v:v[0]):
            a,b=job['start'],job['end']
            tier=collision_options((10.,20.,30.)[tier_index],(10,16,24)[tier_index]) if job['whole_arm'] else TIERS[tier_index]
            if b-a<3 or (budget is not None and attempted_frames+b-a>budget):
                attempts.append(dict(start=a,end=b,tier=tier['name'],status='deferred',reason='fast_repair_work_budget'));continue
            attempted_frames+=b-a;job['tried'].append(tier['name'])
            detail=dict(start=a,end=b,tier=tier['name'],limits=dict(tier),status='local_solver_failed')
            scratch={};seed=out.copy()
            if 'seed' in job:seed[a:b]=job['seed']
            internal=not(job['freeze_start'] and job['freeze_end'])
            try:
                candidate,success=_repair_window(model,t,seed,solver_cfg,a,b,job['sides'],options=dict(tier,
                    _reference=reference,_scratch=scratch,freeze_start=job['freeze_start'],freeze_end=job['freeze_end'],internal_overlap=internal))
                checked,_,_=scan(model,candidate[a:b],t['root_r'][a:b],8)
                if checked['collision_samples']:raise ValueError('dense_interpolated_collision')
                # An unresolved neighbor may remain unresolved; a repaired seam
                # must never introduce or deepen penetration outside this window.
                for lo,hi in ((max(0,a-1),min(n,a+1)),(max(0,b-1),min(n,b+1))):
                    old,_,_=scan(model,out[lo:hi],t['root_r'][lo:hi],8)
                    new,_,_=scan(model,candidate[lo:hi],t['root_r'][lo:hi],8)
                    if new.get('minimum_distance_m',0.)<min(-getattr(model,'cfg',{}).get('penetration_tolerance_m',1e-5),old.get('minimum_distance_m',0.))-2e-6:
                        raise ValueError('overlap_seam_collision_regression')
                    if seam_floor(robot,t,candidate,lo,hi)<min(-.001,seam_floor(robot,t,out,lo,hi))-2e-6:
                        raise ValueError('overlap_seam_floor_regression')
                out[a:b]=candidate[a:b];job['done']=True;job.pop('seed',None)
                detail.update(success);detail['tier']=tier['name'];detail['independent_check']=checked
            except ValueError as error:
                detail['reason']=str(error)
                if 'interrupted' in str(error):detail['status']='interrupted'
                if hasattr(error,'diagnostics'):detail['diagnostics']=error.diagnostics
                if 'candidate_window' in scratch and np.isfinite(scratch['candidate_window']).all():
                    job['seed']=scratch['candidate_window'].copy()
            attempts.append(detail)
    final,_,_=scan(model,out,t['root_r'],8)
    unresolved=[]
    for job in jobs:
        check,_,_=scan(model,out[job['start']:job['end']],t['root_r'][job['start']:job['end']],8)
        if check['collision_samples']:unresolved.append({k:v for k,v in job.items() if k not in ('seed','done')})
    all_tried=bool(unresolved) and all(len(j['tried'])==len(TIERS) for j in unresolved)
    outcome='repaired' if not final['collision_samples'] else ('tiers_exhausted' if all_tried else 'deferred')
    report['graduated_repair']=dict(policy='gesture_capped_overlap_tiers_v2',tiers=([collision_options(v) for v in (10.,20.,30.)] if enabled(solver_cfg) else TIERS),support_tiers=TIERS,attempts=attempts,
        outcome=outcome,unresolved_windows=unresolved,all_unresolved_tried_all_tiers=all_tried,
        used_windows=len(jobs),used_frames=sum(j['end']-j['start'] for j in jobs),attempted_frames=attempted_frames,
        maximum_attempted_frames=budget,maximum_window_frames=limit,
        final_physical_acceptance_unchanged=True,elapsed_s=time.perf_counter()-begin)
    report['after']=final
    return out,report
