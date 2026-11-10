"""Bounded arm-shape repair core, plus the preserved standalone experiment CLI.

The unified pipeline calls refine on original-target windows; main remains an
opt-in comparison command and never exports training data.

Start from a verified collision-free Frax export. Keep root, torso and legs
bitwise unchanged. Use joint fitting for resemblance and reserve tiered bounds for collision, preserving the
existing arm objective, elbow/forearm conventions and anatomical wrist limits.
Every candidate is checked before replacing the reference; no training export.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import time

import numpy as np
import osqp
from scipy import sparse as sp
from scipy.spatial.transform import Rotation

from .v8.arms import ArmObjective
from .v8.anatomy import configure
from .v8.robot import Robot
from .v8.self_collision import CollisionModel, scan, _arm_bodies
from .v8.math3d import skew

TIERS = (10., 20., 30.)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def limits(robot, ids, reference, shoulder_deg, soft_similarity=False):
    # All three shoulder axes share the tier. Distal joints retain their branch
    # neighborhood and the existing anatomical/physical intersection.
    if soft_similarity:
        return (np.broadcast_to(robot.lower[ids],reference.shape).copy(),
                np.broadcast_to(robot.upper[ids],reference.shape).copy())
    caps = np.deg2rad([shoulder_deg]*3+[10.]*4)
    return (np.maximum(robot.lower[ids], reference-caps),
            np.minimum(robot.upper[ids], reference+caps))


def features(robot, targets, xs, side):
    g = robot.sides[side]; v = targets['sides'][side]; result = []
    for i, row in enumerate(xs):
        robot.set(row, targets['root_r'][i])
        def unit(d): return d/max(float(np.linalg.norm(d)), 1e-12)
        result.append(np.array([unit(robot.point(g['elbow'])-robot.point(g['shoulder'])),
                       unit(robot.point(g['wrist'])-robot.point(g['elbow'])),
                       -robot.rot(g['tcp'])[:, 2], robot.rot(g['tcp'])[:, 0],
                       -robot.rot(g['shoulder'])[:, 1], robot.rot(g['forearm'])[:, 0]]))
    actual = np.asarray(result)
    desired = np.stack([v[k] for k in ('upper', 'lower', 'hand_forward', 'hand_width', 'elbow_axis', 'forearm_width')], axis=1)
    return np.rad2deg(np.arccos(np.clip(np.sum(actual*desired, axis=2), -1., 1.)))


def metric_summary(errors):
    return {name:dict(mean=float(np.mean(errors[:,j])), p95=float(np.percentile(errors[:,j],95)),
                      maximum=float(errors[:,j].max())) for j,name in enumerate(
        ('upper','lower','hand_forward','hand_width','elbow_axis','forearm_width'))}


def direction_bound(reference_errors, col):
    if col in (0,1):return reference_errors[...,col]+.02
    if col in (2,3):return np.maximum(reference_errors[...,col]+.2,3.)
    if col==4:return np.minimum(reference_errors[...,col]+.2,np.maximum(reference_errors[...,col],29.9))
    return np.maximum(reference_errors[...,col]+.2,3.)


def regression_guard(robot, targets, baseline, candidate, side, reference_errors, model,
                     collision_floor, check_dense=True, soft_similarity=False):
    ids = np.asarray(robot.sides[side]['arm']); fixed = np.setdiff1d(np.arange(baseline.shape[1]), ids)
    if not np.array_equal(candidate[:,fixed],baseline[:,fixed]): return False,'non_arm_changed',None
    if not np.isfinite(candidate).all(): return False,'nonfinite',None
    q = candidate[:,ids]
    if np.any(q<robot.lower[ids]-2e-6) or np.any(q>robot.upper[ids]+2e-6): return False,'joint_limits',None
    if np.any(abs(np.diff(q,axis=0))*targets['fps']>robot.speed_limits[ids-3]+1e-4): return False,'speed',None
    errors=features(robot,targets,candidate,side)
    # Protect palm orientation and corrected elbow/forearm direction. Tiny
    # near-zero angles get a 3 degree numerical/gesture cone, not a new gate.
    from .similarity import palm_bound
    for col in ((2,3) if soft_similarity else (0,1,2,3,4,5)):
        bound=palm_bound(reference_errors[...,col]) if soft_similarity else direction_bound(reference_errors,col)
        if np.any(errors[:,col]>bound+1e-3):
            return False,'palm_or_anatomical_direction_regression',dict(axis=col,excess_deg=float(np.max(errors[:,col]-bound)))
    for col in (() if soft_similarity else (2,3)):
        if np.percentile(errors[:,col],95)>np.percentile(reference_errors[:,col],95)+.25:
            return False,'palm_p95_regression',None
        if errors[:,col].max()>reference_errors[:,col].max()+.25:
            return False,'palm_peak_regression',None
    if check_dense:
        collision=scan(model,candidate,targets['root_r'],8)[0]
        if collision['collision_samples'] or collision['minimum_distance_m']<min(collision_floor,0.)-2e-6:
            return False,'dense_collision_regression',collision
        # Arms are the only changing links. Check their ground geometry at
        # native frames and the same eight subdivisions used for collisions.
        bodies=_arm_bodies(robot,[side]); rr=targets['root_r']
        delta=Rotation.from_matrix(rr[:-1].transpose(0,2,1)@rr[1:]).as_rotvec()
        for i in range(len(candidate)):
            for k in range(8 if i+1<len(candidate) else 1):
                u=k/8; row=candidate[i] if not k else (1-u)*candidate[i]+u*candidate[i+1]
                rot=rr[i] if not k else rr[i]@Rotation.from_rotvec(delta[i]*u).as_matrix()
                robot.set(row,rot)
                if any(robot.point(b)[2]+np.min(robot.surface_models[b]['hull']@robot.rot(b)[2]) < -1e-5 for b in bodies):
                    return False,'arm_floor_regression',collision
        return True,'safe',collision
    return True,'safe',None


def refine(robot, targets, baseline, solver_cfg, collision_cfg, side='Left', budget_s=200., iterations=6, *, freeze_start=False, freeze_end=False, soft_similarity=False, movable_collision_only=False, branch_hint=None):
    start=time.monotonic(); deadline=start+budget_s
    if not 3<=len(baseline)<=301: raise ValueError('Experiment is bounded to at most six seconds / 301 frames')
    ids=np.asarray(robot.sides[side]['arm']);n=len(baseline);width=7;ref=baseline[:,ids].copy();q=ref.copy()
    hint=None
    if branch_hint is not None:
        branch_hint=np.asarray(branch_hint,float)
        if branch_hint.shape!=baseline.shape or not np.isfinite(branch_hint).all():raise ValueError('Invalid arm branch hint')
        hint=branch_hint[:,ids].copy()
    model=CollisionModel(robot,collision_cfg)
    if movable_collision_only:
        from .refine_routing import movable_submodel
        model=movable_submodel(model,ids)
    original_collision=scan(model,baseline,targets['root_r'],8)[0]
    if original_collision['collision_samples']:raise ValueError('This experiment requires a collision-free Frax baseline')
    bodies=_arm_bodies(robot,[side])
    if any(c['body'] in bodies and np.any(c['mode']>0) for c in targets.get('body_contacts',[])):
        raise ValueError('Active arm support needs a coupled contact solver; refusing to move it')
    errors0=features(robot,targets,baseline,side)
    # Preserve existing objective terms; give palm direction at least the same
    # residual weight as upper/lower arm shape, as requested for resemblance.
    weights=dict(solver_cfg)
    weights['arm_direction'] *= 1.5
    for key in ('hand_direction','hand_width'): weights[key]=max(weights[key],weights['arm_direction'])
    objs=[]
    for i in range(n):
        local=dict(weights)
        local['arm_direction']*=min(3.,max(1.,float(errors0[i,:2].max())/15.))
        obj=ArmObjective(robot,targets,i,side,baseline[i],local)
        if soft_similarity:
            from .similarity import WholeArmObjective
            obj=WholeArmObjective(obj)
        objs.append(obj)
    eye=sp.eye(n*width,format='csc')
    d1=sp.kron(sp.diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n)),sp.eye(width),format='csc')
    d2=sp.kron(sp.diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n)),sp.eye(width),format='csc')
    prior=[.15]*3+[.20]*2+[.25]*2 if soft_similarity else [.05]*3+[.25]*4
    regularizer=sp.vstack([d1*.18,d2*4.,sp.diags(np.tile(prior,n))],format='csc')
    step=np.tile(robot.speed_limits[ids-3]/targets['fps'],n-1)
    # Preserve the actual nonpenetration gate, not every positive gap of the
    # current pose. The outer dense audit remains authoritative.
    clearance_target=min(model.cfg['margin_m'],.0001) if soft_similarity else model.cfg['margin_m']
    collision_targets=[]
    for i,row in enumerate(baseline):
        model.update(row,targets['root_r'][i]);collision_targets.append(np.minimum(model.distances(),clearance_target))
    collision_targets=np.asarray(collision_targets)
    rr=targets['root_r'];drot=Rotation.from_matrix(rr[:-1].transpose(0,2,1)@rr[1:]).as_rotvec()
    subframes=[]
    for i in range(n-1):
        for alpha in (.25,.5,.75):
            rot=rr[i]@Rotation.from_rotvec(drot[i]*alpha).as_matrix()
            row=(1-alpha)*baseline[i]+alpha*baseline[i+1];model.update(row,rot)
            subframes.append((i,alpha,rot,np.minimum(model.distances(),clearance_target)))
    def full(q):
        xs=baseline.copy();xs[:,ids]=q.astype(np.float32).astype(float)
        if freeze_start:xs[:2]=baseline[:2]
        if freeze_end:xs[-2:]=baseline[-2:]
        return xs
    def objective(q):
        values=[obj.evaluate(row) for obj,row in zip(objs,q)]
        residual=np.concatenate([v[0] for v in values]);jacobian=sp.block_diag([v[1] for v in values],format='csc')
        if hint is not None:
            # A different IK branch is a guide, never an adopted unsafe seed.
            # Start from the safe baseline and keep all geometry/palm/speed rows.
            gain=2.*float(solver_cfg['arm_direction'])
            residual=np.r_[residual,gain*(q-hint).ravel()]
            jacobian=sp.vstack([jacobian,gain*eye],format='csc')
        return residual,jacobian
    def cost(q):
        r,_=objective(q);return float(r@r+np.linalg.norm(regularizer@(q-ref).ravel())**2)
    initial_cost=cost(q);history=[];best_cost=initial_cost;g=robot.sides[side];desired=targets['sides'][side]
    search_passes=(None,) if soft_similarity else TIERS
    for tier in search_passes:
        lo,hi=limits(robot,ids,ref,tier,soft_similarity)
        if freeze_start:lo[:2]=hi[:2]=ref[:2]
        if freeze_end:lo[-2:]=hi[-2:]=ref[-2:]
        for iteration in range(2*iterations if soft_similarity else iterations):
            if time.monotonic()>=deadline:
                history.append(dict(tier=tier,status='budget_stop'));break
            residual,J=objective(q);H=J.T@J+regularizer.T@regularizer+eye*.01
            grad=J.T@residual+regularizer.T@(regularizer@(q-ref).ravel())
            blocks=[];bounds=[]
            for i in range(n):
                row=baseline[i].copy();row[ids]=q[i];model.update(row,targets['root_r'][i])
                con=model.limit.compute_qp_inequalities(model.configuration,1.)
                active=np.isfinite(con.h);distances=model.distances()[active]
                C=list(con.G[active][:,robot.cols[ids]]);u=list(np.minimum(con.h[active],distances-collision_targets[i,active]+1e-7))
                for body in bodies:
                    hull=robot.surface_models[body]['hull'];R=robot.rot(body);local=hull[np.argmin(hull@R[2])]
                    if robot.point(body,local)[2]<.02:
                        P,_=robot.jac(body,local);C.append(-P[2,ids]);u.append(robot.point(body,local)[2]+1e-6)
                if not soft_similarity:
                    for start_body,end_body,key,col in [(g['shoulder'],g['elbow'],'upper',0),(g['elbow'],g['wrist'],'lower',1)]:
                        p0,P0=robot.point(start_body),robot.jac(start_body)[0]
                        p1,P1=robot.point(end_body),robot.jac(end_body)[0]
                        d=p1-p0;length=np.linalg.norm(d);direction=d/length;target=desired[key][i]
                        derivative=target@((np.eye(3)-np.outer(direction,direction))@(P1-P0)[:,ids]/length)
                        C.append(-derivative);u.append(float(target@direction)-np.cos(np.deg2rad(direction_bound(errors0[i],col)))+1e-8)
                for body,axis,key,col in [(g['tcp'],[0,0,-1],'hand_forward',2),(g['tcp'],[1,0,0],'hand_width',3),
                                           (g['shoulder'],[0,-1,0],'elbow_axis',4),(g['forearm'],[1,0,0],'forearm_width',5)]:
                    if soft_similarity and col not in (2,3):continue
                    _,A=robot.jac(body);direction=robot.rot(body)@axis;target=desired[key][i]
                    derivative=target@(-skew(direction)@A[:,ids])
                    from .similarity import palm_bound
                    threshold=palm_bound(errors0[i,col]) if soft_similarity else direction_bound(errors0[i],col)
                    C.append(-derivative);u.append(float(target@direction)-np.cos(np.deg2rad(threshold))+1e-8)
                blocks.append(sp.csc_matrix(np.asarray(C).reshape(-1,width)));bounds.append(u)
            C=sp.block_diag(blocks,format='csc');u=np.concatenate(bounds)
            interpolation=[];interpolation_bounds=[]
            for i,alpha,rot,required in subframes:
                row=(1-alpha)*baseline[i]+alpha*baseline[i+1]
                row[ids]=(1-alpha)*q[i]+alpha*q[i+1];model.update(row,rot)
                con=model.limit.compute_qp_inequalities(model.configuration,1.);dist=model.distances()
                active=np.isfinite(con.h)&(dist<.01)
                if not active.any():continue
                G=con.G[active][:,robot.cols[ids]];bound=np.minimum(con.h[active],dist[active]-required[active]+1e-7)
                interpolation.append(sp.hstack([sp.csc_matrix((len(bound),i*width)),sp.csc_matrix((1-alpha)*G),
                    sp.csc_matrix(alpha*G),sp.csc_matrix((len(bound),(n-i-2)*width))],format='csc'))
                interpolation_bounds.append(bound)
            if interpolation:
                C=sp.vstack([C,*interpolation],format='csc');u=np.r_[u,np.concatenate(interpolation_bounds)]
            low=np.maximum((lo-q).ravel(),-np.deg2rad(5));high=np.minimum((hi-q).ravel(),np.deg2rad(5))
            # Float32 input can exceed an exact double bound by a few ulps.
            low=np.minimum(low,0.);high=np.maximum(high,0.)
            A=sp.vstack([eye,d1,C],format='csc');lower=np.r_[low,-step-d1@q.ravel()-1e-7,np.full(len(u),-np.inf)]
            upper=np.r_[high,step-d1@q.ravel()+1e-7,u]
            scale=max(1.,float(abs(H.diagonal()).max()));qp=osqp.OSQP()
            qp.setup(P=sp.triu(H/scale,format='csc'),q=grad/scale,A=A,l=lower,u=upper,
                     eps_abs=1e-7,eps_rel=1e-7,max_iter=12000,verbose=False,polishing=True,time_limit=max(.01,min(8.,deadline-time.monotonic())))
            sol=qp.solve();entry=dict(tier=tier,iteration=iteration,qp_status=sol.info.status,status='rejected')
            if sol.info.status_val not in (1,2) or sol.x is None or not np.isfinite(sol.x).all():
                history.append(entry);break
            at=A@sol.x
            if max(float(np.max(lower-at)),float(np.max(at-upper)))>2e-6:
                entry['reason']='qp_constraint_residual';history.append(entry);break
            delta=sol.x.reshape(n,width)
            for alpha in (1.,.5,.25,.125,.0625,.03125):
                if time.monotonic()>=deadline:entry['reason']='budget_stop';break
                candidate=full(q+alpha*delta);next_q=candidate[:,ids];next_cost=cost(next_q)
                if next_cost>=best_cost-1e-8:entry['reason']='no_objective_improvement';continue
                ok,reason,collision=regression_guard(robot,targets,baseline,candidate,side,errors0,model,original_collision['minimum_distance_m'],soft_similarity=soft_similarity)
                entry['reason']=reason
                if not ok:entry['guard_details']=collision
                if ok:
                    improvement=best_cost-next_cost
                    q=next_q;best_cost=next_cost;entry['improvement']=improvement;entry.pop('guard_details',None);entry.update(status='kept',alpha=alpha,cost=best_cost,
                      min_distance_m=collision['minimum_distance_m']);break
            history.append(entry)
            print(json.dumps(entry),flush=True)
            if entry['status']!='kept' or entry.get('improvement',0.)<1e-4:break
        if time.monotonic()>=deadline:break
    result=full(q);errs=features(robot,targets,result,side)
    return result,dict(policy='whole_arm_resemblance_v2' if soft_similarity else 'experimental_safe_arm_shape_v1',similarity_mode='soft' if soft_similarity else 'conservative',side=side,tiers_deg=[] if soft_similarity else list(TIERS),search_purpose='resemblance',
        distal_offset_cap_deg=(None if soft_similarity else 10.),anatomical_direction_change_guard_deg=None if soft_similarity else .2,root_torso_legs_locked=True,anatomy=robot.wrist_limit_record,
        before=metric_summary(errors0),after=metric_summary(errs),initial_cost=initial_cost,cost=best_cost,
        history=history,wall_s=time.monotonic()-start,baseline_collision=original_collision)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--prepared-human',type=Path,required=True)
    parser.add_argument('--side',choices=['Left','Right'],default='Left')
    parser.add_argument('--budget',type=float,default=200.)
    parser.add_argument('--iterations',type=int,default=6)
    args=parser.parse_args()
    if not np.isfinite(args.budget) or not 0<args.budget<=220 or not 1<=args.iterations<=12:
        parser.error('Use 0 < budget <= 220 seconds and 1..12 iterations')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);base=args.baseline.resolve()
    begin=time.monotonic();run=json.loads((base/'run.json').read_text());meta=json.loads((base/'motion.json').read_text())
    if run['mode']!='frax' or run['status']!='complete':raise ValueError('Need a completed Frax output')
    if any(flag not in {'Left_lower_deg_source_mismatch','Left_upper_deg_source_mismatch','Right_lower_deg_source_mismatch','Right_upper_deg_source_mismatch','Left_hand_forward_deg_source_mismatch','Left_hand_width_deg_source_mismatch','Right_hand_forward_deg_source_mismatch','Right_hand_width_deg_source_mismatch','sustained_arm_pose_mismatch'} for flag in run['flags']):
        raise ValueError('Baseline has non-arm quality failures; refusing this isolated repair')
    for name in ('motion.pkl','motion.json','motion.diagnostics.npz','quality.json'):
        if digest(base/name)!=run['files'][name]:raise ValueError('Baseline artifact changed: '+name)
    from .smplx_source import load,adapt
    from .v8.contacts import prepare
    from .contact_transition import refine as transition
    from .initial_reuse import target_digest
    from .acceptance import evaluate
    cfg=meta['configuration'];robot=Robot(meta['model']);configure(robot,cfg['anatomy'])
    motion=load(args.prepared_human,fps=50.);targets=prepare(adapt(motion),robot,cfg['contact']);transition(targets,robot)
    if target_digest(targets)!=meta['prepared_targets_sha256']:raise ValueError('Prepared human targets differ from baseline')
    payload=pickle.loads((base/'motion.pkl').read_bytes());xs=np.c_[payload['root_pos'],payload['dof_pos']].astype(float)
    if payload['joint_names']!=robot.names or len(xs)!=len(targets['root_r']):raise ValueError('Motion layout mismatch')
    # Use the exported rotation for candidate geometry; torso/root remain exact.
    targets['root_r']=Rotation.from_quat(payload['root_rot']).as_matrix()
    candidate,report=refine(robot,targets,xs,cfg['solver'],cfg['self_collision'],args.side,args.budget,args.iterations)
    pkl=out/'motion.pkl'
    for name in ('motion.diagnostics.npz',):shutil.copyfile(base/name,out/name)
    def write(row):
        data=copy.deepcopy(payload);data['dof_pos']=row[:,3:].astype(np.float32)
        pkl.write_bytes(pickle.dumps(data,protocol=4));np.savez_compressed(out/'motion.npz',**{k:v for k,v in data.items() if v is not None})
        description=copy.deepcopy(meta);description.update(pose_solver='experimental_safe_arm_shape',generation_mode='repair_verified_frax',output_sha256=digest(pkl),experimental=True)
        (out/'motion.json').write_text(json.dumps(description,ensure_ascii=False,indent=2))
        quality=evaluate(pkl,description,None,'experimental_safe_arm_shape')
        return quality
    baseline_quality=json.loads((base/'quality.json').read_text());quality=write(candidate)
    added_flags=sorted(set(quality['flags'])-set(baseline_quality['flags']))
    shape_before=sum(report['before'][k]['mean'] for k in ('upper','lower'))
    shape_after=sum(report['after'][k]['mean'] for k in ('upper','lower'))
    pose_frames_before=baseline_quality['adaptive_pose_gate']['violating_frames']
    pose_frames_after=quality['adaptive_pose_gate']['violating_frames']
    tails_ok=all(report['after'][k]['p95']<=report['before'][k]['p95']+.03 for k in ('upper','lower'))
    kept=not added_flags and tails_ok and pose_frames_after<=pose_frames_before and shape_after<shape_before-1e-4 and report['cost']<report['initial_cost']
    report.update(candidate_pose_frames_before=pose_frames_before,candidate_pose_frames_after=pose_frames_after,shape_tail_guard_passed=tails_ok)
    if not kept:quality=write(xs)
    report.update(candidate_kept=kept,new_flags=added_flags,baseline_flags=baseline_quality['flags'],final_flags=quality['flags'],
        baseline=str(base),baseline_output_sha256=digest(base/'motion.pkl'),prepared_human_sha256=digest(args.prepared_human),
        experimental_implementation_sha256=digest(__file__),total_wall_s=time.monotonic()-begin,
        training_exported=False,scope='Five-second local experiment; unchanged production acceptance; no dynamics validation')
    (out/'quality.json').write_text(json.dumps(quality,ensure_ascii=False,indent=2))
    (out/'experiment.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (out/'run.json').write_text(json.dumps(dict(status='complete',quality=quality['status'],candidate_kept=kept,flags=quality['flags'],total_wall_s=report['total_wall_s']),ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
