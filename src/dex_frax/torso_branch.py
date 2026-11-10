"""Recover a trapped shoulder branch before arm collision compensation.

Source torso orientation supplies a seed, never a pasted exported motion.
Only flagged bounded windows are replanned with fixed root/legs. Physical
limits, speed, joins, source-torso protection and floor/support remain gates;
self collisions still require the normal subsequent repair and final audit.
"""
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .adaptive_refine import inspect, sustained, window_targets
from .local_refine import torso_errors, fit_window, split_windows
from .v8.self_collision import windows, CollisionModel, scan
from .v8.solver import Objective

POLICY=dict(name='source_torso_shoulder_branch_v2',trigger_deg=20.,
            shoulder_limit_margin_deg=5.,padding_s=3.,maximum_window_s=10.,
            overlap_s=2.,maximum_windows=2,budget_s=80.)


def suspect_frames(robot, xs, errors):
    near=np.zeros(len(xs),bool)
    for g in robot.sides.values():
        ids=np.asarray(g['arm'][:3]);q=xs[:,ids]
        near|=np.min(np.minimum(q-robot.lower[ids],robot.upper[ids]-q),axis=1)<np.deg2rad(POLICY['shoulder_limit_margin_deg'])
    return (errors>POLICY['trigger_deg']) & near


def sustained_suspects(robot, xs, errors, fps):
    # A near-limit event identifies a trapped branch. Do not require the
    # shoulder to stay on the same side of a 5-degree threshold every frame.
    error_mask = sustained(errors > POLICY['trigger_deg'], max(2, round(.1*fps)))
    near = suspect_frames(robot, xs, errors)
    edges = np.diff(np.r_[False, error_mask, False].astype(int))
    out = np.zeros(len(xs), dtype=bool)
    for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        if np.any(near[a:b]): out[a:b] = True
    return out


def source_waist(robot,t,xs,a,b,deadline):
    """Fit the source torso using actual FK and physical waist bounds."""
    ids=np.array([3+robot.names.index('waist_'+s+'_joint') for s in ('yaw','roll','pitch')])
    out=xs.copy()
    for i in range(a,b):
        if time.monotonic()>=deadline:raise TimeoutError('torso_branch_deadline')
        def fun(q):
            row=xs[i].copy();row[ids]=q;robot.set(row,t['root_r'][i])
            return (robot.rot(robot.torso)-t['torso_r'][i]).ravel()
        fit=least_squares(fun,np.clip(xs[i,ids],robot.lower[ids]+1e-7,robot.upper[ids]-1e-7),
            bounds=(robot.lower[ids]+1e-7,robot.upper[ids]-1e-7),max_nfev=30)
        out[i,ids]=fit.x
    # Ease into immutable joins, instead of jumping to the source orientation.
    ramp=min(round(.4*t['fps']),max(0,(b-a-4)//2))
    for edge in ('left','right'):
        if (edge=='left' and a==0) or (edge=='right' and b==len(xs)):continue
        for k in range(ramp+2):
            i=a+k if edge=='left' else b-1-k
            u=np.clip((k-1)/max(1,ramp),0,1);u=u*u*u*(10-15*u+6*u*u)
            out[i,ids]=xs[i,ids]+u*(out[i,ids]-xs[i,ids])
    return out


def structural_guard(robot,t,before,after,weights,a,b):
    if after.shape!=before.shape or not np.isfinite(after).all():return 'nonfinite_or_shape'
    fixed=[0,1,2]+sum([g['leg'] for g in robot.sides.values()],[])
    if not np.array_equal(before[:,fixed],after[:,fixed]):return 'root_or_legs_changed'
    if not np.array_equal(before[:a],after[:a]) or not np.array_equal(before[b:],after[b:]):return 'outside_changed'
    if a and not np.array_equal(before[a:a+2],after[a:a+2]):return 'left_join_changed'
    if b<len(before) and not np.array_equal(before[b-2:b],after[b-2:b]):return 'right_join_changed'
    if np.any(after[:,3:]<robot.lower[3:]-np.deg2rad(1e-4)) or np.any(after[:,3:]>robot.upper[3:]+np.deg2rad(1e-4)):return 'joint_limits'
    if np.max(abs(np.diff(after[:,3:],axis=0))*t['fps']-robot.speed_limits)>1e-4:return 'speed_or_join'
    local=window_targets(t,a,b,len(before))
    old=torso_errors(robot,local,before[a:b]);new=torso_errors(robot,local,after[a:b])
    if np.any(new>old+.02):return 'torso_frame_regression'
    if np.sum(np.maximum(new-20.,0.))>=np.sum(np.maximum(old-20.,0.))-.1:return 'no_torso_improvement'
    return None


def floor_samples(robot,t,rows):
    """Match the eight-subdivision geometry checks, including subframe floor."""
    rotations=t['root_r'];delta=Rotation.from_matrix(rotations[:-1].transpose(0,2,1)@rotations[1:]).as_rotvec()
    values=[]
    for i in range(len(rows)):
        for k in range(8 if i+1<len(rows) else 1):
            u=k/8.;q=rows[i] if k==0 else rows[i]*(1-u)+rows[i+1]*u
            rot=rotations[i] if k==0 else rotations[i]@Rotation.from_rotvec(delta[i]*u).as_matrix()
            robot.set(q,rot)
            values.append([robot.point(body)[2]+np.min(hull['hull']@robot.rot(body)[2])
                           for body,hull in robot.surface_models.items()])
    return np.asarray(values)


class TimedObjective:
    def __init__(self,objective,deadline):self.objective,self.deadline=objective,deadline
    def evaluate(self,x):
        if time.monotonic()>=self.deadline:raise TimeoutError('torso_branch_deadline')
        return self.objective.evaluate(x)


def recover(robot,t,xs,weights,collision_config=None):
    start=time.monotonic();deadline=start+POLICY['budget_s'];out=xs.copy();errors=torso_errors(robot,t,xs)
    mask=sustained_suspects(robot,xs,errors,t['fps'])
    ranges=split_windows(windows(set(np.flatnonzero(mask).tolist()),len(xs),round(POLICY['padding_s']*t['fps'])),
        round(POLICY['maximum_window_s']*t['fps'])+1,round(POLICY['overlap_s']*t['fps']))
    ranges.sort(key=lambda ab: -float(np.maximum(errors[ab[0]:ab[1]]-POLICY['trigger_deg'],0.).sum()))
    report=dict(policy=POLICY,trigger_frames=int(mask.sum()),attempts=[],kept=0,
        scope='provisional_pre_collision_branch_recovery',final_acceptance_unchanged=True)
    arms=np.asarray(sum([g['arm'] for g in robot.sides.values()],[]))
    for index,(a,b) in enumerate(ranges):
        detail=dict(start=a,end=b,status='rejected');report['attempts'].append(detail)
        try:
            if index>=POLICY['maximum_windows'] or time.monotonic()>=deadline:raise TimeoutError('torso_branch_budget')
            # Source torso correction must not disturb an active supporting arm.
            if any(np.any(c['mode'][a:b]) for c in t.get('body_contacts',[])):
                raise ValueError('active_body_support')
            detail['phase']='source_waist'
            guided=source_waist(robot,t,out,a,b,deadline)
            local=window_targets(t,a,b,len(out));baseline=inspect(robot,local,out[a:b])
            from .v8.arms import plan_and_refine
            seed=guided.copy()
            detail['phase']='branch_plan'
            seed[a:b],branch=plan_and_refine(robot,local,guided[a:b],dict(weights,_repair_deadline=deadline))
            detail['phase']='join_fit'
            candidate,fit=fit_window(robot,t,guided,weights,a,b,iterations=10,ids_override=arms,seed=seed,
                objective_factory=lambda i:TimedObjective(Objective(robot,t,i,weights),deadline))
            # Screen exactly the float32 joint values that will be exported.
            candidate[:,3:]=candidate[:,3:].astype(np.float32).astype(float)
            # Reassert untouched data and boundary samples after quantization.
            candidate[:a]=out[:a];candidate[b:]=out[b:]
            legs=sum([g['leg'] for g in robot.sides.values()],[]);candidate[:,legs]=out[:,legs]
            if a:candidate[a:a+2]=out[a:a+2]
            if b<len(out):candidate[b-2:b]=out[b-2:b]
            detail['phase']='checks'
            reason=structural_guard(robot,t,out,candidate,weights,a,b)
            if reason:raise ValueError(reason)
            checked=inspect(robot,local,candidate[a:b])
            if np.any(checked['floor']<np.minimum(-.001,baseline['floor']-1e-6)):raise ValueError('floor_regression')
            if np.any(checked['contact']>np.maximum(.001,baseline['contact'])+1e-7):raise ValueError('contact_regression')
            for side in robot.sides:
                for key in ('hand_forward','hand_width'):
                    if np.any(checked['errors'][side][key]>np.maximum(25.,baseline['errors'][side][key])+.001):
                        raise ValueError('palm_envelope')
            # This stage precedes collision repair. Never turn a clear initial
            # window into a colliding one or deepen an existing worst collision.
            if time.monotonic()>=deadline:raise TimeoutError('torso_branch_deadline')
            old_floor=floor_samples(robot,local,out[a:b]);new_floor=floor_samples(robot,local,candidate[a:b])
            if np.any(new_floor<np.minimum(-.001,old_floor-1e-6)):raise ValueError('interpolated_floor_regression')
            model=CollisionModel(robot,collision_config)
            before=scan(model,out[a:b],local['root_r'],8)[0]
            after=scan(model,candidate[a:b],local['root_r'],8)[0]
            if after['collision_samples']>before['collision_samples'] or after['minimum_distance_m']<min(before['minimum_distance_m'],0.)-2e-6:
                raise ValueError('collision_regression')
            if time.monotonic()>=deadline:raise TimeoutError('torso_branch_deadline')
            detail.update(status='kept',minimum_interpolated_floor_m=float(new_floor.min()),fit=fit,branch=branch,before_collision=before,after_collision=after,
                torso_max_before_deg=float(torso_errors(robot,local,out[a:b]).max()),
                torso_max_after_deg=float(torso_errors(robot,local,candidate[a:b]).max()))
            out=candidate;report['kept']+=1
        except (ValueError,TimeoutError) as exc:detail['reason']=str(exc)
    report['elapsed_s']=time.monotonic()-start
    return out,report
