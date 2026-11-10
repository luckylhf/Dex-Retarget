"""Bounded nonlinear shoulder branch proposals, never accepted exports.

Deep penetration can make one linearized collision QP infeasible. Change its
starting branch using measured FK, then return to the existing strict window
solver and independent export auditor. No torso/leg or support relaxation.
"""
import itertools
import time
import numpy as np
from scipy.optimize import least_squares

from .arm_policy import active_arm_support
from .v8.arms import arm_objective
from .v8.self_collision import _repair_window


def shoulder_bounds(robot, ids, reference, degrees):
    lo=robot.lower[ids].copy()+1e-6;hi=robot.upper[ids].copy()-1e-6
    cap=np.deg2rad(degrees)
    lo[:3]=np.maximum(lo[:3],reference[:3]-cap)
    hi[:3]=np.minimum(hi[:3],reference[:3]+cap)
    return lo,hi


def shoulder_seeds(robot, target, frame, side, base, reference, degrees, previous=None):
    """Joint combinations plus FK-based forward/outward/upward directions.

The axis signs are discovered through the current elbow/wrist Jacobians;
there is no assumption that positive J1 always moves a wrist forward.
"""
    ids=np.asarray(robot.sides[side]['arm']);q=base[ids].copy()
    lo,hi=shoulder_bounds(robot,ids,reference[ids],degrees)
    seeds=[np.clip(q,lo,hi),np.clip(reference[ids],lo,hi)]
    if previous is not None:seeds.append(np.clip(previous,lo,hi))
    cap=np.deg2rad(degrees)
    for offset in itertools.product((-1.,0.,1.),repeat=3):
        if not any(offset):continue
        v=q.copy();v[:3]=reference[ids[:3]]+cap*np.asarray(offset)
        seeds.append(np.clip(v,lo,hi))
    robot.set(base,target['root_r'][frame]);g=robot.sides[side]
    J=np.vstack([robot.jac(g[b])[0][:,ids[:3]] for b in ('elbow','wrist')])
    R=target['root_r'][frame];outward=1. if side=='Left' else -1.
    for axis in ((1,0,0),(0,outward,0),(1,outward,0),(1,0,1),(0,outward,1)):
        direction=R@np.asarray(axis,dtype=float);direction/=np.linalg.norm(direction)
        dq=np.linalg.lstsq(J,np.tile(direction,2),rcond=1e-4)[0]
        dq*=cap/max(np.max(abs(dq)),1e-9)
        v=q.copy();v[:3]+=dq;seeds.append(np.clip(v,lo,hi))
    unique=[]
    for seed in seeds:
        if not any(np.max(abs(seed-old))<1e-7 for old in unique):unique.append(seed)
    return unique,lo,hi


class FrameRecovery:
    """Elastic internal geometry residuals; strict acceptance is downstream."""
    def __init__(self,model,t,i,side,base,reference,weights,previous):
        self.model=model;self.r=model.r;self.t=t;self.i=i;self.side=side
        self.base=base.copy();self.ids=np.asarray(self.r.sides[side]['arm'])
        self.reference=reference[self.ids];self.previous=previous
        self.obj=arm_objective(self.r,t,i,side,base,weights)
        self.pairs=np.array([k for k,pair in enumerate(model.pairs)
                             if any(g in model.arms[side] for g in pair)],dtype=int)
        self.cached=None
        # Source palm envelope remains a hard condition at adoption. Optimize
        # toward it here so a geometrically clear but twisted seed is not kept.
        self.r.set(base,t['root_r'][i]);g=self.r.sides[side];v=t['sides'][side]
        axes=(-self.r.rot(g['tcp'])[:,2],self.r.rot(g['tcp'])[:,0])
        self.palm_chords=np.array([max(2*np.sin(np.deg2rad(25.)/2),np.linalg.norm(axis-v[k][i]))
                                   for axis,k in zip(axes,('hand_forward','hand_width'))])

    def evaluate(self,q):
        if self.cached is not None and np.array_equal(q,self.cached[0]):return self.cached[1:]
        from .v8.math3d import skew
        residual,J=self.obj.evaluate(q);x=self.base.copy();x[self.ids]=q
        self.model.update(x,self.t['root_r'][self.i]);ds=self.model.distances()[self.pairs]
        con=self.model.limit.compute_qp_inequalities(self.model.configuration,1.)
        deficit=np.maximum(.004-ds,0.)
        G=con.G[self.pairs][:,self.r.cols[self.ids]].copy();G[deficit<=0]=0.
        # Collision dominates internal search, but cannot authorize an export.
        chunks=[residual,1500.*deficit];rows=[J,1500.*G]
        g=self.r.sides[self.side];v=self.t['sides'][self.side]
        _,A=self.r.jac(g['tcp']);A=A[:,self.ids]
        for axis,key,limit in zip((-self.r.rot(g['tcp'])[:,2],self.r.rot(g['tcp'])[:,0]),
                                   ('hand_forward','hand_width'),self.palm_chords):
            delta=axis-v[key][self.i];norm=np.linalg.norm(delta)
            chunks.append(np.array([80.*max(norm-limit,0.)]))
            rows.append((80.*delta@(-skew(axis)@A)/max(norm,1e-9))[None,:] if norm>limit else np.zeros((1,7)))
        prior=np.array([.05,.05,.05,.15,.15,.25,.25])
        chunks.append(prior*(q-self.reference));rows.append(np.diag(prior))
        if self.previous is not None:
            chunks.append(.8*(q-self.previous));rows.append(np.eye(7)*.8)
        res=np.concatenate(chunks);jac=np.vstack(rows)
        self.cached=(q.copy(),res,jac,ds)
        return res,jac,ds

    def rank(self,q):
        res,_,distances=self.evaluate(q)
        # Prefer actual nonlinear clearance; within that class prefer likeness.
        violation=np.maximum(.001-distances,0.)
        return (bool(violation.max()>0),float(violation@violation),float(res@res))


def propose(model,targets,xs,weights,a,b,options,deadline):
    """Return a strictly validated window or raise, retaining branch diagnostics."""
    if active_arm_support(model.r,targets,a,b):raise ValueError('shoulder_recovery_active_arm_support')
    r=model.r;reference=np.asarray(options['_reference']);seed=xs.copy()
    progress=options.get('_progress',{}).setdefault('recovery',{})
    if 'seed' in progress:seed[a:b]=np.asarray(progress['seed'])
    degrees=float(options['joint_deg']);started=time.monotonic();calls=0;changed=0;unresolved=[]
    details={'policy':'joint_shoulder_multistart_v1','degrees':degrees,'frames':[],
             'reference':'fixed_pre_collision','internal_candidates_only':True}
    for side in r.sides:
        state=progress.setdefault(side,{'next':a})
        ids=np.asarray(r.sides[side]['arm']);previous=None
        side_pairs=np.array([k for k,pair in enumerate(model.pairs) if any(g in model.arms[side] for g in pair)])
        if state['next']>a:previous=seed[state['next']-1,ids].copy()
        for i in range(state['next'],b):
            if time.monotonic()>=deadline:raise ValueError('shoulder_recovery_budget')
            if (a>0 and options.get('freeze_start',True) and i<a+2) or (b<len(xs) and options.get('freeze_end',True) and i>=b-2):
                previous=seed[i,ids].copy();continue
            model.update(seed[i],targets['root_r'][i]);distance=float(model.distances()[side_pairs].min())
            # Track successful preceding branches through the exit transition.
            if distance>=.0035 and (previous is None or np.max(abs(previous-seed[i,ids]))<.06):
                previous=seed[i,ids].copy();continue
            obj=FrameRecovery(model,targets,i,side,seed[i],reference[i],weights,previous)
            candidates,lo,hi=shoulder_seeds(r,targets,i,side,seed[i],reference[i],degrees,previous)
            candidates.sort(key=obj.rank)
            best=candidates[0];best_rank=obj.rank(best)
            # Different shoulder starts, with all seven axes free during each
            # bounded fit. Continue past a mere improvement while still unsafe.
            chosen=[]
            for q in candidates:
                if not chosen or all(np.linalg.norm(q[:3]-v[:3])>.08 for v in chosen):chosen.append(q)
                if len(chosen)==3:break
            for q in chosen:
                if time.monotonic()>=deadline:raise ValueError('shoulder_recovery_budget')
                sol=least_squares(lambda v:obj.evaluate(v)[0],np.clip(q,lo+1e-9,hi-1e-9),
                    jac=lambda v:obj.evaluate(v)[1],bounds=(lo,hi),max_nfev=24,ftol=1e-5,xtol=1e-5,gtol=1e-5)
                calls+=1;rank=obj.rank(sol.x)
                if rank<best_rank:best=sol.x;best_rank=rank
                if not best_rank[0]:break
            seed[i,ids]=best;previous=best.copy();changed+=1
            state['next']=i+1;progress['seed']=seed[a:b].tolist()
            if best_rank[0]:unresolved.append([i,side])
            details['frames'].append({'frame':i,'side':side,'before_m':distance,
                'after_m':float(obj.evaluate(best)[2].min()),'clear':not best_rank[0]})
    details.update(frame_fits=changed,local_solves=calls,unresolved_native=unresolved,
                   seed_elapsed_s=time.monotonic()-started)
    if hasattr(r,'m') and options.get('_recovery_diagnostics',False):
        from .v8.self_collision import scan
        details['seed_native_geometry']=scan(model,seed[a:b],targets['root_r'][a:b],1)[0]
        details['seed_dense_geometry']=scan(model,seed[a:b],targets['root_r'][a:b],8)[0]
        arm_ids=np.asarray(sum([g['arm'] for g in r.sides.values()],[]))
        details['seed_speed_excess_rad_s']=float(np.maximum(abs(np.diff(seed[a:b,arm_ids],axis=0))*targets['fps']-r.speed_limits[arm_ids-3],0.).max()) if len(arm_ids) and b-a>1 else 0.
    
    # Preserve the successful original route. Only failed geometry or a
    # source-palm violation enters the projected, constrained alternative.
    arm_ids=np.asarray(sum([g['arm'] for g in r.sides.values()],[]))
    details['validation_attempts']=[];last_error=None
    routes=[('original',None),('continuous_seed_with_source_palms',(0.,0.,0.))]
    # Try coherent alternative shoulder directions only after prior routes fail.
    # All starts share the same tier reference, deadline and strict validator.
    for axis in range(3):
        for sign in (1.,-1.):
            offset=np.zeros(3);offset[axis]=sign
            routes.append(('continuous_multistart_with_source_palms',tuple(offset)))
    for route_index,(route,offset) in enumerate(routes):
        if route_index<progress.get('route',0):continue
        route_progress=progress.setdefault('validation'+str(route_index),{})
        if route!='original' and time.monotonic()>=deadline:break
        extra=dict(_progress=route_progress,iterations=16 if route=='original' else options.get('recovery_iterations',64),
            _elastic_recovery=True,_project_seed=route!='original',
            _source_palm_constraints=route!='original',watchdog_seconds=max(.01,deadline-time.monotonic()))
        try:
            validation_seed=seed
            if route!='original' and hasattr(r,'m'):
                from .trajectory_recovery import seed as continuous_seed
                validation_seed,details['continuous_seed']=continuous_seed(model,targets,xs,weights,a,b,dict(options,_progress=route_progress,_initial_shoulder_offset=offset),deadline)
                extra['watchdog_seconds']=max(.01,deadline-time.monotonic())
            candidate,validation=_repair_window(model,targets,validation_seed,weights,a,b,list(r.sides),dict(options,**extra))
            if hasattr(r,'m') and options.get('whole_arm'):
                from .palm_constraints import SourcePalms
                palm=SourcePalms(r,targets,options.get('_palm_reference',reference),a,b,list(r.sides),arm_ids)
                if max(palm.evaluate(candidate[i],i-a)[2] for i in range(a,b))>np.deg2rad(.001):
                    raise ValueError('source_palm_envelope')
            details['validation_attempts'].append(dict(route=route,initial_shoulder_offset=offset,status='passed',validation=validation))
            details['validation']=validation
            return candidate,details
        except (ValueError,RuntimeError) as exc:
            last_error=exc
            from .refine_queue import time_sliced
            if time_sliced(exc):raise
            progress['route']=route_index+1
            details['validation_attempts'].append(dict(route=route,initial_shoulder_offset=offset,status='failed',reason=str(exc),diagnostics=getattr(exc,'diagnostics',None)))
    if last_error is None:raise ValueError('shoulder_recovery_budget' if time.monotonic()>=deadline else 'shoulder_recovery_routes_exhausted')
    details['failure_details']=getattr(last_error,'diagnostics',None)
    last_error.recovery_diagnostics=details
    raise last_error
