"""Coherent shoulder escape seeds for deep collision, before strict validation.

One smooth offset per arm axis acts over the measured collision episode. It
cannot move the root/waist/legs, rebase shoulder tiers, or certify an export.
The regular window solver and independent candidate guard remain authoritative.
"""
import itertools
import time
import numpy as np
from scipy.optimize import least_squares
from .shoulder_recovery import FrameRecovery
from .v8.self_collision import _repair_window
from .arm_policy import active_arm_support


def envelope(n,first,last,freeze_start,freeze_end):
    value=np.ones(n)
    def ease(u):return u*u*u*(10-15*u+6*u*u)
    if freeze_start:
        if first<3:raise ValueError('escape_requires_clear_left_context')
        value[:first+1]=ease(np.clip((np.arange(first+1)-1)/(first-1),0,1))
    if freeze_end:
        if last>n-4:raise ValueError('escape_requires_clear_right_context')
        value[last:]=ease(np.clip((n-2-np.arange(last,n))/(n-2-last),0,1))
    return value



def bounded_offsets(base,shape,offset,lower,upper):
    """Clip only the provisional seed; strict trajectory checks still follow.

    Per-frame mechanical corridors avoid requiring every frame to accept the
    same raw offset. Frozen joins retain exact input bytes. Derivatives are
    piecewise analytic, including the active velocity residual below.
    """
    raw=base+shape[:,None]*offset
    value=np.clip(raw,lower,upper)
    derivative=shape[:,None]*((raw>=lower)&(raw<=upper))
    value[shape==0]=base[shape==0]
    return value,derivative


def propose(model,t,xs,weights,a,b,options,deadline):
    r=model.r;seed=xs.copy();reference=np.asarray(options['_reference']);tier=float(options['joint_deg']);report={'policy':'coherent_shoulder_escape_v2','tier_deg':tier,'sides':{}}
    if active_arm_support(r,t,a,b):raise ValueError('escape_routes_active_support_to_contact_solver')
    progress=options.get('_progress')
    state={} if progress is None else progress.setdefault('coherent_escape',{})
    if 'seed' in state:
        saved=np.asarray(state['seed'],float)
        if saved.shape!=(b-a,xs.shape[1]) or not np.isfinite(saved).all():raise ValueError('Invalid coherent escape checkpoint')
        seed[a:b]=saved
    for side,g in r.sides.items():
        if time.monotonic()>=deadline:raise TimeoutError('escape_deadline')
        local_state=state.setdefault(side,{})
        if local_state.get('complete'):
            report['sides'][side]=local_state['report'];continue
        pair_ids=[k for k,p in enumerate(model.pairs) if any(v in model.arms[side] for v in p)]
        if not pair_ids:continue
        depths=[]
        for i in range(a,b):
            model.update(seed[i],t['root_r'][i]);depths.append(float(model.distances()[pair_ids].min()))
        bad=np.flatnonzero(np.asarray(depths)<0.)
        if not len(bad):continue
        shape=envelope(b-a,int(bad[0]),int(bad[-1]),a>0 and options.get('freeze_start',True),b<len(xs) and options.get('freeze_end',True))
        ids=np.asarray(g['arm']);base=seed[a:b,ids].copy();active=shape>1e-9
        cap=np.deg2rad(tier)
        lower=np.broadcast_to(r.lower[ids]+1e-7,base.shape).copy()
        upper=np.broadcast_to(r.upper[ids]-1e-7,base.shape).copy()
        lower[:,:3]=np.maximum(lower[:,:3],reference[a:b,ids[:3]]-cap)
        upper[:,:3]=np.minimum(upper[:,:3],reference[a:b,ids[:3]]+cap)
        if np.any(upper<lower):raise ValueError('escape_empty_physical_corridor')
        low=np.r_[np.full(3,-2*cap),np.full(4,-np.pi)]
        high=-low
        def trajectory(q):return bounded_offsets(base,shape,q,lower,upper)
        chosen=np.unique(np.r_[bad[np.linspace(0,len(bad)-1,min(9,len(bad))).astype(int)],np.argmin(depths)])
        objectives=[FrameRecovery(model,t,a+int(j),side,seed[a+j],reference[a+j],dict(weights,_repair_deadline=deadline),None) for j in chosen]
        step=r.joint_step(t['fps'],weights)[ids-3]
        def fun(q,jac=False):
            if time.monotonic()>=deadline:raise TimeoutError('escape_deadline')
            if local_state.get('optimizing'):local_state['trial_q']=np.asarray(q).tolist()
            residual=[];derivatives=[];rows,D=trajectory(q)
            for j,obj in zip(chosen,objectives):
                val,J,_=obj.evaluate(rows[j]);residual.append(val);derivatives.append(J*D[j])
            delta=np.diff(rows,axis=0)
            violation=np.maximum(abs(delta)-step,0.)
            residual.append((200.*violation).ravel())
            derivatives.append((200.*(violation>0)*np.sign(delta)*np.diff(D,axis=0))[:,:,None]*np.eye(7)[None,:,:])
            J=np.vstack([v.reshape(-1,7) for v in derivatives]);res=np.concatenate(residual)
            return J if jac else res
        def rank(q):
            worst=[];rows,_=trajectory(q)
            for j,obj in zip(chosen,objectives):worst.extend(np.maximum(.0035-obj.evaluate(rows[j])[2],0.))
            return (float(np.max(worst)),float(np.dot(worst,worst)),float(fun(q)@fun(q)))
        if 'guesses' not in local_state:
            guesses=[np.clip(np.zeros(7),low+1e-9,high-1e-9)]
            for signs in itertools.product((-1.,0.,1.),repeat=3):
                q=np.zeros(7);q[:3]=cap*np.asarray(signs);guesses.append(np.clip(q,low+1e-9,high-1e-9))
            guesses=sorted(guesses,key=rank)[:4]
            local_state.update(guesses=[v.tolist() for v in guesses],best=guesses[0].tolist(),best_rank=list(rank(guesses[0])),next_guess=0)
        best=np.asarray(local_state['best']);best_rank=tuple(local_state['best_rank'])
        for k in range(local_state['next_guess'],len(local_state['guesses'])):
            q=np.asarray(local_state.get('trial_q',local_state['guesses'][k]))
            local_state.update(next_guess=k,optimizing=True)
            try:
                fit=least_squares(fun,q,jac=lambda z:fun(z,True),bounds=(low,high),max_nfev=35,ftol=1e-6,xtol=1e-6,gtol=1e-5)
            finally:local_state['optimizing']=False
            quality=rank(fit.x)
            if quality<best_rank:best,best_rank=fit.x,quality
            local_state.update(best=best.tolist(),best_rank=list(best_rank),next_guess=k+1)
            local_state.pop('trial_q',None)
            if best_rank[0]<1e-6:break
        seed[a:b,ids]=trajectory(best)[0]
        report['sides'][side]=dict(offset_deg=np.degrees(best).tolist(),sample_frames=(chosen+a).tolist(),sample_clearance_deficit_m=best_rank[0])
        local_state.update(complete=True,report=report['sides'][side])
        state['seed']=seed[a:b].tolist()
    # A seed is provisional. It must satisfy dense geometry, palms, support,
    # joint speed and the same fixed-reference cap through the strict solver.
    candidate,detail=_repair_window(model,t,seed,weights,a,b,list(r.sides),dict(options,_elastic_recovery=True,_project_seed=True,_source_palm_constraints=True,iterations=32,watchdog_seconds=max(.01,deadline-time.monotonic())))
    report['validation']=detail
    return candidate,report
