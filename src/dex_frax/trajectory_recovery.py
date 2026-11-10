"""Continuous shoulder/arm seeds for a failed local collision branch.

Five shared time knots avoid joining unrelated single-frame solutions. This is
only an internal seed: native, subframe, palm, support and speed checks remain
mandatory in the window solver and independent adoption guard.
"""
import time
import numpy as np
from scipy.optimize import least_squares


def basis(n, freeze_start, freeze_end, knots=5):
    grid=np.linspace(0,n-1,knots)
    B=np.stack([np.interp(np.arange(n),grid,np.eye(knots)[k]) for k in range(knots)],axis=1)
    ramp=min(20,n//3)
    if freeze_start:
        B[:ramp]*=np.sin(np.linspace(0,np.pi/2,ramp))[:,None]**2
        B[:2]=0.
    if freeze_end:
        B[-ramp:]*=np.sin(np.linspace(np.pi/2,0,ramp))[:,None]**2
        B[-2:]=0.
    return B


def seed(model,t,xs,weights,a,b,options,deadline):
    from .shoulder_recovery import FrameRecovery
    r=model.r;reference=np.asarray(options['_reference']);out=xs.copy();n=b-a
    progress=options.get('_progress',{}).setdefault('continuous',{})
    if 'out' in progress:out[a:b]=np.asarray(progress['out'])
    B=basis(n,a>0 and options.get('freeze_start',True),b<len(xs) and options.get('freeze_end',True))
    knots=B.shape[1];records=[]
    for side,g in r.sides.items():
        if side in progress.get('completed',[]):continue
        state=progress.setdefault(side,{})
        ids=np.asarray(g['arm']);base=out[a:b,ids].copy()
        objs=[FrameRecovery(model,t,i,side,out[i],reference[i],weights,None) for i in range(a,b)]
        cap=np.r_[np.full(3,np.deg2rad(options['joint_deg'])),np.full(4,np.pi)]
        lo=np.maximum(r.lower[ids]+1e-6,reference[a:b,ids]-cap)
        hi=np.minimum(r.upper[ids]-1e-6,reference[a:b,ids]+cap)
        step=r.joint_step(t['fps'],weights)[ids-3];cache=[None,None,None]
        def evaluate(z):
            if time.monotonic()>=deadline:raise ValueError('continuous_shoulder_seed_budget')
            if cache[0] is not None and np.array_equal(z,cache[0]):return cache[1],cache[2]
            raw=base+B@z.reshape(knots,7);q=np.clip(raw,lo,hi);free=(raw>lo)&(raw<hi)
            residual=[];jac=[]
            for j,obj in enumerate(objs):
                v,J,_=obj.evaluate(q[j]);residual.append(v)
                jac.append((J[:,None,:]*B[j][None,:,None]*free[j][None,None,:]).reshape(len(v),-1))
            dq=np.diff(q,axis=0);excess=np.maximum(abs(dq)-step,0.)
            residual.append((200.*excess).ravel());J=np.zeros((n-1,7,knots,7))
            for k in range(7):
                J[:,k,:,k]=200.*((excess[:,k]>0)*np.sign(dq[:,k]))[:,None]*(B[1:]*free[1:,k,None]-B[:-1]*free[:-1,k,None])
            jac.append(J.reshape((n-1)*7,-1))
            value=np.concatenate(residual);derivative=np.concatenate(jac)
            cache[:]=[z.copy(),value,derivative]
            cost=float(value@value)
            if cost<state.get('cost',float('inf')):state.update(z=z.tolist(),cost=cost)
            return value,derivative
        offset=np.asarray(options.get('_initial_shoulder_offset',(0.,0.,0.)),dtype=float)
        if offset.shape!=(3,) or not np.isfinite(offset).all() or np.max(abs(offset))>1.:
            raise ValueError('invalid_continuous_shoulder_offset')
        initial=np.tile(np.r_[offset*np.deg2rad(options['joint_deg']),np.zeros(4)],knots)
        if 'z' in state:initial=np.asarray(state['z'])
        fit=least_squares(lambda z:evaluate(z)[0],initial,jac=lambda z:evaluate(z)[1],
            bounds=(-np.tile(cap,knots),np.tile(cap,knots)),max_nfev=30,ftol=1e-5,xtol=1e-5,gtol=1e-5)
        out[a:b,ids]=np.clip(base+B@fit.x.reshape(knots,7),lo,hi)
        # Clipping a slightly out-of-effective-range original join must not
        # mutate immutable input. Feasibility belongs to the strict solver.
        if a>0 and options.get('freeze_start',True):out[a:a+2,ids]=xs[a:a+2,ids]
        if b<len(xs) and options.get('freeze_end',True):out[b-2:b,ids]=xs[b-2:b,ids]
        records.append(dict(side=side,evaluations=fit.nfev,cost=float(fit.cost)))
        progress.setdefault('completed',[]).append(side);progress['out']=out[a:b].tolist()
    return out,dict(policy='continuous_five_knot_arm_seed_v2',initial_shoulder_offset=list(options.get('_initial_shoulder_offset',(0.,0.,0.))),fits=records)
