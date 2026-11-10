"""Bounded root/leg polish for independently measured subframe floor defects.

Arm and waist joints and all root rotations remain fixed. Native support,
physical limits, joint speeds and window joins are constraints. The caller
must independently audit dense collisions, floor, source pose and quality.
"""
import time
import numpy as np
from scipy import sparse as sp
from scipy.spatial.transform import Rotation
from .v8.trajectory import _qp
from .adaptive_refine import window_targets
from .torso_branch import floor_samples


def floor_target(height, immutable):
    # A fixed, already passing join cannot be tightened to the interior margin.
    # Existing penetration below the actual 1 mm gate is never exempted.
    return max(-.001,min(-.00095,float(height)-1e-8)) if immutable else -.00095


def refine(robot,t,xs,weights,a,b,deadline):
    r=robot;n=b-a
    if not 5<=n<=31:raise ValueError('dense_floor_window_budget')
    supported=any(np.any(v['contact'][a:b]>0) for v in t['sides'].values()) or any(np.any(c['mode'][a:b]>0) for c in t.get('body_contacts',[]))
    # Without support, translation alone clears the floor while preserving all
    # relative body geometry. Freeing legs needlessly can deepen hip/arm contact.
    ids=np.asarray(([2] if weights.get("_pelvis_height_only") else [0,1,2])+(sum([g['leg'] for g in r.sides.values()],[]) if supported else []));d=len(ids)
    out=xs.copy();q=xs[a:b,ids].copy();ref=q.copy()
    step=np.r_[np.full(3,np.inf),r.joint_step(t['fps'],weights)][ids]
    D=sp.kron(sp.diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n)),sp.eye(d),format='csc')
    I=sp.eye(n*d,format='csc');H=sp.diags(np.tile(np.where(ids<3,100.,1.),n));history=[]
    local=window_targets(t,a,b,len(xs))
    for iteration in range(12):
        if time.monotonic()>=deadline:raise ValueError('dense_floor_deadline')
        rows=[];lower=[];upper=[]
        def add(j,J,lo,hi,u=0.):
            block=np.zeros((len(J),n*d));block[:,j*d:(j+1)*d]=J*(1-u)
            if u:block[:,(j+1)*d:(j+2)*d]=J*u
            rows.extend(block);lower.extend(np.atleast_1d(lo));upper.extend(np.atleast_1d(hi))
        for j in range(n):
            if time.monotonic()>=deadline:raise ValueError('dense_floor_deadline')
            for k in range(8 if j<n-1 else 1):
                u=k/8.;at=xs[a+j].copy();at[ids]=q[j] if not k else q[j]*(1-u)+q[j+1]*u
                R=t['root_r'][a+j]
                if k:R=R@Rotation.from_rotvec(Rotation.from_matrix(R.T@t['root_r'][a+j+1]).as_rotvec()*u).as_matrix()
                # Unchanged arm/waist joints must also interpolate at subframes.
                if k:
                    fixed=np.setdiff1d(np.arange(xs.shape[1]),ids)
                    at[fixed]=xs[a+j,fixed]*(1-u)+xs[a+j+1,fixed]*u
                r.set(at,R)
                for body,surface in r.surface_models.items():
                    hull=surface['hull'];height=hull@r.rot(body)[2]+r.point(body)[2]
                    if height.min()>.01:continue
                    fixed=lambda v:(a>0 and v<2) or (b<len(xs) and v>=n-2)
                    immutable=fixed(j) and (not k or fixed(j+1))
                    target=floor_target(height.min(),immutable)
                    for vertex in np.argsort(height)[:4]:
                        P,_=r.jac(body,hull[vertex]);add(j,P[2:3,ids],[target-height[vertex]],[np.inf],u)
                if k:continue
                for side,g in r.sides.items():
                    v=t['sides'][side]
                    if v['contact'][a+j]<=0:continue
                    pivot=v['pivots'][a+j];P,_=r.jac(g['foot'],pivot)
                    delta=r.point(g['foot'],pivot)-v['anchors'][a+j]
                    hull=g['hull'];low=hull[np.argmin(hull@r.rot(g['foot'])[2])];G,_=r.jac(g['foot'],low)
                    delta[2]=r.point(g['foot'],low)[2];J=np.vstack([P[:2],G[2]])[:,ids]
                    add(j,J,-.00049-delta,.00049-delta)
                for c in t.get('body_contacts',[]):
                    mode=int(c['mode'][a+j])
                    if not mode:continue
                    body=c['body'];hull=c['robot_hull'];point=hull[np.argmin(hull@r.rot(body)[2])]
                    now=r.point(body,point).copy();P,_=r.jac(body,point)
                    r.set(xs[a+j],t['root_r'][a+j]);old=r.point(body,point).copy();r.set(at,R)
                    delta=now-old;J=P[:,ids]
                    if mode!=1:delta=delta[2:];J=J[2:]
                    add(j,J,-.00049-delta,.00049-delta)
        corridor=np.where(ids<3,.02,.2)
        lo=np.maximum(r.lower[ids],ref-corridor)-q;hi=np.minimum(r.upper[ids],ref+corridor)-q
        if a:lo[:2]=hi[:2]=ref[:2]-q[:2]
        if b<len(xs):lo[-2:]=hi[-2:]=ref[-2:]-q[-2:]
        A=sp.vstack([I,D,sp.csc_matrix(np.asarray(rows))],format='csc')
        low=np.r_[lo.ravel(),-np.tile(step,n-1)-D@q.ravel(),lower]
        high=np.r_[hi.ravel(),np.tile(step,n-1)-D@q.ravel(),upper]
        delta,detail=_qp(H,H@(q-ref).ravel(),A,low,high,np.zeros(n*d),scaling=10)
        q+=delta.reshape(n,d);out[a:b,ids]=q
        minimum=float(floor_samples(r,local,out[a:b]).min())
        history.append(dict(iteration=iteration,minimum_floor_m=minimum,qp=detail))
        if minimum>=-.00098:break
    out[a:b,ids]=out[a:b,ids].astype(np.float32).astype(float)
    if a:out[a:a+2]=xs[a:a+2]
    if b<len(xs):out[b-2:b]=xs[b-2:b]
    return out,dict(policy='subframe_root_leg_floor_polish_v1',legs_free=bool(supported),history=history)
