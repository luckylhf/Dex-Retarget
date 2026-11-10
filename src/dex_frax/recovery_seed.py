"""Feasible temporal initialization for nonlinear collision recovery only."""
import numpy as np
from scipy import sparse as sp
from .v8.trajectory import _qp


def project(q, lower, upper, step, frozen, joins, previous=None, following=None, physical_step=None, tolerance=2e-6):
    q=np.asarray(q,dtype=float);n,d=q.shape
    lo=np.broadcast_to(lower,q.shape).copy();hi=np.broadcast_to(upper,q.shape).copy()
    if previous is not None:lo[0]=np.maximum(lo[0],previous-step);hi[0]=np.minimum(hi[0],previous+step)
    if following is not None:lo[-1]=np.maximum(lo[-1],following-step);hi[-1]=np.minimum(hi[-1],following+step)
    lo[frozen]=hi[frozen]=np.asarray(joins)[frozen]
    D=sp.kron(sp.diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n)),sp.eye(d),format='csc')
    caps=np.broadcast_to(step,(n-1,d)).copy()
    from .v8.self_collision import _frozen_velocity_steps
    caps=_frozen_velocity_steps(caps,abs(np.diff(joins,axis=0)),frozen[:-1]&frozen[1:],physical_step,tolerance)
    A=sp.vstack([sp.eye(n*d),D],format='csc');low=np.r_[lo.ravel(),-caps.ravel()];high=np.r_[hi.ravel(),caps.ravel()]
    values=A@q.ravel()
    if np.all(values>=low-1e-8) and np.all(values<=high+1e-8):return q.copy(),{'executed':False}
    solution,report=_qp(sp.eye(n*d,format='csc'),-q.ravel(),A,low,high,np.asarray(joins).ravel(),scaling=10)
    out=solution.reshape(q.shape);out[frozen]=np.asarray(joins)[frozen]
    report.update(executed=True,maximum_seed_change_rad=float(abs(out-q).max()),
        speed_excess_before_rad_step=float(np.maximum(abs(np.diff(q,axis=0))-step,0.).max()),
        speed_excess_after_rad_step=float(np.maximum(abs(np.diff(out,axis=0))-step,0.).max()))
    return out,report
