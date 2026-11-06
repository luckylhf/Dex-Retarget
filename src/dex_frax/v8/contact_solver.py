"""Bounded whole-body contact refinement with small sequential sparse QPs."""
import numpy as np
from .support_acceptance import support_constraint_failed
import warnings
from scipy.optimize import linprog,OptimizeWarning
from scipy.sparse import csc_matrix,eye,vstack,hstack
from .solver import Objective
from .multicontact import lowest
from .trajectory import _qp


def feasible_projection(candidate,A,lower,upper):
    """Nearest feasible point in infinity norm, with every original row intact.

    The small contact QP can stall on nearly parallel surface rows. This LP
    repairs its finite iterate rather than accepting a larger violation. The
    nonlinear merit/clearance checks still decide whether to take this step.
    """
    candidate=np.asarray(candidate,dtype=float);A=csc_matrix(A);n=A.shape[1]
    if not np.isfinite(candidate).all() or np.max(abs(candidate))>1e3:candidate=np.zeros(n)
    finite_lo=np.isfinite(lower);finite_hi=np.isfinite(upper)
    rows=vstack([A[finite_hi],-A[finite_lo]],format='csc')
    values=np.r_[upper[finite_hi],-lower[finite_lo]]
    rows=vstack([hstack([rows,csc_matrix((len(values),1))]),hstack([eye(n),-np.ones((n,1))]),hstack([-eye(n),-np.ones((n,1))])],format='csc')
    values=np.r_[values,candidate,-candidate]
    with warnings.catch_warnings():
        # SciPy forwards this documented HiGHS option, but warns because its
        # public wrapper does not list threads. Avoid per-worker thread pools.
        warnings.filterwarnings('ignore',message='Unrecognized options detected.*',category=OptimizeWarning)
        result=linprog(np.r_[np.zeros(n),1.],A_ub=rows,b_ub=values,bounds=[(None,None)]*n+[(0,None)],method='highs',options={'threads':1,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not result.success:raise ValueError('Contact linear constraints infeasible: '+result.message)
    point=result.x[:n];at=A@point
    excess=max(float(np.max(lower-at)),float(np.max(at-upper)),0.)
    if not np.isfinite(point).all() or excess>1e-7:raise ValueError('Contact feasibility projection residual: '+str(excess))
    return point,dict(status='independent_linear_feasibility_projection',iterations=int(result.nit),constraint_excess=excess,optimal_status=False,projection_change=float(np.max(abs(point-candidate))))


def refine(robot,t,xs,cfg,progress=None,objective_factory=None):
    if 'body_contacts' not in t:return xs,[]
    out=xs.copy();reports=[];robot.body_contact_qp_summary={'steps':0,'feasible_budget_steps':0,'normalized_retries':0,'feasibility_projections':0,'maximum_constraint_excess':0.};step=robot.joint_step(t['fps'],cfg);previous=None;predicted=None
    feet={g['foot'] for g in robot.sides.values()}
    def qp(P,q,A,lo,hi,initial):
        # Use the same explicit 1e-6 residual ceiling as a solved QP. A budget
        # status alone is not physical infeasibility; nonlinear geometry, clipped
        # joint bounds and final exported per-axis speeds are checked separately.
        try:
            delta,info=_qp(P,q,A,lo,hi,initial,scaling=10,require_optimal=False,feasible_tolerance=1e-6)
        except ValueError:
            # Multiplying the entire objective by one positive scalar preserves
            # its minimizer, while avoiding extreme primal/dual unit imbalance.
            scale=max(1.,float(np.max(abs(P.diagonal()))),float(np.max(abs(q))))
            robot.body_contact_qp_summary['normalized_retries']+=1
            try:
                delta,info=_qp(P/scale,q/scale,A,lo,hi,initial,scaling=10,require_optimal=False,feasible_tolerance=1e-6)
            except ValueError as error:
                candidate=getattr(error,'candidate',None)
                delta,info=feasible_projection(initial if candidate is None else candidate,A,lo,hi)
                robot.body_contact_qp_summary['feasibility_projections']+=1
        summary=robot.body_contact_qp_summary;summary['steps']+=1
        summary['feasible_budget_steps']+=int(not info['optimal_status'])
        summary['maximum_constraint_excess']=max(summary['maximum_constraint_excess'],info['constraint_excess'])
        return delta,info
    for i,base in enumerate(xs):
        robot.set(base,t['root_r'][i])
        body_contacts=[v for v in t['body_contacts'] if v['mode'][i]>0]
        minima={b:robot.point(b,lowest(robot,b))[2] for b in robot.surface_models}
        needed=any(z<-1e-5 for b,z in minima.items() if b not in feet)
        if previous is not None:
            needed=needed or np.any(abs(base[3:]-previous[3:])>step+1e-8)
            # Decay a prior correction through free frames; jumping straight
            # back to the original root creates an artificial support-release pop.
            needed=needed or np.linalg.norm(previous[:3]-xs[i-1,:3])>1e-4
        if not needed:
            out[i]=base;predicted=base if previous is None else base+np.clip(base-previous,-.1,.1);previous=base.copy();continue
        factory=Objective if objective_factory is None else objective_factory
        obj=factory(robot,t,i,cfg,previous,predicted)
        lo=robot.lower.copy();hi=robot.upper.copy()
        if 'repair_lower_bounds' in t:
            lo=np.maximum(lo,t['repair_lower_bounds'][i])
        if previous is not None:
            lo[3:]=np.maximum(lo[3:],previous[3:]-step+1e-8)
            hi[3:]=np.minimum(hi[3:],previous[3:]+step-1e-8)
        x=np.clip(base,lo+1e-8,hi-1e-8)
        contacts={s:v for s,v in t['sides'].items() if v['contact'][i]>0}
        foot_relaxed=False
        def geometry(at):
            robot.set(at,t['root_r'][i]);values=[];rows=[]
            for b,g in robot.surface_models.items():
                # The full-hull acceptance below detects changes in the active set.
                R=robot.rot(b);z=g['hull']@R[2,:]+robot.point(b)[2]
                if z.min()>.15:continue
                ids=np.argsort(z)[:4]
                for local in g['hull'][ids]:
                    P,_=robot.jac(b,local);values.append(robot.point(b,local)[2]);rows.append(P[2])
            return np.asarray(values),np.asarray(rows).reshape(-1,32)
        def foot_constraints(at):
            robot.set(at,t['root_r'][i]);v=[];rows=[]
            for s,c in contacts.items():
                b=robot.sides[s]['foot'];local=c['pivots'][i];P,_=robot.jac(b,local)
                v.extend((robot.point(b,local)-c['anchors'][i])[:2]);rows.extend(P[:2])
                local=lowest(robot,b);P,_=robot.jac(b,local);v.append(robot.point(b,local)[2]);rows.append(P[2])
            return np.asarray(v),np.asarray(rows).reshape(-1,32)
        def merit(at):
            res,J=obj.evaluate(at);z,G=geometry(at)
            return float(res@res)+1e6*float(np.minimum(z,0)@np.minimum(z,0)),res,J,z,G
        value,res,J,z,G=merit(x)
        for iteration in range(14):
            P=csc_matrix(J.T@J)+eye(32,format='csc')*.01;gradient=J.T@res
            trust=np.r_[np.full(3,max(.05,-float(z.min())+.01) if len(z) else .05),np.full(29,.25)]
            lb=np.maximum(lo-x,-trust);ub=np.minimum(hi-x,trust)
            A=vstack([eye(32,format='csc'),csc_matrix(G)],format='csc')
            # A planted sole is exactly on z=0. A positive clearance margin
            # here would contradict its hard zero-height equality.
            lower=np.r_[lb,-z];upper=np.r_[ub,np.full(len(z),np.inf)]
            fv,F=foot_constraints(x)
            if contacts and not foot_relaxed:
                Ac=vstack([A,csc_matrix(F)],format='csc');lc=np.r_[lower,-fv];uc=np.r_[upper,-fv]
                try:delta,_=qp(P,gradient,Ac,lc,uc,np.zeros(32))
                except ValueError:
                    foot_relaxed=True;delta,_=qp(P,gradient,A,lower,upper,np.zeros(32))
            else:delta,_=qp(P,gradient,A,lower,upper,np.zeros(32))
            accepted=False
            for alpha in (1.,.5,.25,.125):
                proposal=np.clip(x+alpha*delta,lo,hi)
                nv,nr,nJ,nz,nG=merit(proposal)
                old_floor=float(z.min()) if len(z) else 1.
                new_floor=float(nz.min()) if len(nz) else 1.
                improves_clearance=old_floor < -1e-5 and new_floor > old_floor+1e-7
                stays_feasible=new_floor >= -1e-5
                if improves_clearance or (stays_feasible and nv<value-1e-8):
                    gain=value-nv;x,value,res,J,z,G=proposal,nv,nr,nJ,nz,nG;accepted=True;break
            if not accepted:break
            if np.max(abs(alpha*delta))<1e-6:break
            if (not len(z) or z.min()>=-1e-5) and 0<=gain<1e-6*max(value,1.):break
        # Finish with a minimum-change feasibility solve if nonlinear surface
        # switches left a small penetration. All joints and the common root
        # remain variables; speed/angle and planted-foot constraints stay hard.
        for cleanup in range(5):
            z,G=geometry(x)
            if not len(z) or z.min()>=-1e-5:break
            A=vstack([eye(32,format='csc'),csc_matrix(G)],format='csc')
            lb=np.r_[lo-x,-z];ub=np.r_[hi-x,np.full(len(z),np.inf)]
            fv,F=foot_constraints(x)
            if contacts:
                A=vstack([A,csc_matrix(F)],format='csc');lb=np.r_[lb,-fv];ub=np.r_[ub,-fv]
            try:delta,_=qp(eye(32,format='csc'),np.zeros(32),A,lb,ub,np.zeros(32))
            except ValueError:break
            proposal=np.clip(x+delta,lo,hi);newz,_=geometry(proposal)
            if len(newz) and newz.min()>z.min()+1e-8:x=proposal
            else:break
        robot.set(x,t['root_r'][i]);minimum=min(robot.point(b,lowest(robot,b))[2] for b in robot.surface_models)
        eq=float(np.max(abs(foot_constraints(x)[0]))) if contacts else 0.
        if support_constraint_failed(minimum,eq):
            reports.append(dict(frame=i,minimum_z=float(minimum),foot_contact_error_m=eq,foot_constraint_relaxed=foot_relaxed,stage='full_body_scene_clearance'))
        out[i]=x;predicted=x if previous is None else x+np.clip(x-previous,-.1,.1);previous=x.copy()
        if progress and ((i+1)%120==0 or i==len(xs)-1):progress(i+1,len(xs))
    return out,reports
