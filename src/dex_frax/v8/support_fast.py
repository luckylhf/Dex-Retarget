"""CPU support refinement; same Dex tasks and feasibility checks.

SOMA's geometric limb stabilizer supplies an optional two-link warm start,
never motor commands. A small differential IK QP uses the real Dex Jacobian.
The original SLSQP remains the fallback. No velocity or geometry limit changes.
"""
import time
import numpy as np
from .support_acceptance import support_constraint_failed
from scipy.optimize import minimize, least_squares
from scipy.sparse import csc_matrix, eye, vstack, triu
import osqp
from .math3d import skew
from .solver import Objective
from .multicontact import lowest, contact_local
from .contact_tasks import body_height_task


def scale_qp(H,gradient,A,lower,upper,mode):
    """Scale the complete quadratic objective by one positive scalar.

    The minimizer, task weight ratios, variables and constraints are unchanged.
    Finite-tolerance iterates can differ; the original physical checks remain.
    D is identity, retained in this interface for the explicit step mapping.
    """
    D=np.ones(len(gradient))
    if mode=='none':return H,gradient,A,lower,upper,D
    if mode=='objective':
        factor=max(float(np.median(np.diag(H))),1e-12)
        return H/factor,gradient/factor,A,lower,upper,D
    raise ValueError('Unknown support QP scaling: '+str(mode))


class SupportObjective:
    """Exact restriction of Objective to root translation and both legs.

Constant upper-body residuals are evaluated once. Their scalar cost is retained
so function-value stopping tests retain the original meaning.
"""
    def __init__(self, robot, t, i, weights, base, ids, previous=None, predicted=None):
        self.r,self.t,self.i,self.w=robot,t,i,weights
        self.base,self.ids=base.copy(),np.asarray(ids)
        self.prev,self.pred=previous,predicted
        self.cached=None;self.calls=0
        full,_=Objective(robot,t,i,weights,previous,predicted).evaluate(base)
        res,_=self.evaluate(base[self.ids])
        self.constant=.5*(full@full-res@res)

    def evaluate(self,y):
        if self.cached is not None and np.array_equal(y,self.cached):return self.result
        r,t,i,w=self.r,self.t,self.i,self.w;ids=self.ids
        x=self.base.copy();x[ids]=y;r.set(x,t['root_r'][i])
        nc=len(ids);eye_=np.eye(nc);res=[];js=[];cache={}
        def body(b):
            if b not in cache:
                J,A=r.jac(b);cache[b]=(r.point(b),J[:,ids],A[:,ids])
            return cache[b]
        def add(v,J,weight):res.append(np.atleast_1d(v)*weight);js.append(np.atleast_2d(J)*weight)
        def point(b,local=None):
            p,J,A=body(b)
            if local is not None:
                offset=r.rot(b)@local;return p+offset,J-skew(offset)@A
            return p,J
        def axis(b,local,target,weight):
            _,_,A=body(b);a=r.rot(b)@local;add(a-target,-skew(a)@A,weight)
        def direction(a,b,target,weight):
            p,P=point(a);q,Q=point(b);d=q-p;length=np.linalg.norm(d);u=d/length
            add(u-target,(np.eye(3)-np.outer(u,u))@(Q-P)/length,weight)
        add(y[:3]-t['root_p'][i],eye_[:3],w['root_position'])
        for side,v in t['sides'].items():
            g=r.sides[side];b=g['foot']
            direction(g['hip'],g['knee'],v['thigh'][i],w['leg_direction'])
            direction(g['knee'],b,v['shin'][i],w['leg_direction'])
            p,J=point(b);add(p-v['foot_target_p'][i],J,w['foot_position'])
            for ax in (0,2):axis(b,np.eye(3)[ax],v['foot_target_r'][i,:,ax],w['foot_orientation'])
            low,L=point(b,r.lowest(side))
            if v['contact'][i]>0:
                cp,C=point(b,v['pivots'][i]);add(np.r_[cp[:2]-v['anchors'][i,:2],low[2]],np.vstack([C[:2],L[2]]),w['contact_position']*v['contact'][i])
            add([min(0.,low[2]-.0005)],L[2:3] if low[2]<.0005 else np.zeros((1,nc)),w['ground_penalty'])
            j=int(np.flatnonzero(ids==g['leg'][2])[0]);add([y[j]],eye_[j:j+1],w['hip_yaw_rest'])
        if 'body_contacts' in t:
            feet={g['foot'] for g in r.sides.values()}
            for b in r.surface_models:
                if b in feet:continue
                local=lowest(r,b);p,J=point(b,local)
                add([min(0.,p[2]-.0005)],J[2:3] if p[2]<.0005 else np.zeros((1,nc)),w['ground_penalty'])
            for contact in t['body_contacts']:
                if not contact['mode'][i]:continue
                p,J=point(contact['body'],contact_local(r,contact))
                confidence=contact['confidence'][i] if 'confidence' in contact else 1.
                weight=w.get('body_contact_position',0.)*contact['source_geometry_confidence']*confidence
                value,derivative=body_height_task(p[2],t,i,contact,w)
                add([value],J[2:3]*derivative,weight)
        if self.prev is not None:
            f=t['fps']/60.
            add(y-self.prev[ids],eye_,w['joint_velocity']*f)
            add(y-self.pred[ids],eye_,w['joint_acceleration']*f*f)
            goal=self.prev[:3]+t['root_p'][i]-t['root_p'][i-1]
            add(y[:3]-goal,eye_[:3],w['root_velocity']*f)
            add(y[:3]-self.pred[:3],eye_[:3],w['root_acceleration']*f*f)
        self.cached=y.copy();self.result=np.concatenate(res),np.vstack(js);self.calls+=1
        return self.result

    def fun(self,y):
        res,J=self.evaluate(y);return .5*float(res@res)+self.constant,J.T@res


def two_link_knee(hip,knee,ankle,target):
    """Cosine-rule knee position with current bend-plane hint, as in SOMA.

    Clamp to reachability; protect zero/straight limbs. This is a position
    proposal only: the subsequent real-model fit enforces actual joint axes.
    """
    a,b,c,target=map(lambda v:np.asarray(v,dtype=float),(hip,knee,ankle,target))
    l1=np.linalg.norm(b-a);l2=np.linalg.norm(c-b);d=target-a;distance=np.linalg.norm(d)
    if min(l1,l2,distance)<1e-8:return b.copy()
    axis=d/distance;distance=np.clip(distance,abs(l1-l2)+1e-6,l1+l2-1e-6)
    along=(l1*l1+distance*distance-l2*l2)/(2*distance)
    hint=b-a-axis*np.dot(b-a,axis)
    if np.linalg.norm(hint)<1e-7:
        ref=np.eye(3)[np.argmin(abs(axis))];hint=ref-axis*np.dot(ref,axis)
    return a+axis*along+hint/np.linalg.norm(hint)*np.sqrt(max(0.,l1*l1-along*along))


def geometric_seed(robot,t,i,base,ids,lo,hi):
    """SOMA-style geometric positions fitted through the actual Dex model."""
    robot.set(base,t['root_r'][i]);targets=[]
    for side,g in robot.sides.items():
        foot=t['sides'][side]['foot_target_p'][i]
        knee=two_link_knee(robot.point(g['hip']),robot.point(g['knee']),robot.point(g['foot']),foot)
        targets.extend([(g['knee'],knee),(g['foot'],foot)])
    cache=None;result=None
    def evaluate(y):
        nonlocal cache,result
        if cache is not None and np.array_equal(y,cache):return result
        x=base.copy();x[ids]=y;robot.set(x,t['root_r'][i]);res=[];jac=[]
        for b,target in targets:
            J,_=robot.jac(b);res.extend(robot.point(b)-target);jac.extend(J[:,ids])
        res.extend((y-base[ids])*.03);jac.extend(np.eye(len(ids))*.03)
        cache=y.copy();result=np.asarray(res),np.asarray(jac);return result
    ans=least_squares(lambda y:evaluate(y)[0],np.clip(base[ids],lo+1e-8,hi-1e-8),jac=lambda y:evaluate(y)[1],bounds=(lo,hi),max_nfev=5)
    return ans.x


def constrain_support(robot,t,xs,cfg,progress=None):
    """Checked bypass -> bounded differential IK -> original SQP fallback."""
    ids=np.array([0,1,2]+sum([g['leg'] for g in robot.sides.values()],[]));out=xs.copy();reports=[]
    leg_indices=ids[3:]-3;step=robot.joint_step(t['fps'],cfg)[leg_indices]
    robot.support_branch_relaxations=[]
    prev=None;pred=None
    counts=dict(bypass=0,light=0,fallback=0,geometry_seeds=0,qp_steps=0,objective_evaluations=0,frames=[])
    counts.update(objective_builds=0,light_stop_reasons={},fallback_triggers={},
                  fallback_successes=0,fallback_failures=0,
                  route_seconds={k:0. for k in ('bypass','light','fallback')},
                  qp_setup_s=0.,qp_solve_s=0.,fallback_s=0.)
    started=time.perf_counter()
    for i,base in enumerate(xs):
        frame_started=time.perf_counter();obj=None
        def state(y):
            x=base.copy();x[ids]=y;robot.set(x,t['root_r'][i]);return x
        state(base[ids]);active={s:set(np.argsort(g['hull']@robot.rot(g['foot'])[2])[:8].tolist()) for s,g in robot.sides.items()}
        geometry_cache={}
        def geometry(y,equality=False,all_vertices=False,jacobian=True):
            key=(equality,all_vertices,tuple((s,tuple(sorted(a))) for s,a in active.items()),jacobian)
            cached=geometry_cache.get(key)
            if cached is not None and np.array_equal(y,cached[0]):return cached[1]
            # Reuse already evaluated values when the optimizer computed rows.
            if not jacobian:
                cached=geometry_cache.get(key[:-1]+(True,))
                if cached is not None and np.array_equal(y,cached[0]):return cached[1][0],None
            state(y);values=[];rows=[]
            for side,g in robot.sides.items():
                v=t['sides'][side];b=g['foot'];p=robot.point(b);R=robot.rot(b)
                if jacobian:P,A=robot.jac(b)
                if equality:
                    if v['contact'][i]>0:
                        d=R@v['pivots'][i];values.extend((p+d-v['anchors'][i])[:2])
                        low=R@robot.lowest(side);values.append(p[2]+low[2])
                        if jacobian:rows.extend((P-skew(d)@A)[:2,ids]);rows.append((P-skew(low)@A)[2,ids])
                else:
                    if v['contact'][i]>0 and not all_vertices:continue
                    vertices=g['hull'] if all_vertices else g['hull'][sorted(active[side])];d=vertices@R.T
                    values.extend(p[2]+d[:,2])
                    if jacobian:rows.extend((P[2]+np.cross(d,[0.,0.,1.])@A)[:,ids])
            result=np.asarray(values),np.asarray(rows).reshape(-1,len(ids)) if jacobian else None
            geometry_cache[key]=(y.copy(),result);return result
        has_contact=any(v['contact'][i]>0 for v in t['sides'].values())
        lo=robot.lower[ids].copy();hi=robot.upper[ids].copy()
        lo[3:]=np.maximum(lo[3:],base[ids][3:]-.30);hi[3:]=np.minimum(hi[3:],base[ids][3:]+.30)
        if prev is not None:
            vlo=np.maximum(robot.lower[ids[3:]],prev[ids[3:]]-step);vhi=np.minimum(robot.upper[ids[3:]],prev[ids[3:]]+step)
            slo=np.maximum(lo[3:],vlo);shi=np.minimum(hi[3:],vhi);valid=shi>slo+1e-6
            if not valid.all():robot.support_branch_relaxations.append(dict(frame=i,joints=[robot.names[k] for k in leg_indices[~valid]]))
            lo[3:]=np.where(valid,slo,vlo);hi[3:]=np.where(valid,shi,vhi)
        def checks(y):
            floor=float(geometry(y,all_vertices=True,jacobian=False)[0].min());eq=float(np.max(abs(geometry(y,True,jacobian=False)[0]))) if has_contact else 0.
            valid=np.isfinite(y).all() and np.all(y>=lo-1e-9) and np.all(y<=hi+1e-9) and floor>=-1e-5 and eq<=1e-4
            return bool(valid),floor,eq
        seed=np.clip(base[ids],lo+1e-7,hi-1e-7);y=seed.copy();route='light';qp_count=0
        stop_reason='bypass';qp_status=None;qp_iterations=0;fallback_triggers=[]
        pre_fallback_floor=None;pre_fallback_eq=None;fallback_seconds=0.
        continuous=prev is None or i==0 or np.linalg.norm(prev[:3]-xs[i-1,:3])<=1e-4
        if cfg.get('support_fast_bypass',True) and np.array_equal(y,base[ids]) and checks(y)[0] and continuous:
            route='bypass'
        else:
            obj=SupportObjective(robot,t,i,cfg,base,ids,prev,pred);counts['objective_builds']+=1
            stop_reason='iteration_budget'
            if cfg.get('support_geometric_seed',False):
                proposal=geometric_seed(robot,t,i,base,ids,lo,hi);counts['geometry_seeds']+=1
                if obj.fun(proposal)[0]<obj.fun(y)[0]:y=proposal
            for iteration in range(cfg.get('support_light_iterations',6)):
                # QP handles redundant active rows. Include the entire shoe:
                # otherwise the minimum vertex can switch after each update.
                res,J=obj.evaluate(y);eq,E=geometry(y,True);z,G=geometry(y,all_vertices=True)
                H=J.T@J+np.eye(len(ids))*.01;gradient=J.T@res
                trust=np.r_[np.full(3,.05),np.full(len(ids)-3,.15)]
                blocks=[eye(len(ids),format='csc')];lower=[np.maximum(lo-y,-trust)];upper=[np.minimum(hi-y,trust)]
                if len(eq):blocks.append(csc_matrix(E));lower.append(-eq);upper.append(-eq)
                if len(z):blocks.append(csc_matrix(G));lower.append(-z);upper.append(np.full(len(z),np.inf))
                Hq,gq,Aq,lq,uq,D=scale_qp(H,gradient,vstack(blocks,format='csc'),np.concatenate(lower),np.concatenate(upper),cfg.get('support_qp_scaling','none'))
                qp=osqp.OSQP();qp.setup(P=triu(csc_matrix(Hq),format='csc'),q=gq,A=Aq,l=lq,u=uq,verbose=False,eps_abs=1e-8,eps_rel=1e-8,max_iter=4000,polishing=False)
                answer=qp.solve(raise_error=False);qp_count+=1
                qp_status=str(answer.info.status);qp_iterations+=int(answer.info.iter)
                counts['qp_setup_s']+=float(answer.info.setup_time);counts['qp_solve_s']+=float(answer.info.solve_time)
                if answer.x is None or answer.info.status_val not in (1,2) or not np.isfinite(answer.x).all():
                    stop_reason='qp_'+qp_status.lower().replace(' ','_');break
                delta=answer.x*D
                old_cost=obj.fun(y)[0];old_violation=np.sum(abs(eq))+np.sum(np.maximum(-z,0))
                accepted=False
                for alpha in (1.,.5,.25):
                    proposal=np.clip(y+alpha*delta,lo,hi);pe,_=geometry(proposal,True);pz,_=geometry(proposal,all_vertices=True)
                    violation=np.sum(abs(pe))+np.sum(np.maximum(-pz,0))
                    if obj.fun(proposal)[0]+1e5*violation < old_cost+1e5*old_violation+1e-10:
                        y=proposal;accepted=True;break
                if not accepted:stop_reason='line_search_rejected';break
                # Feasibility is checked on the whole hull, not just QP rows.
                if checks(y)[0] and (np.max(abs(alpha*delta))<1e-4 or iteration>=2):
                    stop_reason='feasible_exit';break
        ok,floor,eq=checks(y)
        if not ok:
            fallback_started=time.perf_counter()
            pre_fallback_floor,pre_fallback_eq=floor,eq
            if not np.isfinite(y).all() or not np.isfinite([floor,eq]).all():fallback_triggers.append('nonfinite')
            if np.any(y<lo-1e-9) or np.any(y>hi+1e-9):fallback_triggers.append('joint_or_velocity_bounds')
            if floor < -1e-5:fallback_triggers.append('shoe_penetration')
            if eq > 1e-4:fallback_triggers.append('contact_residual')
            route='fallback';cs=[]
            if any(v['contact'][i]==0 for v in t['sides'].values()):cs.append({'type':'ineq','fun':lambda y:geometry(y)[0],'jac':lambda y:geometry(y)[1]})
            if has_contact:cs.append({'type':'eq','fun':lambda y:geometry(y,True)[0],'jac':lambda y:geometry(y,True)[1]})
            # The old SQP recipe, using the exact restricted objective.
            ans=minimize(obj.fun,seed,jac=True,bounds=list(zip(lo,hi)),constraints=cs,method='SLSQP',options={'maxiter':90,'ftol':1e-8})
            for _ in range(5):
                state(ans.x);added=False
                for side,g in robot.sides.items():
                    zz=g['hull']@robot.rot(g['foot'])[2]+robot.point(g['foot'])[2];bad=np.flatnonzero(zz < -2e-6)
                    new=set(bad[np.argsort(zz[bad])[:8]].tolist())-active[side]
                    if new:active[side].update(new);added=True
                if not added:break
                ans=minimize(obj.fun,ans.x,jac=True,bounds=list(zip(lo,hi)),constraints=cs,method='SLSQP',options={'maxiter':65,'ftol':1e-8})
            ok,floor,eq=checks(ans.x)
            if not ok:
                ans=minimize(obj.fun,ans.x,jac=True,bounds=list(zip(lo,hi)),constraints=cs,method='SLSQP',options={'maxiter':220,'ftol':1e-9})
            y=ans.x;ok,floor,eq=checks(y)
            # Final acceptance tolerates 1 mm; optimization and fallback still aim tighter.
            ok=(np.isfinite(y).all() and np.all(y>=lo-1e-9) and np.all(y<=hi+1e-9)
                and not support_constraint_failed(floor,eq))
            if not ok:reports.append(dict(frame=i,success=bool(ans.success),minimum_z=floor,contact_error_m=eq,message=str(ans.message)))
            fallback_seconds=time.perf_counter()-fallback_started
            counts['fallback_s']+=fallback_seconds
            counts['fallback_successes' if ok else 'fallback_failures']+=1
        x=state(y);out[i]=x;pred=x if prev is None else x+np.clip(x-prev,-.1,.1);prev=x.copy()
        counts[route]+=1;counts['qp_steps']+=qp_count;counts['objective_evaluations']+=obj.calls if obj is not None else 0
        counts['light_stop_reasons'][stop_reason]=counts['light_stop_reasons'].get(stop_reason,0)+1
        for reason in fallback_triggers:counts['fallback_triggers'][reason]=counts['fallback_triggers'].get(reason,0)+1
        elapsed=time.perf_counter()-frame_started;counts['route_seconds'][route]+=elapsed
        counts['frames'].append(dict(frame=i,route=route,qp_steps=qp_count,minimum_z=floor,contact_error_m=eq,
                                    light_stop_reason=stop_reason,qp_status=qp_status,qp_iterations=qp_iterations,
                                    fallback_triggers=fallback_triggers,pre_fallback_minimum_z=pre_fallback_floor,
                                    pre_fallback_contact_error_m=pre_fallback_eq,final_feasible=ok,
                                    elapsed_s=elapsed,fallback_s=fallback_seconds))
        if progress and ((i+1)%120==0 or i==len(xs)-1):progress(i+1,len(xs))
    counts['elapsed_s']=time.perf_counter()-started;robot.support_fast_report=counts
    return out,reports
