"""Bounded whole-body IK, analytic MuJoCo Jacobians, shared pelvis translation.

The pelvis rotation comes from calibrated source anatomy. Translation and all
29 hinges are solved together, so both support legs constrain the SAME root.
No output-stage root-height lift or pasted historical leg trajectory is used.
"""
import numpy as np
from .support_acceptance import support_constraint_failed
from scipy.optimize import least_squares, minimize
from .math3d import unit,skew
from .anatomy import wrist_barrier
from .task_weights import hand_weights
from .contact_tasks import body_height_task

class Objective:
    def __init__(self,robot,t,i,weights,previous=None,predicted=None):
        self.robot,self.t,self.i,self.w=robot,t,i,weights
        self.prev,self.pred=previous,predicted
        self.cache_x=None

    def evaluate(self,x):
        if self.w.get('_repair_deadline') is not None:
            import time
            if time.monotonic()>=self.w['_repair_deadline']:raise TimeoutError('window_solver_deadline')
        if self.cache_x is not None and np.array_equal(x,self.cache_x):return self.cache
        rob,t,i,w=self.robot,self.t,self.i,self.w
        rob.set(x,t['root_r'][i]);res=[];jac=[];nc=len(x);eye=np.eye(nc)
        jp={};jr={}
        def body(b):
            if b not in jp:jp[b],jr[b]=rob.jac(b)
            return rob.point(b),jp[b],jr[b]
        def add(value,j,weight):
            res.append(np.atleast_1d(value)*weight);jac.append(np.atleast_2d(j)*weight)
        def point(b,local=None):
            p,J,A=body(b)
            if local is not None:
                v=rob.rot(b)@local;p=p+v;J=J-skew(v)@A
            return p,J
        def axis(b,local,target,weight):
            _,_,A=body(b);a=rob.rot(b)@local
            add(a-target,-skew(a)@A,weight)
        def direction(a,b,target,weight):
            pa,Ja=point(a);pb,Jb=point(b);delta=pb-pa;length=np.linalg.norm(delta);v=delta/length
            add(v-target,(np.eye(3)-np.outer(v,v))@(Jb-Ja)/length,weight)
        add(x[:3]-t['root_p'][i],eye[:3],w['root_position'])
        for ax in (0,2):axis(rob.torso,np.eye(3)[ax],t['torso_r'][i,:,ax],w['torso_orientation'])
        for side,v in t['sides'].items():
            g=rob.sides[side]
            hand_direction_weight,hand_width_weight=hand_weights(w,v,i)
            direction(g['hip'],g['knee'],v['thigh'][i],w['leg_direction'])
            direction(g['knee'],g['foot'],v['shin'][i],w['leg_direction'])
            fp,F=point(g['foot']);add(fp-v['foot_target_p'][i],F,w['foot_position'])
            for ax in (0,2):axis(g['foot'],np.eye(3)[ax],v['foot_target_r'][i,:,ax],w['foot_orientation'])
            if v['contact'][i]>0:
                cp,C=point(g['foot'],v['pivots'][i])
                low,L=point(g['foot'],rob.lowest(side))
                add(np.r_[cp[:2]-v['anchors'][i,:2],low[2]],np.vstack([C[:2],L[2]]),w['contact_position']*v['contact'][i])
            # Enforce actual visual-mesh ground clearance inside the solve.
            low,L=point(g['foot'],rob.lowest(side))
            add([min(0.,low[2]-.0005)],L[2:3] if low[2]<.0005 else np.zeros((1,nc)),w['ground_penalty'])
            direction(g['shoulder'],g['elbow'],v['upper'][i],w['arm_direction'])
            direction(g['elbow'],g['wrist'],v['lower'][i],w['arm_direction'])
            # Morphology-matched wrist XYZ in the shoulder frame. Solving this
            # along with elbow/forearm direction discourages wrist compensation.
            sh,S=point(g['shoulder']);wr,W=point(g['wrist'])
            wr_goal=sh+g['upper_length']*v['upper'][i]+g['lower_length']*v['lower'][i]
            add(wr-wr_goal,W-S,6.)
            axis(g['tcp'],np.array([0.,0.,-1.]),v['hand_forward'][i],hand_direction_weight)
            axis(g['tcp'],np.array([1.,0.,0.]),v['hand_width'][i],hand_width_weight)
            axis(g['forearm'],np.array([1.,0.,0.]),v['forearm_width'][i],w['forearm_twist'])
            axis(g['shoulder'],np.array([0.,-1.,0.]),v['elbow_axis'][i],w['elbow_axis'])
            ai=g['arm'];li=g['leg']
            elbow_goal=np.clip(g['elbow_geometric_zero']-v['signed_elbow_bend'][i],rob.lower[ai[3]],rob.upper[ai[3]])
            add([x[ai[3]]-elbow_goal],eye[ai[3]:ai[3]+1],w['elbow_bend'])
            add(x[ai[5:7]],eye[ai[5:7]],w['wrist_rest'])
            barrier,derivative=wrist_barrier(x[ai[5:7]],g)
            add(barrier,eye[ai[5:7]]*derivative[:,None],w['wrist_soft_limit'])
            add([x[li[2]]],eye[li[2]:li[2]+1],w['hip_yaw_rest'])
        if 'body_contacts' in t:
            from .multicontact import lowest,contact_local
            feet={g['foot'] for g in rob.sides.values()}
            for b in rob.surface_models:
                if b in feet:continue
                local=lowest(rob,b);height=rob.point(b,local)[2]
                if height < .0005:
                    _,J=point(b,local);add([height-.0005],J[2:3],w['ground_penalty'])
                else:add([0.],np.zeros((1,nc)),w['ground_penalty'])
            for contact in t['body_contacts']:
                mode=contact['mode'][i]
                if not mode:continue
                b=contact['body'];local=contact_local(rob,contact);P,J=point(b,local)
                # Source skin proximity is an uncertain morphology cue. It is
                # not a measured robot plant, and supplies no robot XY anchor.
                confidence=contact['confidence'][i] if 'confidence' in contact else 1.
                weight=w.get('body_contact_position',0.)*contact['source_geometry_confidence']*confidence
                value,derivative=body_height_task(P[2],t,i,contact,w)
                add([value],J[2:3]*derivative,weight)
        if self.prev is not None:
            # Frame-rate-independent approximate velocity and acceleration
            # regularizers. No filtering after IK, which would reopen contacts.
            f=t['fps']/60.
            add(x-self.prev,eye,w['joint_velocity']*f)
            add(x-self.pred,eye,w['joint_acceleration']*f*f)
            root_velocity_goal=self.prev[:3]+t['root_p'][i]-t['root_p'][i-1]
            add(x[:3]-root_velocity_goal,eye[:3],w['root_velocity']*f)
            add(x[:3]-self.pred[:3],eye[:3],w['root_acceleration']*f*f)
        out=(np.concatenate(res),np.vstack(jac))
        self.cache_x=x.copy();self.cache=out
        return out
    def fun(self,x):return self.evaluate(x)[0]
    def jac(self,x):return self.evaluate(x)[1]

def solve(robot,t,cfg,progress=None):
    n=len(t['hips']);xs=np.empty((n,32));records=[]
    x=np.r_[t['root_p'][0],np.zeros(29)]
    for g in robot.sides.values():x[g['arm'][3]]=-.4;x[g['leg'][3]]=.15
    lower=robot.lower.copy();upper=robot.upper.copy()
    lower[3:]+=1e-7;upper[3:]-=1e-7
    joint_step=robot.joint_step(t['fps'],cfg)
    if cfg.get('defer_arm_speed_to_final', False):
        # The offline arm planner replaces all 14 provisional arm trajectories.
        # Prevent temporary arm lag from unnecessarily recruiting the torso.
        arm_indices = [idx-3 for g in robot.sides.values() for idx in g['arm']]
        joint_step[arm_indices] = np.deg2rad(cfg['numerical_joint_speed_deg_s'])/t['fps']
    for i in range(n):
        previous=xs[i-1] if i else None
        predicted=previous if i<2 else previous+np.clip(previous-xs[i-2],-.10,.10)
        if i:x=previous.copy();x[:3]+=t['root_p'][i]-t['root_p'][i-1]
        obj=Objective(robot,t,i,cfg,previous,predicted)
        frame_lower=lower.copy();frame_upper=upper.copy()
        if i:
            frame_lower[3:]=np.maximum(lower[3:],previous[3:]-joint_step)
            frame_upper[3:]=np.minimum(upper[3:],previous[3:]+joint_step)
        result=least_squares(obj.fun,np.clip(x,frame_lower+1e-6,frame_upper-1e-6),jac=obj.jac,bounds=(frame_lower,frame_upper),
            max_nfev=80 if i==0 else cfg['max_iterations'],ftol=1e-5,xtol=1e-6,gtol=1e-5,x_scale='jac')
        # Source can begin in an extended pose with a poor elbow/yaw branch.
        # Search only the first frame; later frames retain temporal continuity.
        if i==0:
            for yaw in (-1.2,1.2):
                seed=result.x.copy()
                for g in robot.sides.values():seed[g['arm'][2]]=yaw;seed[g['arm'][4]]=-yaw
                alt=least_squares(obj.fun,np.clip(seed,lower+1e-6,upper-1e-6),jac=obj.jac,bounds=(lower,upper),max_nfev=90,ftol=1e-6,gtol=1e-5,xtol=1e-6,x_scale='jac')
                if alt.cost<result.cost:result=alt
        x=result.x;xs[i]=x
        records.append([float(result.cost),int(result.nfev),int(result.status)])
        if progress is not None and ((i+1)%120==0 or i==n-1):progress(i+1,n)
    return xs,records

def constrain_support(robot,t,xs,cfg,progress=None):
    """SQP contact pass: hard XYZ support equalities + mesh-ground inequalities.

    The official limb stabilizer solves legs and pelvis after correcting foot
    targets. Here a bounded constrained solve replaces the two-bone shortcut
    because Dex has lateral hip offsets and a rigid, non-point sole.
    """
    ids=np.array([0,1,2]+sum([g['leg'] for g in robot.sides.values()],[]));out=xs.copy();reports=[]
    leg_joint_indices = ids[3:] - 3
    leg_step = robot.joint_step(t['fps'],cfg)[leg_joint_indices]
    robot.support_branch_relaxations = []
    prev=None;pred=None
    for i in range(len(xs)):
        base=xs[i].copy();obj=Objective(robot,t,i,cfg,prev,pred)
        robot.set(base,t['root_r'][i])
        active={s:set(np.argsort(g['hull']@robot.rot(g['foot'])[2,:])[:8].tolist()) for s,g in robot.sides.items()}
        def state(y):
            x=base.copy();x[ids]=y;robot.set(x,t['root_r'][i]);return x
        def fun(y):
            x=base.copy();x[ids]=y;res,J=obj.evaluate(x)
            return .5*(res@res),J[:,ids].T@res
        def geometry(y,equality=False,all_vertices=False):
            state(y);values=[];rows=[]
            for side,g in robot.sides.items():
                v=t['sides'][side];b=g['foot'];P,A=robot.jac(b);p=robot.point(b);R=robot.rot(b)
                if equality:
                    if v['contact'][i]>0:
                        d=R@v['pivots'][i];values.extend((p+d-v['anchors'][i])[:2]);rows.extend((P-skew(d)@A)[:2,ids])
                        low=R@robot.lowest(side);values.append(p[2]+low[2]);rows.append((P-skew(low)@A)[2,ids])
                else:
                    # For a support foot, min(mesh_z)==0 already entails ALL
                    # vertex inequalities. Adding the active vertex again as
                    # an inequality makes the SQP constraints rank-degenerate.
                    if v['contact'][i]>0 and not all_vertices:continue
                    vertices=g['hull'] if all_vertices else g['hull'][sorted(active[side])]
                    d=vertices@R.T
                    # All convex-hull vertices stay above the plane. This is
                    # smooth per vertex and has no argmin switching Jacobian.
                    values.extend(p[2]+d[:,2])
                    rows.extend((P[2]+np.cross(d,[0.,0.,1.])@A)[:,ids])
            return np.asarray(values),np.asarray(rows).reshape(-1,len(ids))
        cs=[]
        if any(v['contact'][i]==0 for v in t['sides'].values()):
            cs.append({'type':'ineq','fun':lambda y:geometry(y)[0],'jac':lambda y:geometry(y)[1]})
        has_contact=any(v['contact'][i]>0 for v in t['sides'].values())
        if has_contact:cs.append({'type':'eq','fun':lambda y:geometry(y,True)[0],'jac':lambda y:geometry(y,True)[1]})
        lo=robot.lower[ids].copy();hi=robot.upper[ids].copy()
        # Contact refinement stays on the already-fitted anatomical IK branch.
        # An unconstrained SQP step can rotate a hip by >180 deg while keeping
        # an endpoint in place. Bound its correction, not the source motion.
        lo[3:]=np.maximum(lo[3:],base[ids][3:]-.30)
        hi[3:]=np.minimum(hi[3:],base[ids][3:]+.30)
        if prev is not None:
            velocity_lo = np.maximum(robot.lower[ids[3:]], prev[ids[3:]] - leg_step)
            velocity_hi = np.minimum(robot.upper[ids[3:]], prev[ids[3:]] + leg_step)
            safe_lo = np.maximum(lo[3:], velocity_lo)
            safe_hi = np.minimum(hi[3:], velocity_hi)
            valid = safe_hi > safe_lo + 1e-6
            if not valid.all():
                robot.support_branch_relaxations.append(dict(frame=i,
                    joints=[robot.names[k] for k in leg_joint_indices[~valid]]))
            # The branch corridor is an IK preference. When it becomes
            # disjoint, retain the hard physical range and adjacent speed box.
            lo[3:] = np.where(valid, safe_lo, velocity_lo)
            hi[3:] = np.where(valid, safe_hi, velocity_hi)
        bounds=list(zip(lo,hi))
        seed=base[ids].copy()
        seed=np.clip(seed,lo+1e-7,hi-1e-7)
        ans=minimize(fun,seed,jac=True,bounds=bounds,constraints=cs,method='SLSQP',options={'maxiter':90,'ftol':1e-8})
        # Active-set constraint generation: only potential support vertices enter
        # the small QP, but the acceptance test always covers the ENTIRE hull.
        for _ in range(5):
            state(ans.x);added=False
            for side,g in robot.sides.items():
                z=g['hull']@robot.rot(g['foot'])[2,:]+robot.point(g['foot'])[2]
                bad=np.flatnonzero(z < -2e-6)
                new=set(bad[np.argsort(z[bad])[:8]].tolist())-active[side]
                if new:active[side].update(new);added=True
            if not added:break
            ans=minimize(fun,ans.x,jac=True,bounds=bounds,constraints=cs,method='SLSQP',options={'maxiter':65,'ftol':1e-8})
        x=state(ans.x);floor=float(geometry(ans.x,all_vertices=True)[0].min());eq=float(np.max(abs(geometry(ans.x,True)[0]))) if has_contact else 0.
        if floor < -1e-5 or eq>1e-4:
            ans=minimize(fun,ans.x,jac=True,bounds=bounds,constraints=cs,method='SLSQP',options={'maxiter':220,'ftol':1e-9})
            x=state(ans.x);floor=float(geometry(ans.x,all_vertices=True)[0].min());eq=float(np.max(abs(geometry(ans.x,True)[0]))) if has_contact else 0.
        if support_constraint_failed(floor,eq):
            # Inconsistent inferred contact is reported, not silently repaired
            # by changing the root after the solve. Retain best feasible trial.
            reports.append({'frame':i,'success':bool(ans.success),'minimum_z':floor,'contact_error_m':eq,'message':str(ans.message)})
        out[i]=x
        pred=x if prev is None else x+np.clip(x-prev,-.10,.10);prev=x.copy()
        if progress and ((i+1)%120==0 or i==len(xs)-1):progress(i+1,len(xs))
    return out,reports
