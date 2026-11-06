"""Offline arm IK branch planning: candidate poses + temporal path selection.

Dex's long J5/J6 links and shoulder Euler branches make a greedy per-frame
minimum unreliable. At sparse time knots solve several admissible branches,
choose a low-motion path over the WHOLE clip, then refine at full frame rate.
This is a Dex extension, not part of NVIDIA's implementation.
"""
import numpy as np
import time
from scipy.optimize import least_squares
from scipy.sparse import block_diag,diags,kron,eye,vstack
from .math3d import unit,skew
from .trajectory import constrained_refine
from .anatomy import wrist_barrier
from .task_weights import hand_weights
from .contact_tasks import body_height_task

class ArmObjective:
    def __init__(self,robot,t,i,side,base,weights,prior=None,prior_weight=0.):
        self.r,self.t,self.i,self.side,self.base,self.w=robot,t,i,side,np.asarray(base,dtype=float).copy(),weights
        self.ids=robot.sides[side]['arm'];self.prior=prior;self.pw=prior_weight;self.cx=None
    def evaluate(self,q):
        if self.cx is not None and np.array_equal(q,self.cx):return self.cache
        r,t,i=self.r,self.t,self.i;g=r.sides[self.side];v=t['sides'][self.side];w=self.w
        hand_direction_weight,hand_width_weight=hand_weights(w,v,i)
        x=self.base.copy();x[self.ids]=q;r.set(x,t['root_r'][i]);res=[];js=[];cache={}
        def body(b):
            if b not in cache:
                J,A=r.jac(b);cache[b]=(r.point(b),J[:,self.ids],A[:,self.ids])
            return cache[b]
        def add(v,J,weight):
            scale=np.asarray(weight);res.append(np.atleast_1d(v)*scale)
            js.append(np.atleast_2d(J)*(scale[:,None] if scale.ndim else scale))
        def direction(a,b,target,weight):
            p,P,_=body(a);h,H,_=body(b);d=h-p;ln=np.linalg.norm(d);u=d/ln
            add(u-target,(np.eye(3)-np.outer(u,u))@(H-P)/ln,weight)
        def axis(b,local,target,weight):
            _,_,A=body(b);a=r.rot(b)@local;add(a-target,-skew(a)@A,weight)
        direction(g['shoulder'],g['elbow'],v['upper'][i],w['arm_direction'])
        direction(g['elbow'],g['wrist'],v['lower'][i],w['arm_direction'])
        axis(g['tcp'],[0,0,-1],v['hand_forward'][i],hand_direction_weight)
        axis(g['tcp'],[1,0,0],v['hand_width'][i],hand_width_weight)
        axis(g['forearm'],[1,0,0],v['forearm_width'][i],w['forearm_twist'])
        axis(g['shoulder'],[0,-1,0],v['elbow_axis'][i],w['elbow_axis'])
        egoal=np.clip(g['elbow_geometric_zero']-v['signed_elbow_bend'][i],r.lower[self.ids[3]],r.upper[self.ids[3]])
        eye=np.eye(7);add([q[3]-egoal],eye[3:4],w['elbow_bend'])
        add(q[5:7],eye[5:7],w['wrist_rest'])
        barrier,derivative=wrist_barrier(q[5:7],g)
        add(barrier,eye[5:7]*derivative[:,None],w['wrist_soft_limit'])
        # Weak joint-center prior resolves unobservable twist at straight limbs.
        add(q,eye,.025)
        if self.prior is not None:
            proximal=self.w.get('arm_event_tracking',False) and self.w.get('arm_adaptive_candidates',False)
            add(q-self.prior,eye,self.pw*(np.array([.4,.4,.4,1.,1.,1.,1.]) if proximal else 1.))
        # Include the same ground/support tasks in branch selection and the
        # whole-clip fit, so a late greedy cleanup need not undo the chosen arm.
        if w.get('arm_ground_penalty',0.)>0:
            for b,hull in g['ground_hulls'].items():
                local=hull[np.argmin(hull@r.rot(b)[2])];offset=r.rot(b)@local
                p,P,A=body(b);height=p[2]+offset[2]
                add([min(height,0.)],(P-skew(offset)@A)[2:3] if height<0 else np.zeros((1,7)),w['arm_ground_penalty'])
            for contact in t.get('body_contacts',[]):
                if contact['body'] not in g['ground_hulls']:continue
                hull=np.asarray(contact['robot_hull']);b=contact['body']
                local=hull[np.argmin(hull@r.rot(b)[2])];offset=r.rot(b)@local
                p,P,A=body(b);height=p[2]+offset[2]
                confidence=contact['confidence'][i] if 'confidence' in contact else float(contact['mode'][i]>0)
                weight=w.get('body_contact_position',0.)*contact['source_geometry_confidence']*confidence
                value,derivative=body_height_task(height,t,i,contact,w)
                add([value],(P-skew(offset)@A)[2:3]*derivative,weight)
        self.cx=q.copy();self.cache=np.concatenate(res),np.vstack(js)
        return self.cache
    def fun(self,q):return self.evaluate(q)[0]
    def jac(self,q):return self.evaluate(q)[1]

def plan_and_refine(robot,t,xs,cfg,progress=None,_only_sides=None):
    from ..arm_tracking import SearchEvents,acceptable,track,local_temporal,eligible,hard_frames,original_window
    from ..resource_events import phase
    n=len(xs);fps=t['fps'];step=max(1,round(fps/15))
    event_tracking=cfg.get('arm_event_tracking',False) and cfg.get('arm_adaptive_candidates',False)
    knots=np.unique(np.r_[np.arange(0,n,step),n-1]);out=xs.copy();report={}
    for side,g in robot.sides.items():
        if _only_sides is not None and side not in _only_sides:continue
        timings={};phase_start=time.perf_counter();candidate_calls=0;full_knots=0;search_reasons=[]
        side_event=event_tracking
        hard=hard_frames(robot,t,xs,cfg,side) if side_event else np.ones(n,dtype=bool)
        if side_event and hard.any():
            padding=max(2,round(.4*fps))
            hard=np.convolve(np.pad(hard,(padding,padding)),np.ones(2*padding+1),mode='valid')>0
        # A small difficult clip is already a bounded recovery window. Run the
        # original fit once; do not first fit, reject, and solve it a second time.
        if side_event and original_window(hard):
            solved,legacy=plan_and_refine(robot,t,out,dict(cfg,arm_event_tracking=False),progress,{side})
            out[:,g['arm']]=solved[:,g['arm']];report[side]=legacy[side]
            report[side]['local_legacy_window']=dict(start=0,end=n,reason='bounded_dense_difficulty',context_fraction=float(hard.mean()))
            continue
        weights=cfg if side_event else dict(cfg,arm_event_tracking=False)
        ids=g['arm'];lo=robot.lower[ids]+1e-6;hi=robot.upper[ids]-1e-6;sgn=1 if side=='Left' else -1
        joint_step=robot.joint_step(fps,cfg)[np.asarray(ids)-3]
        nodes=[];costs=[];previous=[];events=SearchEvents();tracking_calls=0;reused_knots=0;proximal_fits=0;reused_frames=0;frame_solver_calls=0
        phase('arms_candidates')
        for ki,i in enumerate(knots):
            obj=ArmObjective(robot,t,i,side,xs[i],weights)
            seeds=[xs[i,ids]]+previous;fixed=[]
            # Cover both admissible shoulder branches and forearm twist signs.
            for pitch,roll,yaw,twist in [(0,.4,0,0),(-1.,1.4,1.8,-1.8),(1.,2.6,-1.8,1.8),(-1.,2.6,-1.8,1.8),(1.,.4,1.8,-1.8)]:
                fixed.append(np.array([pitch,sgn*roll,yaw,-1.,twist,0.,0.]))
            # The near-straight anatomical branch may lie on either side of
            # the geometric zero because source bone offsets are non-collinear.
            neutral=xs[i,ids].copy();neutral[2]=0.;neutral[4]=0.
            neutral[3]=g['elbow_geometric_zero']-t['sides'][side]['signed_elbow_bend'][i]
            adaptive=cfg.get('arm_adaptive_candidates',False)
            solved={}
            def solve_seed(seed):
                nonlocal candidate_calls
                clipped=np.clip(seed,lo+.005,hi-.005);key=clipped.tobytes()
                if key not in solved:
                    solved[key]=least_squares(obj.fun,clipped,jac=obj.jac,bounds=(lo,hi),max_nfev=35,ftol=1e-5,xtol=1e-6,gtol=1e-5)
                    candidate_calls+=1
                return solved[key]
            reason='full_default'
            tracked=None
            if side_event and not hard[i]:
                tracked,calls,kind=track(obj,seeds,lo,hi);tracking_calls+=calls
                reason=events.reason(obj,tracked.x,tracked.cost,ki,ki==len(knots)-1)
                reused_knots+=int(kind=='reuse' and reason=='continuation');proximal_fits+=int(kind=='proximal')
            elif adaptive and (not side_event or not hard[i]):
                probes=[solve_seed(s) for s in seeds+[neutral]];best=min(probes,key=lambda a:a.cost)
                v=t['sides'][side];reason='continuation'
                if ki==0 or ki==len(knots)-1 or ki%8==0:reason='periodic'
                elif abs(v['signed_elbow_bend'][i])<np.deg2rad(15):reason='near_straight'
                elif np.min(np.minimum(best.x-lo,hi-best.x))<.12:reason='near_limit'
                elif best.cost>costs[-1].min()*1.25+.3:reason='cost_rise'
                elif any(np.dot(v[k][i],v[k][knots[ki-1]])<np.cos(np.deg2rad(10)) for k in ('upper','lower','hand_forward','hand_width')):reason='target_change'
            full=not adaptive or reason!='continuation';full_knots+=int(full);search_reasons.append(reason)
            candidates=[]
            # Preserve original seed ordering when expanding a complete search.
            seed_list=seeds+(fixed if full else [])+[neutral]
            if side_event and not hard[i] and not full:
                candidates=[(tracked.x,float(tracked.cost))]
                seed_list=[]
                # Retain a good input branch as an alternative for global DP.
                raw=np.clip(xs[i,ids],lo+1e-6,hi-1e-6)
                if acceptable(obj,raw,xs[i,ids]) and np.linalg.norm(raw-tracked.x)>=.08:
                    candidates.append((raw,float(obj.fun(raw)@obj.fun(raw))/2))
            for seed in seed_list:
                result=solve_seed(seed)
                if not any(np.linalg.norm(result.x-q)<.08 for q,_ in candidates):candidates.append((result.x,float(result.cost)))
            candidates.sort(key=lambda z:z[1]);candidates=candidates[:5]
            previous=[q for q,c in candidates[:2]];nodes.append(np.array([q for q,c in candidates]));costs.append(np.array([c for q,c in candidates]))
            if progress and (ki+1)%120==0:progress(side,'branches',ki+1,len(knots))
        timings['candidate_search_s']=time.perf_counter()-phase_start;phase_start=time.perf_counter()
        total=costs[0].copy();backs=[]
        for k in range(1,len(knots)):
            dt=(knots[k]-knots[k-1])/fps
            dq=nodes[k][None,:,:]-nodes[k-1][:,None,:]
            transition=.12*np.sum(dq*dq,axis=2)/max(dt,.01)
            # A branch jump of pi is extremely expensive, while smooth fast
            # gestures remain possible. This cost is not a hardware limit.
            capacity=joint_step*fps*dt
            excess=np.maximum(abs(dq)-capacity,0)
            transition+=3.*np.sum(excess**2,axis=2)
            # Consider each motor's reachability before committing to a branch.
            transition+=.2*np.sum((excess/np.maximum(capacity,1e-6))**2,axis=2)
            scores=total[:,None]+transition
            back=np.argmin(scores,axis=0);backs.append(back)
            total=costs[k]+scores[back,np.arange(len(nodes[k]))]
        chosen=[int(np.argmin(total))]
        for b in reversed(backs):chosen.append(int(b[chosen[-1]]))
        chosen=chosen[::-1];path=np.array([nodes[k][v] for k,v in enumerate(chosen)])
        seeds=np.stack([np.interp(np.arange(n),knots,path[:,j]) for j in range(7)],axis=1)
        timings['path_selection_s']=time.perf_counter()-phase_start;phase_start=time.perf_counter()
        phase('arms_frame_fit')
        # The selected offline path is the initialization AND a weak prior.
        # Its future context avoids a forward greedy trap near a joint limit.
        for i in range(n):
            obj=ArmObjective(robot,t,i,side,out[i],weights,seeds[i],.12)
            if side_event and not hard[i]:
                result,calls,kind=track(obj,[seeds[i],xs[i,ids]],lo,hi)
                frame_solver_calls+=calls;reused_frames+=int(kind=='reuse');proximal_fits+=int(kind=='proximal')
            else:
                result=least_squares(obj.fun,np.clip(seeds[i],lo+1e-5,hi-1e-5),jac=obj.jac,bounds=(lo,hi),max_nfev=25,ftol=1e-6,xtol=1e-6,gtol=1e-5)
                frame_solver_calls+=1
            out[i,ids]=result.x
            if progress and (i+1)%120==0:progress(side,'frame_fit',i+1,n)
        # Joint-space temporal optimization across the entire clip. Filtering
        # angles alone would lose hand orientation; pose residuals participate
        # in every temporal iteration, including the near-straight singularity.
        timings['frame_fit_s']=time.perf_counter()-phase_start;phase_start=time.perf_counter()
        phase('arms_temporal')
        local=local_temporal(robot,t,out,weights,side) if side_event else None
        if local is not None:
            out,guard=local
            timings['temporal_fit_s']=time.perf_counter()-phase_start;timings['constrained_fit_s']=0.
            report[side]=dict(knot_count=len(knots),candidate_counts=[len(v) for v in nodes],timings=timings,
                candidate_solver_calls=candidate_calls+tracking_calls,full_search_knots=full_knots,search_reasons=search_reasons,
                reused_knots=reused_knots,reused_frames=reused_frames,frame_solver_calls=frame_solver_calls,proximal_fits=proximal_fits,
                temporal_solver_status=1,temporal_nfev=0,constrained_temporal_refinement=guard,
                branch_speed_policy='event_tracking_named_axis_v3',
                max_knot_step_deg=float(np.rad2deg(np.max(abs(np.diff(path,axis=0))))),
                max_final_step_deg=float(np.rad2deg(np.max(abs(np.diff(out[:,ids],axis=0))))))
            if progress:progress(side,'refined',n,n)
            continue
        # A failed fast temporal bypass needs the original continuity fit once.
        # Do not rerun whole-clip branch search, and do not force a new branch
        # at arbitrary window cuts merely to avoid this necessary fit.
        objectives=[ArmObjective(robot,t,i,side,out[i],weights) for i in range(n)]
        f=fps/60.
        D1=kron(diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n)),eye(7),format='csr')*.18*f
        D2=kron(diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n)),eye(7),format='csr')*6.0*f*f
        def temporal_fun(z):
            qs=z.reshape(n,7)
            return np.r_[np.concatenate([o.fun(q) for o,q in zip(objectives,qs)]),D1@z,D2@z]
        def temporal_jac(z):
            qs=z.reshape(n,7)
            return vstack([block_diag([o.jac(q) for o,q in zip(objectives,qs)],format='csr'),D1,D2],format='csr')
        if progress:progress(side,'temporal_fit',0,n)
        smooth=least_squares(temporal_fun,np.clip(out[:,ids],lo+1e-5,hi-1e-5).ravel(),jac=temporal_jac,
            bounds=(np.tile(lo,n),np.tile(hi,n)),max_nfev=22,ftol=1e-5,xtol=1e-5,gtol=1e-4,
            tr_solver='lsmr',tr_options={'maxiter':60},x_scale='jac')
        timings['temporal_fit_s']=time.perf_counter()-phase_start;phase_start=time.perf_counter()
        if progress:progress(side,'constrained_fit',0,n)
        out[:,ids],guard=constrained_refine(smooth.x.reshape(n,7),temporal_fun,temporal_jac,lo,hi,
            joint_step)
        timings['constrained_fit_s']=time.perf_counter()-phase_start
        report[side]={'knot_count':len(knots),'candidate_counts':[len(v) for v in nodes],
            'timings':timings,'candidate_solver_calls':candidate_calls+tracking_calls,'full_search_knots':full_knots,'search_reasons':search_reasons,
            'reused_knots':reused_knots,'reused_frames':reused_frames,'frame_solver_calls':frame_solver_calls,'proximal_fits':proximal_fits,
            'temporal_solver_status':int(smooth.status),'temporal_nfev':int(smooth.nfev),
            'constrained_temporal_refinement':guard,
            'branch_speed_policy':'event_tracking_named_axis_v3' if side_event else 'named_axis_reachability_v2',
            'event_tracking_eligible':side_event,
            'max_knot_step_deg':float(np.rad2deg(np.max(abs(np.diff(path,axis=0))))),
            'max_final_step_deg':float(np.rad2deg(np.max(abs(np.diff(out[:,ids],axis=0)))))}
        if progress:progress(side,'refined',n,n)
    if event_tracking:
        for side,g in robot.sides.items():
            if side not in report or report[side]['branch_speed_policy']!='event_tracking_named_axis_v3':continue
            if not hard_frames(robot,t,out,cfg,side,include_support=False).any():continue
            if n>200:
                from ..arm_tracking import recover_windows
                out,local_guard=recover_windows(robot,t,xs,out,cfg,side,progress)
                report[side]['local_guard_fallback']=local_guard
                continue
            reference=out.copy();reference[:,g['arm']]=xs[:,g['arm']]
            recovered,legacy=plan_and_refine(robot,t,reference,dict(cfg,arm_event_tracking=False),progress,{side})
            old_report=report[side];report[side]=legacy[side]
            report[side]['event_guard_fallback']=True
            report[side]['local_legacy_window']=dict(start=0,end=n)
            report[side]['candidate_solver_calls']+=old_report['candidate_solver_calls']
            report[side]['timings']['guarded_attempt_s']=sum(old_report['timings'].values())
            out[:,g['arm']]=recovered[:,g['arm']]
    return out,report
