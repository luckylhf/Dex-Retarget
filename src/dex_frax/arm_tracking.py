"""Guarded arm continuation with event searches and bounded temporal windows."""
import numpy as np
from scipy.optimize import least_squares
from types import SimpleNamespace


def geometry(obj,q):
    r,t,i=obj.r,obj.t,obj.i;g=r.sides[obj.side];v=t['sides'][obj.side]
    x=obj.base.copy();x[obj.ids]=q;r.set(x,t['root_r'][i])
    def angle(a,b):return float(np.rad2deg(np.arccos(np.clip(np.dot(a,b),-1.,1.))))
    values=[]
    for a,b,key in [('shoulder','elbow','upper'),('elbow','wrist','lower')]:
        d=r.point(g[b])-r.point(g[a]);d/=max(np.linalg.norm(d),1e-12);values.append(angle(d,v[key][i]))
    for b,axis,key in [(g['tcp'],[0,0,-1],'hand_forward'),(g['tcp'],[1,0,0],'hand_width'),
                       (g['forearm'],[1,0,0],'forearm_width'),(g['shoulder'],[0,-1,0],'elbow_axis')]:
        values.append(angle(r.rot(b)@axis,v[key][i]))
    goal=np.clip(g['elbow_geometric_zero']-v['signed_elbow_bend'][i],r.lower[obj.ids[3]],r.upper[obj.ids[3]])
    values.append(float(np.rad2deg(abs(q[3]-goal))))
    soft=np.maximum(g['wrist_soft_lower']-q[5:],0)+np.maximum(q[5:]-g['wrist_soft_upper'],0)
    return dict(angles=np.array(values),wrist=r.point(g['wrist']).copy(),elbow=r.point(g['elbow']).copy(),soft=soft)


def acceptable(obj,q,reference):
    value=geometry(obj,q);baseline=geometry(obj,reference)
    caps=np.array([5.,5.,3.,3.,180.,15.,180.])
    # A fast bypass must preserve the existing distal goals, not merely avoid
    # physical limits. Palms are checked along both independent hand axes.
    bound=np.minimum(caps,baseline['angles']+np.array([.5,.5,.5,.5,.25,.5,.25]))
    return bool(np.all(value['angles']<=bound+1e-7) and np.all(value['soft']<=baseline['soft']+1e-8)
        and np.linalg.norm(value['wrist']-baseline['wrist'])<=.01
        and np.linalg.norm(value['elbow']-baseline['elbow'])<=.01)


def support_signature(obj):
    g=obj.r.sides[obj.side];v=obj.t['sides'][obj.side];i=obj.i
    confidence=v.get('ground_contact_confidence')
    return (bool(v['contact'][i]>.5),bool(confidence is not None and confidence[i]>.5),
        tuple(int(c['mode'][i]) for c in obj.t.get('body_contacts',[]) if c['body'] in g['ground_hulls']))


def hard_frames(robot,t,xs,cfg,side,include_support=True):
    """Local source-pose/support mask; one hard span never disables a whole arm."""
    from .v8.arms import ArmObjective
    from .adaptive_refine import sustained
    g=robot.sides[side];bad=[]
    for i,row in enumerate(xs):
        obj=ArmObjective(robot,t,i,side,row,cfg);angles=geometry(obj,row[g['arm']])['angles']
        bad.append(np.any(angles[[0,1,2,3,5]]>np.array([20.,20.,25.,25.,30.])))
    mask=sustained(bad,max(2,round(.1*t['fps'])))
    if include_support:
        for c in t.get('body_contacts',[]):
            if c['body'] in g['ground_hulls']:mask|=np.asarray(c['mode'])>0
    return mask


def original_window(hard):
    """Protect densely difficult short windows; never revert a long clip."""
    n=len(hard)
    return bool(np.any(hard) and (n<=200 or (n<=600 and np.mean(hard)>=1/3)))


def eligible(robot,t,xs,cfg,side):
    return not hard_frames(robot,t,xs,cfg,side).any()


class SearchEvents:
    def __init__(self):self.last=None;self.straight=False;self.near=False;self.last_full=-1000;self.bad=0
    def reason(self,obj,q,cost,ki,last):
        v=obj.t['sides'][obj.side];i=obj.i;sig=support_signature(obj)
        bend=abs(np.rad2deg(v['signed_elbow_bend'][i]));straight=bend<(18. if self.straight else 12.)
        lo=obj.r.lower[obj.ids];hi=obj.r.upper[obj.ids]
        near=bool(np.min(np.minimum(q-lo,hi-q))<.12)
        changed=self.last is not None and any(np.dot(v[k][i],v[k][self.last['frame']])<np.cos(np.deg2rad(10)) for k in ('upper','lower','hand_forward','hand_width'))
        bad=not acceptable(obj,q,obj.base[obj.ids]);self.bad=self.bad+1 if bad else 0
        reason='continuation'
        if self.last is None:reason='initial'
        elif last:reason='last_anchor'
        elif sig!=self.last['support']:reason='contact_change'
        elif changed:reason='target_change'
        elif near and not self.near:reason='near_limit_enter'
        elif straight!=self.straight:reason='straight_transition'
        elif ki-self.last_full>=4 and (self.bad>=2 or cost>self.last['cost']*1.25+.3):reason='residual_rise'
        self.last=dict(frame=i,support=sig,cost=cost);self.straight=straight;self.near=near
        if reason!='continuation':self.last_full=ki
        return reason


def track(obj,seeds,lo,hi):
    candidates=[np.clip(np.asarray(s),lo+1e-6,hi-1e-6) for s in seeds]
    best=min(candidates,key=lambda q:float(obj.fun(q)@obj.fun(q)))
    reference=obj.base[obj.ids]
    if acceptable(obj,best,reference):return SimpleNamespace(x=best,cost=float(obj.fun(best)@obj.fun(best))/2),0,'reuse'
    # J1-J3 first; distal axes and their goals are preserved in this attempt.
    q=best.copy()
    def combine(z):at=q.copy();at[:3]=z;return at
    shoulder=least_squares(lambda z:obj.fun(combine(z)),q[:3],jac=lambda z:obj.jac(combine(z))[:,:3],
        bounds=(lo[:3],hi[:3]),max_nfev=8,ftol=1e-5,xtol=1e-6,gtol=1e-5)
    proximal=combine(shoulder.x)
    if acceptable(obj,proximal,reference):return SimpleNamespace(x=proximal,cost=shoulder.cost),1,'proximal'
    result=least_squares(obj.fun,proximal,jac=obj.jac,bounds=(lo,hi),max_nfev=25,ftol=1e-5,xtol=1e-6,gtol=1e-5)
    return result,2,'joint_refit'


def local_temporal(robot,t,xs,cfg,side):
    from .local_refine import fit_window
    from .v8.self_collision import windows
    from .v8.arms import ArmObjective
    g=robot.sides[side];ids=np.asarray(g['arm']);q=xs[:,ids];n=len(q)
    step=robot.joint_step(t['fps'],cfg)[ids-3];bad=np.zeros(n,dtype=bool)
    velocity=np.any(abs(np.diff(q,axis=0))>step*(1-1e-5),axis=1);bad[:-1]|=velocity;bad[1:]|=velocity
    if n>2:bad[1:-1]|=np.any(abs(np.diff(q,n=2,axis=0))*t['fps']**2>400.,axis=1)
    for i in range(n):
        obj=ArmObjective(robot,t,i,side,xs[i],cfg)
        # The strict 3-degree bypass gate is for skipping a frame solve, not
        # for re-running full search after that solve has already converged.
        angles=geometry(obj,q[i])['angles']
        bad[i]|=np.any(angles[[0,1,2,3,5]]>[20.,20.,25.,25.,30.])
    ranges=windows(set(np.flatnonzero(bad).tolist()),n,round(.3*t['fps']))
    from .graduated_collision import split_windows
    ranges=[piece for a,b in ranges for piece in split_windows(a,b,200,20)]
    class Objective:
        def __init__(self,i):self.inner=ArmObjective(robot,t,i,side,xs[i],cfg,xs[i,ids],.12)
        def evaluate(self,x):
            value,J=self.inner.evaluate(x[ids]);full=np.zeros((len(value),32));full[:,ids]=J;return value,full
    out=xs.copy();reports=[]
    try:
        for a,b in ranges:
            out,report=fit_window(robot,t,out,cfg,a,b,iterations=8,ids_override=ids,
                objective_factory=Objective,local_checks=True,allow_support=True)
            # Local joins are fixed; preserve hand/limb goals at changed frames.
            for i in range(a,b):
                obj=ArmObjective(robot,t,i,side,xs[i],cfg)
                old=geometry(obj,xs[i,ids]);new=geometry(obj,out[i,ids])
                if np.any(new['angles']>old['angles']+.5) or np.any(new['soft']>old['soft']+1e-8):
                    raise ValueError('local_arm_pose_regression')
            reports.append(report)
        if np.any(abs(np.diff(out[:,ids],axis=0))>step+1e-9):return None
        return out,dict(method='guarded_arm_windows_v1',windows=reports,iterations=sum(r['iterations'] for r in reports))
    except ValueError:return None


def recover_windows(robot,t,reference,candidate,cfg,side,progress=None):
    """Original multistart on bounded bad spans, followed by exact local joins.

    Failed candidates are rolled back locally and remain visible to final audit.
    This function never recursively runs the whole long arm trajectory.
    """
    from .v8.arms import ArmObjective,plan_and_refine
    from .v8.self_collision import windows
    from .adaptive_refine import window_targets
    from .graduated_collision import split_windows
    from .local_refine import fit_window
    ids=np.asarray(robot.sides[side]['arm']);n=len(reference);step=robot.joint_step(t['fps'],cfg)[ids-3]
    bad=hard_frames(robot,t,candidate,cfg,side)
    for i in range(n):
        obj=ArmObjective(robot,t,i,side,reference[i],cfg)
        old=geometry(obj,reference[i,ids]);new=geometry(obj,candidate[i,ids])
        bad[i]|=np.any(new['angles']>np.maximum(old['angles'],[20,20,25,25,180,30,180])+.5) or np.any(new['soft']>old['soft']+1e-6)
    speed=np.any(abs(np.diff(candidate[:,ids],axis=0))>step*(1-1e-5),axis=1)
    bad[:-1]|=speed;bad[1:]|=speed
    if n>2:bad[1:-1]|=np.any(abs(np.diff(candidate[:,ids],n=2,axis=0))*t['fps']**2>400.,axis=1)
    spans=windows(set(np.flatnonzero(bad).tolist()),n,max(3,round(.4*t['fps'])))
    spans=[piece for a,b in spans for piece in split_windows(a,b,200,20)]
    out=candidate.copy();reports=[]
    # Restore only affected frames before fitting so a failed trial cannot leak.
    for a,b in spans:out[a:b,ids]=reference[a:b,ids]
    class Objective:
        def __init__(self,i):self.inner=ArmObjective(robot,t,i,side,reference[i],cfg)
        def evaluate(self,x):
            value,J=self.inner.evaluate(x[ids]);full=np.zeros((len(value),32));full[:,ids]=J;return value,full
    for a,b in spans:
        detail=dict(start=a,end=b,status='local_solver_failed')
        try:
            sub=window_targets(t,a,b,n)
            solved,legacy=plan_and_refine(robot,sub,out[a:b],dict(cfg,arm_event_tracking=False),progress,{side})
            seed=out.copy();seed[a:b,ids]=solved[:,ids]
            trial,fit=fit_window(robot,t,out,cfg,a,b,iterations=12,ids_override=ids,
                objective_factory=Objective,local_checks=True,allow_support=True,seed=seed)
            # Source-pose goals, including distal anatomy, cannot be hidden by
            # merely staying close to a bad robot reference.
            for i in range(a,b):
                obj=ArmObjective(robot,t,i,side,reference[i],cfg)
                old=geometry(obj,reference[i,ids]);legacy_pose=geometry(obj,solved[i-a,ids]);new=geometry(obj,trial[i,ids])
                if np.any(new['angles']>np.maximum(np.maximum(old['angles'],legacy_pose['angles']),[20,20,25,25,180,30,180])+.5):
                    raise ValueError('local_source_pose_regression')
                if np.any(new['soft']>np.maximum(old['soft'],legacy_pose['soft'])+np.deg2rad(.25)):
                    raise ValueError('local_wrist_regression')
            out[:,ids]=trial[:,ids];detail.update(status='repaired',join=fit,legacy=legacy[side])
        except ValueError as error:detail['reason']=str(error)
        reports.append(detail)
    return out,dict(method='bounded_original_search_v4',windows=reports,
        unresolved_windows=sum(r['status']!='repaired' for r in reports),whole_clip_fallback=False)
