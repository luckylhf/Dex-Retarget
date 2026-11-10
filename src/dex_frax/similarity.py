"""Soft source resemblance objective; final acceptance remains independent."""
import numpy as np

PALM_AXIS_LIMIT_DEG = 25.  # Source-axis candidate envelope; not a joint-angle limit.
PALM_COMFORT_DEG = 5.  # Small perceptual errors have reduced, nonzero reward.
POSE_FLAGS = {'sustained_arm_pose_mismatch'} | {
    side+'_'+axis+'_deg_source_mismatch'
    for side in ('Left','Right') for axis in ('upper','lower','hand_forward','hand_width')}


class DifficultShapeObjective:
    """Retain V8's full objective, adding smooth-piecewise penalties above 20°.

ArmObjective's first two residual triples are weighted unit-vector errors.
Their chord lengths yield angles and analytic angle derivatives, without new
FK calls or replacing elbow, palm, wrist, and anatomy terms.
"""
    def __init__(self, objective, direction_weight):
        self.objective=objective;self.weight=float(direction_weight)

    def evaluate(self,q):
        residual,jacobian=self.objective.evaluate(q)
        penalties=[];rows=[];w=self.weight
        for start in (0,3):
            r=residual[start:start+3];J=jacobian[start:start+3]
            cosine=np.clip(1.-float(r@r)/(2*w*w),-1.,1.)
            angle=np.arccos(cosine);excess=max(0.,angle-np.deg2rad(20.))
            gain=2*w
            penalties.append(gain*excess)
            rows.append(gain*(r@J)/(w*w*max(float(np.sin(angle)),1e-6)) if excess>0 else np.zeros(J.shape[1]))
        return np.r_[residual,penalties],np.vstack([jacobian,rows])


def palm_bound(reference):
    # A pre-existing bad source estimate is not permission to worsen it.
    return np.maximum(reference, PALM_AXIS_LIMIT_DEG)


def palm_envelope(before, after):
    for side in before:
        for key in ('hand_forward','hand_width'):
            if np.any(after[side][key]>palm_bound(before[side][key])+1e-3):
                return 'palm_absolute_envelope:'+side+':'+key
    return None


def score(errors):
    """Whole-arm mean and worst-tail resemblance, without rewarding perfect palms.

The elbow scale remains 30 degrees; moving its acceptance boundary to 40
must not silently reduce its existing resemblance weight.
"""
    def tail(values):
        v=np.asarray(values).ravel();n=max(1,int(np.ceil(.05*len(v))))
        return float(np.mean(np.partition(v,len(v)-n)[-n:]))
    value=0.
    for key in ('upper','lower'):
        e=np.asarray(errors[key])/20.
        value+=float(np.mean(e*e)+4*np.mean(np.maximum(e-1.,0.)**2))+2*tail(e*e)
    for key in ('hand_forward','hand_width'):
        angle=np.asarray(errors[key]);e=(.2*np.minimum(angle,PALM_COMFORT_DEG)+np.maximum(angle-PALM_COMFORT_DEG,0.))/10.
        value+=float(np.mean(e*e))+.5*tail(e*e)
    angle=np.asarray(errors['elbow_axis']);e=angle/30.
    value+=.5*(float(np.mean(e*e))+tail(e*e))+4*float(np.mean((np.maximum(angle-40.,0.)/20.)**2))
    return value


class WholeArmObjective:
    """Preserve anatomy/branch terms; reduce tiny palm gains and penalize bad axes.

ArmObjective starts with six 3-vector residuals: upper, lower, palm forward,
palm width, forearm width, elbow axis. All Jacobians are analytic.
"""
    def __init__(self, objective):
        self.objective=objective

    def __getattr__(self,name):
        return getattr(self.objective,name)

    def evaluate(self,q):
        from .v8.task_weights import hand_weights
        obj=self.objective
        base,jac=obj.evaluate(q);res=base.copy();J=jac.copy()
        palm_weights=hand_weights(obj.w,obj.t['sides'][obj.side],obj.i)
        for start,w in zip((6,9),palm_weights):
            r=res[start:start+3].copy();old=J[start:start+3].copy();m=float(np.linalg.norm(r))
            band=2*w*np.sin(np.deg2rad(PALM_COMFORT_DEG)/2)
            if m<=band or m<1e-12:
                res[start:start+3]=.2*r;J[start:start+3]=.2*old
            else:
                scale=1.-.8*band/m
                res[start:start+3]=scale*r
                J[start:start+3]=scale*old+(.8*band/m**3)*np.outer(r,r@old)
        penalties=[];rows=[]
        for start,weight,limit,gain in ((0,obj.w['arm_direction'],20.,2*obj.w['arm_direction']),
                                      (3,obj.w['arm_direction'],20.,2*obj.w['arm_direction']),
                                      (15,obj.w['elbow_axis'],40.,1.5*obj.w['arm_direction'])):
            if weight<=0:continue
            r=base[start:start+3];old=jac[start:start+3]
            angle=np.arccos(np.clip(1.-float(r@r)/(2*weight**2),-1.,1.))
            excess=max(0.,angle-np.deg2rad(limit))
            penalties.append(gain*excess)
            rows.append(gain*(r@old)/(weight**2*max(float(np.sin(angle)),1e-6)) if excess else np.zeros(J.shape[1]))
        return np.r_[res,penalties],np.vstack([J,*rows])

    def fun(self,q):return self.evaluate(q)[0]
    def jac(self,q):return self.evaluate(q)[1]
