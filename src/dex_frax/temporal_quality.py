"""Screen source-relative reversals and velocity pulses without motor accel claims."""
import numpy as np

POLICY=dict(name='source_relative_reversals_and_impulses_v2',window_s=.24,minimum_reversals=4,
            minimum_step_deg=1.,source_step_ratio=.5,hardware_acceleration_limit=False)


def region(name):
    if any(k in name for k in ('shoulder','elbow','wrist')):return ('Left' if '_l_' in name else 'Right')+'_arm'
    if any(k in name for k in ('hip','knee','ankle')):return ('Left' if '_l_' in name else 'Right')+'_leg'
    return 'torso'


def source_steps(dg,area,n):
    if area=='torso':keys=['torso_target','root_r']
    else:
        side,part=area.split('_');keys=[side+'_'+k for k in (('upper','lower','hand_forward','hand_width','elbow_axis') if part=='arm' else ('thigh','shin','foot_target_r'))]
    values=[]
    for key in keys:
        if key not in dg:continue
        a=np.asarray(dg[key],float)
        if len(a)!=n:continue
        if a.ndim==3:
            from scipy.spatial.transform import Rotation
            value=Rotation.from_matrix(a[:-1].transpose(0,2,1)@a[1:]).magnitude()
        else:
            a=a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)
            value=np.arccos(np.clip(np.sum(a[:-1]*a[1:],axis=-1),-1.,1.))
        values.append(np.degrees(value))
    return np.max(values,axis=0) if values else None


def assess(q,fps,names,dg):
    q=np.asarray(q,float);n=len(q);step=np.degrees(np.diff(q,axis=0));width=max(5,round(POLICY['window_s']*fps))
    regions={region(name) for name in names};source={k:source_steps(dg,k,n) for k in regions}
    bad=np.zeros(n,bool);joints={}
    for j,name in enumerate(names):
        d=step[:,j];rev=(d[:-1]*d[1:]<0)&(np.minimum(abs(d[:-1]),abs(d[1:]))>=POLICY['minimum_step_deg'])
        reference=source[region(name)];added=rev.copy()
        if reference is not None:
            added&=np.maximum(reference[:-1],reference[1:]) < POLICY['source_step_ratio']*np.minimum(abs(d[:-1]),abs(d[1:]))
        else:added[:]=False  # Missing evidence cannot prove conversion-added jitter.
        mask=np.zeros(n,bool)
        for a in np.flatnonzero(added):
            b=min(len(added),a+width)
            if np.count_nonzero(added[a:b])>=POLICY['minimum_reversals']:mask[a:min(n,b+2)]=True
        impulse=np.zeros(n,bool)
        if reference is not None and len(d)>=3:
            excess=abs(d[1:-1]-.5*(d[:-2]+d[2:]))
            local_source=np.maximum.reduce([reference[:-2],reference[1:-1],reference[2:]])
            isolated=(excess>=POLICY['minimum_step_deg']) & (abs(d[:-2]-d[2:])<POLICY['source_step_ratio']*excess)
            isolated &= np.maximum(abs(d[:-2]),abs(d[2:])) < POLICY['source_step_ratio']*abs(d[1:-1])
            isolated &= local_source<POLICY['source_step_ratio']*excess
            for k in np.flatnonzero(isolated)+1:impulse[k:k+2]=True
        combined=mask|impulse
        bad|=combined
        acceleration=abs(np.diff(q[:,j],n=2))*fps**2;jerk=abs(np.diff(q[:,j],n=3))*fps**3
        joints[name]=dict(region=region(name),added_reversal_frames=np.flatnonzero(mask).tolist(),
            added_velocity_impulse_frames=np.flatnonzero(impulse).tolist(),violating_frames=np.flatnonzero(combined).tolist(),
            acceleration_p95_rad_s2=float(np.percentile(acceleration,95)) if len(acceleration) else 0.,
            acceleration_max_rad_s2=float(acceleration.max()) if len(acceleration) else 0.,
            jerk_p95_rad_s3=float(np.percentile(jerk,95)) if len(jerk) else 0.,jerk_max_rad_s3=float(jerk.max()) if len(jerk) else 0.,source_comparison_available=reference is not None)
    return dict(policy=POLICY,violating_frames=np.flatnonzero(bad).tolist(),joints=joints,
                scope='quality heuristic for sustained reversals or isolated velocity pulses absent from source-region motion; not a proof of cause or hardware safety')
