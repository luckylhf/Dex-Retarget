"""Opt-in local slowdown of prepared targets; never re-estimate calibration.

The output remains uniformly sampled. A monotone source-time map makes every
pose comparison refer to the corresponding original instant, not a shifted
frame number. This changes timing, not the source pose or acceptance limits.
"""
import copy
import numpy as np
from scipy.spatial.transform import Rotation,Slerp

POLICY='prepared_target_local_root_slowdown_v2'
DIRECTIONS={'thigh','shin','upper','lower','elbow_axis','hand_forward','hand_width','forearm_width'}
DISCRETE={'contact','phase','mode','anchors','pivots','source_expected_contact','source_airborne','confidence','ground_contact_confidence'}
STATIC={'robot_hull','hull','world_origin','local_axis'}


def time_map(root_r,root_p,fps,angular_limit,linear_limit):
    if not np.isfinite([fps,angular_limit,linear_limit]).all() or min(fps,angular_limit,linear_limit)<=0:
        raise ValueError('Invalid retiming rates')
    n=len(root_r)
    if n<2 or np.shape(root_r)!=(n,3,3) or np.shape(root_p)!=(n,3):raise ValueError('Invalid retiming targets')
    angles=Rotation.from_matrix(root_r[:-1].transpose(0,2,1)@root_r[1:]).magnitude()
    lengths=np.linalg.norm(np.diff(root_p,axis=0),axis=1)
    dt=np.maximum.reduce([np.full(n-1,1/fps),angles/angular_limit,lengths/linear_limit])
    changed=np.flatnonzero(dt>1/fps+1e-12)
    if not len(changed):return np.arange(n,dtype=float),dict(policy=POLICY,changed=False,source_duration_s=(n-1)/fps,output_duration_s=(n-1)/fps,stretched_intervals=[])
    clock=np.r_[0.,np.cumsum(dt)]
    count=int(np.ceil(clock[-1]*fps-1e-9))+1
    output_time=np.arange(count)/fps
    source_index=np.interp(output_time,clock,np.arange(n))
    # The final <=1 frame is a hold, not a global rescaling or a dropped tail.
    exact=np.abs(source_index-np.rint(source_index))<1e-9
    source_index[exact]=np.rint(source_index[exact])
    source_index[-1]=n-1
    return source_index,dict(policy=POLICY,changed=True,source_duration_s=(n-1)/fps,output_duration_s=(count-1)/fps,
        source_frames=n,output_frames=count,stretched_intervals=changed.tolist(),
        angular_target_rad_s=float(angular_limit),linear_target_m_s=float(linear_limit),
        source_angular_max_rad_s=float(angles.max()*fps),source_linear_max_m_s=float(lengths.max()*fps),
        final_hold_s=float(output_time[-1]-clock[-1]))


def apply(targets,robot,angular_limit,linear_limit):
    fps=float(targets['fps']);n=len(targets['root_r'])
    indices,report=time_map(targets['root_r'],targets['root_p'],fps,angular_limit,linear_limit)
    if not report['changed']:return targets,indices/fps,report
    nearest=np.clip(np.floor(indices+.5).astype(int),0,n-1)
    exact=indices==np.rint(indices)
    left=np.floor(indices).astype(int);right=np.minimum(left+1,n-1);alpha=indices-left
    def visit(value,key=None):
        if isinstance(value,np.ndarray):
            if key in STATIC or not value.ndim or len(value)!=n:return value.copy()
            if key in DISCRETE or value.dtype.kind in 'biuUS':return value[nearest].copy()
            if value.shape[1:]==(3,3):
                out=Slerp(np.arange(n),Rotation.from_matrix(value))(indices).as_matrix()
                out[exact]=value[left[exact]]
                return out
            shape=(len(indices),)+(1,)*(value.ndim-1)
            out=value[left]*(1-alpha.reshape(shape))+value[right]*alpha.reshape(shape)
            if key in DIRECTIONS:
                length=np.linalg.norm(out,axis=-1,keepdims=True)
                if np.any(length<1e-8):raise ValueError('Ambiguous antipodal source direction during retiming')
                out=out/length
            out[exact]=value[left[exact]]
            return out
        if isinstance(value,dict):return {k:visit(v,k) for k,v in value.items()}
        if isinstance(value,list):return [visit(v) for v in value]
        return copy.deepcopy(value)
    out=visit(targets)
    for side,v in out['sides'].items():
        v['segments']=[]
        for segment in targets['sides'][side].get('segments',[]):
            hits=np.flatnonzero((nearest>=segment['start'])&(nearest<segment['end']))
            if len(hits):v['segments'].append(dict(segment,start=int(hits[0]),end=int(hits[-1]+1)))
        active=(v['contact']>0)&~exact
        # Discrete anchors/pivots must not blend across contact switches.
        # Rebuild only the existing active-foot geometric target, not contact
        # detection, scale, floor, confidence, or source calibration.
        if np.any(active):
            R=v['foot_target_r'][active];g=robot.sides[side]
            v['foot_target_p'][active]=v['anchors'][active]-np.einsum('nij,nj->ni',R,v['pivots'][active])
            v['foot_target_p'][active,2]=-np.min(R[:,2,:]@g['hull'].T,axis=1)
        for key in ('heel','toe'):
            v[key+'_speed']=np.linalg.norm(np.gradient(v[key],axis=0)*fps,axis=1)
    report.update(downstream_ik_required=True,interpolated_trajectory_is_seed_only=True,calibration='preserved from full original sequence',contact_phases='nearest original phase; no redetection',reference='original source at source_time_s')
    return out,indices/fps,report
