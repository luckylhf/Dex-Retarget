"""Recompute export FK and report geometry, contact, anatomical and time errors."""
import argparse,json,pickle
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .robot import Robot
from .math3d import unit
from .anatomy import configure
from .review import local_windows
from .multicontact import lowest,contact_local

def stats(a):
    a=np.asarray(a)
    return {'median':float(np.median(a)),'p95':float(np.percentile(a,95)),'max':float(np.max(a))} if a.size else None
def ang(a,b):return np.degrees(np.arccos(np.clip(np.sum(unit(a)*unit(b),axis=-1),-1,1)))

def audit_model_path(path, meta, payload):
    from ..cli import sha
    from ..paths import ASSETS, asset_fingerprints, is_runtime_asset

    original = Path(meta['model'])
    expected = meta.get('model_sha256') or payload.get('model_sha256')
    if payload.get('model_sha256') and expected != payload['model_sha256']:
        raise ValueError('Motion and metadata model hashes differ')
    if original.is_file():
        if expected and sha(original) != expected:
            raise ValueError('Recorded model XML hash mismatch')
        return original

    candidate = ASSETS / 'mjcf/dex_29.xml'
    if not expected or not candidate.is_file() or sha(candidate) != expected:
        raise ValueError('Recorded model is unavailable. Set DEX_FRAX_ASSETS to matching robot assets.')
    identity = path.parent / 'identity.json'
    if identity.is_file():
        old_assets = json.loads(identity.read_text())['assets']
        expected_assets = {name: digest for name, digest in old_assets.items() if is_runtime_asset(name)}
        if asset_fingerprints() != expected_assets:
            raise ValueError('Configured robot asset hashes differ from the recorded assets')
    return candidate


def audit(path,write=True,cache=None,changed=None):
    path=Path(path);data=pickle.loads(path.read_bytes());meta=json.loads(path.with_suffix('.json').read_text())
    with np.load(path.with_suffix('.diagnostics.npz'),allow_pickle=False) as archive:dg={k:archive[k] for k in archive.files}
    rob=Robot(audit_model_path(path,meta,data))
    if data['joint_names']!=rob.names or data['quat_order']!='xyzw':raise ValueError('Export contract mismatch')
    physical_lo=rob.lower.copy();physical_hi=rob.upper.copy()
    if 'anatomy' in meta['configuration']:configure(rob,meta['configuration']['anatomy'])
    q=data['dof_pos'].astype(float);n=len(q);rr=Rotation.from_quat(data['root_rot']).as_matrix();fps=data['fps']
    from ..audit_cache import digest,valid_rows,remember,dense_floor as cached_floor
    cache={} if cache is None else cache
    key=digest(meta,dg,rob.path)
    valid=valid_rows(cache,data,changed,key)
    if valid:
        values=cache['frames']
    else:
        values=dict(
            errors={s:{k:[None]*n for k in ('upper_deg','lower_deg','hand_forward_deg','hand_width_deg','forearm_width_deg','elbow_axis_deg','signed_elbow_goal_error_deg','thigh_deg','shin_deg','foot_pose_cm','sole_height_cm','toe_height_cm','heel_height_cm')} for s in rob.sides},
            contact_xyz={s:[None]*n for s in rob.sides},allp={s:[None]*n for s in rob.sides},
            flat_tilt={s:[None]*n for s in rob.sides},flat_heel={s:[None]*n for s in rob.sides},
            root_orientation=[None]*n,torso_orientation=[None]*n,torso_up=[None]*n,
            body_heights=[None]*n,body_errors={c['body_name']:{'normal_cm':[None]*n,'fixed_xy_cm':[None]*n} for c in meta.get('body_contact_models',[])})
    errors=values['errors'];contact_xyz=values['contact_xyz'];allp=values['allp']
    flat_tilt=values['flat_tilt'];flat_heel=values['flat_heel']
    root_orientation=values['root_orientation'];torso_orientation=values['torso_orientation'];torso_up=values['torso_up']
    body_heights=values['body_heights'];body_errors=values['body_errors']
    contact_vel={s:[] for s in rob.sides};floating=longest=0
    root_speed=np.linalg.norm(np.gradient(data['root_pos'].astype(float),1/fps,axis=0),axis=1)
    for i in (range(*changed) if valid else range(n)):
        rob.set(np.r_[data['root_pos'][i],q[i]],rr[i])
        root_orientation[i]=np.degrees(Rotation.from_matrix(dg['root_r'][i].T@rr[i]).magnitude())
        torso_orientation[i]=np.degrees(Rotation.from_matrix(dg['torso_target'][i].T@rob.rot(rob.torso)).magnitude())
        torso_up[i]=ang(dg['torso_target'][i,:,2],rob.rot(rob.torso)[:,2])
        for s,g in rob.sides.items():
            d=errors[s];v=lambda k:dg[s+'_'+k][i]
            for a,b,k in [('shoulder','elbow','upper'),('elbow','wrist','lower'),('hip','knee','thigh'),('knee','foot','shin')]:
                d[k+'_deg'][i]=ang(rob.point(g[b])-rob.point(g[a]),v(k))
            d['hand_forward_deg'][i]=ang(-rob.rot(g['tcp'])[:,2],v('hand_forward'))
            d['hand_width_deg'][i]=ang(rob.rot(g['tcp'])[:,0],v('hand_width'))
            d['forearm_width_deg'][i]=ang(rob.rot(g['forearm'])[:,0],v('forearm_width'))
            if s+'_elbow_axis' in dg:
                d['elbow_axis_deg'][i]=ang(-rob.rot(g['shoulder'])[:,1],v('elbow_axis'))
                goal=np.clip(g['elbow_geometric_zero']-v('signed_elbow_bend'),rob.lower[g['arm'][3]],rob.upper[g['arm'][3]])
                d['signed_elbow_goal_error_deg'][i]=abs(np.rad2deg(q[i,g['arm'][3]-3]-goal))
            d['foot_pose_cm'][i]=np.linalg.norm(rob.point(g['foot'])-v('foot_target_p'))*100
            d['sole_height_cm'][i]=rob.point(g['foot'],rob.lowest(s))[2]*100
            for k in ('toe','heel'):d[k+'_height_cm'][i]=rob.point(g['foot'],g[k])[2]*100
            if v('phase')==1:
                flat_tilt[s][i]=np.degrees(np.arccos(np.clip(rob.rot(g['foot'])[2,2],-1,1)))
                flat_heel[s][i]=rob.point(g['foot'],g['heel'])[2]*100
            if v('contact')>0:
                p=rob.point(g['foot'],v('pivots'));error=p-v('anchors');error[2]=rob.point(g['foot'],rob.lowest(s))[2];contact_xyz[s][i]=error*100
            allp[s][i]={k:rob.point(g['foot'],g[k]) for k in ('toe','heel','sole')}
        all_low=min(rob.point(b,lowest(rob,b))[2] for b in rob.surface_models)
        body_heights[i]=all_low
        for contact in meta.get('body_contact_models',[]):
            name=contact['body_name'];mode=dg['body_'+name+'_mode'][i]
            if not mode:continue
            body=rob.bid(name);point=rob.point(body,contact_local(rob,contact))
            body_errors[name]['normal_cm'][i]=abs(point[2])*100
            if mode==1:
                anchor=dg['body_'+name+'_anchors'][i]
                body_errors[name]['fixed_xy_cm'][i]=np.linalg.norm(point[:2]-anchor[:2])*100
    cache['frames']=values
    for i,all_low in enumerate(body_heights):
        unexplained=(all_low>.05 if 'source_expected_contact' in dg else min(errors[side]['sole_height_cm'][i] for side in rob.sides)>5) and root_speed[i]<.2
        floating=floating+1 if unexplained else 0
        longest=max(longest,floating)
    def present(items):return [v for v in items if v is not None]
    segments={}
    for s,g in rob.sides.items():
        segments[s]=[]
        for seg in meta['estimated_contact_segments'][s]:
            a,b=seg['start'],seg['end'];p=np.array([v[seg['phase']] for v in allp[s][a:b]])
            velocities=np.linalg.norm(np.diff(p,axis=0),axis=1)*fps*100
            contact_vel[s].extend(velocities)
            segments[s].append({**seg,'start_s':a/fps,'end_s':(b-1)/fps,
                'pivot_speed_cm_s':stats(velocities),'xy_diameter_cm':float(np.max(np.linalg.norm(p[:,:2]-np.median(p[:,:2],axis=0),axis=1))*200),
                'z_cm':stats(abs(p[:,2])*100)})
    limit_violation=np.maximum(physical_lo[3:]-q,0)+np.maximum(q-physical_hi[3:],0)
    effective_violation=np.maximum(rob.lower[3:]-q,0)+np.maximum(q-rob.upper[3:],0)
    result={'frames':n,'fps':fps,'finite':bool(np.isfinite(q).all() and np.isfinite(data['root_pos']).all()),
        'joint_limit_violation_deg':float(np.rad2deg(limit_violation.max())),
        'effective_anatomical_limit_violation_deg':float(np.rad2deg(effective_violation.max())),
        'joint_step_deg':stats(np.rad2deg(abs(np.diff(q,axis=0))).ravel()),
        'joint_speed_deg_s':stats(np.rad2deg(abs(np.diff(q,axis=0))).ravel()*fps),
        'root_step_cm':stats(np.linalg.norm(np.diff(data['root_pos'],axis=0),axis=1)*100),
        'root_target_difference_cm':stats(np.linalg.norm(data['root_pos']-dg['root_target'],axis=1)*100),
        'root_orientation_deg':stats(root_orientation),'torso_orientation_deg':stats(torso_orientation),
        'torso_up_deg':stats(torso_up),'stationary_floating_longest_seconds':longest/fps,
        'sides':{},'contact_segments':segments,
        'minimum_body_visual_cm':float(min(body_heights)*100),
        'body_penetration_frames_below_minus_2cm':int(np.sum(np.asarray(body_heights)<-.02)),
        'floating_basis':'all-body surface; sustained stationary clearance, not feet-only' if 'source_expected_contact' in dg else 'legacy feet-only',
        'body_contacts':{name:{k:stats(present(v)) for k,v in values.items()} for name,values in body_errors.items()}}
    for s,d in errors.items():
        ce=np.asarray(present(contact_xyz[s]));g=rob.sides[s]
        result['sides'][s]={k:stats(v) for k,v in d.items()}
        result['sides'][s]['local_pose_windows']={k:max(local_windows(d[k],fps),key=lambda v:v['p95']) for k in ('hand_forward_deg','hand_width_deg','upper_deg','lower_deg','elbow_axis_deg')}
        result['sides'][s]['minimum_visual_sole_cm']=float(min(d['sole_height_cm']))
        result['sides'][s]['penetration_frames_over_1mm']=int(np.sum(np.array(d['sole_height_cm'])<-.1))
        result['sides'][s]['contact_xyz_cm']=stats(np.linalg.norm(ce,axis=1)) if ce.size else None
        result['sides'][s]['contact_z_cm']=stats(abs(ce[:,2])) if ce.size else None
        result['sides'][s]['contact_speed_cm_s']=stats(contact_vel[s])
        result['sides'][s]['estimated_flat_support_tilt_deg']=stats(present(flat_tilt[s]))
        result['sides'][s]['estimated_flat_support_heel_height_cm']=stats(present(flat_heel[s]))
        result['sides'][s]['J5_J6_J7_abs_deg']={rob.names[j-3]:stats(np.rad2deg(abs(q[:,j-3]))) for j in g['arm'][4:]}
        if hasattr(rob,'wrist_limit_record'):
            wrist=q[:,np.asarray(g['arm'][5:7])-3]
            excess=np.maximum(g['wrist_soft_lower']-wrist,0)+np.maximum(wrist-g['wrist_soft_upper'],0)
            result['sides'][s]['wrist_signed_ranges_deg']=np.rad2deg(np.stack([wrist.min(axis=0),wrist.max(axis=0)])).tolist()
            result['sides'][s]['wrist_soft_excess_deg']=stats(np.rad2deg(excess).ravel())
            result['sides'][s]['wrist_soft_excess_frame_counts']=(excess>1e-6).sum(axis=0).tolist()
            bends=ang(dg[s+'_upper'],dg[s+'_lower']);near=bends<20
            result['sides'][s]['near_straight_elbow_axis_deg']=stats(np.asarray(d['elbow_axis_deg'])[near])
    if 'baseline' in meta:
        baseline=pickle.loads(Path(meta['baseline']).read_bytes())
        arm_ids=np.concatenate([np.asarray(g['arm'])-3 for g in rob.sides.values()]);body_ids=np.setdiff1d(np.arange(q.shape[1]),arm_ids)
        result['baseline_body_identical']={k:bool(np.array_equal(data[k],baseline[k])) for k in ('root_pos','root_rot')}
        result['baseline_body_identical']['non_arm_dof_pos']=bool(np.array_equal(data['dof_pos'][:,body_ids],baseline['dof_pos'][:,body_ids]))
        if not all(result['baseline_body_identical'].values()):raise AssertionError('Baseline body changed')
    from .self_collision import audit_export
    from .self_collision import CollisionModel,scan
    arm_model=CollisionModel(rob,meta['configuration'].get('self_collision'))
    result['self_collision']=scan(arm_model,np.c_[data['root_pos'],q],rr,8,cache=cache.setdefault('arm_collision',{}),changed=changed if valid else None)[0]
    from .self_collision import CollisionModel,scan
    full_model=CollisionModel(rob,dict(meta['configuration'].get('self_collision') or {},coverage='full'))
    result['full_self_collision']=scan(full_model,np.c_[data['root_pos'],q],rr,8,cache=cache.setdefault('full_collision',{}),changed=changed if valid else None)[0]
    import hashlib
    pair_names=sorted(sorted(pair) for pair in full_model.names)
    result['full_self_collision']['pair_set_sha256']=hashlib.sha256(json.dumps(pair_names,separators=(',',':')).encode()).hexdigest()
    delta=data['root_pos'].astype(float)-dg['root_target']
    result['root_translation_error_cm']=dict(xy=stats(np.linalg.norm(delta[:,:2],axis=1)*100),z=stats(abs(delta[:,2])*100),reference='scaled and grounded prepared human root target; height compensation included')
    from ..torso_branch import floor_samples
    dense_floor=cached_floor(rob,rr,np.c_[data['root_pos'],q],cache,changed,valid)
    result['interpolated_floor']={
        'interpolation_steps':8,'samples':len(dense_floor),
        'minimum_body_visual_cm':float(dense_floor.min()*100),
        'samples_below_minus_1mm':int(np.any(dense_floor<-.001,axis=1).sum()),
        'violating_intervals':np.unique(np.flatnonzero(np.any(dense_floor<-.001,axis=1))//8).tolist(),
        'scope':'native plus linearly interpolated joints/root translation and SLERP root rotation; not continuous certification'}
    from ..support_quality import evaluate_contacts
    result['body_support']=evaluate_contacts(meta,dg,body_errors,n,fps)
    from ..temporal_quality import assess
    result['temporal_quality']=assess(q,fps,rob.names,dg)
    cache['metrics']=dict(torso=np.asarray(torso_orientation),errors={side:{k.removesuffix('_deg'):np.asarray(v) for k,v in vals.items() if k in ('upper_deg','lower_deg','hand_forward_deg','hand_width_deg','elbow_axis_deg')} for side,vals in errors.items()})
    cache['work']=dict(native_frames_recomputed=(changed[1]-changed[0]) if valid else n,cache_hit=valid)
    remember(cache,data,key)
    if write:
        np.savez_compressed(path.with_suffix('.pose_errors.npz'),**{s+'_'+k:np.asarray(v) for s,d in errors.items() for k,v in d.items()})
        path.with_suffix('.check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('pkl');a=p.parse_args();r=audit(a.pkl)
    print(json.dumps({k:v for k,v in r.items() if k!='contact_segments'},ensure_ascii=False,indent=2))
