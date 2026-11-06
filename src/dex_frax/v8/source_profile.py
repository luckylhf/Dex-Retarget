"""Rig definitions do not depend on a clip containing a standing/flat pose."""
import hashlib,json
from pathlib import Path
import numpy as np
from .math3d import frame_xz


def lafan_foot(motion,p,r,side):
    profile=json.loads(Path(__file__).with_name('lafan1_profile.json').read_text())
    ankle=side+'Foot';toe=side+profile['orientation_joint_suffix']
    index=motion.names.index(toe)
    if motion.parents[index]!=motion.names.index(ankle):
        raise ValueError('LAFAN profile requires Toe parented to Foot: '+side)
    length=float(np.linalg.norm(motion.offsets[index]))
    if abs(length-profile['expected_ankle_to_toe_m'])>profile['bone_length_tolerance_m']:
        raise ValueError('Unrecognized LAFAN foot dimensions; provide a matching source profile')
    up=r[toe]@np.array(profile['sole_up_in_toe'],float)
    forward=r[toe]@np.array(profile['forward_in_toe'],float)
    frame=frame_xz(forward,up)
    # The original LAFAN Toe has a fixed orientation relative to Foot. Verify
    # that fact rather than deriving a sole normal from low/slow world poses.
    relative=np.swapaxes(r[ankle],1,2)@r[toe]
    delta=relative-relative[0]
    if np.max(abs(delta))>1e-4:
        raise ValueError('Articulated or altered LAFAN toe rig needs an explicit geometry profile')
    surface=json.loads(Path(__file__).with_name('reference').joinpath('lafan1_surfaces.json').read_text())
    foot=surface['feet'][side]
    local_up=np.array(foot['sole_up_local_in_foot']);local_forward=np.array(foot['forward_local_in_foot'])
    frame=frame_xz(r[ankle]@local_forward,r[ankle]@local_up)
    heel=p[ankle]+r[ankle]@np.array(foot['heel_local_in_foot'])
    toe_point=p[ankle]+r[ankle]@np.array(foot['toe_local_in_foot'])
    signature=dict(names=motion.names,parents=motion.parents.tolist(),offsets_m=np.round(motion.offsets,5).tolist())
    record=dict(method=profile['id'],orientation_joint=toe,gmr_forward_in_toe=profile['forward_in_toe'],gmr_sole_up_in_toe=profile['sole_up_in_toe'],local_up_in_ankle=local_up.tolist(),ankle_to_toe_m=length,heel_geometry_status=foot['geometry'],reference_model_sha256=surface['fbx_sha256'],fixed_surface_correction_from_gmr_deg=foot['gmr_to_surface_correction_deg'],skeleton_sha256=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest(),standing_frames_required=False)
    return frame,heel,toe_point,record,profile['scene_ground']
