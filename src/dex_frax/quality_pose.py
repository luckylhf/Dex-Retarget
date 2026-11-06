"""Screen pose against source anatomy, geometry, and the training motion contract."""
import json
import pickle
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dex_frax.paths import ASSETS


def quality(pkl, meta, audit):
    limits = json.loads(Path(__file__).with_suffix('.json').read_text())
    data = pickle.loads(Path(pkl).read_bytes())
    fps=float(data['fps']);q=data['dof_pos'].astype(float);p=data['root_pos'].astype(float)
    flags=[]
    if len(q)<limits['minimum_frames']:flags.append('too_short_for_training_window')
    if not audit['finite']:flags.append('nonfinite_motion')
    if audit['effective_anatomical_limit_violation_deg']>limits['joint_limit_tolerance_deg']:
        flags.append('physical_or_anatomical_joint_limit')
    if meta['support_constraint_failures']:flags.append('support_constraint_failed')
    speed=np.linalg.norm(np.diff(p,axis=0),axis=1)*fps
    if speed.max()>limits['root_speed_m_s']:flags.append('root_linear_speed_above_8m_s')
    R=Rotation.from_quat(data['root_rot']).as_matrix()
    omega=np.linalg.norm(Rotation.from_matrix(R[1:]@R[:-1].transpose(0,2,1)).as_rotvec(),axis=1)*fps
    if omega.max()>limits['root_angular_speed_rad_s']:flags.append('root_angular_speed_above_8rad_s')
    manifest=json.loads((ASSETS/'model.json').read_text())
    velocity=np.gradient(q,1/fps,axis=0)
    motor_limits=np.array([manifest['joint_limits'][n]['velocity'] for n in data['joint_names']], dtype=float)
    if np.any(abs(velocity)>motor_limits+1e-4):flags.append('joint_velocity_exceeds_model_limit')
    if audit['root_orientation_deg']['max']>limits['root_orientation_max_deg']:
        flags.append('source_pelvis_orientation_mismatch')
    if audit['torso_up_deg']['max']>limits['torso_up_max_deg']:
        flags.append('source_torso_up_mismatch')
    if audit['torso_orientation_deg']['p95']>limits['torso_orientation_p95_deg']:
        flags.append('source_torso_orientation_mismatch')
    if audit['stationary_floating_longest_seconds']>=limits['stationary_floating_seconds']:
        flags.append('stationary_floating_over_half_second')
    if audit.get('minimum_body_visual_cm',0.) < limits['minimum_sole_m']*100:
        flags.append('body_penetration_below_minus_2cm')
    for side, values in audit['sides'].items():
        if values['minimum_visual_sole_cm']/100<limits['minimum_sole_m']:
            flags.append(side+'_foot_penetration_below_minus_2cm')
        for key in ('upper_deg','lower_deg'):
            if values[key]['p95']>limits['arm_direction_p95_deg']:flags.append(side+'_'+key+'_source_mismatch')
        for key in ('hand_forward_deg','hand_width_deg'):
            if values[key]['p95']>limits['hand_axis_p95_deg']:flags.append(side+'_'+key+'_source_mismatch')
    return dict(flags=flags, frames=len(q), backend='pose_quality', criteria=limits,
                root_speed_max_m_s=float(speed.max()), root_angular_speed_max_rad_s=float(omega.max()),
                source_pose_audit={k:audit[k] for k in ('root_orientation_deg','torso_orientation_deg','torso_up_deg')},
                stationary_floating_longest_seconds=audit['stationary_floating_longest_seconds'],
                support_constraint_failures=meta['support_constraint_failures'],
                initialization='whole-sequence solve; no GMR priming or table-retry semantics',
                dynamic_status='not_tested_by_controller')
