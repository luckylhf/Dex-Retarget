"""V8 acceptance for the per-axis velocity-constrained pose solver."""
import json
import pickle
from pathlib import Path

import numpy as np

from dex_frax.paths import ASSETS
from .identity import CONVERTER, QUALITY_POLICY

VELOCITY_TOLERANCE_RAD_S = 1e-4
SPEED_FLAG = 'joint_velocity_exceeds_model_limit'


def adjacent_speed_report(q, fps, names, motor_limits):
    """Compare each adjacent interval to its named joint's velocity limit (rad/s)."""
    q = np.asarray(q, dtype=np.float64)
    limits = np.asarray(motor_limits, dtype=np.float64)
    fps = float(fps)
    if q.ndim != 2 or len(q) < 2 or q.shape[1] != len(names) or not len(names):
        raise ValueError('Expected at least two frames and one column per joint')
    if not np.isfinite(q).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid joint angles or frame rate')
    if limits.shape != (len(names),) or not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError('Expected one finite positive velocity limit per joint')

    speeds = np.abs(np.diff(q, axis=0)) * fps
    violations = speeds > limits + VELOCITY_TOLERANCE_RAD_S
    ratios = speeds / limits
    i, j = np.unravel_index(np.argmax(ratios), ratios.shape)
    central_speed = np.abs(np.gradient(q, 1.0 / fps, axis=0))
    counts = violations.sum(axis=0)
    return dict(policy=QUALITY_POLICY, fps=fps, intervals=len(speeds),
                tolerance_rad_s=VELOCITY_TOLERANCE_RAD_S,
                violating_intervals=int(violations.any(axis=1).sum()),
                central_excess_frames=int((central_speed > limits + VELOCITY_TOLERANCE_RAD_S).any(axis=1).sum()),
                violations_by_joint={name:int(count) for name,count in zip(names, counts) if count},
                worst_joint=names[j], worst_interval=int(i), worst_time_s=float(i / fps),
                worst_end_time_s=float((i + 1) / fps), peak_ratio=float(ratios[i, j]),
                speed_rad_s=float(speeds[i, j]), limit_rad_s=float(limits[j]))


def check_speed(path):
    """Inspect a trusted local Dex preview PKL, preview NPZ or training NPZ without rewriting it."""
    path = Path(path)
    manifest = json.loads((ASSETS/'model.json').read_text())
    if path.suffix.lower() == '.pkl':
        data = pickle.loads(path.read_bytes())
        q, fps, names = data['dof_pos'], data['fps'], list(data['joint_names'])
    elif path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as data:
            if 'model_sha256' in data and str(data['model_sha256']) != manifest['model_sha256']:
                raise ValueError('Motion model hash differs from the supplied assets')
            q = data['joint_pos'] if 'joint_pos' in data else data['dof_pos']
            fps, names = float(data['fps']), data['joint_names'].tolist()
    else:
        raise ValueError('Expected a .pkl or .npz motion')
    if names != manifest['joint_names']:
        raise ValueError('Motion joint names/order differ from the supplied assets')
    limits = [float(manifest['joint_limits'][name]['velocity']) for name in names]
    return adjacent_speed_report(q, fps, names, limits)


def quality(pkl, meta, audit):
    """Apply pose/geometry gates and V8 adjacent-interval speed acceptance."""
    from .quality_pose import quality as pose_quality
    result = pose_quality(pkl, meta, audit)
    report = check_speed(pkl)
    flags = [flag for flag in result['flags'] if flag != SPEED_FLAG]
    if report['violating_intervals']:
        flags.append(SPEED_FLAG)
    collision=audit.get('self_collision')
    if collision is None:
        flags.append('arm_self_collision_not_checked')
    elif collision['collision_samples']:
        flags.append('arm_self_collision_unresolved')
    result['self_collision']=collision
    result.update(backend=CONVERTER, quality_policy=QUALITY_POLICY, flags=flags, step_velocity=report)
    return result
