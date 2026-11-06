"""Hand-relative wrist ranges, intersected with the actual robot limits.

NASA's range endpoints are population reference ranges, not robot ratings.
Lower endpoints mark a soft zone; upper endpoints bound this retargeting
profile. The URDF/MJCF remains unchanged. This is not a coupled physiological
workspace model: NASA's table supplies single-axis ranges only.
"""
import numpy as np

def configure(robot,profile):
    if not hasattr(robot,'physical_lower'):
        robot.physical_lower=robot.lower.copy();robot.physical_upper=robot.upper.copy()
    robot.lower=robot.physical_lower.copy();robot.upper=robot.physical_upper.copy()
    records={}
    for side,g in robot.sides.items():
        maximum=profile['maximum_deg'];soft=profile['soft_deg']
        # At neutral, hand +X points toward the thumb, -Z points distally.
        # J6 positive moves distally toward -X: ulnar deviation, both hands.
        # J7 positive moves toward +Y: left extension / right flexion.
        negative,positive=('flexion','extension') if side=='Left' else ('extension','flexion')
        lo=np.deg2rad([-maximum['radial'],-maximum[negative]])
        hi=np.deg2rad([maximum['ulnar'],maximum[positive]])
        ids=g['arm'][5:7]
        robot.lower[ids]=np.maximum(robot.physical_lower[ids],lo)
        robot.upper[ids]=np.minimum(robot.physical_upper[ids],hi)
        if np.any(robot.lower[ids]>=robot.upper[ids]):raise ValueError('Empty anatomical/physical wrist intersection')
        g['wrist_soft_lower']=np.maximum(robot.lower[ids],np.deg2rad([-soft['radial'],-soft[negative]]))
        g['wrist_soft_upper']=np.minimum(robot.upper[ids],np.deg2rad([soft['ulnar'],soft[positive]]))
        records[side]={'joints':[robot.names[j-3] for j in ids],
            'physical_lower_deg':np.rad2deg(robot.physical_lower[ids]).tolist(),
            'physical_upper_deg':np.rad2deg(robot.physical_upper[ids]).tolist(),
            'effective_lower_deg':np.rad2deg(robot.lower[ids]).tolist(),
            'effective_upper_deg':np.rad2deg(robot.upper[ids]).tolist(),
            'soft_lower_deg':np.rad2deg(g['wrist_soft_lower']).tolist(),
            'soft_upper_deg':np.rad2deg(g['wrist_soft_upper']).tolist(),
            'J6_positive':'ulnar','J7_positive':positive,
            'frame':'forearm-to-hand Ry(J6) Rx(J7), neutral mounting verified'}
    robot.wrist_limit_record=records
    return records

def wrist_barrier(q,side_geometry):
    """Signed distance outside the preferred interval, with its derivative."""
    lo,hi=side_geometry['wrist_soft_lower'],side_geometry['wrist_soft_upper']
    outside=(q<lo)|(q>hi)
    return q-np.clip(q,lo,hi),outside.astype(float)

def wrist_from_relative_rotation(R):
    """Dex's verified two-axis mounting; no world-space up/down assumptions."""
    distal=-np.asarray(R)[..., :,2]
    return np.stack([np.arctan2(-distal[...,0],-distal[...,2]),
                     np.arcsin(np.clip(distal[...,1],-1.,1.))],axis=-1)
