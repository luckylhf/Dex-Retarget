"""Ease inferred pivot-type orientation changes without releasing a plant."""
import numpy as np
from scipy.spatial.transform import Rotation


def refine(t, robot, duration_s=.1):
    reports = []
    for side, v in t['sides'].items():
        g = robot.sides[side]
        for previous, current in zip(v['segments'][:-1], v['segments'][1:]):
            a, b = current['start'], current['end']
            if previous['end'] != a:
                continue
            r = v['foot_target_r']
            start = r[a-1].copy()
            jump = Rotation.from_matrix(start.T @ r[a]).magnitude()
            if jump <= np.deg2rad(3.):
                continue
            count = min(b-a, max(2, round(duration_s*t['fps'])))
            # Quintic easing reaches the existing target at the window end.
            # Keep contact weights, anchors and actual convex-floor geometry.
            u = np.arange(1, count+1)/count
            w = u**3*(10.+u*(-15.+6.*u))
            delta = Rotation.from_matrix(start.T @ r[a:a+count]).as_rotvec()
            r[a:a+count] = start @ Rotation.from_rotvec(delta*w[:, None]).as_matrix()
            pos = v['foot_target_p']
            pos[a:a+count] = v['anchors'][a:a+count] - np.einsum('nij,nj->ni', r[a:a+count], v['pivots'][a:a+count])
            pos[a:a+count, 2] = -np.min(r[a:a+count, 2, :] @ g['hull'].T, axis=1)
            reports.append(dict(side=side, start=a, end=a+count, from_phase=previous['phase'],
                                to_phase=current['phase'], original_jump_deg=float(np.rad2deg(jump))))
    return dict(policy='contact_pivot_orientation_easing_v1', duration_s=duration_s, windows=reports,
                anchors_unchanged=True, contact_weights_unchanged=True)
