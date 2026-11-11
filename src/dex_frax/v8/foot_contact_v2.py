"""Independent support selection and confidence-weighted sole orientation."""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from .math3d import runs


DEFAULTS = {
    "end_lift_m": 0.012,
    "flat_tilt_deg": 25.0,
    "flat_speed_m_s": 0.15,
    "side_lift_m": 0.015,
    "side_full_lift_m": 0.035,
    "maximum_side_release": 0.65,
}


def support_labels(heel, toe, heel_stable, toe_stable, minimum_frames, settings):
    """Choose a pivot from positive lift evidence, independent of sole tilt."""
    labels = (heel_stable | toe_stable).astype(np.int8)
    heel_minus_toe = heel[:, 2] - toe[:, 2]
    lifted = settings["end_lift_m"]
    for label, evidence in (
        (2, toe_stable & ~heel_stable & (heel_minus_toe >= lifted)),
        (3, heel_stable & ~toe_stable & (heel_minus_toe <= -lifted)),
    ):
        for start, end in runs(evidence):
            if end - start >= minimum_frames:
                labels[start:end] = label
    return labels


def orientation_targets(raw, labels, foot, heel_stable, toe_stable, heel_speed, toe_speed,
                        ground, fps, cfg):
    settings = cfg["v2"]
    active = labels > 0
    tilt = np.rad2deg(np.arccos(np.clip(raw[:, 2, 2], -1, 1)))
    heel_z, toe_z = foot["heel"][:, 2], foot["toe"][:, 2]
    flat_evidence = active & heel_stable & toe_stable
    flat_evidence &= np.abs(heel_z - toe_z) < cfg["height_spread_m"]
    angle_score = np.clip(1 - tilt / settings["flat_tilt_deg"], 0, 1)
    speed_score = np.clip(1 - np.maximum(heel_speed, toe_speed) / settings["flat_speed_m_s"], 0, 1)
    flat_confidence = flat_evidence * angle_score * speed_score

    side_evidence = np.zeros(len(raw))
    corners = foot.get("sole_corners")
    if corners is not None:
        # Posed skin points, paired on the same lateral side at heel and forefoot.
        heel_a, heel_b, toe_a, toe_b = corners[:, :, 2].T
        heel_gap, toe_gap = heel_a - heel_b, toe_a - toe_b
        low_edge = np.maximum(np.minimum(heel_a, heel_b), np.minimum(toe_a, toe_b))
        high_edge_gap = np.minimum(abs(heel_gap), abs(toe_gap))
        consistent = heel_gap * toe_gap > 0
        near_ground = low_edge < ground + cfg["height_tolerance_m"]
        start, full = settings["side_lift_m"], settings["side_full_lift_m"]
        side_evidence = consistent * near_ground * np.clip((high_edge_gap - start) / (full - start), 0, 1)

    # Smooth corrections, not the original motion. Lateral evidence can release
    # some roll while planted; it cannot choose a contact phase or create an anchor.
    sigma = max(0.5, fps * cfg["transition_s"] / 2)
    roll_level = active * (1 - settings["maximum_side_release"] * side_evidence)
    roll_level = gaussian_filter1d(roll_level, sigma, mode="nearest", truncate=2)
    pitch_level = gaussian_filter1d(flat_confidence, sigma, mode="nearest", truncate=2)
    changed = (roll_level > 0) | (pitch_level > 0)
    r = raw[changed]
    angles = np.column_stack((
        np.arctan2(r[:, 2, 1], r[:, 2, 2]) * (1 - roll_level[changed]),
        np.arctan2(-r[:, 2, 0], np.hypot(r[:, 0, 0], r[:, 1, 0])) * (1 - pitch_level[changed]),
        np.arctan2(r[:, 1, 0], r[:, 0, 0]),
    ))
    result = raw.copy()
    if changed.any():
        result[changed] = Rotation.from_euler("xyz", angles).as_matrix()
    return result, flat_confidence, side_evidence, roll_level
