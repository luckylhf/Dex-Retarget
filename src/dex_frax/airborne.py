"""Preserve inferred source flight; a ground plane is not a scene model."""
import numpy as np

from .v8.math3d import runs


POLICY = 'source_airborne_preservation_v1'


def airborne_mask(targets, frames):
    """Missing flight evidence keeps the existing conservative checks."""
    if 'source_airborne' not in targets:
        return np.zeros(frames, dtype=bool)
    value = np.asarray(targets['source_airborne'])
    if value.shape != (frames,) or not np.all((value == 0) | (value == 1)):
        raise ValueError('source_airborne must contain one boolean per frame')
    return value.astype(bool)


def preserve_targets(targets):
    """Remove ground attraction and its transition tails on source flight frames.

    Raw foot targets already include robot scale and sole geometry. Clearance,
    collision, speed and source-pose constraints remain active downstream.
    """
    mask = airborne_mask(targets, len(targets['root_p']))
    if not np.any(mask):
        return targets
    for foot in targets['sides'].values():
        foot['foot_target_p'][mask] = foot['raw_foot_p'][mask]
        foot['foot_target_r'][mask] = foot['raw_foot_r'][mask]
        for key in ('contact', 'phase', 'foot_flat_confidence', 'foot_roll_level_weight',
                    'ground_contact_confidence'):
            if key in foot:
                foot[key][mask] = 0
        segments = []
        for segment in foot.get('segments', []):
            a, b = segment['start'], segment['end']
            for start, end in runs(~mask[a:b]):
                part = dict(segment, start=a+start, end=a+end)
                if start:
                    part.pop('continuous_from_previous', None)
                segments.append(part)
        foot['segments'] = segments
    for contact in targets.get('body_contacts', []):
        contact['mode'][mask] = 0
        contact['confidence'][mask] = 0
    if 'source_expected_contact' in targets:
        targets['source_expected_contact'][mask] = False
    return targets


def floating_report(clearance_m, root_speed_m_s, fps, diagnostics):
    """Screen stationary clearance only where source flight does not explain it."""
    clearance = np.asarray(clearance_m)
    candidate = (clearance > .05) & (np.asarray(root_speed_m_s) < .2)
    mask = airborne_mask(diagnostics, len(clearance))
    unexplained = candidate & ~mask
    longest = max((b-a for a, b in runs(unexplained)), default=0)
    return dict(policy=POLICY, source_airborne_available='source_airborne' in diagnostics,
                candidate_frames=int(candidate.sum()),
                source_airborne_exempt_frames=int((candidate & mask).sum()),
                unexplained_frames=int(unexplained.sum()),
                longest_seconds=float(longest/fps))
