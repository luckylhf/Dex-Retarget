"""Uncertain extra surface proximity must not dominate an established foot plant.

This changes a soft attraction task only. Mesh clearance and planted-foot XYZ
constraints remain separate. Without a planted foot the original body-support
task is retained, including seated/lying transitions.
"""
import math


def body_height_task(height, targets, frame, contact, cfg):
    """Return the soft height residual and its derivative with respect to height.

    Reference-skin proximity is not a measured load-bearing contact. When feet
    already provide a plant, use a bounded residual for incompatible extra body
    hints. Its scale is the same source-height uncertainty used by the detector,
    converted to robot metres. Near zero it retains the original slope; far
    away its gradient decays instead of pulling the waist/arms into the plane.
    Blend by the existing foot transition weights to avoid a new mode switch.
    """
    policy=cfg.get('body_contact_loss','linear')
    if policy=='linear':return float(height),1.
    if policy!='bounded_when_feet_planted':raise ValueError('Unknown body contact loss: '+policy)
    tolerance=float(targets['body_contact_height_tolerance_m'])
    if not math.isfinite(tolerance) or tolerance<=0:raise ValueError('Invalid body-contact height uncertainty')
    support=min(1.,max(0.,max((float(v['contact'][frame]) for v in targets['sides'].values()),default=0.)))
    ratio=float(height)/tolerance;normalizer=math.hypot(1.,ratio)
    bounded=float(height)/normalizer;derivative=1./normalizer**3
    return ((1.-support)*float(height)+support*bounded,
            (1.-support)+support*derivative)
