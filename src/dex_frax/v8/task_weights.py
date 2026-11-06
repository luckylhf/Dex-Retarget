"""Retain hand orientation when inferred ground support changes arm priorities."""

def hand_weights(cfg,side,frame):
    confidence=side.get('ground_contact_confidence')
    confidence=0. if confidence is None else float(confidence[frame])
    primary=max(cfg['arm_direction'],cfg['hand_direction'],cfg['hand_width'])
    return tuple(cfg[key]+confidence*(primary-cfg[key]) for key in ['hand_direction','hand_width'])
