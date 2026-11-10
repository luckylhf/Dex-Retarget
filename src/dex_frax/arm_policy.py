"""Whole-arm resemblance policy; collision tiers are not similarity tiers."""
POLICY = 'whole_arm_resemblance_v2'
SHOULDER_TIERS = (10., 20., 30.)


def enabled(weights):
    return weights.get('arm_repair_policy') == POLICY


def collision_options(degrees, iterations=12):
    if degrees not in SHOULDER_TIERS:
        raise ValueError('Expected a 10/20/30 degree collision tier')
    return dict(name={10.:'small',20.:'medium',30.:'large'}[degrees],
                joint_deg=degrees, free_distal=True, whole_arm=True, gesture_caps_active=False,
                hand_rotation_deg=degrees,
                hand_position_m={10.:.03,20.:.08,30.:.15}[degrees],
                tracking_weight=1., iterations=iterations,
                qp_iterations=20000, coupled=False)


def active_arm_support(robot, targets, a, b):
    """Support scenes retain the existing coupled contact policy."""
    if robot is None:
        return False
    from .v8.self_collision import _arm_bodies
    import numpy as np
    bodies=set(_arm_bodies(robot,list(robot.sides)))
    return any(c['body'] in bodies and np.any(np.asarray(c['mode'])[a:b]>0)
               for c in targets.get('body_contacts',[]))
