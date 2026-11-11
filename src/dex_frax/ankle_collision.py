"""User-declared same-leg foot/shin exclusion inside both ankle joint limits.

This is a collision classification rule, not a measured geometry clearance.
Pairs stay present so an out-of-range ankle never receives the exclusion.
"""
import numpy as np
import mink
from mink.limits.limit import Constraint

POLICY = 'same_leg_foot_shin_within_ankle_limits_v1'
# Only absorb float32 representation of an XML endpoint (under 0.000006 deg).
LIMIT_NUMERICAL_TOLERANCE_RAD = 1e-7


class SameLegAnkleRule:
    def __init__(self, model, pairs):
        pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
        bodies = model.geom_bodyid[pairs]
        self.size = len(pairs)
        self.sides = []
        for side, suffix in (('Left', 'l'), ('Right', 'r')):
            shin = model.body('knee_pitch_' + suffix + '_link').id
            foot = model.body('ankle_roll_' + suffix + '_link').id
            indices = np.flatnonzero(((bodies[:, 0] == shin) & (bodies[:, 1] == foot)) |
                                     ((bodies[:, 1] == shin) & (bodies[:, 0] == foot)))
            joints = [model.joint('ankle_' + axis + '_' + suffix + '_joint').id
                      for axis in ('pitch', 'roll')]
            if not np.all(model.jnt_limited[joints]):
                raise ValueError('same_leg_ankle_exclusion_requires_bounded_joints')
            limits = model.jnt_range[joints].copy()
            if not np.isfinite(limits).all():
                raise ValueError('same_leg_ankle_exclusion_requires_finite_limits')
            self.sides.append(dict(side=side, indices=indices,
                                   qadr=model.jnt_qposadr[joints].copy(), limits=limits,
                                   joints=[model.joint(j).name for j in joints],
                                   pairs=[[model.geom(a).name, model.geom(b).name]
                                          for a, b in pairs[indices]]))

    def excluded(self, qpos):
        mask = np.zeros(self.size, dtype=bool)
        for side in self.sides:
            if not len(side['indices']):
                continue
            angles = np.asarray(qpos)[side['qadr']]
            limits = side['limits']
            if (np.isfinite(angles).all() and
                    np.all(angles >= limits[:, 0] - LIMIT_NUMERICAL_TOLERANCE_RAD) and
                    np.all(angles <= limits[:, 1] + LIMIT_NUMERICAL_TOLERANCE_RAD)):
                mask[side['indices']] = True
        return mask

    def describe(self):
        return dict(policy=POLICY, basis='user_declared_collision_exclusion_not_measured_clearance',
                    condition='both same-side ankle pitch and roll within inclusive XML joint limits',
                    numerical_tolerance_rad=LIMIT_NUMERICAL_TOLERANCE_RAD,
                    applies_to='same-side knee_pitch shin body versus ankle_roll foot body only',
                    sides=[dict(side=s['side'], joints=s['joints'], limits_rad=s['limits'].tolist(),
                                conditional_pair_count=len(s['indices']), pairs=s['pairs'])
                           for s in self.sides])


class AnkleCollisionAvoidanceLimit(mink.CollisionAvoidanceLimit):
    """The same per-pose exclusion for every consumer of Mink collision rows."""
    def __init__(self, model, geom_pairs, **kwargs):
        super().__init__(model, geom_pairs, **kwargs)
        self.ankle_rule = SameLegAnkleRule(model, self.geom_id_pairs)

    def excluded_pairs(self, configuration):
        return self.ankle_rule.excluded(configuration.data.qpos)

    def compute_qp_inequalities(self, configuration, dt):
        result = super().compute_qp_inequalities(configuration, dt)
        excluded = self.excluded_pairs(configuration)
        if excluded.any():
            result.G[excluded] = 0.
            result.h[excluded] = np.inf
        return Constraint(G=result.G, h=result.h)
