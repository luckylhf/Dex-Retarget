"""frax kinematics with V8's prescribed pelvis rotation and 32 unknowns.

The reduced coordinates are [world pelvis XYZ, 29 named hinges]. Pelvis
orientation is an input per frame, exactly as in V8, rather than a soft task.
Fixed link offsets are derived from the unchanged MuJoCo asset and verified
against frax at random poses before conversion.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from frax import Robot as FraxRobot
from .paths import ASSETS


@jax.tree_util.register_static
class DexKinematics:
    nq = nv = 32

    def __init__(self, robot):
        self.frax = FraxRobot(str(ASSETS / "urdf/dex_29.urdf"), joint_ordering=robot.names)
        if self.frax.joint_names != robot.names:
            raise ValueError("frax joint ordering differs from Dex model")
        robot.set(np.zeros(32), np.eye(3))
        m = robot.m
        joint_to_idx = {int(j): i for i, j in enumerate(robot.jids)}
        indices, offsets = [], []
        for body in range(m.nbody):
            parent = body
            while parent and not any(j in joint_to_idx for j in range(int(m.body_jntadr[parent]), int(m.body_jntadr[parent] + m.body_jntnum[parent]))):
                parent = int(m.body_parentid[parent])
            matches = [j for j in range(int(m.body_jntadr[parent]), int(m.body_jntadr[parent] + m.body_jntnum[parent])) if j in joint_to_idx]
            idx = joint_to_idx[matches[0]] if matches else -1
            base_R = robot.rot(parent) if idx >= 0 else np.eye(3)
            base_p = robot.point(parent) if idx >= 0 else np.zeros(3)
            tf = np.eye(4)
            tf[:3, :3] = base_R.T @ robot.rot(body)
            tf[:3, 3] = base_R.T @ (robot.point(body) - base_p)
            indices.append(idx + 1)
            offsets.append(tf)
        self.indices = np.array(indices)
        self.offsets = np.array(offsets)

    def transforms(self, x, root_r):
        local = self.frax.joint_to_world_transforms(x[3:])
        local = jnp.concatenate([jnp.eye(4)[None], local])[self.indices] @ self.offsets
        world = jnp.broadcast_to(jnp.eye(4), local.shape)
        world = world.at[:, :3, :3].set(root_r @ local[:, :3, :3])
        return world.at[:, :3, 3].set(x[:3] + jnp.einsum("ij,bj->bi", root_r, local[:, :3, 3]))

    def integrate(self, q, v, dt):
        return q + v * dt

    def velocity_to_qdot_map(self, q):
        return jnp.eye(self.nv, dtype=q.dtype)

    def verify(self, robot, count=12):
        from scipy.spatial.transform import Rotation
        rng = np.random.default_rng(20261105)
        fk = jax.jit(self.transforms)
        max_pos = max_rot = 0.
        # World body is intentionally excluded: it is not attached to pelvis.
        for i in range(count):
            q = np.zeros(29) if i == 0 else rng.uniform(np.maximum(robot.lower[3:], -.8), np.minimum(robot.upper[3:], .8))
            x = np.r_[rng.normal(size=3), q]
            r = Rotation.from_rotvec(rng.normal(size=3) * .4).as_matrix()
            robot.set(x, r)
            tf = np.asarray(fk(x, r))
            max_pos = max(max_pos, float(np.max(abs(tf[1:, :3, 3] - robot.d.xpos[1:]))))
            max_rot = max(max_rot, float(np.max(abs(tf[1:, :3, :3] - robot.d.xmat[1:].reshape(-1, 3, 3)))))
        if max(max_pos, max_rot) > 1e-8:
            raise ValueError(f"frax/MuJoCo FK mismatch: position={max_pos}, rotation={max_rot}")
        return dict(poses=count, max_position_m=max_pos, max_rotation_matrix_error=max_rot)
