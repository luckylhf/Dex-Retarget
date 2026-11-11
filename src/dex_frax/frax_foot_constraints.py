"""Exact declared foot geometry rows in the refine foot-stage Frax QP.

CPU callbacks use the same MuJoCo/Mink geometry as the independent output audit.
This is an initial solve constraint, not a rebase of a repair's fixed reference.
"""
import threading
import numpy as np
import jax
import jax.numpy as jnp
from frax.ik.constraints import InequalityConstraint


from .frax_arm_constraints import ArmGeometry


@jax.tree_util.register_static
class FootGeometry(ArmGeometry):
    """The complete owned leg class, including the incoming swept segment."""
    def __init__(self,kinematics,robot,gain,columns=None):
        super().__init__(kinematics,robot,gain,columns=columns,pair_group='leg_collision')
        self.margin=.0001
        self.report.update(policy='declared_complete_leg_geometry_in_initial_frax_swept_v2',
            clearance_target_m=self.margin,scope=self.geometry.names)


@jax.tree_util.register_static
class FootSupport(InequalityConstraint):
    """Actual sole floor and the existing 0.5 mm stance support box."""
    def __init__(self,kinematics,robot,gain,columns=None,space=None):
        self.robot=kinematics if space is None else space
        self.kinematics=kinematics;self.sides=robot.sides;self.gain=float(gain)
        self.columns=np.arange(32) if columns is None else np.asarray(columns,int)
        self.num_rows=14;self.penalty=1e8

    def rows(self,q,ref):
        def values(x):
            if len(self.columns)!=32:x=ref['priority_pose'].at[self.columns].set(x)
            if 'priority_mask' in ref:x=jnp.where(ref['priority_mask'],ref['priority_pose'],x)
            tf=self.kinematics.transforms(x,ref['root_r']);p=tf[:,:3,3];r=tf[:,:3,:3];out=[]
            for side,g in self.sides.items():
                v=ref['sides'][side];body=g['foot'];point=p[body]+r[body]@v['pivots']
                height=p[body,2]+jnp.min(jnp.asarray(g['hull'])@r[body,2])
                out.extend([height,point[0]-v['anchors'][0],point[1]-v['anchors'][1]])
            return jnp.stack(out)
        value=values(q);J=jax.jacfwd(values)(q);floor=jnp.array([0,3])
        active=jnp.repeat(jnp.array([ref['sides'][s]['contact']>0 for s in self.sides]),3)
        G=jnp.concatenate([-J[floor],jnp.where(active[:,None],J,0.),jnp.where(active[:,None],-J,0.)])
        h=jnp.concatenate([value[floor]-.0001,jnp.where(active,.0005-value,1.),jnp.where(active,.0005+value,1.)])
        return G,self.gain*h
