"""Exact declared foot geometry rows in the refine foot-stage Frax QP.

CPU callbacks use the same MuJoCo/Mink geometry as the independent output audit.
This is an initial solve constraint, not a rebase of a repair's fixed reference.
"""
import threading
import numpy as np
import jax
import jax.numpy as jnp
from frax.ik.constraints import InequalityConstraint


@jax.tree_util.register_static
class FootGeometry(InequalityConstraint):
    def __init__(self,kinematics,robot,gain):
        from .v8.robot import Robot
        from .v8.self_collision import CollisionModel
        from .refine_routing import collision_groups,submodel
        # The callback owns its mutable FK state; it never races the caller's FK.
        local=Robot(robot.path);full=CollisionModel(local,{'coverage':'full'})
        owned=collision_groups(full)['leg_collision']
        selected=[k for k in owned if any(name.startswith('retarget_foot_') or
            (name.startswith('ankle_roll_') and name.endswith('_visual')) for name in full.names[k])]
        self.geometry=submodel(full,selected);self.lock=threading.Lock()
        self.robot=kinematics;self.gain=float(gain);self.num_rows=len(selected);self.penalty=1e8
        self.margin=.0001
        if not self.num_rows:raise ValueError('Missing initial foot collision geometry')
        self.report=dict(policy='declared_foot_geometry_in_initial_frax_v1',pair_count=self.num_rows,
            clearance_target_m=self.margin,scope=self.geometry.names,solver='Frax differential IK; exact CPU geometry rows')

    def measure(self,q,rotation):
        with self.lock:
            m=self.geometry;m.update(np.asarray(q),np.asarray(rotation))
            con=m.limit.compute_qp_inequalities(m.configuration,1.)
            active=np.isfinite(con.h)
            G=np.asarray(con.G[:,m.r.cols],dtype=np.float64).copy()
            gap=m.distances()-self.margin
            G[~active]=0.;gap[~active]=1.
            return G,gap

    def rows(self,q,ref):
        G,gap=jax.pure_callback(self.measure,
            (jax.ShapeDtypeStruct((self.num_rows,32),q.dtype),jax.ShapeDtypeStruct((self.num_rows,),q.dtype)),
            q,ref['root_r'],vmap_method='sequential')
        # Frozen upstream coordinates cannot be used to satisfy a foot row and
        # then silently projected back after the QP step.
        if 'priority_mask' in ref:G=jnp.where(ref['priority_mask'][None,:],0.,G)
        return G,self.gain*gap


@jax.tree_util.register_static
class FootSupport(InequalityConstraint):
    """Actual sole floor and the existing 0.5 mm stance support box."""
    def __init__(self,kinematics,robot,gain):
        self.robot=kinematics;self.sides=robot.sides;self.gain=float(gain)
        self.num_rows=14;self.penalty=1e8

    def rows(self,q,ref):
        def values(x):
            if 'priority_mask' in ref:x=jnp.where(ref['priority_mask'],ref['priority_pose'],x)
            tf=self.robot.transforms(x,ref['root_r']);p=tf[:,:3,3];r=tf[:,:3,:3];out=[]
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
