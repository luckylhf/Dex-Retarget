"""Declared arm collision rows inside the ordered initial Frax solve.

The callback measures the unchanged full collision model. Only pairs affected
by arm joints enter this stage; final independent export acceptance is unchanged.
"""
import threading
import numpy as np
import mujoco
import jax
import jax.numpy as jnp
from frax.ik.constraints import InequalityConstraint
from mink.limits.collision_avoidance_limit import compute_contact_normal_jacobian


def release_activity(proximity,history,decay):
    memory=jnp.maximum(proximity,history*decay)
    # Approaching or maintaining proximity must not damp the source fit.
    return memory,jnp.maximum(memory-proximity,0.)


def source_arm_steps(t):
    """Soft recovery dead zone from the unchanged temporal policy's source data.

    Use the same five regional directions and neighbouring source intervals as
    its isolated-pulse screen. This is an objective dead zone, not a motor cap.
    """
    from .temporal_quality import POLICY
    sides=[];n=len(t['root_p'])
    for side in ('Left','Right'):
        values=t['sides'][side]
        steps=[]
        for key in ('upper','lower','hand_forward','hand_width','elbow_axis'):
            vectors=np.asarray(values[key],float)
            vectors=vectors/np.maximum(np.linalg.norm(vectors,axis=-1,keepdims=True),1e-12)
            steps.append(np.arccos(np.clip(np.sum(vectors[:-1]*vectors[1:],axis=-1),-1.,1.)))
        regional=np.max(steps,axis=0) if n>1 else np.zeros(1)
        intervals=np.arange(n)-1
        nearby=np.maximum.reduce([regional[np.clip(intervals+offset,0,len(regional)-1)] for offset in (-1,0,1)])
        sides.append(np.maximum(nearby/POLICY['source_step_ratio'],np.deg2rad(POLICY['minimum_step_deg'])))
    return np.stack(sides,axis=-1)


@jax.tree_util.register_static
class ArmGeometry(InequalityConstraint):
    def __init__(self, kinematics, robot, gain, columns=None, pair_group='collision'):
        from .v8.robot import Robot
        from .v8.self_collision import CollisionModel
        from .refine_routing import collision_groups,submodel
        local=Robot(robot.path)
        full=CollisionModel(local,{'coverage':'full'})
        selected=collision_groups(full)[pair_group]
        self.geometry=submodel(full,selected)
        self.lock=threading.Lock()
        self.robot=kinematics;self.gain=float(gain)
        self.columns=np.arange(32) if columns is None else np.asarray(columns,int)
        self.num_columns=len(self.columns)
        self.num_rows=len(self.geometry.pairs);self.penalty=1e8
        self.margin=float(full.cfg['margin_m'])
        self.interpolation_steps=8
        if not self.num_rows:raise ValueError('Missing initial arm collision geometry')
        if np.any(self.geometry.m.geom_bodyid[self.geometry.pairs]<=0):
            raise ValueError('Arm self-collision rows must not contain world geometry')
        self._pairs=np.asarray(self.geometry.pairs,int)
        self._bounded=np.all(self.geometry.m.geom_rbound[self._pairs]>0.,axis=1)
        self._radii=self.geometry.m.geom_rbound[self._pairs].sum(axis=1)
        self._distances=np.empty((self.interpolation_steps,self.num_rows))
        self._witness=np.empty((self.interpolation_steps,self.num_rows,6))
        self._fromto=np.empty(6);self._normal=np.empty(3)
        self._jac1=np.empty((3,self.geometry.m.nv));self._jac2=np.empty_like(self._jac1)
        self._qpos=np.zeros(self.geometry.m.nq);self._qpos[3]=1.
        from .refine_routing import descendant_bodies
        bodies=self.geometry.m.geom_bodyid[self._pairs]
        self._side_pairs=[]
        for g in robot.sides.values():
            arm_bodies=descendant_bodies(robot,g['arm'])
            self._side_pairs.append(np.any(np.isin(bodies,list(arm_bodies)),axis=1))
        self._activity_q=None;self._activity_value=None
        self.report=dict(policy='declared_arm_geometry_in_initial_frax_swept_v3_fast',pair_count=self.num_rows,
            clearance_target_m=self.margin,scope=self.geometry.names,
            interpolation_steps=self.interpolation_steps,
            row_policy='worst sampled clearance per pair, relinearized each IK iteration',
            variables=self.num_columns,
            upstream_protection='zero Jacobian columns for protected joints',
            solver='existing Frax differential IK; no new candidate route')

    def _update(self,q):
        # Every declared pair is robot/robot. Use a canonical rigid root and
        # update only the geometry's MjData, avoiding the duplicate Robot FK.
        self._qpos[self.geometry.r.qadr]=q[3:]
        self.geometry.configuration.update(self._qpos)

    def _distances_and_witness(self,part):
        m=self.geometry;data=m.configuration.data;limit=m.cfg['detection_m']
        a,b=self._pairs.T
        near=(~self._bounded)|(np.linalg.norm(data.geom_xpos[a]-data.geom_xpos[b],axis=1)-self._radii<limit)
        near &= ~m.limit.excluded_pairs(m.configuration)
        row=self._distances[part];row.fill(limit)
        for k in np.flatnonzero(near):
            value=mujoco.mj_geomDistance(m.m,data,int(a[k]),int(b[k]),limit,self._fromto)
            row[k]=value
            if abs(value-limit)>=1e-12:self._witness[part,k]=self._fromto

    def _measure(self,q,previous,first):
        q=np.asarray(q);previous=np.asarray(previous);m=self.geometry
        fractions=np.array([1.]) if first else np.arange(1,self.interpolation_steps+1)/self.interpolation_steps
        for part,alpha in enumerate(fractions):
            self._update(previous*(1-alpha)+q*alpha)
            self._distances_and_witness(part)
        distances=self._distances[:len(fractions)];chosen=np.argmin(distances,axis=0)
        minimum=distances[chosen,np.arange(self.num_rows)]
        near=(minimum<m.cfg['detection_m'])&(abs(minimum-m.cfg['detection_m'])>=1e-12)
        G=np.zeros((self.num_rows,self.num_columns));gap=np.ones(self.num_rows)
        for part in np.unique(chosen[near]):
            alpha=fractions[part];self._update(previous*(1-alpha)+q*alpha)
            for k in np.flatnonzero((chosen==part)&near):
                a,b=self._pairs[k];value=distances[part,k]
                derivative=compute_contact_normal_jacobian(m.m,m.configuration.data,int(a),int(b),
                    self._witness[part,k],self._normal,self._jac1,self._jac2)
                sign=-1. if value>=0. else 1.
                G[k]=sign*derivative[m.r.cols[self.columns]]*alpha
                gap[k]=value-self.margin
        return G,gap

    def measure(self,q,rotation):
        with self.lock:return self._measure(q,q,True)

    def rows(self,q,ref):
        if self.num_columns!=32:q=ref['priority_pose'].at[self.columns].set(q)
        if 'priority_pose' in ref:
            q=jnp.where(ref['priority_mask'],ref['priority_pose'],q)
        G,gap=jax.pure_callback(self.measure_swept,
            (jax.ShapeDtypeStruct((self.num_rows,self.num_columns),q.dtype),jax.ShapeDtypeStruct((self.num_rows,),q.dtype)),
            q,ref['root_r'],ref.get('previous',q),jnp.asarray(ref.get('first',1.)),vmap_method='sequential')
        if 'priority_mask' in ref:G=jnp.where(ref['priority_mask'][self.columns][None,:],0.,G)
        return G,self.gain*gap

    def activity(self,previous):
        """Previous-pose proximity, used only as a soft continuity preference."""
        with self.lock:
            previous=np.asarray(previous)
            if self._activity_q is not None and np.array_equal(previous[3:],self._activity_q):
                return self._activity_value.copy()
            self._update(previous);self._distances_and_witness(0)
            minimum=np.array([self._distances[0,mask].min() for mask in self._side_pairs])
            value=np.clip((2*self.margin-minimum)/self.margin,0.,1.)
            value=value*value*(3.-2.*value)
            self._activity_q=previous[3:].copy();self._activity_value=value.copy()
            return value

    def measure_swept(self,q,rotation,previous,first):
        """Linearize the incoming segment with respect to its current endpoint.

        All pairs belong to the robot. A common rigid root transform preserves
        signed distance, so evaluating interpolated hinges under the current
        root rotation is equivalent to the auditor's root SLERP for these rows.
        The previous endpoint is fixed; the endpoint Jacobian gets factor alpha.
        """
        with self.lock:
            return self._measure(q,previous,bool(np.asarray(first).item()))
