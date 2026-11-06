from pathlib import Path
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from .math3d import skew

from dex_frax.paths import ASSETS

DEFAULT_MODEL=str(ASSETS/'mjcf/dex_29.xml')
DEFAULT_URDF=str(ASSETS/'urdf/dex_29.urdf')

class Robot:
    def __init__(self,path=DEFAULT_MODEL):
        from dex_frax.model_cache import load
        self.path=str(Path(path).resolve());self.m=load(self.path);self.d=mujoco.MjData(self.m)
        m=self.m
        self.jids=np.array([i for i in range(m.njnt) if m.jnt_type[i]==mujoco.mjtJoint.mjJNT_HINGE])
        if len(self.jids)!=29 or m.nq!=36: raise ValueError('Expected Dex 29 revolute joints plus free pelvis')
        self.names=[m.joint(int(i)).name for i in self.jids]
        manifest = json.loads((ASSETS / 'model.json').read_text())
        self.speed_limits = np.array([float(manifest['joint_limits'][name]['velocity']) for name in self.names])
        if not np.isfinite(self.speed_limits).all() or np.any(self.speed_limits <= 0):
            raise ValueError('Invalid per-axis joint velocity limits')
        self.qadr=m.jnt_qposadr[self.jids];self.dadr=m.jnt_dofadr[self.jids]
        self.lower=np.r_[[-np.inf]*3,m.jnt_range[self.jids,0]]
        self.upper=np.r_[[np.inf]*3,m.jnt_range[self.jids,1]]
        self.cols=np.r_[0,1,2,self.dadr]
        self._root_matrix=None;self._root_quat=None;self._fk_qpos=None;self._jac_cache={}
        self.cache_stats=dict(set_calls=0,fk_calls=0,jac_calls=0,jac_hits=0,rotation_conversions=0)
        self.set(np.r_[0.,0.,1.,np.zeros(29)],np.eye(3))
        self.sides={};self.torso=self.bid('waist_pitch_link')
        for side,suf in [('Left','l'),('Right','r')]:
            g={k:self.bid(name+'_'+suf+'_link') for k,name in [('hip','hip_pitch'),('knee','knee_pitch'),('foot','ankle_roll'),('shoulder','shoulder_yaw'),('elbow','elbow_pitch'),('forearm','elbow_yaw'),('wrist','wrist_pitch'),('hand','wrist_roll')]}
            g['tcp']=self.bid(side.lower()+'_tcp_link')
            g['arm']=[3+self.names.index(x+'_'+suf+'_joint') for x in ['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow_pitch','elbow_yaw','wrist_pitch','wrist_roll']]
            g['leg']=[3+self.names.index(x+'_'+suf+'_joint') for x in ['hip_pitch','hip_roll','hip_yaw','knee_pitch','ankle_pitch','ankle_roll']]
            gid=m.geom('ankle_roll_'+suf+'_link_visual').id;mid=m.geom_dataid[gid]
            v=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]].copy()
            world=v@self.d.geom_xmat[gid].reshape(3,3).T+self.d.geom_xpos[gid]
            local=(world-self.d.xpos[g['foot']])@self.rot(g['foot'])
            g['vertices']=local
            zmin=float(local[:,2].min());sole=local[local[:,2]<zmin+.006]
            # Ground plane support polygon in the visual foot's own frame.
            # Sampling the actual mesh avoids using ankle origin as the sole.
            from scipy.spatial import ConvexHull
            hull=ConvexHull(local)
            g['hull']=local[hull.vertices]
            g['toe']=np.r_[np.percentile(sole[:,0],95),np.median(sole[:,1]),zmin]
            g['heel']=np.r_[np.percentile(sole[:,0],5),np.median(sole[:,1]),zmin]
            g['sole']=(g['toe']+g['heel'])/2
            g['thigh_length']=float(np.linalg.norm(self.d.xpos[g['knee']]-self.d.xpos[g['hip']]))
            g['shin_length']=float(np.linalg.norm(self.d.xpos[g['foot']]-self.d.xpos[g['knee']]))
            g['upper_length']=float(np.linalg.norm(self.d.xpos[g['elbow']]-self.d.xpos[g['shoulder']]))
            g['lower_length']=float(np.linalg.norm(self.d.xpos[g['wrist']]-self.d.xpos[g['elbow']]))
            u=self.point(g['elbow'])-self.point(g['shoulder']);u/=np.linalg.norm(u)
            v=self.point(g['wrist'])-self.point(g['elbow']);v/=np.linalg.norm(v)
            # Actual joint-origin geometry, not a hard-coded 12.9 degree offset.
            flex_axis=-self.rot(g['shoulder'])[:,1]
            g['elbow_geometric_zero']=float(np.arctan2(flex_axis@np.cross(u,v),u@v))
            self.sides[side]=g
        self.neutral_pos=self.d.xpos.copy()
        from .multicontact import surfaces,region_hull
        self.surface_models=surfaces(self)
        for side,g in self.sides.items():
            g['ground_hulls']={b:self.surface_models[b]['hull'] for b in [g['shoulder'],g['forearm'],g['tcp']]}
            g['ground_hulls'][g['tcp']]=region_hull(self,side+'_hand',g['tcp'])

    def joint_step(self, fps, cfg):
        # Array is in joint_names order, without the three root translations.
        numerical_limit = np.deg2rad(cfg['numerical_joint_speed_deg_s'])
        return np.minimum(self.speed_limits, numerical_limit) / fps

    def bid(self,name):
        i=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_BODY,name)
        if i<0: raise ValueError(name)
        return i
    def rot(self,b):return self.d.xmat[b].reshape(3,3)
    def set(self,x,root_r):
        from scipy.spatial.transform import Rotation
        self.cache_stats['set_calls']+=1
        if self._root_matrix is None or not np.array_equal(root_r,self._root_matrix):
            self._root_matrix=np.array(root_r,copy=True)
            self._root_quat=Rotation.from_matrix(root_r).as_quat()[[3,0,1,2]]
            self.cache_stats['rotation_conversions']+=1
        self.d.qpos[:3]=x[:3];self.d.qpos[3:7]=self._root_quat;self.d.qpos[self.qadr]=x[3:]
        if self._fk_qpos is not None and np.array_equal(self.d.qpos,self._fk_qpos):return
        mujoco.mj_kinematics(self.m,self.d);mujoco.mj_comPos(self.m,self.d)
        self._fk_qpos=self.d.qpos.copy();self._jac_cache.clear();self.cache_stats['fk_calls']+=1
    def point(self,b,local=None):
        return self.d.xpos[b].copy() if local is None else self.d.xpos[b]+self.rot(b)@local
    def jac(self,b,local=None):
        self.cache_stats['jac_calls']+=1
        key=(int(b),None if local is None else tuple(local))
        if key in self._jac_cache:
            self.cache_stats['jac_hits']+=1
            return tuple(a.copy() for a in self._jac_cache[key])
        jp=np.zeros((3,self.m.nv));jr=np.zeros_like(jp)
        mujoco.mj_jac(self.m,self.d,jp,jr,self.point(b,local),b)
        result=jp[:,self.cols],jr[:,self.cols]
        self._jac_cache[key]=tuple(a.copy() for a in result)
        return result
    def lowest(self,side):
        g=self.sides[side];v=g['hull'];return v[np.argmin(v@self.rot(g['foot'])[2,:])]
