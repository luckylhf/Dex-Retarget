import numpy as np
from scipy.spatial.transform import Rotation

def unit(v):
    return v / np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-10)

def frame_yz(y,z):
    z=unit(z);y=unit(y-(y*z).sum(axis=-1,keepdims=True)*z)
    return np.stack((unit(np.cross(y,z)),y,z),axis=-1)

def frame_xz(x,z):
    z=unit(z);x=unit(x-(x*z).sum(axis=-1,keepdims=True)*z)
    return np.stack((x,unit(np.cross(z,x)),z),axis=-1)

def skew(v):
    x,y,z=v
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])

def runs(mask):
    e=np.diff(np.r_[False,mask,False].astype(int))
    return list(zip(np.flatnonzero(e==1),np.flatnonzero(e==-1)))

def smoother(x):
    x=np.clip(x,0.,1.)
    return x*x*x*(x*(6*x-15)+10)

def blend_rot(a,b,w):
    # Scalar or per-frame weights; shortest-path SO(3) interpolation.
    relative=Rotation.from_matrix(np.swapaxes(a,-1,-2)@b).as_rotvec()
    return a@Rotation.from_rotvec(relative*np.asarray(w)[...,None]).as_matrix()
