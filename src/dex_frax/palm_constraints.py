"""The existing source-palm envelope expressed inside collision SQP."""
import numpy as np
from .similarity import PALM_AXIS_LIMIT_DEG
from .v8.math3d import skew


class SourcePalms:
    def __init__(self,robot,targets,reference,a,b,sides,ids,interior_deg=0.,fixed_frames=()):
        self.r,self.t,self.a,self.sides,self.ids=robot,targets,a,list(sides),np.asarray(ids)
        self.interior=np.full(b-a,np.deg2rad(interior_deg))
        self.interior[list(fixed_frames)]=0.
        self.limits=np.empty((b-a,len(sides),2))
        for j in range(b-a):
            robot.set(reference[a+j],targets['root_r'][a+j])
            for k,side in enumerate(self.sides):
                R=robot.rot(robot.sides[side]['tcp']);v=targets['sides'][side]
                for h,(axis,key) in enumerate(((-R[:,2],'hand_forward'),(R[:,0],'hand_width'))):
                    old=np.arccos(np.clip(axis@v[key][a+j],-1.,1.))
                    self.limits[j,k,h]=max(np.deg2rad(PALM_AXIS_LIMIT_DEG),float(old))

    def evaluate(self,row,j):
        r=self.r;i=self.a+j;r.set(row,self.t['root_r'][i]);C=[];upper=[];excess=0.
        for k,side in enumerate(self.sides):
            body=r.sides[side]['tcp'];R=r.rot(body);_,A=r.jac(body);A=A[:,self.ids]
            for h,(axis,key) in enumerate(((-R[:,2],'hand_forward'),(R[:,0],'hand_width'))):
                target=self.t['sides'][side][key][i];dot=float(np.clip(target@axis,-1.,1.));limit=self.limits[j,k,h]
                C.append(-(target@(-skew(axis)@A)));upper.append(dot-np.cos(max(0.,limit-self.interior[j])))
                excess=max(excess,float(np.arccos(dot))-limit)
        return np.asarray(C),np.asarray(upper),max(excess,0.)
