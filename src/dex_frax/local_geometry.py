"""Local arm floor/support rows, enforced during fitting and checked nonlinearly."""
import numpy as np
from scipy.sparse import block_diag,csc_matrix


class Geometry:
    def __init__(self,robot,t,xs,a,b,ids,moves):
        self.r,self.t,self.xs,self.a,self.b,self.ids=robot,t,xs,a,b,ids
        self.bodies=[body for body in robot.surface_models if moves(body)]
        self.frames=[]
        for i in range(a,b):
            robot.set(xs[i],t['root_r'][i]);floors={};contacts=[]
            for body in self.bodies:
                hull=robot.surface_models[body]['hull']
                height=float((hull@robot.rot(body)[2]+robot.point(body)[2]).min())
                floors[body]=min(-.00099,height-5e-7)
            for c in t.get('body_contacts',[]):
                body=c['body'];mode=int(c['mode'][i])
                if not mode or body not in self.bodies:continue
                hull=np.asarray(c['robot_hull']);local=hull[np.argmin(hull@robot.rot(body)[2])].copy()
                contacts.append((body,local,robot.point(body,local).copy(),mode,hull.copy()))
            self.frames.append((floors,contacts))

    def evaluate(self,z,linear=False):
        r=self.r;blocks=[];lows=[];highs=[];penalty=0.
        for j,q in enumerate(z.reshape(-1,len(self.ids))):
            i=self.a+j;x=self.xs[i].copy();x[self.ids]=q;r.set(x,self.t['root_r'][i])
            rows=[];lo=[];hi=[];floors,contacts=self.frames[j]
            for body,target in floors.items():
                hull=r.surface_models[body]['hull'];heights=hull@r.rot(body)[2]+r.point(body)[2]
                deficit=max(target-float(heights.min()),0.);penalty+=deficit**2
                if linear and heights.min()<.02:
                    for k in np.argsort(heights)[:4]:
                        P,_=r.jac(body,hull[k]);rows.append(P[2,self.ids]);lo.append(target-heights[k]);hi.append(np.inf)
            for body,local,point,mode,hull in contacts:
                if mode!=1:local=hull[np.argmin(hull@r.rot(body)[2])]
                value=r.point(body,local)-point
                if mode!=1:value=value[2:]
                penalty+=float(np.maximum(abs(value)-.00049,0.)@np.maximum(abs(value)-.00049,0.))
                if linear:
                    P,_=r.jac(body,local);P=P if mode==1 else P[2:]
                    rows.extend(P[:,self.ids]);lo.extend(-.00049-value);hi.extend(.00049-value)
            if linear:
                blocks.append(csc_matrix(np.asarray(rows).reshape(-1,len(self.ids))))
                lows.extend(lo);highs.extend(hi)
        if linear:return block_diag(blocks,format='csc'),np.asarray(lows),np.asarray(highs)
        return 1e8*penalty
