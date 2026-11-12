"""Native/support and swept geometry rows for the existing sparse local fit."""
import numpy as np
import mujoco
from scipy import sparse
from scipy.spatial.transform import Rotation
from mink.limits.collision_avoidance_limit import compute_contact_normal_jacobian
from .support_quality import preserve_contact


class Geometry:
    def __init__(self,robot,t,xs,a,b,ids,moves,restore=False,fixed_frames=(),*,initial_leg_repair=False,collision_cfg=None):
        self.r,self.t,self.xs,self.a,self.b,self.ids=robot,t,xs,a,b,np.asarray(ids)
        self.restore=restore;self.fixed=set(fixed_frames);self.n=b-a;self.width=len(ids)
        self.initial_leg_repair=initial_leg_repair
        self.elastic_rows=[]
        self.bodies=[body for body in robot.surface_models if moves(body)]
        self.collision=None;self.palms=None
        joints=[j for j in ids if j>=3]
        sides=[side for side,g in robot.sides.items() if set(ids)&set(g['arm'])]
        if joints:
            from .v8.self_collision import CollisionModel
            from .refine_routing import movable_submodel,collision_groups
            self.collision=movable_submodel(CollisionModel(robot,dict(collision_cfg or {},coverage='full')),joints)
            passive=np.zeros(len(self.collision.pairs),bool)
            if not sides:passive[collision_groups(self.collision)['collision']]=True
            if sides:
                from .palm_constraints import SourcePalms
                self.palms=SourcePalms(robot,t,xs,a,b,sides,ids,interior_deg=.1,fixed_frames=sorted(self.fixed))
        deltas=Rotation.from_matrix(t['root_r'][a:b-1].transpose(0,2,1)@t['root_r'][a+1:b]).as_rotvec()
        self.samples=[];self.native=[]
        for j in range(self.n):
            for part in range(8 if j+1<self.n else 1):
                u=part/8.;rotation=t['root_r'][a+j] if not part else t['root_r'][a+j]@Rotation.from_rotvec(deltas[j]*u).as_matrix()
                row=xs[a+j].copy() if not part else xs[a+j]*(1-u)+xs[a+j+1]*u
                robot.set(row,rotation)
                immutable=j in self.fixed and (not part or j+1 in self.fixed)
                before_floor=np.array([float((robot.surface_models[body]['hull']@robot.rot(body)[2]+robot.point(body)[2]).min()) for body in self.bodies])
                floor_target=np.full(len(self.bodies),-.0008) if restore and not immutable else np.minimum(-.00099,before_floor-1e-7)
                target=None;before_distance=None
                if self.collision is not None:
                    self.collision.update(row,rotation);before_distance=self.collision.distances().copy()
                    target=np.minimum(.0001,before_distance-1e-7)
                    if restore and not immutable:target[~passive]=.0001
                self.samples.append((j,u,rotation,row,floor_target,target,before_floor,before_distance))
            robot.set(xs[a+j],t['root_r'][a+j]);feet=[];body_contacts=[]
            for side,g in robot.sides.items():
                if not moves(g['foot']) or t['sides'][side]['contact'][a+j]<=0:continue
                v=t['sides'][side];point=robot.point(g['foot'],v['pivots'][a+j]);low=(g['hull']@robot.rot(g['foot'])[2]+robot.point(g['foot'])[2]).min()
                error=np.r_[point[:2]-v['anchors'][a+j,:2],low]
                # Leave an interior corridor below the unchanged 1 mm gate.
                # Pinning a passing contact to its current norm can stall SQP
                # at a curved boundary during otherwise feasible smoothing.
                bound=.0008 if restore and j not in self.fixed else max(.00099,float(np.linalg.norm(error)))
                tolerance=np.full(3,bound)
                feet.append((side,tolerance,bound,max(.001,float(np.linalg.norm(error)))))
            for c in t.get('body_contacts',[]):
                if c['body'] not in self.bodies or not preserve_contact(c,a+j):continue
                body=c['body'];hull=np.asarray(c['robot_hull']);local=hull[np.argmin(hull@robot.rot(body)[2])]
                body_contacts.append((body,hull,local.copy(),robot.point(body,local).copy(),int(c['mode'][a+j])))
            self.native.append((feet,body_contacts))
        # Unfixed boundary poses also constrain their incoming/outgoing seams.
        for j,frame,neighbor in ((-1,a-1,a),(self.n-1,b-1,b)):
            if not (0<=frame<len(xs) and 0<=neighbor<len(xs)):continue
            delta=Rotation.from_matrix(t['root_r'][frame].T@t['root_r'][neighbor]).as_rotvec()
            for part in range(1,8):
                u=part/8.;rotation=t['root_r'][frame]@Rotation.from_rotvec(delta*u).as_matrix();row=xs[frame]*(1-u)+xs[neighbor]*u
                robot.set(row,rotation);bf=np.array([float((robot.surface_models[body]['hull']@robot.rot(body)[2]+robot.point(body)[2]).min()) for body in self.bodies]);ft=np.minimum(-.00099,bf-1e-7)
                bd=dt=None
                if self.collision is not None:self.collision.update(row,rotation);bd=self.collision.distances().copy();dt=np.minimum(.0001,bd-1e-7)
                self.samples.append((j,u,rotation,row,ft,dt,bf,bd))
        if self.collision is not None:
            self._normal=np.empty(3);self._jac1=np.empty((3,self.collision.m.nv));self._jac2=np.empty_like(self._jac1);self._fromto=np.empty(6)
        self.last={}

    def evaluate(self,z,linear=False):
        r=self.r;qs=z.reshape(self.n,self.width);rows=[];columns=[];values=[];lower=[];upper=[];penalty=0.;worst_floor={};worst_pair={}
        nonreg=0.;floor_reg=0.;contact_reg=0.;floor_min=float('inf');contact_max=0.;collision_min=float('inf');interior_floor=float('inf');interior_collision=float('inf')
        def add(J,lo,hi,j,u=0.):
            J=np.atleast_2d(J);lo=np.broadcast_to(np.atleast_1d(lo),(len(J),));hi=np.broadcast_to(np.atleast_1d(hi),(len(J),))
            for k,row in enumerate(J):
                index=len(lower);nz=np.flatnonzero(abs(row)>1e-14)
                if j>=0:
                    rows.extend([index]*len(nz));columns.extend((j*self.width+nz).tolist());values.extend((row[nz]*(1-u)).tolist())
                if u and j+1<self.n:
                    rows.extend([index]*len(nz));columns.extend(((j+1)*self.width+nz).tolist());values.extend((row[nz]*u).tolist())
                lower.append(float(lo[k]));upper.append(float(hi[k]))
        def pose(j,u,base):
            x=base.copy()
            if j<0:x[self.ids]=self.xs[self.a-1,self.ids]*(1-u)+qs[0]*u
            elif j+1>=self.n and u:x[self.ids]=qs[-1]*(1-u)+self.xs[self.b,self.ids]*u
            else:x[self.ids]=qs[j] if not u else qs[j]*(1-u)+qs[j+1]*u
            return x
        for index,(j,u,rotation,base,ft,dt,bf,bd) in enumerate(self.samples):
            x=pose(j,u,base);r.set(x,rotation)
            for k,body in enumerate(self.bodies):
                hull=r.surface_models[body]['hull'];heights=hull@r.rot(body)[2]+r.point(body)[2];v=int(np.argmin(heights));height=float(heights[v]);gap=height-ft[k]
                floor_min=min(floor_min,height)
                if 0<=j<self.n and not (j==self.n-1 and u):interior_floor=min(interior_floor,height)
                penalty+=max(-gap,0.)**2;floor_reg=max(floor_reg,min(bf[k],-.001)-height);nonreg=max(nonreg,floor_reg)
                key=(j,body)
                if linear and height<.02:
                    if not u:
                        for vertex in np.argsort(heights)[:4]:worst_floor[(j,body,int(vertex))]=(float(heights[vertex]-ft[k]),index,hull[vertex].copy())
                    elif key not in worst_floor or gap<worst_floor[key][0]:worst_floor[key]=(gap,index,hull[v].copy())
            if self.collision is not None:
                self.collision.update(x,rotation);distance=self.collision.distances();deficit=np.maximum(dt-distance,0.);penalty+=float(deficit@deficit)
                collision_min=min(collision_min,float(distance.min()))
                if 0<=j<self.n and not (j==self.n-1 and u):interior_collision=min(interior_collision,float(distance.min()))
                nonreg=max(nonreg,float((np.minimum(bd,0.)-distance).max()))
                if linear:
                    for k in np.flatnonzero(distance<.02):
                        key=(j,int(k),index);gap=distance[k]-dt[k]
                        if key not in worst_pair or gap<worst_pair[key][0]:worst_pair[key]=(float(gap),index,float(distance[k]))
            if u:continue
            feet,contacts=self.native[j]
            for side,tolerance,bound,original_bound in feet:
                g=r.sides[side];v=self.t['sides'][side];point=r.point(g['foot'],v['pivots'][self.a+j]);hull=g['hull'];low=hull[np.argmin(hull@r.rot(g['foot'])[2])]
                value=np.r_[point[:2]-v['anchors'][self.a+j,:2],r.point(g['foot'],low)[2]];norm=float(np.linalg.norm(value));contact_max=max(contact_max,norm)
                penalty+=max(norm-bound,0.)**2;contact_reg=max(contact_reg,norm-original_bound);nonreg=max(nonreg,contact_reg)
                if linear:
                    P,_=r.jac(g['foot'],v['pivots'][self.a+j]);G,_=r.jac(g['foot'],low);J=np.vstack([P[:2],G[2]])[:,self.ids];add(J,-tolerance-value,tolerance-value,j)
                    if norm>1e-10:add((value/norm)@J,-np.inf,bound-norm,j)
            for body,hull,local,point,mode in contacts:
                if mode!=1:local=hull[np.argmin(hull@r.rot(body)[2])]
                value=r.point(body,local)-point
                if mode!=1:value=value[2:]
                excess=np.maximum(abs(value)-.00049,0.);penalty+=float(excess@excess);nonreg=max(nonreg,float(abs(value).max())-.0005)
                if linear:
                    P,_=r.jac(body,local);P=P if mode==1 else P[2:];add(P[:,self.ids],-.00049-value,.00049-value,j)
            if self.palms is not None:
                C,h,excess=self.palms.evaluate(x,j);penalty+=excess**2
                nonreg=max(nonreg,excess-np.deg2rad(.001))
                if linear:add(C,-np.inf,h,j)
        if linear:
            self.elastic_rows=[]
            # Keep every near collision witness from the eight audit samples.
            # One representative per pair can trade penetration between
            # fractions, stalling or failing the independent dense guard.
            # Floor vertices keep their per-body relinearized witness policy.
            for key,(gap,index,local) in worst_floor.items():
                body=key[1];j,u,rotation,base,*_=self.samples[index];x=pose(j,u,base);r.set(x,rotation)
                P,_=r.jac(body,local);add(P[2,self.ids],-gap,np.inf,j,u)
            for (_,k,_),(gap,index,distance) in worst_pair.items():
                j,u,rotation,base,*_=self.samples[index];x=pose(j,u,base)
                model=self.collision;model.update(x,rotation);a,b=map(int,model.pairs[k]);value=mujoco.mj_geomDistance(model.m,model.configuration.data,a,b,model.cfg['detection_m'],self._fromto)
                derivative=compute_contact_normal_jacobian(model.m,model.configuration.data,a,b,self._fromto,self._normal,self._jac1,self._jac2)
                G=(-1. if value>=0. else 1.)*derivative[r.cols[self.ids]]
                if self.initial_leg_repair:
                    _,_,_,_,_,target,_,baseline=self.samples[index]
                    # An owned row has a positive restoration target. Only a
                    # pair already penetrating at this sample can take slack.
                    if target[k]>0. and baseline[k]<-model.cfg['penetration_tolerance_m']:
                        self.elastic_rows.append(len(lower))
                add(G,-np.inf,max(gap,-.008) if self.restore else gap,j,u)
            return sparse.csc_matrix((values,(rows,columns)),shape=(len(lower),len(z))),np.asarray(lower),np.asarray(upper)
        self.last=dict(nonregression_excess=max(nonreg,0.),minimum_floor_m=floor_min,maximum_contact_m=contact_max,minimum_collision_m=collision_min,
            interior_floor_m=interior_floor,interior_collision_m=interior_collision,
            safety_goal_met=(interior_floor>=-.00098 and interior_collision>=-1e-5 and contact_max<=.00098 and nonreg<=2e-6))
        if self.initial_leg_repair:
            self.last.update(floor_regression_m=floor_reg,contact_regression_m=contact_reg)
        return 1e8*penalty
def project_root_height(robot, targets, rows, start, end):
    """Solve the vertical translation subproblem without changing any hinge.

    A shared upward translation cannot introduce self-collision, change limb
    directions, or add joint jitter. Each planted foot retains the existing
    1 mm envelope (or its already-failing measured error). Other support points
    keep the original 0.5 mm preservation corridor. Common adoption still checks
    every dense seam and all final gates.
    """
    from scipy import sparse
    from osqp import OSQP
    n=end-start; upper=np.zeros(n); goal=np.zeros(n)
    for j,i in enumerate(range(start,end)):
        robot.set(rows[i],targets['root_r'][i])
        floor=min(float((shape['hull']@robot.rot(body)[2]+robot.point(body)[2]).min())
                  for body,shape in robot.surface_models.items())
        goal[j]=max(0.,.0001-floor)
        cap=goal[j]
        for side,g in robot.sides.items():
            values=targets['sides'][side]
            if values['contact'][i]<=0:continue
            point=robot.point(g['foot'],values['pivots'][i]);xy=point[:2]-values['anchors'][i,:2]
            height=float((g['hull']@robot.rot(g['foot'])[2]+robot.point(g['foot'])[2]).min())
            original=np.sqrt(float(xy@xy)+height*height)
            radius=np.sqrt(max(max(.00099,original)**2-float(xy@xy),0.))
            cap=min(cap,max(0.,radius-height))
        from .support_quality import preserve_contact
        if any(preserve_contact(contact,i) for contact in targets.get('body_contacts',[])):
            cap=min(cap,.0005)
        upper[j]=cap
    if start:upper[:2]=0.
    if end<len(rows):upper[-2:]=0.
    if upper.max()<2e-6:return None
    I=sparse.eye(n,format='csc')
    D=sparse.diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n),format='csc')
    problem=OSQP();problem.setup(P=sparse.triu(I+2.*D.T@D,format='csc'),q=-np.minimum(goal,upper),
        A=I,l=np.zeros(n),u=upper,verbose=False,eps_abs=1e-9,eps_rel=1e-9,max_iter=10000,polishing=True)
    result=problem.solve()
    if result.info.status_val not in (1,2):return None
    delta=np.clip(result.x,0.,upper)
    if delta.max()<2e-6:return None
    candidate=rows.copy();candidate[start:end,2]+=delta
    return candidate,dict(vertical_support_subproblem=True,maximum_root_lift_m=float(delta.max()),
                           changed_frames=int(np.count_nonzero(delta>1e-6)))

