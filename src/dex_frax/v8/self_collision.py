"""CPU arm self-collision scan and bounded local trajectory recovery.

Mink supplies signed-distance displacement Jacobians. Already penetrating input
needs an additional recovery bound; Mink's normal approach limit alone cannot
remove it. Only checked windows are committed, and unresolved motion is graded B.
This covers the declared arm/torso/hip/head/opposite-arm pairs, not all contacts.
"""
import copy
import time
import numpy as np
import mujoco
from scipy import sparse as sp
from scipy.spatial.transform import Rotation
from .arms import ArmObjective

POLICY = 'mink_arm_local_recovery_v2'
DEFAULTS = dict(enabled=True, margin_m=.0035, detection_m=.08,
                penetration_tolerance_m=1e-5, padding_s=.32,
                max_window_s=3., max_iterations=24, interpolation_steps=4)


class CollisionModel:
    def __init__(self, robot, config=None):
        import mink
        self.r = robot
        self.cfg = dict(DEFAULTS, **(config or {}))
        if not (0 <= self.cfg['penetration_tolerance_m'] < self.cfg['margin_m'] < self.cfg['detection_m']):
            raise ValueError('Invalid self-collision distances')
        # Never mutate the shared read-only model used by batch workers.
        self.m = m = copy.copy(robot.m)
        ids = np.flatnonzero(m.geom_group == 1)
        self.arms = {}
        for side, suffix in [('Left', 'l'), ('Right', 'r')]:
            names = [f'{p}_{suffix}_link_collision' for p in
                     ('shoulder_yaw', 'elbow_pitch', 'elbow_yaw', 'wrist_pitch', 'wrist_roll')]
            names += [side.lower() + '_tcp_link_collision']
            available = {m.geom(i).name: int(i) for i in ids}
            if not set(names) <= available.keys():
                raise ValueError('Dex arm collision geometry missing: ' + str(set(names) - available.keys()))
            self.arms[side] = [available[name] for name in names]
        body = [int(i) for i in ids if i not in sum(self.arms.values(), [])
                and not any(n in m.geom(i).name for n in ('shoulder', 'ankle', 'knee'))]
        if not body:
            raise ValueError('Dex torso collision geometry missing')
        groups = [(self.arms[s], body) for s in self.arms]
        groups += [(self.arms['Left'], self.arms['Right'])]
        m.geom_contype[ids] = 1
        m.geom_conaffinity[ids] = 1
        self.limit = mink.CollisionAvoidanceLimit(m, groups,
            minimum_distance_from_collisions=self.cfg['margin_m'],
            collision_detection_distance=self.cfg['detection_m'])
        self.pairs = np.array(self.limit.geom_id_pairs, dtype=int)
        if not len(self.pairs):
            raise ValueError('Empty self-collision pair set')
        self.names = [[m.geom(a).name, m.geom(b).name] for a, b in self.pairs]
        self.configuration = mink.Configuration(m)

    def update(self, x, rotation):
        self.r.set(x, rotation)
        self.configuration.update(self.r.d.qpos)

    def distances(self):
        a, b = self.pairs.T
        m, data = self.m, self.configuration.data
        limit = self.cfg['detection_m']
        # Conservative bounding-sphere test; unbounded geoms always survive.
        bounded = (m.geom_rbound[a] > 0) & (m.geom_rbound[b] > 0)
        near = (~bounded) | (np.linalg.norm(data.geom_xpos[a]-data.geom_xpos[b], axis=1)
                            - m.geom_rbound[a]-m.geom_rbound[b] < limit)
        d = np.full(len(a), limit)
        for k in np.flatnonzero(near):
            d[k] = mujoco.mj_geomDistance(m, data, int(a[k]), int(b[k]), limit, None)
        return d

    def linear(self, columns):
        con = self.limit.compute_qp_inequalities(self.configuration, 1.)
        active = np.isfinite(con.h)
        d = self.distances()[active]
        # Retain Mink's approach damping outside the margin; demand recovery
        # inside it. G is displacement-based, so multiplying it by fps is wrong.
        upper = np.minimum(con.h[active], d-self.cfg['margin_m'])
        return con.G[active][:, self.r.cols[columns]], upper


def scan(model, xs, rotations, subdivisions=None):
    """Full exported trajectory, plus fixed subframe FK samples. Not a CCD proof."""
    xs = np.asarray(xs)
    if len(xs) != len(rotations) or not np.isfinite(xs).all() or not np.isfinite(rotations).all():
        raise ValueError('Invalid collision trajectory')
    steps = model.cfg['interpolation_steps'] if subdivisions is None else subdivisions
    if not isinstance(steps, int) or steps < 1:
        raise ValueError('interpolation_steps must be a positive integer')
    tol = model.cfg['penetration_tolerance_m']
    minimum = model.cfg['detection_m']; worst = None; events = []
    native = []; bad_intervals = set(); impacted = set()
    delta = Rotation.from_matrix(rotations[:-1].transpose(0,2,1) @ rotations[1:]).as_rotvec() if len(xs)>1 else []
    for i, x in enumerate(xs):
        for k in range(steps if i+1 < len(xs) else 1):
            alpha = k/steps
            rot = rotations[i] if k == 0 else rotations[i] @ Rotation.from_rotvec(delta[i]*alpha).as_matrix()
            model.update(x if k == 0 else x*(1-alpha)+xs[i+1]*alpha, rot)
            d = model.distances(); j = int(np.argmin(d)); value = float(d[j])
            if k == 0: native.append(value)
            if value < minimum:
                minimum = value; worst = dict(frame=i, fraction=alpha, pair=model.names[j], distance_m=value)
            bad = np.flatnonzero(d < -tol)
            if len(bad):
                impacted.add(i)
                if k: impacted.add(i+1); bad_intervals.add(i)
                events.append(dict(frame=i, fraction=alpha, pairs=bad.tolist(), minimum_m=value))
    report = dict(policy=POLICY, pair_count=len(model.pairs), frames=len(xs),
                  native_colliding_frames=int(np.sum(np.asarray(native)<-tol)),
                  interpolated_colliding_intervals=len(bad_intervals),
                  collision_samples=len(events), minimum_distance_m=minimum,
                  worst=worst, interpolation_steps=steps,
                  scope='arms versus torso/hips/head and opposite arm; convex geometry; not continuous collision certification')
    return report, events, impacted


def windows(impacted, n, padding):
    result = []
    for i in sorted(impacted):
        a, b = max(0,i-padding), min(n,i+padding+1)
        if result and a <= result[-1][1]: result[-1][1] = max(b,result[-1][1])
        else: result.append([a,b])
    return result


def _arm_bodies(robot, sides):
    roots = {int(robot.m.jnt_bodyid[np.flatnonzero(robot.m.jnt_dofadr == robot.dadr[robot.sides[s]['arm'][0]-3])[0]]) for s in sides}
    def belongs(body):
        while body:
            if body in roots: return True
            body = int(robot.m.body_parentid[body])
        return False
    return [b for b in robot.surface_models if belongs(b)]


def _repair_window(model, t, xs, solver_cfg, a, b, sides, options=None):
    import osqp
    r = model.r; cfg = model.cfg; n = b-a; fps = float(t['fps'])
    coupled=bool(options and options.get('coupled'))
    if coupled:sides=list(r.sides)
    repair_started=time.perf_counter()
    ids = np.arange(32) if coupled else np.array(sum([r.sides[s]['arm'] for s in sides], [])); width = len(ids)
    arm_ids=np.array(sum([r.sides[s]['arm'] for s in sides], []))
    bodies = list(r.surface_models) if coupled else _arm_bodies(r, sides)
    floor_tolerance=.001 if coupled else 1e-5
    # Preserve load-bearing/rolling/sliding arm support. Such scenes need a
    # coupled contact solver; this local arm-only stage must not release them.
    for contact in t.get('body_contacts', []):
        if not options and contact['body'] in bodies and np.any(np.asarray(contact['mode'])[a:b] > 0):
            raise ValueError('active_arm_support_requires_coupled_repair')
    reference=options.get('_reference',xs) if options else xs
    q0 = reference[a:b,ids].copy();q=xs[a:b,ids].copy();join_q=xs[a:b,ids].copy()
    fixed_frames=set(([0,1] if a>0 and (options or {}).get('freeze_start',True) else [])+([n-2,n-1] if b<len(xs) and (options or {}).get('freeze_end',True) else []))
    frozen=np.zeros((n,width),bool)
    if fixed_frames:frozen[list(fixed_frames)]=True
    if options and not coupled:
        for side in sides:
            arm_bodies=set(_arm_bodies(r,[side]));columns=np.isin(ids,r.sides[side]['arm'])
            for contact in t.get('body_contacts',[]):
                if contact['body'] in arm_bodies:
                    active=np.flatnonzero(np.asarray(contact['mode'])[a:b]>0)
                    frozen[np.ix_(active,np.flatnonzero(columns))]=True
    margin_targets={}
    if options:
        for j in fixed_frames:
            model.update(xs[a+j],t['root_r'][a+j]);margin_targets[(j,0.)]=np.maximum(0.,np.minimum(cfg['margin_m'],model.distances()-1e-8))
        for j in range(n):
            if j in fixed_frames or not frozen[j].any():continue
            model.update(xs[a+j],t['root_r'][a+j]);con=model.limit.compute_qp_inequalities(model.configuration,1.)
            unaffected=np.sum(abs(con.G[:,r.cols[ids[~frozen[j]]]]),axis=1)<1e-12
            target=np.full(len(model.pairs),cfg['margin_m']);d=model.distances()
            target[unaffected]=np.maximum(0.,np.minimum(cfg['margin_m'],d[unaffected]-1e-8));margin_targets[(j,0.)]=target
    lo, hi = r.lower[ids]+1e-6, r.upper[ids]-1e-6
    step = np.r_[np.full(3,np.inf),r.joint_step(fps,solver_cfg)][ids]
    weights=dict(solver_cfg)
    # Tier relaxation changes search bounds, never the source-pose weights.
    if coupled:
        from .solver import Objective
        objs=[Objective(r,t,i,weights,xs[i-1] if i else None,xs[i-1] if i else None) for i in range(a,b)]
    else:objs = [[ArmObjective(r,t,i,s,xs[i],weights) for s in sides] for i in range(a,b)]
    baselines=[]
    if options:
        for i in range(a,b):
            r.set(reference[i],t['root_r'][i]);hands={s:(r.point(r.sides[s]['tcp']),r.rot(r.sides[s]['tcp']).copy()) for s in sides}
            contacts=[]
            if coupled:
                for contact in t.get('body_contacts',[]):
                    mode=int(contact['mode'][i])
                    if not mode:continue
                    body=contact['body'];hull=np.asarray(contact.get('robot_hull',r.surface_models[body]['hull']))
                    local=hull[np.argmin(hull@r.rot(body)[2])].copy()
                    contacts.append((body,hull,local,r.point(body,local),mode))
            baselines.append((hands,contacts))
    I = sp.eye(n*width, format='csc')
    D1 = sp.kron(sp.diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n)),sp.eye(width),format='csc')
    D2 = sp.kron(sp.diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n)),sp.eye(width),format='csc')
    prior=np.full(width,.12)
    if options:
        shoulder_ids=np.array(sum([r.sides[s]['arm'][:3] for s in sides],[]))
        prior[np.isin(ids,shoulder_ids)]*=.4
    # Prefer J1-J3 changes without freezing the distal axes or weakening their
    # palm/forearm/elbow goals. All axes remain available when jointly needed.
    smooth = sp.vstack([D1*.18*(fps/60),D2*6*(fps/60)**2,sp.diags(np.tile(prior,n))],format='csc')
    def pose(q, j):
        x=xs[a+j].copy();x[ids]=q[j];return x
    def objective(q):
        residual=[];blocks=[]
        if coupled:
            for obj,row in zip(objs,q):
                res,J=obj.evaluate(row);residual.append(res);blocks.append(J)
            return np.concatenate(residual),sp.block_diag(blocks,format='csc')
        for frame, row in zip(objs,q):
            for k,o in enumerate(frame):
                res,J=o.evaluate(row[k*7:(k+1)*7]);residual.append(res);blocks.append(J)
        return np.concatenate(residual), sp.block_diag(blocks,format='csc')
    def floor_geometry(x, i, jacobian=False,rotation=None,activation=.10,count=4):
        r.set(x,t['root_r'][i] if rotation is None else rotation);values=[];rows=[]
        for body in bodies:
            hull=r.surface_models[body]['hull'];heights=hull@r.rot(body)[2]+r.point(body)[2]
            values.append(float(heights.min()))
            if jacobian and heights.min()<activation:
                for local in hull[np.argsort(heights)[:count]]:
                    J,_=r.jac(body,local);rows.append((-J[2,ids],float(r.point(body,local)[2])))
        return min(values,default=1.),rows
    rotation_deltas=Rotation.from_matrix(t['root_r'][a:b-1].transpose(0,2,1)@t['root_r'][a+1:b]).as_rotvec() if options else []
    def target_margin(j,alpha=0.):
        key=(j,alpha)
        if options and alpha and np.any(frozen[j]&frozen[j+1]) and key not in margin_targets:
            rotation=t['root_r'][a+j]@Rotation.from_rotvec(rotation_deltas[j]*alpha).as_matrix()
            model.update(xs[a+j]*(1-alpha)+xs[a+j+1]*alpha,rotation)
            con=model.limit.compute_qp_inequalities(model.configuration,1.)
            free=~(frozen[j]&frozen[j+1]);unaffected=np.sum(abs(con.G[:,r.cols[ids[free]]]),axis=1)<1e-12
            target=np.full(len(model.pairs),cfg['margin_m']);d=model.distances()
            target[unaffected]=np.maximum(0.,np.minimum(cfg['margin_m'],d[unaffected]-1e-8));margin_targets[key]=target
        return margin_targets.get(key,cfg['margin_m'])
    def clearance_rows(j,alpha=0.):
        if not options:return model.linear(ids)
        # target_margin may query the baseline; restore the current point after it.
        target=target_margin(j,alpha)
        if alpha:
            rotation=t['root_r'][a+j]@Rotation.from_rotvec(rotation_deltas[j]*alpha).as_matrix()
            model.update(pose(q,j)*(1-alpha)+pose(q,j+1)*alpha,rotation)
        else:model.update(pose(q,j),t['root_r'][a+j])
        con=model.limit.compute_qp_inequalities(model.configuration,1.);active=np.isfinite(con.h)
        distances=model.distances();required=np.broadcast_to(target,distances.shape)
        return con.G[active][:,r.cols[ids]],np.minimum(con.h[active],np.maximum(distances[active]-required[active],-.008))
    def intermediate(q):
        if not options:return
        for j in range(n-1):
            for part in range(1,8):
                alpha=part/8;rotation=t['root_r'][a+j]@Rotation.from_rotvec(rotation_deltas[j]*alpha).as_matrix()
                at=pose(q,j)*(1-alpha)+pose(q,j+1)*alpha
                yield j,alpha,at,rotation
    def extra_constraints(x,j):
        """Bound gesture change and preserve existing support while clearing it."""
        if not options:return np.zeros((0,width)),np.zeros(0),0.,0.
        from .math3d import skew
        r.set(x,t['root_r'][a+j]);C=[];upper=[];pose_excess=0.;support_excess=0.
        def box(value,J,tolerance):
            C.extend([*J,*(-J)]);upper.extend(np.r_[tolerance-value,tolerance+value])
        hands,contacts=baselines[j]
        for side,(point0,rot0) in hands.items():
            body=r.sides[side]['tcp'];P,A=r.jac(body);delta=r.point(body)-point0
            box(delta,P[:,ids],options['hand_position_m']/np.sqrt(3))
            angle=Rotation.from_matrix(r.rot(body)@rot0.T).magnitude()
            pose_excess=max(pose_excess,np.linalg.norm(delta)-options['hand_position_m'],angle-np.deg2rad(options['hand_rotation_deg']))
            for axis in range(3):
                direction=r.rot(body)[:,axis];dot=float(rot0[:,axis]@direction)
                derivative=rot0[:,axis]@(-skew(direction)@A[:,ids])
                C.append(-derivative);upper.append(dot-np.cos(np.deg2rad(options['hand_rotation_deg'])))
        tolerance=.0005
        if coupled:
            for side,g in r.sides.items():
                if t['sides'][side]['contact'][a+j]<=0:continue
                v=t['sides'][side];local=v['pivots'][a+j];P,_=r.jac(g['foot'],local)
                error=r.point(g['foot'],local)[:2]-v['anchors'][a+j,:2]
                hull=g['hull'];low=hull[np.argmin(hull@r.rot(g['foot'])[2])];G,_=r.jac(g['foot'],low)
                value=np.r_[error,r.point(g['foot'],low)[2]];J=np.vstack([P[:2],G[2]])[:,ids]
                box(value,J,tolerance);support_excess=max(support_excess,float(np.max(abs(value)))-tolerance)
            for body,hull,local,point0,mode in contacts:
                if mode==1:
                    P,_=r.jac(body,local);value=r.point(body,local)-point0;box(value,P[:,ids],tolerance)
                else:
                    local=hull[np.argmin(hull@r.rot(body)[2])];P,_=r.jac(body,local)
                    value=np.array([r.point(body,local)[2]-point0[2]]);box(value,P[2:3,ids],tolerance)
                support_excess=max(support_excess,float(np.max(abs(value)))-tolerance)
        return np.asarray(C).reshape(-1,width),np.asarray(upper),max(0.,pose_excess),max(0.,support_excess)
    def metrics(q):
        distances=[];ground=[];pose_error=[];support_error=[];deficits=[]
        for j in range(n):
            x=pose(q,j);model.update(x,t['root_r'][a+j]);d=model.distances();distances.append(d);deficits.append(np.maximum(target_margin(j)-d,0))
            ground.append(floor_geometry(x,a+j)[0])
            _,_,p,s=extra_constraints(x,j);pose_error.append(p);support_error.append(s)
        for j,alpha,at,rotation in intermediate(q):
            model.update(at,rotation);d=model.distances();distances.append(d);deficits.append(np.maximum(target_margin(j,alpha)-d,0))
            ground.append(floor_geometry(at,a+j,rotation=rotation)[0])
        res,_=objective(q);d=np.asarray(distances);z=np.asarray(ground)
        cost=float(res@res+np.linalg.norm(smooth@(q-q0).ravel())**2)
        violation=float(np.asarray(deficits).sum())
        merit=cost+1e5*violation+1e6*float(np.maximum(-z,0).sum())+1e7*sum(support_error)+1e4*sum(pose_error)
        return merit,float(d.min()),float(z.min()),max(pose_error,default=0.),max(support_error,default=0.),float(np.asarray(deficits).max())
    history=[]
    # Caller-owned scratch retains finite iterates only for the next tier.
    # It is never an accepted output and is checked again from scratch.
    scratch=(options or {}).get('_scratch')
    for iteration in range(options.get('iterations',cfg['max_iterations']) if options else cfg['max_iterations']):
        # Deterministic work budget: OS scheduling cannot consume pose quality.
        if options and options.get('watchdog_seconds') and time.perf_counter()-repair_started>options['watchdog_seconds']:
            raise ValueError('collision_repair_interrupted_watchdog')
        residual,J=objective(q);H=J.T@J+smooth.T@smooth+I*.01
        gradient=J.T@residual+smooth.T@(smooth@(q-q0).ravel())
        blocks=[];bounds=[]
        for j in range(n):
            x=pose(q,j);model.update(x,t['root_r'][a+j]);C,upper=clearance_rows(j)
            _,floor_rows=floor_geometry(x,a+j,True)
            if floor_rows:
                C=np.vstack([C,np.array([v[0] for v in floor_rows])]);upper=np.r_[upper,[v[1]+(floor_tolerance if options else 0.) for v in floor_rows]]
            E,limits,_,_=extra_constraints(x,j)
            if len(limits):C=np.vstack([C,E]);upper=np.r_[upper,limits]
            blocks.append(sp.csc_matrix(C));bounds.append(upper)
        C=sp.block_diag(blocks,format='csc');upper=np.concatenate(bounds)
        interpolation_blocks=[];interpolation_bounds=[]
        for j,alpha,at,rotation in intermediate(q):
            model.update(at,rotation);d=model.distances();G,h=clearance_rows(j,alpha)
            # Mink orders active rows by the finite limit bounds.
            con=model.limit.compute_qp_inequalities(model.configuration,1.);near=d[np.isfinite(con.h)]<cfg['margin_m']+.001
            G,h=G[near],h[near]
            _,floor_rows=floor_geometry(at,a+j,True,rotation,.002,1)
            if floor_rows:
                G=np.vstack([G,np.array([v[0] for v in floor_rows])]);h=np.r_[h,[v[1]+floor_tolerance for v in floor_rows]]
            if len(h):
                piece=sp.hstack([sp.csc_matrix((len(h),j*width)),sp.csc_matrix(G*(1-alpha)),sp.csc_matrix(G*alpha),sp.csc_matrix((len(h),(n-j-2)*width))],format='csc')
                interpolation_blocks.append(piece);interpolation_bounds.append(h)
        if interpolation_blocks:
            C=sp.vstack([C,*interpolation_blocks],format='csc');upper=np.r_[upper,np.concatenate(interpolation_bounds)]
        lb=np.maximum(np.tile(lo,n)-q.ravel(),-.18);ub=np.minimum(np.tile(hi,n)-q.ravel(),.18)
        if options:
            cap=np.full(width,np.deg2rad(options.get('body_joint_deg',5.)))
            cap[np.isin(ids,arm_ids)]=np.deg2rad(options['joint_deg']);cap[ids<3]=options.get('root_position_m',.02)
            lb=np.maximum(lb,(q0-q-cap).ravel());ub=np.minimum(ub,(q0-q+cap).ravel())
            if coupled:
                for side,g in r.sides.items():
                    k=int(g['leg'][3]);positions=np.arange(n)*width+k
                    lb[positions]=np.maximum(lb[positions],np.where(q0[:,k]>=0.,-q[:,k],-np.inf))
        # Moving internal overlap endpoints still obey their outside motor speeds.
        if options:
            for j,neighbor in ((0,a-1),(n-1,b)):
                if 0<=neighbor<len(xs):
                    k=np.arange(width)+j*width
                    lb[k]=np.maximum(lb[k],xs[neighbor,ids]-step-q[j])
                    ub[k]=np.minimum(ub[k],xs[neighbor,ids]+step-q[j])
        # Preserve both endpoint pose and endpoint velocity when joining an
        # unchanged prefix/suffix. At actual sequence boundaries allow repair.
        positions=np.flatnonzero(frozen.ravel());values=(join_q-q).ravel()[positions]
        lb[positions]=ub[positions]=values
        # Root translations stay bounded but have no invented motor speed rating.
        speed_rows=np.tile(ids>=3,n-1)
        Dv=D1[speed_rows];velocity_step=np.tile(step,n-1)[speed_rows]
        A=sp.vstack([I,Dv,C],format='csc')
        lower=np.r_[lb,-velocity_step-Dv@q.ravel(),np.full(len(upper),-np.inf)]
        upper=np.r_[ub,velocity_step-Dv@q.ravel(),upper]
        scale=1/np.maximum(np.sqrt(A.multiply(A).sum(axis=1)).A.ravel(),1e-9)
        A=(sp.diags(scale)@A).tocsc();lower*=scale;upper*=scale
        diagnostics=None
        if options:
            fixed_cols=np.flatnonzero(abs(lb-ub)<1e-12);free_cols=np.flatnonzero(abs(lb-ub)>=1e-12)
            constant=np.asarray(A[:,fixed_cols]@lb[fixed_cols]).ravel()
            movable=np.asarray(abs(A[:,free_cols]).sum(axis=1)).ravel()
            bad=np.flatnonzero((movable<1e-10)&((constant<lower-1e-7)|(constant>upper+1e-7)))
            diagnostics=dict(iteration=iteration,fixed_variable_conflicts=[dict(row=int(k),lower=float(lower[k]),upper=float(upper[k]),constant=float(constant[k])) for k in bad[:12]],conflict_count=len(bad),rows=A.shape[0],variables=A.shape[1])
            if len(bad):
                error=ValueError('collision_fixed_support_conflict');error.diagnostics=diagnostics;raise error
        extra={}
        objective_scale=max(float(abs(H.diagonal()).max()),1.)
        opt=osqp.OSQP();opt.setup(P=sp.triu(H/objective_scale,format='csc'),q=gradient/objective_scale,A=A,l=lower,u=upper,
            eps_abs=1e-7,eps_rel=1e-7,max_iter=(options or {}).get('qp_iterations',20000),adaptive_rho_interval=50,verbose=False,polishing=True,**extra)
        sol=opt.solve()
        if sol.info.status_val not in (1,2) or not np.isfinite(sol.x).all():
            error=ValueError('collision_qp_'+sol.info.status)
            if options:error.diagnostics=diagnostics
            raise error
        at=A@sol.x
        if max(float(np.max(lower-at)),float(np.max(at-upper)))>1e-6:
            raise ValueError('collision_qp_constraint_residual')
        delta=sol.x.reshape(n,width);old=metrics(q);accepted=False
        for alpha in (1.,.5,.25,.125,.0625):
            trial=q+alpha*delta;new=metrics(trial)
            if new[0]<old[0]+1e-9 and (not options or (new[3]<2e-6 and new[4]<2e-6)):
                q=trial;accepted=True;break
        if accepted and scratch is not None:
            seed=xs[a:b].copy();seed[:,ids]=q;scratch['candidate_window']=seed
        history.append(dict(iteration=iteration,accepted=accepted,minimum_m=new[1]))
        if not accepted:break
        if new[5]<=2e-6 and new[2]>=-floor_tolerance and abs(delta).max()<2e-4:break
    candidate=xs.copy();candidate[a:b,ids]=q
    # Validate the exact float32 joint representation later exported to PKL.
    candidate[a:b,ids]=candidate[a:b,ids].astype(np.float32).astype(float)
    q=candidate[a:b,ids];merit,dmin,zmin,*_=metrics(q)
    if (dmin<(-cfg['penetration_tolerance_m'] if options else .003-2e-6)) or zmin < -floor_tolerance:
        error=ValueError('nonlinear_collision_or_floor_clearance')
        if options:error.diagnostics=dict(minimum_distance_m=dmin,minimum_floor_m=zmin,iterations=len(history),last_iteration=history[-1] if history else None,hand_pose_excess=metrics(q)[3],support_excess=metrics(q)[4])
        raise error
    if options:
        result=metrics(q)
        if result[5]>2e-6:raise ValueError('nonlinear_collision_clearance_margin')
        if result[3]>2e-6:raise ValueError('repair_hand_pose_budget')
        if result[4]>2e-6:raise ValueError('repair_support_contact_changed')
    if np.any(q<lo-2e-6) or np.any(q>hi+2e-6):raise ValueError('arm_angle_limit')
    if np.any(abs(np.diff(candidate[max(0,a-1):min(len(xs),b+1),ids],axis=0))[:,ids>=3]>step[ids>=3]+2e-6):
        raise ValueError('arm_velocity_or_boundary_limit')
    checked,_,_=scan(model,candidate[max(0,a-1):min(len(xs),b+1)],t['root_r'][max(0,a-1):min(len(xs),b+1)])
    if checked['collision_samples'] and not (options or {}).get('internal_overlap',False):
        error=ValueError('interpolated_collision');error.diagnostics=checked;raise error
    return candidate,dict(start=a,end=b,sides=sides,status='repaired',iterations=len(history),
                          minimum_m=dmin,interpolated_minimum_m=checked['minimum_distance_m'],
                          maximum_correction_deg=float(np.rad2deg(abs(q-q0)[:,ids>=3]).max()),coupled=bool(coupled))


def _window_sides(model, xs, rotations, a, b, collision_pair_ids):
    """Select every arm needed by the window's clearance constraints.

    A nonpenetrating opposite arm can still violate the recovery margin.
    Freezing it while imposing its distance inequality creates an impossible
    zero-Jacobian row. Include its joints rather than dropping the constraint
    or reducing the margin. The repair's support guard still applies to every
    selected arm. Only already-impacted windows reach this function.
    """
    def sides_for(pair_ids):
        geoms = {geom for k in pair_ids for geom in model.pairs[k]}
        return [side for side, group in model.arms.items() if geoms.intersection(group)]

    collision_pair_ids = set(collision_pair_ids)
    collision_sides = sides_for(collision_pair_ids)
    margin_pair_ids = set()
    for i in range(a, b):
        model.update(xs[i], rotations[i])
        margin_pair_ids.update(np.flatnonzero(model.distances() < model.cfg['margin_m']).tolist())
    sides = sides_for(collision_pair_ids | margin_pair_ids)
    return sides, dict(policy='collision_and_window_clearance_v2',
                       collision_sides=collision_sides,
                       margin_added_sides=[s for s in sides if s not in collision_sides],
                       margin_pairs=[model.names[k] for k in sorted(margin_pair_ids)])


def refine(robot, t, xs, solver_cfg, config=None, progress=None):
    cfg=dict(DEFAULTS,**(config or {}));start=time.perf_counter()
    if not cfg['enabled']:return xs,dict(policy=POLICY,status='disabled')
    model=CollisionModel(robot,cfg);before,events,impacted=scan(model,xs,t['root_r'])
    out=xs.copy();reports=[];fps=float(t['fps'])
    for a,b in windows(impacted,len(xs),max(2,round(cfg['padding_s']*fps))):
        if progress:progress(a,len(xs))
        relevant=[e for e in events if a<=e['frame']<b]
        pair_ids={k for e in relevant for k in e['pairs']}
        sides,selection=_window_sides(model,out,t['root_r'],a,b,pair_ids)
        try:
            if b-a>max(3,round(cfg['max_window_s']*fps)):
                raise ValueError('window_exceeds_local_repair_budget')
            if b-a<3:raise ValueError('window_too_short')
            candidate,report=_repair_window(model,t,out,solver_cfg,a,b,sides)
            out=candidate
        except ValueError as error:
            report=dict(start=a,end=b,sides=sides,status='unresolved',reason=str(error))
        report['side_selection']=selection
        reports.append(report)
    after,_,_=scan(model,out,t['root_r']) if impacted else (before,None,None)
    return out,dict(policy=POLICY,status='completed',before=before,after=after,windows=reports,
                    config=cfg,elapsed_s=time.perf_counter()-start)


def audit_export(robot, payload, config=None):
    xs=np.c_[payload['root_pos'],payload['dof_pos']].astype(float)
    rotations=Rotation.from_quat(payload['root_rot']).as_matrix()
    return scan(CollisionModel(robot,config),xs,rotations)[0]
