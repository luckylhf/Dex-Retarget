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
        from ..ankle_collision import AnkleCollisionAvoidanceLimit
        self.r = robot
        self.cfg = dict(DEFAULTS, **(config or {}))
        if not (0 <= self.cfg['penetration_tolerance_m'] < self.cfg['margin_m'] < self.cfg['detection_m']):
            raise ValueError('Invalid self-collision distances')
        # Never mutate the shared read-only model used by batch workers.
        self.foot_coverage=None
        if self.cfg.get('coverage')=='full':
            from ..foot_collision import augmented
            self.m,self.foot_coverage=augmented(robot)
        else:self.m=copy.copy(robot.m)
        m=self.m
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
        self.scope='declared_upper_body'
        self.exclusions=[]
        if self.cfg.get('coverage')=='full':
            self.scope='nonadjacent_full_body'
            # The roll housing is inside the pelvis/waist gimbal assembly.
            # Same/welded and direct parent bodies are excluded by Mink.
            assembly={frozenset(('pelvis_collision','waist_roll_link_collision'))}
            groups=[]
            for k,a in enumerate(ids):
                for b in ids[k+1:]:
                    names=[m.geom(int(a)).name,m.geom(int(b)).name]
                    if frozenset(names) in assembly:
                        self.exclusions.append(dict(pair=names,reason='pelvis/waist nested gimbal housing'))
                    else:groups.append(([int(a)],[int(b)]))

        if self.foot_coverage is not None:
            from ..foot_collision import groups as foot_groups
            extra,foot_ids=foot_groups(m,self.foot_coverage,ids);groups+=extra
            ids=np.unique(np.r_[ids,foot_ids])
            self.scope='full_body_with_full_feet_and_conditional_same_leg_foot_shin'
        m.geom_contype[ids] = 1
        m.geom_conaffinity[ids] = 1
        self.limit = AnkleCollisionAvoidanceLimit(m, groups,
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
        # Declared exclusions are classification rules, not positive measured
        # clearances. Detection-distance sentinels keep pair indexing stable.
        near &= ~self.limit.excluded_pairs(self.configuration)
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


def scan(model, xs, rotations, subdivisions=None, cache=None, changed=None):
    """Full exported trajectory, plus fixed subframe FK samples. Not a CCD proof."""
    xs = np.asarray(xs)
    if len(xs) != len(rotations) or not np.isfinite(xs).all() or not np.isfinite(rotations).all():
        raise ValueError('Invalid collision trajectory')
    steps = model.cfg['interpolation_steps'] if subdivisions is None else subdivisions
    if not isinstance(steps, int) or steps < 1:
        raise ValueError('interpolation_steps must be a positive integer')
    tol = model.cfg['penetration_tolerance_m']
    cache={} if cache is None else cache
    conditional_exclusions=model.limit.ankle_rule.describe()
    key=(steps,tuple(map(tuple,model.pairs.tolist())),tuple(sorted((k,str(v)) for k,v in model.cfg.items())),str(conditional_exclusions))
    valid=False
    if changed is not None and cache.get('key')==key and cache.get('xs',np.empty(0)).shape==xs.shape:
        a,b=changed
        valid=(0<=a<b<=len(xs) and np.array_equal(cache['xs'][:a],xs[:a]) and np.array_equal(cache['xs'][b:],xs[b:])
            and np.array_equal(cache['rr'][:a],rotations[:a]) and np.array_equal(cache['rr'][b:],rotations[b:]))
    from ..audit_cache import dense_range
    if valid:
        lo,hi,_,_=dense_range(changed,len(xs),steps)
        samples=list(cache['samples'])
    else:
        lo,hi=0,len(xs);samples=[None]*((len(xs)-1)*steps+1)
    delta = Rotation.from_matrix(rotations[:-1].transpose(0,2,1) @ rotations[1:]).as_rotvec() if len(xs)>1 else []
    for i in range(lo,hi):
        for k in range(steps if i+1 < len(xs) else 1):
            alpha=k/steps
            rot=rotations[i] if k==0 else rotations[i]@Rotation.from_rotvec(delta[i]*alpha).as_matrix()
            model.update(xs[i] if k==0 else xs[i]*(1-alpha)+xs[i+1]*alpha,rot)
            d=model.distances();j=int(np.argmin(d))
            samples[i*steps+k]=(float(d[j]),j,np.flatnonzero(d < -tol).tolist())
    minimum=model.cfg['detection_m'];worst=None;events=[];native=[];bad_intervals=set();impacted=set()
    for index,(value,j,bad) in enumerate(samples):
        i,k=divmod(index,steps);alpha=k/steps
        if k==0:native.append(value)
        if value<minimum:
            minimum=value;worst=dict(frame=i,fraction=alpha,pair=model.names[j],distance_m=value)
        if bad:
            impacted.add(i)
            if k:impacted.add(i+1);bad_intervals.add(i)
            events.append(dict(frame=i,fraction=alpha,pairs=bad,minimum_m=value))
    cache.update(key=key,xs=xs.copy(),rr=rotations.copy(),samples=samples)
    report = dict(policy=POLICY, pair_count=len(model.pairs), frames=len(xs),
                  native_colliding_frames=int(np.sum(np.asarray(native)<-tol)),
                  interpolated_colliding_intervals=len(bad_intervals),
                  collision_samples=len(events), minimum_distance_m=minimum,
                  worst=worst, interpolation_steps=steps,
                  scope=getattr(model,'scope','declared_upper_body')+'; convex geometry, adjacent/welded bodies excluded; sampled, not continuous certification',
                  structural_exclusions=getattr(model,'exclusions',[]),foot_coverage=getattr(model,'foot_coverage',None),
                  conditional_exclusions=conditional_exclusions)
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


def _frozen_velocity_steps(steps, actual, frozen, max_steps=None, tolerance=2e-6):
    """Represent existing serialized fixed joins within the final tolerance."""
    out=np.array(steps,copy=True)
    limit=out if max_steps is None else np.broadcast_to(max_steps,out.shape)
    roundoff=frozen&(actual>out)&(actual<=limit+tolerance)
    out[roundoff]=actual[roundoff]
    return out


def _repair_window(model, t, xs, solver_cfg, a, b, sides, options=None):
    import osqp
    r = model.r; cfg = model.cfg; n = b-a; fps = float(t['fps'])
    passive=np.zeros(len(model.pairs),bool)
    if options and options.get('_passive_collision_indices') is not None:
        indices=np.asarray(options['_passive_collision_indices'],int)
        if np.any((indices<0)|(indices>=len(passive))):raise ValueError('invalid_passive_collision_indices')
        passive[indices]=True
    owned=~passive
    if not np.any(owned):raise ValueError('repair_requires_owned_collision_pairs')
    geometry_reference=(options or {}).get('_geometry_reference',xs)
    passive_targets={}
    coupled=bool(options and options.get('coupled'))
    if coupled:sides=list(r.sides)
    repair_started=time.perf_counter()
    ids = np.asarray(options['_moving_ids'],int) if coupled and options.get('_moving_ids') is not None else (np.arange(32) if coupled else np.array(sum([r.sides[s]['arm'] for s in sides], [])))
    if len(np.unique(ids))!=len(ids) or np.any((ids<0)|(ids>=32)):raise ValueError('invalid_moving_columns')
    width=len(ids)
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
    physical_step = np.r_[np.full(3,np.inf),r.speed_limits/fps][ids]
    seed_projection=None
    if (options or {}).get('_elastic_recovery') and (options or {}).get('_project_seed') and not coupled:
        # Single-frame collision-free poses may jump between distal branches.
        # Starting a line search from that infeasible path can reject every
        # speed-feasible step. Restore feasibility before nonlinear geometry.
        from ..recovery_seed import project
        cap=np.full(width,np.deg2rad(options['joint_deg']))
        distal=np.isin(ids,sum([r.sides[s]['arm'][3:] for s in sides],[]))
        if options.get('free_distal'):cap[distal]=np.inf
        elif 'distal_joint_deg' in options:cap[distal]=np.deg2rad(options['distal_joint_deg'])
        lower=np.maximum(lo,q0-cap);upper=np.minimum(hi,q0+cap)
        q,seed_projection=project(q,lower,upper,step,frozen,join_q,
            xs[a-1,ids] if a else None,xs[b,ids] if b<len(xs) else None,
            physical_step=physical_step,tolerance=1e-4/fps)
    weights=dict(solver_cfg)
    # Tier relaxation changes search bounds, never the source-pose weights.
    if coupled:
        from .solver import Objective
        objs=[Objective(r,t,i,weights,xs[i-1] if i else None,xs[i-1] if i else None) for i in range(a,b)]
    else:
        from .arms import arm_objective
        objs = [[arm_objective(r,t,i,s,xs[i],weights) for s in sides] for i in range(a,b)]
    baselines=[]
    if options:
        for i in range(a,b):
            r.set(reference[i],t['root_r'][i]);hands={s:(r.point(r.sides[s]['tcp']),r.rot(r.sides[s]['tcp']).copy()) for s in sides}
            contacts=[]
            if coupled:
                for contact in t.get('body_contacts',[]):
                    mode=int(contact['mode'][i])
                    from ..support_quality import preserve_contact
                    if not mode or not preserve_contact(contact,i):continue
                    body=contact['body'];hull=np.asarray(contact.get('robot_hull',r.surface_models[body]['hull']))
                    local=hull[np.argmin(hull@r.rot(body)[2])].copy()
                    contacts.append((body,hull,local,r.point(body,local),mode))
            baselines.append((hands,contacts))
    source_palms=None
    if (options or {}).get('whole_arm') and (options or {}).get('_source_palm_constraints'):
        from ..palm_constraints import SourcePalms
        source_palms=SourcePalms(r,t,options.get('_palm_reference',reference),a,b,sides,ids,
            interior_deg=.1,fixed_frames=np.flatnonzero(np.all(frozen,axis=1)))
    I = sp.eye(n*width, format='csc')
    D1 = sp.kron(sp.diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n)),sp.eye(width),format='csc')
    D2 = sp.kron(sp.diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n)),sp.eye(width),format='csc')
    prior=np.full(width,.12)
    if options:
        shoulder_ids=np.array(sum([r.sides[s]['arm'][:3] for s in sides],[]))
        prior[np.isin(ids,shoulder_ids)]*=.4
    if options and options.get('whole_arm'):
        for side in sides:
            arm=r.sides[side]['arm']
            prior[np.isin(ids,arm[:3])]=.05
            prior[np.isin(ids,arm[3:5])]=.15
            prior[np.isin(ids,arm[5:])]=.25
    # Prefer J1-J3 changes without freezing the distal axes or weakening their
    # palm/forearm/elbow goals. All axes remain available when jointly needed.
    prior_rows=np.broadcast_to(prior,(n,width)).copy()
    if (options or {}).get('_joint_prior') is not None:
        prior_rows=np.asarray(options['_joint_prior'],float)
        if prior_rows.shape!=(n,width) or not np.isfinite(prior_rows).all() or np.any(prior_rows<=0):raise ValueError('invalid_joint_preference')
    smooth = sp.vstack([D1*.18*(fps/60),D2*6*(fps/60)**2,sp.diags(prior_rows.ravel())],format='csc')
    def pose(q, j):
        x=xs[a+j].copy();x[ids]=q[j];return x
    def objective(q):
        residual=[];blocks=[]
        if coupled:
            for j,(obj,row) in enumerate(zip(objs,q)):
                res,J=obj.evaluate(pose(q,j));residual.append(res);blocks.append(J[:,ids])
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
        target=margin_targets.get(key,cfg['margin_m'])
        if passive.any():
            if key not in passive_targets:
                if alpha:
                    rotation=t['root_r'][a+j]@Rotation.from_rotvec(rotation_deltas[j]*alpha).as_matrix()
                    baseline=geometry_reference[a+j]*(1-alpha)+geometry_reference[a+j+1]*alpha
                else:rotation=t['root_r'][a+j];baseline=geometry_reference[a+j]
                model.update(baseline,rotation)
                passive_targets[key]=np.minimum(model.distances()[passive],.0001)
            target=np.broadcast_to(target,(len(model.pairs),)).copy()
            target[passive]=passive_targets[key]
        return target
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
        upper=np.minimum(con.h[active],np.maximum(distances[active]-required[active],-.008))
        # An unrelated pre-existing penetration must not be forced through a
        # limb that cannot resolve it. Preserve its measured baseline instead.
        selected_passive=passive[active]
        upper[selected_passive]=(distances[active]-required[active])[selected_passive]
        return con.G[active][:,r.cols[ids]],upper
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
        # A colliding reference is not a hand-position target. Free-arm
        # resemblance uses source directions; legacy support keeps its caps.
        constrained_hands=hands if options.get('gesture_caps_active',True) else {}
        for side,(point0,rot0) in constrained_hands.items():
            body=r.sides[side]['tcp'];P,A=r.jac(body);delta=r.point(body)-point0
            box(delta,P[:,ids],options['hand_position_m']/np.sqrt(3))
            angle=Rotation.from_matrix(r.rot(body)@rot0.T).magnitude()
            pose_excess=max(pose_excess,np.linalg.norm(delta)-options['hand_position_m'],angle-np.deg2rad(options['hand_rotation_deg']))
            for axis in range(3):
                direction=r.rot(body)[:,axis];dot=float(rot0[:,axis]@direction)
                derivative=rot0[:,axis]@(-skew(direction)@A[:,ids])
                C.append(-derivative);upper.append(dot-np.cos(np.deg2rad(options['hand_rotation_deg'])))
        # Immutable joins may already satisfy export's 1 mm support gate while
        # missing the optimizer's 0.5 mm interior target. Do not declare such a
        # fixed, already-valid row infeasible. Moving poses retain the target.
        tolerance=.001 if frozen[j].all() else .0005
        # A small interior target leaves room for nonlinear contact curvature;
        # the original measured support tolerance remains unchanged.
        linear_tolerance=tolerance*.9 if coupled and options.get('_elastic_recovery') and not frozen[j].all() else tolerance
        if coupled:
            for side,g in r.sides.items():
                if t['sides'][side]['contact'][a+j]<=0:continue
                v=t['sides'][side];local=v['pivots'][a+j];P,_=r.jac(g['foot'],local)
                error=r.point(g['foot'],local)[:2]-v['anchors'][a+j,:2]
                hull=g['hull'];low=hull[np.argmin(hull@r.rot(g['foot'])[2])];G,_=r.jac(g['foot'],low)
                value=np.r_[error,r.point(g['foot'],low)[2]];J=np.vstack([P[:2],G[2]])[:,ids]
                box(value,J,linear_tolerance);support_excess=max(support_excess,float(np.max(abs(value)))-tolerance)
            for body,hull,local,point0,mode in contacts:
                if mode==1:
                    P,_=r.jac(body,local);value=r.point(body,local)-point0;box(value,P[:,ids],linear_tolerance)
                else:
                    local=hull[np.argmin(hull@r.rot(body)[2])];P,_=r.jac(body,local)
                    value=np.array([r.point(body,local)[2]-point0[2]]);box(value,P[2:3,ids],linear_tolerance)
                support_excess=max(support_excess,float(np.max(abs(value)))-tolerance)
        return np.asarray(C).reshape(-1,width),np.asarray(upper),max(0.,pose_excess),max(0.,support_excess)
    def metrics(q):
        distances=[];ground=[];pose_error=[];support_error=[];deficits=[]
        for j in range(n):
            x=pose(q,j);model.update(x,t['root_r'][a+j]);d=model.distances();distances.append(d);deficits.append(np.maximum(target_margin(j)-d,0))
            ground.append(floor_geometry(x,a+j)[0])
            _,_,p,s=extra_constraints(x,j)
            if source_palms is not None:p=max(p,source_palms.evaluate(x,j)[2])
            pose_error.append(p);support_error.append(s)
        for j,alpha,at,rotation in intermediate(q):
            model.update(at,rotation);d=model.distances();distances.append(d);deficits.append(np.maximum(target_margin(j,alpha)-d,0))
            ground.append(floor_geometry(at,a+j,rotation=rotation)[0])
        res,_=objective(q);d=np.asarray(distances);z=np.asarray(ground)
        cost=float(res@res+np.linalg.norm(smooth@(q-q0).ravel())**2)
        violation=float(np.asarray(deficits).sum())
        merit=cost+1e5*violation+1e6*float(np.maximum(-z,0).sum())+1e7*sum(support_error)+1e4*sum(pose_error)
        passive_excess=float(np.asarray(deficits)[:,passive].max()) if passive.any() else 0.
        return merit,float(d[:,owned].min()),float(z.min()),max(pose_error,default=0.),max(support_error,default=0.),float(np.asarray(deficits).max()),passive_excess
    progress=(options or {}).get('_progress')
    saved=(progress or {}).get('window_qp',{})
    history=[];start_iteration=0
    if saved and saved.get('ids')==ids.tolist() and saved.get('shape')==[n,width]:
        resumed=np.asarray(saved['q'],float)
        if resumed.shape==q.shape and np.isfinite(resumed).all():
            q=resumed;history=saved.get('history',[]);start_iteration=int(saved.get('iteration',0))
    def save_iteration(iteration):
        if progress is not None:
            progress['window_qp']=dict(ids=ids.tolist(),shape=[n,width],q=q.tolist(),iteration=iteration,history=history)
    # Caller-owned scratch retains finite iterates only for the next tier.
    # It is never an accepted output and is checked again from scratch.
    scratch=(options or {}).get('_scratch')
    iteration_limit=options.get('iterations',cfg['max_iterations']) if options else cfg['max_iterations']
    for iteration in range(start_iteration,iteration_limit):
        save_iteration(iteration)
        # Deterministic work budget: OS scheduling cannot consume pose quality.
        if options and options.get('watchdog_seconds') and time.perf_counter()-repair_started>options['watchdog_seconds']:
            raise ValueError('collision_repair_interrupted_watchdog')
        residual,J=objective(q);H=J.T@J+smooth.T@smooth+I*.01
        gradient=J.T@residual+smooth.T@(smooth@(q-q0).ravel())
        blocks=[];bounds=[];palm_blocks=[];palm_bounds=[];support_blocks=[];support_bounds=[]
        for j in range(n):
            x=pose(q,j);model.update(x,t['root_r'][a+j]);C,upper=clearance_rows(j)
            _,floor_rows=floor_geometry(x,a+j,True)
            if floor_rows:
                C=np.vstack([C,np.array([v[0] for v in floor_rows])]);upper=np.r_[upper,[v[1]+(floor_tolerance if options else 0.) for v in floor_rows]]
            E,limits,_,_=extra_constraints(x,j)
            if coupled:
                # Elastic recovery may soften geometry restoration, never the
                # support anchors that its nonlinear line search must preserve.
                support_blocks.append(sp.csc_matrix(E));support_bounds.append(limits)
            elif len(limits):C=np.vstack([C,E]);upper=np.r_[upper,limits]
            blocks.append(sp.csc_matrix(C));bounds.append(upper)
            if source_palms is not None:
                PC,PH,_=source_palms.evaluate(x,j);palm_blocks.append(sp.csc_matrix(PC));palm_bounds.append(PH)
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
        trust=np.inf if (options or {}).get("_elastic_recovery") else .18
        lb=np.maximum(np.tile(lo,n)-q.ravel(),-trust);ub=np.minimum(np.tile(hi,n)-q.ravel(),trust)
        if options:
            cap=np.full(width,np.deg2rad(options.get('body_joint_deg',5.)))
            cap[np.isin(ids,arm_ids)]=np.deg2rad(options['joint_deg']);cap[ids<3]=options.get('root_position_m',.02)
            if 'distal_joint_deg' in options:
                distal=sum([r.sides[s]['arm'][3:] for s in sides],[])
                cap[np.isin(ids,distal)]=np.deg2rad(options['distal_joint_deg'])
            lb=np.maximum(lb,(q0-q-cap).ravel());ub=np.minimum(ub,(q0-q+cap).ravel())
            if options.get('free_distal'):
                distal=sum([r.sides[s]['arm'][3:] for s in sides],[])
                cap[np.isin(ids,distal)]=np.inf
                lb=np.maximum(np.tile(lo,n)-q.ravel(),-trust)
                ub=np.minimum(np.tile(hi,n)-q.ravel(),trust)
                lb=np.maximum(lb,(q0-q-cap).ravel());ub=np.minimum(ub,(q0-q+cap).ravel())
            if coupled:
                for side,g in r.sides.items():
                    matches=np.flatnonzero(ids==int(g['leg'][3]))
                    if len(matches):
                        k=int(matches[0]);positions=np.arange(n)*width+k
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
        if options:
            # Fixed, already motor-valid input is immutable even when it is
            # faster than the optimizer's numerical cap. Adjustable intervals
            # keep the numerical cap; true motor violations are not exempted.
            fixed_speed=(frozen[:-1]&frozen[1:]).ravel()[speed_rows]
            fixed_value=abs(Dv@q.ravel())
            velocity_step=_frozen_velocity_steps(velocity_step,fixed_value,fixed_speed,np.tile(physical_step,n-1)[speed_rows],1e-4/fps)
        palms=sp.block_diag(palm_blocks,format='csc') if palm_blocks else sp.csc_matrix((0,n*width))
        palm_upper=np.concatenate(palm_bounds) if palm_bounds else np.empty(0)
        # Palm envelope rows precede C so elastic geometry restoration cannot
        # soften them. The same source envelope is independently checked later.
        supports=sp.block_diag(support_blocks,format='csc') if support_blocks else sp.csc_matrix((0,n*width))
        support_upper=np.concatenate(support_bounds) if support_bounds else np.empty(0)
        A=sp.vstack([I,Dv,palms,supports,C],format='csc')
        lower=np.r_[lb,-velocity_step-Dv@q.ravel(),np.full(len(palm_upper)+len(support_upper)+len(upper),-np.inf)]
        upper=np.r_[ub,velocity_step-Dv@q.ravel(),palm_upper,support_upper,upper]
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
        qp_H=H/objective_scale;qp_g=gradient/objective_scale
        elastic=bool(options and options.get('_elastic_recovery'))
        if elastic:
            # Internal feasibility restoration only. Geometry slacks let a
            # penetrating seed leave conflicting contact normals. Joint, speed
            # and join bounds remain hard; all original nonlinear checks below
            # still reject any unsafe result before this function returns.
            count=C.shape[0];head=A.shape[0]-count
            penalty=1e5/objective_scale
            qp_H=sp.block_diag([qp_H,sp.eye(count)*penalty],format='csc')
            qp_g=np.r_[qp_g,np.full(count,100./objective_scale)]
            A=sp.vstack([sp.hstack([A,sp.vstack([sp.csc_matrix((head,count)),-sp.eye(count)],format='csc')]),
                         sp.hstack([sp.csc_matrix((count,n*width)),sp.eye(count)])],format='csc')
            lower=np.r_[lower,np.zeros(count)];upper=np.r_[upper,np.full(count,np.inf)]
        if elastic and options.get('watchdog_seconds'):
            extra['time_limit']=max(.01,options['watchdog_seconds']-(time.perf_counter()-repair_started))
        if np.any(lower>upper+1e-10):raise ValueError('collision_candidate_empty_bounds')
        opt=osqp.OSQP();opt.setup(P=sp.triu(qp_H,format='csc'),q=qp_g,A=A,l=lower,u=upper,
            eps_abs=1e-7,eps_rel=1e-7,max_iter=(options or {}).get('qp_iterations',20000),adaptive_rho_interval=50,verbose=False,polishing=True,**extra)
        sol=opt.solve()
        feasible_inexact=(elastic and sol.info.status_val in (7,8) and sol.x is not None and
            np.isfinite(sol.x).all() and max(float(np.max(lower-A@sol.x)),float(np.max(A@sol.x-upper)))<=1e-6)
        if (sol.info.status_val not in (1,2) and not feasible_inexact) or sol.x is None or not np.isfinite(sol.x).all():
            error=ValueError('collision_qp_'+sol.info.status)
            if options:error.diagnostics=diagnostics
            raise error
        at=A@sol.x
        if max(float(np.max(lower-at)),float(np.max(at-upper)))>1e-6:
            raise ValueError('collision_qp_constraint_residual')
        delta=sol.x[:n*width].reshape(n,width);old=metrics(q);accepted=False
        for alpha in ((1.,.5,.25,.125,.0625,.03125,.015625) if coupled and elastic else (1.,.5,.25,.125,.0625)):
            trial=q+alpha*delta;new=metrics(trial)
            palm_progress=max(2e-6,old[3]-1e-7) if source_palms is not None else 2e-6
            restoring_palm=source_palms is not None and old[3]>2e-6 and new[3]<old[3]-1e-7
            support_progress=max(2e-6,old[4]-1e-7) if coupled and elastic else 2e-6
            if (new[0]<old[0]+1e-9 or restoring_palm) and (not options or (new[3]<=palm_progress and new[4]<=support_progress)):
                q=trial;accepted=True;break
        if accepted and scratch is not None:
            seed=xs[a:b].copy();seed[:,ids]=q;scratch['candidate_window']=seed
        history.append(dict(iteration=iteration,accepted=accepted,minimum_m=new[1]));save_iteration(iteration+1)
        if not accepted:break
        if new[5]<=2e-6 and new[2]>=-floor_tolerance and ((options or {}).get('whole_arm') or abs(delta).max()<2e-4):break
    candidate=xs.copy();candidate[a:b,ids]=q
    # Validate the exact float32 joint representation later exported to PKL.
    candidate[a:b,ids]=candidate[a:b,ids].astype(np.float32).astype(float)
    q=candidate[a:b,ids];merit,dmin,zmin,*_=metrics(q)
    if (dmin<(-cfg['penetration_tolerance_m'] if options else .003-2e-6)) or zmin < -floor_tolerance:
        error=ValueError('nonlinear_collision_or_floor_clearance')
        if options:error.diagnostics=dict(minimum_distance_m=dmin,minimum_floor_m=zmin,iterations=len(history),last_iteration=history[-1] if history else None,seed_projection=seed_projection,hand_pose_excess=metrics(q)[3],support_excess=metrics(q)[4])
        raise error
    if options:
        result=metrics(q)
        # The free-arm policy keeps the original export safety gate at dense
        # samples. A target margin is not a new stricter acceptance threshold.
        if not options.get('whole_arm') and result[5]>2e-6:raise ValueError('nonlinear_collision_clearance_margin')
        if result[3]>2e-6:raise ValueError('repair_hand_pose_budget')
        if result[4]>2e-6:raise ValueError('repair_support_contact_changed')
        if result[6]>2e-6:raise ValueError('passive_collision_regression')
    if np.any(q<lo-2e-6) or np.any(q>hi+2e-6):raise ValueError('arm_angle_limit')
    if np.any(abs(np.diff(candidate[max(0,a-1):min(len(xs),b+1),ids],axis=0))[:,ids>=3]>physical_step[ids>=3]+1e-4/fps):
        raise ValueError('arm_velocity_or_boundary_limit')
    verification_model=model
    if passive.any():
        from ..refine_routing import submodel
        verification_model=submodel(model,np.flatnonzero(owned))
    checked,_,_=scan(verification_model,candidate[max(0,a-1):min(len(xs),b+1)],t['root_r'][max(0,a-1):min(len(xs),b+1)])
    if checked['collision_samples'] and not (options or {}).get('internal_overlap',False):
        error=ValueError('interpolated_collision');error.diagnostics=checked;raise error
    return candidate,dict(start=a,end=b,sides=sides,status='repaired',iterations=len(history),
                          minimum_m=dmin,interpolated_minimum_m=checked['minimum_distance_m'],
                          maximum_correction_deg=float(np.rad2deg(abs(q-q0)[:,ids>=3]).max()),coupled=bool(coupled),seed_projection=seed_projection,owned_pair_count=int(owned.sum()),passive_pairs_protected=int(passive.sum()))


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
            from ..arm_policy import enabled,collision_options,active_arm_support
            options=None
            if enabled(solver_cfg) and not active_arm_support(robot,t,a,b):
                options=dict(collision_options(10.,cfg['max_iterations']),_reference=xs)
            candidate,report=_repair_window(model,t,out,solver_cfg,a,b,sides,options=options)
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
    # Final export must cover at least the samples used by repair guards.
    model=CollisionModel(robot,config)
    return scan(model,xs,rotations,max(8,int(model.cfg['interpolation_steps'])))[0]
