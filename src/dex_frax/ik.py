"""JIT-compiled differential IK using frax, with V8 anatomical targets.

This is a new optimizer, not a claim of numerical equivalence to V8 TRF.
All task derivatives are obtained through frax FK. Dynamic QP boxes retain
the original clip-to-clip per-axis speed bounds throughout inner iterations.
"""
import time
import hashlib
import json
import os
from collections import OrderedDict
import jax
import jax.numpy as jnp
import numpy as np
from frax.ik import DiffIKSolver, ResidualTask
from frax.ik.constraints import InequalityConstraint
from .kinematics import DexKinematics

_MODELS = OrderedDict()
_KERNELS = OrderedDict()


def clear_runtime_cache():
    _MODELS.clear()
    _KERNELS.clear()


def _remember(cache, key, value, capacity):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > capacity:
        cache.popitem(last=False)


def initial_branch_count(t):
    mask=np.asarray(t.get('priority_mask',np.zeros(32,bool)))
    # Three seeds differ only at shoulder-yaw/elbow-yaw. When all upper joints
    # are protected, they are projected to the identical state before stepping.
    return 1 if np.all(mask[18:]) else 3


def runtime_keys(robot, t, cfg, iterations, initial_iterations, improved, assets, block_frames=128):
    """Every static residual/constraint input participates in cache identity."""
    geometry = hashlib.sha256()
    for name in ('body_pos','body_quat','body_parentid','jnt_pos','jnt_axis','jnt_bodyid','geom_pos','geom_quat','geom_bodyid','mesh_vert'):
        geometry.update(np.asarray(getattr(robot.m,name)).tobytes())
    model = dict(assets=assets, names=robot.names, lower=robot.lower.tolist(),
                 upper=robot.upper.tolist(), speeds=robot.speed_limits.tolist(),geometry=geometry.hexdigest())
    model_key = hashlib.sha256(json.dumps(model,sort_keys=True).encode()).hexdigest()
    kernel = dict(model=model_key,initial_branches=initial_branch_count(t), foot_geometry=bool(t.get('foot_self_collision',False)), block_frames=block_frames, algorithm='carried_block_scan_v1',fps=float(t['fps']), weights=cfg,
                  iterations=iterations, initial_iterations=initial_iterations, improved=improved,
                  input_shapes=[(list(a.shape[1:]),str(a.dtype)) for a in jax.tree.leaves(targets(t))],
                  contacts=[dict(body=int(c['body']),hull=np.asarray(c.get('robot_hull',robot.surface_models[c['body']]['hull'])).tolist()) for c in t['body_contacts']])
    return model_key, hashlib.sha256(json.dumps(kernel,sort_keys=True).encode()).hexdigest()


@jax.tree_util.register_static
class FrameBounds(InequalityConstraint):
    def __init__(self, robot, lower, upper, step, gain):
        self.robot, self.gain = robot, gain
        self.lower, self.upper, self.step = map(np.asarray, (lower, upper, step))
        self.G = np.vstack([np.eye(32)[3:], -np.eye(32)[3:]])
        self.num_rows, self.penalty = 58, 1e8

    def rows(self, q, ref):
        lo = jnp.maximum(self.lower, ref["previous"][3:] - self.step)
        hi = jnp.minimum(self.upper, ref["previous"][3:] + self.step)
        lo = jnp.where(ref["first"], self.lower, lo)
        hi = jnp.where(ref["first"], self.upper, hi)
        return self.G, self.gain * jnp.concatenate([hi - q[3:], q[3:] - lo])


def targets(t):
    keys = ("thigh", "shin", "foot_target_p", "foot_target_r", "contact", "pivots", "anchors",
            "upper", "lower", "hand_forward", "hand_width", "forearm_width", "elbow_axis",
            "signed_elbow_bend", "ground_contact_confidence")
    out = {k: np.asarray(t[k]) for k in ("root_p", "root_r", "torso_r")}
    out['body_tolerance'] = np.full(len(t['hips']),float(t['body_contact_height_tolerance_m']))
    if 'priority_pose' in t:
        out['priority_pose']=np.asarray(t['priority_pose'])
        out['priority_mask']=np.broadcast_to(np.asarray(t['priority_mask'],bool),out['priority_pose'].shape).copy()
    out["sides"] = {s: {k: np.asarray(v[k]) for k in keys} for s, v in t["sides"].items()}
    out["body_weights"] = np.stack([
        np.asarray(c["confidence"]) * (np.asarray(c["mode"]) != 0) * c["source_geometry_confidence"]
        for c in t["body_contacts"]], axis=-1) if t["body_contacts"] else np.zeros((len(t["hips"]), 0))
    return out


def make_residual(model, robot, t, w):
    fps = t["fps"]
    surface = [(b, np.array(v["hull"])) for b, v in robot.surface_models.items()]
    contacts = [(c["body"], np.asarray(c.get("robot_hull", robot.surface_models[c["body"]]["hull"]))) for c in t["body_contacts"]]

    def residual(x, ref):
        if 'priority_pose' in ref:x=jnp.where(ref['priority_mask'],ref['priority_pose'],x)
        tolerance = ref['body_tolerance']
        tf = model.transforms(x, ref["root_r"])
        p, r = tf[:, :3, 3], tf[:, :3, :3]
        res = []
        def add(value, weight):
            res.append(jnp.atleast_1d(value).reshape(-1) * weight)
        def axis(b, local, target, weight):
            add(r[b] @ jnp.array(local) - target, weight)
        def direction(a, b, target, weight):
            delta = p[b] - p[a]
            add(delta / jnp.maximum(jnp.linalg.norm(delta), 1e-12) - target, weight)
        def low(b, hull):
            return p[b, 2] + jnp.min(jnp.asarray(hull) @ r[b, 2])
        add(x[:3] - ref["root_p"], w["root_position"])
        for ax in (0, 2):
            axis(robot.torso, np.eye(3)[ax], ref["torso_r"][:, ax], w["torso_orientation"])
        for side, g in robot.sides.items():
            v = ref["sides"][side]
            direction(g["hip"], g["knee"], v["thigh"], w["leg_direction"])
            direction(g["knee"], g["foot"], v["shin"], w["leg_direction"])
            add(p[g["foot"]] - v["foot_target_p"], w["foot_position"])
            for ax in (0, 2):
                axis(g["foot"], np.eye(3)[ax], v["foot_target_r"][:, ax], w["foot_orientation"])
            cp = p[g["foot"]] + r[g["foot"]] @ v["pivots"]
            height = low(g["foot"], g["hull"])
            add(jnp.concatenate([cp[:2] - v["anchors"][:2], height[None]]), w["contact_position"] * v["contact"])
            direction(g["shoulder"], g["elbow"], v["upper"], w["arm_direction"])
            direction(g["elbow"], g["wrist"], v["lower"], w["arm_direction"])
            add(p[g["wrist"]] - p[g["shoulder"]] - g["upper_length"] * v["upper"] - g["lower_length"] * v["lower"], 6.)
            primary = max(w["arm_direction"], w["hand_direction"], w["hand_width"])
            fw = w["hand_direction"] + v["ground_contact_confidence"] * (primary - w["hand_direction"])
            ww = w["hand_width"] + v["ground_contact_confidence"] * (primary - w["hand_width"])
            axis(g["tcp"], [0., 0., -1.], v["hand_forward"], fw)
            axis(g["tcp"], [1., 0., 0.], v["hand_width"], ww)
            axis(g["forearm"], [1., 0., 0.], v["forearm_width"], w["forearm_twist"])
            axis(g["shoulder"], [0., -1., 0.], v["elbow_axis"], w["elbow_axis"])
            ai, li = np.array(g["arm"]), np.array(g["leg"])
            goal = jnp.clip(g["elbow_geometric_zero"] - v["signed_elbow_bend"], robot.lower[ai[3]], robot.upper[ai[3]])
            add(x[ai[3]] - goal, w["elbow_bend"])
            add(x[ai[5:]], w["wrist_rest"])
            add(x[ai[5:]] - jnp.clip(x[ai[5:]], g["wrist_soft_lower"], g["wrist_soft_upper"]), w["wrist_soft_limit"])
            add(x[li[2]], w["hip_yaw_rest"])
            # Prefer flexion over the mechanically permitted hyperextension
            # branch. This is a soft retargeting preference, not a model limit.
            if w.get("knee_extension_preference", 0.):
                add(jnp.minimum(x[li[3]] - w.get("knee_preferred_flexion_rad", 0.), 0.), w["knee_extension_preference"])
        # Same mesh convex hulls as V8; no G1 sphere substitution.
        for b, hull in surface:
            add(jnp.minimum(low(b, hull) - .0005, 0.), w["ground_penalty"])
        support = jnp.clip(jnp.maximum(ref["sides"]["Left"]["contact"], ref["sides"]["Right"]["contact"]), 0., 1.)
        for j, (b, hull) in enumerate(contacts):
            h = low(b, hull)
            bounded = h / jnp.sqrt(1. + (h / tolerance) ** 2)
            value = (1. - support) * h + support * bounded
            add(value, w["body_contact_position"] * ref["body_weights"][j])
        active = 1. - ref["first"]
        f = fps / 60.
        add(x - ref["previous"], w["joint_velocity"] * f * active)
        add(x - ref["predicted"], w["joint_acceleration"] * f * f * active)
        add(x[:3] - ref["root_velocity_goal"], w["root_velocity"] * f * active)
        add(x[:3] - ref["predicted"][:3], w["root_acceleration"] * f * f * active)
        return jnp.concatenate(res)
    return residual


class FraxIK:
    def __init__(self, iterations=8, initial_iterations=100, improved=True, assets=None, use_cache=True,block_frames=None):
        self.iterations, self.initial_iterations = iterations, initial_iterations
        self.improved = improved
        self.assets, self.use_cache = assets, use_cache
        self.report = {}
        self.initial = None
        self.block_frames=int(block_frames or os.environ.get('DEX_FRAX_IK_BLOCK_FRAMES','128'))
        if self.block_frames<1:raise ValueError('IK block length must be positive')
        if not jax.config.jax_compilation_cache_dir:
            from .resources import state_root
            folder=state_root()/'jax_cache'
            try:
                folder.mkdir(parents=True,exist_ok=True,mode=0o700)
                jax.config.update('jax_compilation_cache_dir',str(folder))
                jax.config.update('jax_persistent_cache_min_compile_time_secs',0.)
            except OSError as error:self.report['persistent_cache']=dict(enabled=False,reason=str(error))
        if jax.config.jax_compilation_cache_dir:self.report['persistent_cache']=dict(enabled=True,path=jax.config.jax_compilation_cache_dir)

    def __call__(self, robot, t, cfg, progress=None):
        from .resources import CompileLease
        if self.assets is None:
            from .paths import ASSETS
            self.assets={str(p.relative_to(ASSETS)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ASSETS.rglob('*')) if p.is_file()}
        keys=runtime_keys(robot,t,cfg,self.iterations,self.initial_iterations,self.improved,self.assets,self.block_frames)
        warm=self.use_cache and keys[0] in _MODELS and keys[1] in _KERNELS
        from .resource_events import phase
        phase('waiting_compile',compile_possible=not warm)
        with CompileLease(required=not warm) as lease:
            result=self._solve(robot,t,cfg,progress,lease,keys)
            self.report['compile_queue_wait_s']=lease.wait_s
            return result

    def _solve(self, robot, t, cfg, progress, lease,keys):
        from .resource_events import phase
        phase('ik_prepare',compile_possible=not (self.use_cache and keys[1] in _KERNELS))
        start = time.perf_counter()
        assets = self.assets
        if assets is None:
            from .paths import ASSETS
            assets = {str(p.relative_to(ASSETS)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(ASSETS.rglob('*')) if p.is_file()}
        model_key,kernel_key = keys
        model_hit = self.use_cache and model_key in _MODELS
        if model_hit:
            model,parity = _MODELS[model_key]
            _MODELS.move_to_end(model_key)
        else:
            model = DexKinematics(robot)
            parity = model.verify(robot)
            if self.use_cache: _remember(_MODELS,model_key,(model,parity),4)
        self.report['foot_self_collision_enabled']=bool(t.get('foot_self_collision',False))
        self.report["fk_parity"] = {**parity, 'reused_verified_model': bool(model_hit)}
        refs = targets(t)
        n, fps = len(t["hips"]), float(t["fps"])
        x0 = np.r_[t["root_p"][0], np.zeros(29)]
        for g in robot.sides.values():
            x0[g["arm"][3]], x0[g["leg"][3]] = -.4, .15
        if 'priority_pose' in t:x0=np.asarray(t['priority_pose'][0]).copy()
        kernel_hit=self.use_cache and kernel_key in _KERNELS
        begin=time.perf_counter()
        if kernel_hit:
            compiled,initialize,rows=_KERNELS[kernel_key]
            _KERNELS.move_to_end(kernel_key)
        else:
            phase('ik_compile',compile_possible=True)
            compiled,initialize,rows=self._compile(model,robot,t,cfg,refs,x0,fps)
            if self.use_cache:_remember(_KERNELS,kernel_key,(compiled,initialize,rows),int(os.environ.get('DEX_FRAX_KERNEL_CACHE_SIZE','2')))
        compile_s=time.perf_counter()-begin
        from .resources import release_memory
        release_memory()
        prep=time.perf_counter()-start-compile_s
        lease.release()
        phase('ik_solve')
        begin=time.perf_counter()
        carry=initialize(jnp.asarray(x0),jnp.asarray(refs['root_p'][0]))
        poses=[];logs=[]
        for offset in range(0,n,self.block_frames):
            count=min(self.block_frames,n-offset)
            block=jax.tree.map(lambda a:np.pad(a[offset:offset+count],[(0,self.block_frames-count)]+[(0,0)]*(a.ndim-1),mode='edge'),refs)
            carry,(q,records)=compiled(carry,block,np.arange(self.block_frames)<count)
            q,records=jax.device_get((q,records));poses.append(q[:count]);logs.append(records[:count])
        x=np.concatenate(poses);records=np.concatenate(logs)
        solve_s=time.perf_counter()-begin
        if not np.isfinite(x).all():raise ValueError('frax produced nonfinite poses')
        max_speed_excess=float(np.maximum(abs(np.diff(x[:,3:],axis=0))*fps-robot.speed_limits,0.).max())
        self.initial=np.array(x)
        self.report.update(backend='frax_diff_ik_cpu_fp64',frames=n,iterations=self.iterations,
            initial_iterations=self.initial_iterations,prepare_s=prep,compile_s=compile_s,solve_s=solve_s,
            total_s=time.perf_counter()-start,qp_nonconverged_steps=int(records[:,2].sum()),
            qp_nonconverged_output_frames=np.flatnonzero(records[:,2]>0).tolist(),
            qp_selected_iteration_count=int(self.initial_iterations+(n-1)*self.iterations),initial_branch_count=initial_branch_count(t),
            max_speed_excess_rad_s=max_speed_excess,residual_rows=rows,root_orientation='prescribed_by_v8',cpu_trf_fallback=False,
            execution=dict(policy='carried_fixed_blocks_v1',block_frames=self.block_frames,blocks=len(poses),state_continuity=True,padded_frames_not_solved=True))
        self.report['runtime_cache']=dict(model_hit=bool(model_hit),kernel_hit=bool(kernel_hit),model_key=model_key,kernel_key=kernel_key,enabled=self.use_cache)
        self.report['knee_extension_preference']=3. if self.improved else 0.
        self.report['knee_preferred_flexion_deg']=1. if self.improved else 0.
        print(f'frax: {n} frames, compile {compile_s:.2f}s, solve {solve_s:.2f}s, QP nonconverged {self.report["qp_nonconverged_steps"]}',flush=True)
        if progress:progress(n,n)
        return np.asarray(x),np.asarray(records).tolist()

    def _compile(self,model,robot,t,cfg,refs,x0,fps):
        seed={**jax.tree.map(lambda a:a[0],refs),'first':1.,'previous':x0,'predicted':x0,'root_velocity_goal':x0[:3]}
        weights = dict(cfg)
        if self.improved:
            weights["knee_extension_preference"] = 3.
            weights["knee_preferred_flexion_rad"] = np.deg2rad(1.)
        residual = make_residual(model, robot, t, weights)
        rows = jax.eval_shape(residual,jnp.asarray(x0),seed).shape[0]
        task = ResidualTask(model, residual, rows, gain=fps)
        bounds = FrameBounds(model, robot.lower[3:] + 1e-7, robot.upper[3:] - 1e-7, robot.joint_step(fps, cfg), fps)
        constraints=[bounds]
        if t.get('foot_self_collision',False):
            from .frax_foot_constraints import FootGeometry,FootSupport
            feet=FootGeometry(model,robot,fps);constraints.extend([feet,FootSupport(model,robot,fps)])
            self.report['initial_foot_geometry']=feet.report
        solver = DiffIKSolver(model, [task], constraints, regularization=1., solver_tol=1e-7)
        dt = .5 / fps

        def iterate(state, ref, count):
            # Half Newton steps reduce overshoot at changing mesh minima.
            def body(_, carry):
                state, iters, failed = carry
                if 'priority_pose' in ref:state=state._replace(q=jnp.where(ref['priority_mask'],ref['priority_pose'],state.q))
                state, _, info = solver.step(state, ref, dt)
                if 'priority_pose' in ref:state=state._replace(q=jnp.where(ref['priority_mask'],ref['priority_pose'],state.q))
                return state, iters + info.qp_iters, failed + jnp.logical_not(info.qp_converged).astype(jnp.int32)
            return jax.lax.fori_loop(0, count, body, (state, jnp.int32(0), jnp.int32(0)))

        def initialize(initial_q,root):
            return solver.init(initial_q),initial_q,root,jnp.int32(0)

        def first_frame(carry,first):
            initial_q=carry[0].q
            ref0 = {**first, "first": 1., "previous": initial_q, "predicted": initial_q, "root_velocity_goal": initial_q[:3]}
            # Three independent first-frame branches; no CPU IK fallback.
            branches=initial_branch_count(t)
            seeds = jnp.tile(initial_q, (branches, 1))
            for k, yaw in enumerate((-1.2, 1.2) if branches==3 else (), 1):
                for g in robot.sides.values():
                    seeds = seeds.at[k,g['arm'][2]].set(yaw).at[k,g['arm'][4]].set(-yaw)
            states, iterations, failures = jax.vmap(lambda q: iterate(solver.init(q), ref0, self.initial_iterations))(jnp.asarray(seeds))
            costs = jax.vmap(lambda q: jnp.sum(residual(q, ref0) ** 2))(states.q)
            best = jnp.argmin(costs)
            state = jax.tree.map(lambda a: a[best], states)
            record=jnp.array([costs[best]/2.,iterations[best],failures[best]])
            return (state,state.q,first['root_p'],jnp.int32(1)),(state.q,record)

        def next_frame(carry,inputs):
            state,prev2,root_prev,index=carry;previous=state.q
            predicted=jnp.where(index<2,previous,previous+jnp.clip(previous-prev2,-.10,.10))
            root_goal=previous[:3]+inputs['root_p']-root_prev
            ref={**inputs,'first':0.,'previous':previous,'predicted':predicted,'root_velocity_goal':root_goal}
            state=state._replace(q=state.q.at[:3].set(root_goal))
            state,count,failed=iterate(state,ref,self.iterations)
            record=jnp.array([jnp.sum(residual(state.q,ref)**2)/2.,count,failed])
            return (state,previous,inputs['root_p'],index+1),(state.q,record)

        def run(carry,block,valid):
            def frame(carry,inputs):
                ref,active=inputs
                def execute(carry):
                    return jax.lax.cond(carry[3]==0,lambda c:first_frame(c,ref),lambda c:next_frame(c,ref),carry)
                return jax.lax.cond(active,execute,lambda c:(c,(c[0].q,jnp.zeros(3))),carry)
            return jax.lax.scan(frame,carry,(block,valid))
        sample=jax.tree.map(lambda a:np.repeat(a[:1],self.block_frames,axis=0),refs)
        compiled=jax.jit(run).lower(initialize(jnp.asarray(x0),jnp.asarray(refs['root_p'][0])),sample,np.ones(self.block_frames,bool)).compile()
        return compiled,initialize,rows
