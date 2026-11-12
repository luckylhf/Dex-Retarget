"""Source pelvis -> feasible waist trajectory -> limb IK with protected body."""
import time
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation


def waist_trajectory(robot,t,seed,cfg=None,collision_cfg=None,report=None):
    ids=np.array([3+robot.names.index('waist_'+axis+'_joint') for axis in ('yaw','roll','pitch')])
    from .v8.self_collision import CollisionModel
    from .refine_routing import collision_groups,submodel
    full=CollisionModel(robot,dict(collision_cfg or {},coverage='full'))
    model=submodel(full,collision_groups(full)['body_collision'])
    out=seed.copy();step=robot.joint_step(t['fps'],cfg or {'numerical_joint_speed_deg_s':720.})[ids-3]
    repaired=[];remaining=[];minimum=float('inf')
    for i in range(len(out)):
        lo=robot.lower[ids]+1e-7;hi=robot.upper[ids]-1e-7
        if i:lo=np.maximum(lo,out[i-1,ids]-step);hi=np.minimum(hi,out[i-1,ids]+step)
        def objective(q):
            row=out[i].copy();row[ids]=q;robot.set(row,t['root_r'][i])
            return (robot.rot(robot.torso)-t['torso_r'][i]).ravel()
        fit=least_squares(objective,np.clip(seed[i,ids],lo+1e-9,hi-1e-9),bounds=(lo,hi),max_nfev=30,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        chosen=fit.x
        def geometry(q):
            # Body-only preflight, before arm fitting. Include the whole incoming
            # interpolation segment; a valid endpoint alone is insufficient.
            row=out[i].copy();row[ids]=q
            if not i:
                model.update(row,t['root_r'][i]);return model.distances()
            rotation=Rotation.from_matrix(t['root_r'][i-1].T@t['root_r'][i]).as_rotvec()
            values=[]
            for alpha in np.arange(1,9)/8:
                pose=(1-alpha)*out[i-1]+alpha*row
                rot=t['root_r'][i-1]@Rotation.from_rotvec(alpha*rotation).as_matrix()
                model.update(pose,rot);values.extend(model.distances())
            return np.asarray(values)
        if geometry(chosen).min()<0.:
            bounded=minimize(lambda q:float(objective(q)@objective(q)),chosen,method='SLSQP',
                bounds=list(zip(lo,hi)),constraints=[dict(type='ineq',fun=lambda q:geometry(q)-.0001)],
                options=dict(maxiter=60,ftol=1e-10))
            # A failed nonlinear attempt is never called a safety certificate.
            # Keep the best measured proposal; the independent full audit routes
            # any residual body collision to REVIEW, not shoulder compensation.
            if np.isfinite(bounded.x).all() and np.all(bounded.x>=lo-1e-9) and np.all(bounded.x<=hi+1e-9) and geometry(bounded.x).min()>geometry(chosen).min()+1e-8:
                chosen=bounded.x;repaired.append(i)
        clearance=float(geometry(chosen).min());minimum=min(minimum,clearance)
        if clearance<0.:remaining.append(i)
        out[i,ids]=chosen
    if report is not None:report.update(policy='body_owned_geometry_before_limb_ik_v1',pair_count=len(model.pairs),interpolation_steps=8,adjusted_frames=repaired,unresolved_frames=remaining,minimum_distance_m=minimum)
    return out,ids


INITIAL_POLICY='pelvis_waist_adaptive_feet_arms_geometry_v10'


def solve_protected(robot,t,cfg,baseline,base_solver,root_mode='source',progress=None,*,auto_lower=False,collision_cfg=None):
    from .ik import FraxIK
    from .initial_reuse import target_digest
    start=time.monotonic();seed=baseline.copy();target_sha=target_digest(t)
    if root_mode=='source':seed[:,:3]=t['root_p']
    elif root_mode!='feasible':raise ValueError('Unknown pelvis reference')
    legs=np.asarray(sum([g['leg'] for g in robot.sides.values()],[]))
    arms=np.asarray(sum([g['arm'] for g in robot.sides.values()],[]))
    # Waist orientation and body/body distances are independent of the legs
    # and a common root translation. Choose it before the single coupled foot
    # solve, so leg/body geometry already sees the final protected waist.
    if progress:print('initial: waist',flush=True)
    body_check={};rows,waist=waist_trajectory(robot,t,seed,cfg,report=body_check)
    reports={};all_records=[]
    stages=[('feet',np.r_[2,legs] if root_mode=='feasible' else legs),('arms',arms)]
    lower_routing=None
    if auto_lower and root_mode=='feasible':
        from .initial_lower import route
        candidate,lower_routing=route(robot,t,rows,cfg,collision_cfg)
        if candidate is not None:
            rows=candidate
            stages=stages[1:]
        if progress:print('initial: '+lower_routing['selected']+' ('+lower_routing['reason']+')',flush=True)
    for name,moving in stages:
        if progress:print('initial: '+name,flush=True)
        mask=np.ones(32,bool);mask[moving]=False;reference=rows.copy()
        local=dict(t,priority_pose=reference,priority_mask=mask,foot_self_collision=name=='feet',arm_self_collision=name=='arms')
        solver=FraxIK(base_solver.iterations,base_solver.initial_iterations,improved=False,assets=base_solver.assets)
        local_cfg=dict(cfg)
        if name=='feet':local_cfg['leg_direction']=max(float(cfg.get('leg_direction',0.)),float(cfg.get('arm_direction',0.)),2.*float(cfg.get('foot_orientation',0.)))
        rows,records=solver(robot,local,local_cfg,progress) if progress else solver(robot,local,local_cfg)
        if not np.array_equal(rows[:,mask],reference[:,mask]):raise ValueError('Protected upstream stage moved in '+name+' solve')
        if target_digest(t)!=target_sha:raise ValueError('Protected stage mutated prepared targets')
        reports[name]=dict(solver.report,stage_leg_direction_weight=local_cfg.get('leg_direction'),prepared_targets_sha256=target_sha)
        all_records.append(records)
    from .local_refine import torso_errors
    report=dict(reports['arms']);report['qp_nonconverged_steps']=sum(v.get('qp_nonconverged_steps',0) for v in reports.values())
    return rows,all_records[-1],dict(body_geometry_preflight=body_check,torso_reference_error_deg=torso_errors(robot,t,rows).tolist(),
        policy=INITIAL_POLICY,root_mode=root_mode,elapsed_s=time.monotonic()-start,limb_solver=report,
        prepared_targets_sha256=target_sha,downstream_stages_recomputed_on_actual_timeline=True,ordered_limb_solvers=reports,
        stage_order=['pelvis','waist','feet' if 'feet' in reports else 'local_feet','arms'],
        lower_routing=lower_routing,foot_solver_passes=int('feet' in reports),
        root_height_optimized_with_feet=root_mode=='feasible' and ('feet' in reports or any(
            trial['reason']=='kept' for trial in lower_routing['trials'])),feet_locked_during_arm_solve=True)



class PelvisPriorityIK:
    """Refine initial solve; fast keeps its independent unprotected Frax path.

    The prescribed anatomical pelvis rotation and Frax horizontal path are
    retained; support may compensate pelvis height. The limited, speed-bounded waist is fixed before limb IK.
    Infeasible support/collision outcomes remain REVIEW under the normal audit.
    """
    improved=False

    def __init__(self,iterations=8,initial_iterations=100,root_mode="feasible",baseline_solver=None):
        from .ik import FraxIK
        self.base=baseline_solver or FraxIK(iterations,initial_iterations,improved=False)
        self.settings=FraxIK(iterations,initial_iterations,improved=False)
        self.report={}
        self.root_mode=root_mode
        self.initial_collision_cfg=None

    def __call__(self,robot,t,cfg,progress=None):
        baseline,records=self.base(robot,t,cfg,progress)
        self.baseline=baseline.copy()
        self.settings.assets=self.base.assets
        rows,new_records,detail=solve_protected(robot,t,cfg,baseline,self.settings,self.root_mode,progress,
                                                auto_lower=True,collision_cfg=self.initial_collision_cfg)
        self.report=dict(detail['limb_solver'],body_priority=detail,
                         raw_frax_report=self.base.report)
        self.report['qp_nonconverged_steps']+=self.base.report.get('qp_nonconverged_steps',0)
        self.initial=rows.copy()
        if progress:progress(len(rows),len(rows))
        return rows,new_records
