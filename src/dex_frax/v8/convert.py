#!/usr/bin/env python3
"""Convert SOMA/LAFAN/mocap BVH to a named-joint Dex PKL, entirely on CPU."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import argparse,hashlib,json,pickle,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .bvh import load
from .source import adapt
from .robot import Robot,DEFAULT_MODEL,DEFAULT_URDF
from .contacts import prepare
from .solver import solve,constrain_support as legacy_support
from .support_fast import constrain_support as hybrid_support
from .arms import plan_and_refine
from .contact_solver import refine as refine_body_contacts
from .anatomy import configure
from .self_collision import refine as refine_self_collision, audit_export as audit_self_collision

ROOT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def support_solver(cfg):
    mode=cfg.get('support_refinement','hybrid_cpu_v1')
    if mode=='hybrid_cpu_v1':return hybrid_support
    if mode=='legacy_slsqp':return legacy_support
    raise ValueError('Unknown support_refinement: '+str(mode))

def convert(bvh,output,source='auto',model=DEFAULT_MODEL,fps=None,config=None,max_frames=None,no_contacts=False,initial_solver=None,refine=True,prepared_input=None):
    from ..resource_events import phase
    phase('preprocessing',compile_possible=initial_solver is not None and not getattr(initial_solver,'strict',False))
    start=time.monotonic();output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError('Refusing to overwrite '+str(output))
    cfg=json.loads((ROOT/'config.json').read_text()) if config is None else json.loads(Path(config).read_text())
    constrain_support=support_solver(cfg['solver'])
    from ..input_errors import MotionInputError
    try:
        if source == 'smplx':
            from ..smplx_source import load as load_smplx, adapt as adapt_smplx
            if prepared_input is None: raise ValueError('SMPL-X requires the dex-frax convert preparation entry')
            motion=load_smplx(prepared_input,fps=fps)
            source_targets=adapt_smplx(motion)
        else:
            motion=load(bvh,fps=fps)
            if max_frames:motion.local_p=motion.local_p[:max_frames];motion.local_r=motion.local_r[:max_frames]
            if len(motion.local_p)<3:raise ValueError('Conversion needs at least 3 frames; one-frame poses are calibration-only')
            source_targets=adapt(motion,source)
    except (ValueError,KeyError,IndexError,StopIteration) as error:
        raise MotionInputError(str(error)) from error
    robot=Robot(model);configure(robot,cfg['anatomy']);t=prepare(source_targets,robot,cfg['contact'],enabled=not no_contacts)
    if getattr(initial_solver, 'improved', False):
        from ..contact_transition import refine as refine_transitions
        initial_solver.report['contact_transitions']=refine_transitions(t,robot)
    from ..initial_reuse import target_digest
    prepared_targets_sha256=target_digest(t)
    stage_times={};stage_start=time.monotonic()
    phase('initial',compile_possible=initial_solver is not None and not getattr(initial_solver,'strict',False))
    if initial_solver is None:phase('v8_ik')
    x,records=(solve if initial_solver is None else initial_solver)(robot,t,cfg['solver'],lambda i,n:print(f'{output.stem}: {i}/{n} {time.monotonic()-start:.1f}s',flush=True))
    stage_times['initial_ik']=time.monotonic()-stage_start
    initial_x=x.copy()
    if refine:
        phase('support')
        stage_start=time.monotonic()
        x,support_reports=constrain_support(robot,t,x,cfg['solver'],lambda i,n:print(f'{output.stem}: support {i}/{n} {time.monotonic()-start:.1f}s',flush=True))
        stage_times['support']=time.monotonic()-stage_start
        phase('arms')
        stage_start=time.monotonic()
        x,arm_report=plan_and_refine(robot,t,x,cfg['solver'],lambda side,phase,i,n:print(f'{output.stem}: {side} {phase} {i}/{n} {time.monotonic()-start:.1f}s',flush=True))
        stage_times['arms']=time.monotonic()-stage_start
        phase('body_contacts')
        stage_start=time.monotonic()
        x,body_contact_reports=refine_body_contacts(robot,t,x,cfg['solver'],lambda i,n:print(f'{output.stem}: body_contacts {i}/{n} {time.monotonic()-start:.1f}s',flush=True))
        stage_times['body_contacts']=time.monotonic()-stage_start
        support_reports.extend(body_contact_reports)
        phase('self_collision')
        stage_start=time.monotonic()
        x,collision_report=refine_self_collision(robot,t,x,cfg['solver'],cfg.get('self_collision'),
            lambda i,n:print(f'{output.stem}: self_collision {i}/{n}',flush=True))
        from ..graduated_collision import recover
        x,collision_report=recover(robot,t,x,cfg['solver'],cfg.get('self_collision'),collision_report)
        stage_times['self_collision']=time.monotonic()-stage_start
    else:
        phase('fast_refinement')
        support_reports=[];arm_report={'executed':False};collision_report={'executed':False}
        robot.support_branch_relaxations=[];robot.body_contact_qp_summary={'executed':False}
        if getattr(initial_solver, 'improved', False):
            from ..local_refine import refine as refine_local
            stage_start=time.monotonic()
            x,local_report=refine_local(robot,t,x,cfg['solver'])
            initial_solver.report['local_refinement']=local_report
            stage_times['local_pose']=time.monotonic()-stage_start
            from ..adaptive_refine import refine as refine_adaptive
            stage_start=time.monotonic()
            x,adaptive_report=refine_adaptive(robot,t,x,cfg['solver'])
            initial_solver.report['adaptive_refinement']=adaptive_report
            arm_report={'executed':any(r['stage']=='arms' and r['status']=='accepted' for r in adaptive_report['repairs']),
                        'scope':'adaptive','details_location':'frax.adaptive_refinement.repairs'}
            robot.body_contact_qp_summary={'executed':any(r['stage']=='body' and r['status']=='accepted' for r in adaptive_report['repairs']),
                                          'scope':'adaptive','details_location':'frax.adaptive_refinement.repairs'}
            stage_times['adaptive_v8']=time.monotonic()-stage_start
            stage_start=time.monotonic()
            from ..fast_collision import refine as refine_fast_collision
            x,collision_report=refine_fast_collision(robot,t,x,cfg['solver'],cfg.get('self_collision'))
            from ..graduated_collision import recover
            x,collision_report=recover(robot,t,x,cfg['solver'],cfg.get('self_collision'),collision_report,fast=True)
            stage_times['self_collision']=time.monotonic()-stage_start
    phase('export_check')
    np.savez_compressed(output.with_suffix('.initial.npz'),x=initial_x,root_r=t['root_r'],fps=motion.fps,joint_names=robot.names,solver_records=np.asarray(records))
    quat=Rotation.from_matrix(t['root_r']).as_quat()
    for i in range(1,len(quat)):
        if quat[i]@quat[i-1]<0:quat[i]*=-1
    payload={'fps':motion.fps,'root_pos':x[:,:3].astype(np.float32),'root_rot':quat.astype(np.float32),
        'dof_pos':x[:,3:].astype(np.float32),'joint_names':robot.names,'quat_order':'xyzw',
        'local_body_pos':None,'link_body_list':None,'model_sha256':sha(robot.path),'robot':'tienkung_dex_29'}
    collision_report['export_check']=audit_self_collision(robot,payload,cfg.get('self_collision'))
    from ..quality_v8 import adjacent_speed_report
    velocity_check = adjacent_speed_report(payload['dof_pos'], motion.fps, robot.names, robot.speed_limits)
    # Adjacent speed failures are retained in quality flags; no training export.
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_bytes(pickle.dumps(payload,protocol=4))
    np.savez_compressed(output.with_suffix('.npz'),**{k:v for k,v in payload.items() if v is not None})
    diagnostics={'root_target':t['root_p'],'root_r':t['root_r'],'torso_target':t['torso_r'],'solver_records':records}
    support_stats=getattr(robot,'support_fast_report',None)
    if support_stats is not None:
        diagnostics['support_route']=np.array([{'bypass':0,'light':1,'fallback':2}[v['route']] for v in support_stats['frames']],dtype=np.uint8)
        diagnostics['support_qp_steps']=np.array([v['qp_steps'] for v in support_stats['frames']],dtype=np.int32)
        diagnostics['support_light_stop_reason']=np.array([v['light_stop_reason'] for v in support_stats['frames']])
        diagnostics['support_fallback_triggers']=np.array(['|'.join(v['fallback_triggers']) for v in support_stats['frames']])
        diagnostics['support_final_feasible']=np.array([v['final_feasible'] for v in support_stats['frames']],dtype=bool)
        # Detailed diagnostics remain JSON-readable without pickle-enabled NPZ.
        output.with_suffix('.support.json').write_text(json.dumps(support_stats,ensure_ascii=False,indent=2))
    for s,v in t['sides'].items():
        for k in ('foot_target_p','foot_target_r','raw_foot_p','raw_foot_r','contact','phase','anchors','pivots','heel','toe','heel_speed','toe_speed','thigh','shin','upper','lower','elbow_axis','signed_elbow_bend','hand_forward','hand_width','forearm_width','ground_contact_confidence'):
            diagnostics[s+'_'+k]=v[k]
    diagnostics['source_expected_contact']=t['source_expected_contact'];diagnostics['source_airborne']=t['source_airborne']
    for contact in t['body_contacts']:
        for k in ['mode','confidence','anchors','source_surface_height','source_surface_point']:diagnostics['body_'+contact['body_name']+'_'+k]=contact[k]
    np.savez_compressed(output.with_suffix('.diagnostics.npz'),**diagnostics)
    from ..identity import CONVERTER_RELEASE,SOLVER_POLICY
    meta={'converter_version':8,'generation_mode':'frax_v8_from_original_bvh' if initial_solver is not None else 'v8_reference_from_original_bvh','pose_solver':'frax_diff_ik' if initial_solver is not None else 'v8_trf','converter_release':CONVERTER_RELEASE,'solver_policy':SOLVER_POLICY,
        'input':str(Path(bvh).resolve()),'input_sha256':sha(bvh),'source_type':t['source_type'],
        'input_frames':motion.input_frames,'input_fps':motion.input_fps,'output_frames':len(x),'output_fps':motion.fps,
        'model':robot.path,'model_sha256':sha(robot.path),'viewer_urdf':DEFAULT_URDF,'viewer_urdf_sha256':sha(DEFAULT_URDF),
        'joint_names':robot.names,'quat_order':'xyzw','units':'metres/radians','cpu_only':True,
        'official_reference_commit':'cbdb2b494a2af82137bf9b6f15122d60167281c4','official_runtime':False,
        'prepared_targets_sha256':prepared_targets_sha256,'configuration':cfg,'contacts_enabled':not no_contacts,'source_floor_m':t['floor'],'leg_scale':t['scale'],
        'ground_calibration':t['ground_calibration'],'multicontact':t['multicontact_record'],
        'body_contact_solver':robot.body_contact_qp_summary,
        'body_contact_task_policy':'bounded extra-body normal cues with existing foot support, linear body support without foot plants; independent hard mesh clearance and foot XYZ constraints',
        'body_contact_models':[{k:v for k,v in c.items() if k not in ('mode','confidence','anchors','source_surface_height','source_surface_point')} for c in t['body_contacts']],
        'world_origin':t['world_origin'].tolist(),'source_calibration':{s:{k:v[k] for k in ('foot_calibration','hand_method','elbow_calibration')} for s,v in t['sides'].items()},
        'wrist_limits':robot.wrist_limit_record,
        'estimated_contact_segments':{s:v['segments'] for s,v in t['sides'].items()},
        'arm_branch_planning':arm_report,
        'self_collision':collision_report,
        'support_constraint_failures':support_reports,
        'support_branch_relaxations':robot.support_branch_relaxations,
        'support_refinement':cfg['solver'].get('support_refinement','hybrid_cpu_v1'),
        'support_refinement_statistics':{k:v for k,v in support_stats.items() if k!='frames'} if support_stats is not None else None,
        'joint_velocity_limits_rad_s':dict(zip(robot.names,robot.speed_limits.tolist())),
        'solver_velocity_check':velocity_check,
        'stage_times_s':stage_times,'frax':getattr(initial_solver,'report',None),'v8_refinement':refine,'elapsed_s':time.monotonic()-start,'output_sha256':sha(output),
        'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted(list(ROOT.glob('*.py'))+[ROOT/'config.json'])},
        'limits':'Kinematic reference only; contact inferred; flat floor assumed; no dynamics/self-collision certification.'}
    if source == 'smplx':
        meta.update(generation_mode='smplx_parameters_to_dex',input_adapter=motion.metadata)
    output.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    return meta

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('bvh');p.add_argument('output');p.add_argument('--source',choices=['auto','soma','lafan','mocap'],default='auto')
    p.add_argument('--model',default=DEFAULT_MODEL);p.add_argument('--fps',type=float,default=None,help='Default preserves source fps; resampling uses local-rotation SLERP')
    p.add_argument('--config');p.add_argument('--max-frames',type=int);p.add_argument('--no-contacts',action='store_true',help='Ablation only')
    a=p.parse_args();m=convert(**vars(a));print(json.dumps({k:m[k] for k in ('output_frames','output_fps','elapsed_s')},ensure_ascii=False))
