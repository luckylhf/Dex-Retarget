"""Compare same-source exported motions, including changed target provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
from scipy.spatial.transform import Rotation
from dex_frax.local_refine import continuity
from dex_frax.v8.robot import Robot
from dex_frax.v8.self_collision import CollisionModel, scan


def evaluate(path, dense=False):
    path = Path(path)
    p = pickle.loads(path.read_bytes())
    meta = json.loads(path.with_suffix('.json').read_text())
    audit = json.loads(path.with_suffix('.check.json').read_text())
    quality = json.loads((path.parent/'quality.json').read_text())
    run_path = path.parent/'run.json'
    runtime = json.loads(run_path.read_text()) if run_path.exists() else None
    q = p['dof_pos'].astype(float)
    result = dict(path=str(path.resolve()), motion_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  source_sha256=meta['input_sha256'], model_sha256=meta['model_sha256'],
                  frames=len(q), fps=p['fps'], joints=p['joint_names'],
                  total_wall_s=runtime['total_wall_s'] if runtime else None,
                  converter_elapsed_s=meta['elapsed_s'], stage_times_s=meta.get('stage_times_s'),
                  torso_deg=audit['torso_orientation_deg'],
                  foot_anchor_max_mm=max(v['contact_xyz_cm']['max']*10 for v in audit['sides'].values() if v['contact_xyz_cm']),
                  sole_penetration_max_mm=max(0., -10*min(v['minimum_visual_sole_cm'] for v in audit['sides'].values())),
                  collision=audit['self_collision'],
                  joint_limit_violation_deg=audit['joint_limit_violation_deg'],
                  anatomical_limit_violation_deg=audit['effective_anatomical_limit_violation_deg'],
                  body_minimum_mm=audit['minimum_body_visual_cm']*10.,
                  floating_longest_s=audit['stationary_floating_longest_seconds'],
                  root_speed_max_m_s=quality['root_speed_max_m_s'],
                  root_angular_speed_max_rad_s=quality['root_angular_speed_max_rad_s'],
                  arm_errors={s:{k:v[k] for k in ('upper_deg','lower_deg','hand_forward_deg','hand_width_deg',
                                                 'forearm_width_deg','elbow_axis_deg','signed_elbow_goal_error_deg')}
                              for s,v in audit['sides'].items()},
                  contact_speed_cm_s={s:v['contact_speed_cm_s'] for s,v in audit['sides'].items()},
                  frax_runtime=runtime.get('frax') if runtime else None,
                  max_joint_step_deg=float(np.rad2deg(abs(np.diff(q,axis=0))).max()),
                  continuity=continuity(q,p['fps']), flags=quality['flags'],
                  speed_violations=quality['step_velocity']['violating_intervals'],
                  knees={name:dict(min_deg=float(np.rad2deg(q[:,j]).min()),negative_frames=int(np.sum(q[:,j]<0)))
                         for j,name in enumerate(p['joint_names']) if 'knee' in name})
    if dense:
        r = Robot(); x = np.c_[p['root_pos'],q]
        result['dense_collision_32'] = scan(CollisionModel(r),x,Rotation.from_quat(p['root_rot']).as_matrix(),32)[0]
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('motions',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dense',action='store_true')
    args=parser.parse_args()
    rows=[evaluate(p,args.dense) for p in args.motions]
    for key in ('source_sha256','model_sha256','frames','fps','joints'):
        if any(r[key] != rows[0][key] for r in rows[1:]):
            raise ValueError('Comparison mismatch: '+key)
    targets=[]
    with np.load(args.motions[0].with_suffix('.diagnostics.npz')) as first:
        for path in args.motions[1:]:
            with np.load(path.with_suffix('.diagnostics.npz')) as other:
                keys=[k for k in first.files if k in other.files and k != 'solver_records' and not k.startswith('support_')]
                different=[k for k in keys if not np.array_equal(first[k],other[k])]
                targets.append(dict(path=str(path), compared_arrays=len(keys), changed_arrays=different))
    result=dict(runs=rows,targets=targets,scope='Same original nobody BVH. Single-run wall times; target easing intentionally changes foot pose targets. No full-library, dynamics or hardware validation.')
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(args.output)


if __name__=='__main__':
    main()
