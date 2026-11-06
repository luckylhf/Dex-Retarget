"""Private regression evidence, including common gates and source-axis checks."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
from scipy.spatial.transform import Rotation
from dex_frax.adaptive_refine import policy,pose_mask
from dex_frax.local_refine import continuity
from dex_frax.v8.robot import Robot
from dex_frax.v8.self_collision import CollisionModel,scan


def read(path):return json.loads(path.read_text())


def row(folder,dense=False):
    run=read(folder/'run.json')
    if run['status']!='complete':return dict(status=run['status'])
    a=read(folder/'motion.check.json');q=read(folder/'quality.json');m=read(folder/'motion.json')
    p=pickle.loads((folder/'motion.pkl').read_bytes())
    with np.load(folder/'motion.pose_errors.npz') as d:
        errors={s:{k:d[s+'_'+k+'_deg'] for k in ('upper','lower','hand_forward','hand_width','elbow_axis')}
                for s in ('Left','Right')}
    flags=set(q['flags']);bad=int(pose_mask(errors,50.,policy()).sum())
    if bad:flags.add('sustained_arm_pose_mismatch')
    result=dict(status='complete',quality=q['status'],common_flags=sorted(flags),common_arm_bad_frames=bad,
        seconds=run['total_wall_s'],frames=run['frames'],torso=a['torso_orientation_deg'],
        contact_max_mm=max((v['contact_xyz_cm']['max']*10 for v in a['sides'].values() if v['contact_xyz_cm']),default=0.),
        body_minimum_mm=a['minimum_body_visual_cm']*10,
        native_collision_frames=a['self_collision']['native_colliding_frames'],
        speed_violations=q['step_velocity']['violating_intervals'],
        continuity=continuity(p['dof_pos'],50.),
        arms={s:{k:v[k] for k in ('elbow_axis_deg','upper_deg','lower_deg','hand_forward_deg','hand_width_deg')}
              for s,v in a['sides'].items()},
        source_elbow_calibration={s:v['elbow_calibration'] for s,v in m['source_calibration'].items()},
        cache=(run.get('frax') or {}).get('runtime_cache'),
        motion_sha256=hashlib.sha256((folder/'motion.pkl').read_bytes()).hexdigest())
    if dense:
        model=CollisionModel(Robot())
        report=scan(model,np.c_[p['root_pos'],p['dof_pos']],Rotation.from_quat(p['root_rot']).as_matrix(),32)[0]
        result['collision_32']=report
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('suite',type=Path);parser.add_argument('--dense',action='store_true')
    parser.add_argument('--candidate',default='candidate_v040')
    parser.add_argument('--output',default='comparison.json')
    args=parser.parse_args();root=args.suite.resolve();manifest=read(root/'manifest.json');rows=[]
    for case in manifest['cases']:
        ident=case['id'];base=root/'results/baseline_v030';candidate=root/'results'/args.candidate/ident
        result=dict(case=ident,baseline=row(base/(ident+'_frax')),v8=row(base/(ident+'_v8')),candidate=row(candidate,args.dense))
        with np.load(base/(ident+'_frax')/'motion.diagnostics.npz') as a,np.load(candidate/'motion.diagnostics.npz') as b:
            keys=[k for k in a.files if k in b.files and k!='solver_records' and not k.startswith('support_')]
            result['changed_target_arrays']=[k for k in keys if not np.array_equal(a[k],b[k])]
        rows.append(result);print(ident,result['candidate']['common_flags'],flush=True)
        report=dict(scope=manifest['scope'],common_pose_gate=policy(),cases=rows,
            timing_scope='regression jobs overlapped some development checks; use isolated_same_clip for speed claims')
        (root/args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(root/args.output)


if __name__=='__main__':main()
