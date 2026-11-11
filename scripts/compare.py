"""Summarize complete, same-input runs without hiding local pose differences."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import numpy as np


def result_stage(folder):
    folder = Path(folder).resolve()
    if (folder / 'quality.json').is_file():
        return folder
    selection = json.loads((folder / 'selection.json').read_text())
    selected = selection.get('selected_output') or selection.get('preview_output')
    if not selected:
        selected = next(
            (attempt['output'] for attempt in reversed(selection['attempts'])
             if attempt['status'] == 'complete' and attempt.get('quality') == 'review'),
            None,
        )
    if not selected:
        raise ValueError(f'No completed motion in {folder}')
    output_root = folder.parent if (folder.parent / 'batch.json').is_file() else folder
    stage = (output_root / selected).resolve()
    if not stage.is_relative_to(folder):
        raise ValueError(f'Motion output is outside {folder}')
    return stage


def summarize(folder):
    folder = result_stage(folder)
    read = lambda n: json.loads((folder/n).read_text())
    run, check, quality, identity = map(read, ["run.json", "motion.check.json", "quality.json", "identity.json"])
    if run["status"] != "complete":
        raise ValueError("Incomplete run: " + str(folder))
    collision = check["full_self_collision"]
    feet = list(check["sides"].values())
    return dict(directory=str(folder), mode=run.get("profile") or run["mode"], frames=run["frames"],
                source_sha256=identity["input_sha256"], model_sha256=quality["model_sha256"],
                status=run["status"], quality=quality["status"], total_wall_s=run["total_wall_s"],
                stages=run["stage_times_s"], frax=run["frax"], flags=run["flags"],
                speed_violations=quality["step_velocity"]["violating_intervals"],
                joint_limit_violation_deg=check["joint_limit_violation_deg"],
                max_contact_error_mm=max((v["contact_xyz_cm"]["max"]*10 for v in feet if v["contact_xyz_cm"]),default=0.),
                max_sole_penetration_mm=max(0., -min(v["minimum_visual_sole_cm"] for v in feet)*10),
                collision_frames=collision["native_colliding_frames"],
                collision_samples=collision["collision_samples"],
                collision_pair_set_sha256=collision["pair_set_sha256"],
                collision_subdivisions=collision["interpolation_steps"],
                audit_contract=quality["audit_contract"],
                sustained_pose_frames=quality["adaptive_pose_gate"]["violating_frames"],
                jitter_frames=len(quality["temporal_quality"]["violating_frames"]),
                deepest_collision_mm=max(0., -collision["minimum_distance_m"]*1000),
                torso_orientation_deg=check["torso_orientation_deg"],
                hand_forward_p95_deg={k: v["hand_forward_deg"]["p95"] for k,v in check["sides"].items()},
                hand_width_p95_deg={k: v["hand_width_deg"]["p95"] for k,v in check["sides"].items()},
                motion_sha256=hashlib.sha256((folder/"motion.npz").read_bytes()).hexdigest())


def compare_motions(baseline, candidate):
    before = Path(baseline['directory'])
    after = Path(candidate['directory'])
    for key in ('source_sha256', 'model_sha256', 'frames',
                'collision_pair_set_sha256', 'collision_subdivisions'):
        if baseline[key] != candidate[key]:
            raise ValueError('Comparison mismatch: ' + key)
    def criteria(row):
        return {key: value for key, value in row['audit_contract'].items()
                if key not in ('fingerprint', 'implementation')}

    if criteria(baseline) != criteria(candidate):
        raise ValueError('Comparison audit criteria mismatch')
    with np.load(before/'motion.diagnostics.npz', allow_pickle=False) as old, np.load(after/'motion.diagnostics.npz', allow_pickle=False) as new:
        target_keys = {key for key in set(old.files) | set(new.files)
                       if key != 'solver_records' and not key.startswith('support_')}
        changed = sorted(key for key in target_keys
                         if key not in old or key not in new or not np.array_equal(old[key], new[key]))
    with np.load(before/'motion.npz', allow_pickle=False) as old, np.load(after/'motion.npz', allow_pickle=False) as new:
        for key in ('joint_names', 'fps', 'source_time_s', 'source_fps'):
            if (key in old) != (key in new) or (key in old and not np.array_equal(old[key], new[key])):
                raise ValueError('Comparison time or joint order mismatch: ' + key)
        error = np.rad2deg(np.abs(new['dof_pos'].astype(float) - old['dof_pos']))
        frame, joint = np.unravel_index(np.argmax(error), error.shape)
        return dict(joint_error_p95_deg=float(np.percentile(error,95)),
                    max_joint_error_deg=float(error[frame,joint]), worst_joint=str(old['joint_names'][joint]),
                    worst_frame=int(frame), worst_time_s=float(frame/old['fps']),
                    byte_identical=baseline['motion_sha256']==candidate['motion_sha256'],
                    changed_target_arrays=changed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = [summarize(p) for p in args.runs]
    differences = [dict(candidate=row['directory'], baseline=rows[-1]['directory'],
                        **compare_motions(rows[-1], row)) for row in rows[:-1]]
    report=dict(host=platform.platform(), machine=platform.machine(), runs=rows, differences=differences,
                scope="Same source motion, output timeline, and kinematic checks")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
