"""Summarize complete, same-input runs without hiding local pose differences."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import platform
import numpy as np


def summarize(folder):
    folder = Path(folder)
    read = lambda n: json.loads((folder/n).read_text())
    run, check, quality, identity = map(read, ["run.json", "motion.check.json", "quality.json", "identity.json"])
    if run["status"] != "complete":
        raise ValueError("Incomplete run: " + str(folder))
    collision = check["self_collision"]
    feet = list(check["sides"].values())
    return dict(directory=str(folder.resolve()), mode=run["mode"], frames=run["frames"],
                source_sha256=identity["input_sha256"], status=run["quality"], total_wall_s=run["total_wall_s"],
                stages=run["stage_times_s"], frax=run["frax"], flags=run["flags"],
                speed_violations=quality["step_velocity"]["violating_intervals"],
                joint_limit_violation_deg=check["joint_limit_violation_deg"],
                max_contact_error_mm=max((v["contact_xyz_cm"]["max"]*10 for v in feet if v["contact_xyz_cm"]),default=0.),
                max_sole_penetration_mm=max(0., -min(v["minimum_visual_sole_cm"] for v in feet)*10),
                collision_frames=collision["native_colliding_frames"],
                deepest_collision_mm=max(0., -collision["minimum_distance_m"]*1000),
                torso_orientation_deg=check["torso_orientation_deg"],
                hand_forward_p95_deg={k: v["hand_forward_deg"]["p95"] for k,v in check["sides"].items()},
                hand_width_p95_deg={k: v["hand_width_deg"]["p95"] for k,v in check["sides"].items()},
                motion_sha256=hashlib.sha256((folder/"motion.pkl").read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = [summarize(p) for p in args.runs]
    if len({(r["source_sha256"],r["frames"]) for r in rows}) != 1:
        raise ValueError("Different source inputs or frame counts")
    baseline = args.runs[-1]
    base = pickle.loads((baseline/"motion.pkl").read_bytes())
    differences = []
    with np.load(baseline/"motion.diagnostics.npz",allow_pickle=False) as d:
        target_keys = [k for k in d.files if k not in ("solver_records",) and not k.startswith("support_")]
        base_targets = {k: d[k] for k in target_keys}
    for folder in args.runs[:-1]:
        data = pickle.loads((folder/"motion.pkl").read_bytes())
        if data["joint_names"] != base["joint_names"]:
            raise ValueError("Joint order mismatch")
        err = np.rad2deg(abs(data["dof_pos"].astype(float)-base["dof_pos"]))
        i,j = np.unravel_index(np.argmax(err),err.shape)
        with np.load(folder/"motion.diagnostics.npz",allow_pickle=False) as d:
            mismatch = [k for k in target_keys if not np.array_equal(d[k],base_targets[k])]
        differences.append(dict(mode=folder.name, versus=baseline.name, joint_error_p95_deg=float(np.percentile(err,95)),
                                max_joint_error_deg=float(err[i,j]), worst_joint=data["joint_names"][j],
                                worst_frame=int(i), worst_time_s=float(i/50), identical_target_arrays=len(target_keys)-len(mismatch),
                                different_target_arrays=mismatch))
    report=dict(host=platform.platform(), machine=platform.machine(), runs=rows, differences=differences,
                scope="One full nobody clip; sequential local runs. No full-library, dynamic, or hardware validation.")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
