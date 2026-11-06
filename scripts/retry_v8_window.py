"""Explicit local retry of an existing V8 result; preserve original artifacts.

The original repair selects arms by *penetration*, but its QP applies the
clearance margin to all pairs, including the fixed opposite arm. This command
can enable both arms in a requested window without changing the V8 optimizer,
geometry, limits, margins or original trajectories outside that window.
"""
import argparse
import copy
import json
import pickle
import shutil
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dex_frax.cli import atomic_json, sha
from dex_frax.v8.robot import Robot
from dex_frax.v8.anatomy import configure
from dex_frax.v8.bvh import load
from dex_frax.v8.source import adapt
from dex_frax.v8.contacts import prepare
from dex_frax.v8.self_collision import CollisionModel, scan, _repair_window, audit_export
from dex_frax.v8.audit import audit
from dex_frax.quality_v8 import quality


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--start", type=int, default=1081)
    p.add_argument("--end", type=int, default=1158)
    p.add_argument("--sides", nargs="+", choices=["Left", "Right"], default=["Left", "Right"])
    args = p.parse_args()
    start = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.baseline / "motion.pkl"
    payload = pickle.loads(source.read_bytes())
    meta = json.loads(source.with_suffix(".json").read_text())
    cfg = meta["configuration"]
    if sha(meta["input"]) != meta["input_sha256"]:
        raise ValueError("Original BVH hash differs")
    r = Robot(meta["model"])
    configure(r, cfg["anatomy"])
    motion = load(meta["input"], fps=payload["fps"])
    t = prepare(adapt(motion, meta["source_type"]), r, cfg["contact"])
    xs = np.c_[payload["root_pos"], payload["dof_pos"]].astype(float)
    if len(xs) != len(t["root_r"]) or not 0 <= args.start < args.end <= len(xs):
        raise ValueError("Frame window mismatch")
    # Preserve the baseline's actual exported root rotations exactly.
    t["root_r"] = Rotation.from_quat(payload["root_rot"]).as_matrix()
    model = CollisionModel(r, cfg["self_collision"])
    before, _, _ = scan(model, xs, t["root_r"], subdivisions=8)
    report = dict(baseline=str(source.resolve()), baseline_sha256=sha(source),
                  start=args.start, end=args.end, sides=args.sides,
                  optimizer="unchanged_v8_repair_window", before=before,
                  collision_config=cfg["self_collision"], script_sha256=sha(__file__))
    atomic_json(args.output / "retry.json", report)
    try:
        result, window = _repair_window(model, t, xs, cfg["solver"], args.start, args.end, args.sides)
    except ValueError as exc:
        report.update(status="unresolved", reason=str(exc), elapsed_s=time.perf_counter()-start)
        atomic_json(args.output / "retry.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    candidate = copy.deepcopy(payload)
    candidate["dof_pos"] = result[:, 3:].astype(np.float32)
    ids = np.array(sum([r.sides[s]["arm"] for s in args.sides], [])) - 3
    other = np.setdiff1d(np.arange(29), ids)
    invariants = dict(root_pos=np.array_equal(candidate["root_pos"], payload["root_pos"]),
                      root_rot=np.array_equal(candidate["root_rot"], payload["root_rot"]),
                      nonselected_joints=np.array_equal(candidate["dof_pos"][:, other], payload["dof_pos"][:, other]),
                      prefix=np.array_equal(candidate["dof_pos"][:args.start], payload["dof_pos"][:args.start]),
                      suffix=np.array_equal(candidate["dof_pos"][args.end:], payload["dof_pos"][args.end:]),
                      join_two_frames=np.array_equal(candidate["dof_pos"][args.start:args.start+2], payload["dof_pos"][args.start:args.start+2]))
    if not all(invariants.values()):
        raise AssertionError(invariants)
    after, _, _ = scan(CollisionModel(r, cfg["self_collision"]), np.c_[candidate["root_pos"],candidate["dof_pos"]].astype(float), t["root_r"], subdivisions=8)
    report.update(status="repaired", window=window, after=after, invariant_checks=invariants)
    for name in ("motion.diagnostics.npz", "motion.initial.npz", "motion.support.json"):
        if (args.baseline/name).exists():
            shutil.copy2(args.baseline/name, args.output/name)
    path = args.output / "motion.pkl"
    path.write_bytes(pickle.dumps(candidate, protocol=4))
    np.savez_compressed(path.with_suffix(".npz"), **{k:v for k,v in candidate.items() if v is not None})
    meta["baseline"] = str(source.resolve())
    meta["baseline_sha256"] = sha(source)
    meta["prior_self_collision"] = meta["self_collision"]
    meta["self_collision"] = dict(policy=before["policy"], status="completed", before=before, after=after,
                                    windows=[window], config=cfg["self_collision"], export_check=audit_export(r,candidate,cfg["self_collision"]))
    meta["generation_mode"] = "local_both_arm_retry_of_v8_output"
    meta["output_sha256"] = sha(path)
    meta["local_retry"] = report
    atomic_json(path.with_suffix(".json"), meta)
    check = audit(path)
    assessed = quality(path, meta, check)
    for side, v in check["sides"].items():
        if v["minimum_visual_sole_cm"] < -.1:
            assessed["flags"].append(side+"_sole_penetration_over_1mm")
        if v["contact_xyz_cm"] and v["contact_xyz_cm"]["max"] > .1:
            assessed["flags"].append(side+"_anchor_error_over_1mm")
    if check["minimum_body_visual_cm"] < -.1:
        assessed["flags"].append("full_body_mesh_penetration_over_1mm")
    if after["collision_samples"]:
        assessed["flags"].append("dense_interpolated_collision_unresolved")
    assessed["status"] = "review" if assessed["flags"] else "accepted_kinematic_candidate"
    assessed["validation_scope"] = "one existing V8 full clip after local arm retry; no dynamic validation"
    atomic_json(args.output/"quality.json", assessed)
    report.update(quality=assessed["status"], flags=assessed["flags"], elapsed_s=time.perf_counter()-start,
                  output_sha256=sha(path), speed=assessed["step_velocity"])
    atomic_json(args.output/"retry.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
