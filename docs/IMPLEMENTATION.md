# Implementation boundary

Version 0.7.0 changes both orchestration and the local arm optimizer. Single-file
auto always retains frax -> hybrid -> V8. Batch auto uses BVH Frames * Frame Time:
under 60 seconds uses the cascade; at least 60 seconds starts at independent V8.
Queue priority is frax, hybrid, V8, with longest-first V8 tail filling. Full-BVH
SHA256 aliases avoid duplicate work; conflicting source hints are rejected.
No source-speed or collision-only shortcut suppresses the final V8 attempt.

Completed-stage PSS envelopes are separated by solver, source and retained JAX
state. Cold compilation is learned separately and shared only when a task may
actually compile. Strict initial reuse fails into V8 instead of unexpectedly
compiling under a reuse budget. Worker phase events and bounded tests do not
change numerical IK parameters. Existing host/cgroup guards remain mandatory.

The local v8/arms.py copy adds guarded branch tracking, shoulder-first J1-J3
fitting, event-driven candidate search and bounded arm-only temporal windows.
J4-J7 anatomy, hand axes, wrist limits and speed checks remain. Supported arms
and sustained pose failures use the previous algorithm; rejected new results
rerun only the affected side. The sibling official V8 checkout is untouched.
The common quality policy is unchanged. See PIPELINE_070_VALIDATION.md for
measured limits; cached window tests do not establish full-library quality.

Version 0.6.1 changes resource scheduling only: a continuous per-clip cascade
replaces global stage barriers, and resource_policy.json is checked every two
seconds. Valid changes update active and queued reservations without restarting
workers. Invalid updates retain the previous policy and are logged. The host
reserve, disk reserve, cgroup containment, IK, and quality gates remain fixed.
The release identity changes; the 0.6.0 numerical solver policy is preserved.

The V8 source is from the sibling `Dex-SuperMimic`. The initial snapshot was
`V8_cpu_support_20261105`; version 0.1.1 synchronizes the explicitly authorized
`V8_cpu_coupled_arms_20261105` update from that checkout. Default collision
windows now include arms required by below-margin pairs. Original hashes and
receipts are preserved, and current hashes are recorded separately.

Through 0.5.1 the V8 numerical modules were preserved; only the asset import in `robot.py`
changed. `v8/convert.py` accepts an injected initial solver, records timings,
saves initial trajectories, supports an unrefined ablation, and retains failed
previews for explicit audit. Its identities are replaced, not mislabeled V8.
The quality module imports the new identity and asset paths. The root CLI
adds strict geometry and solver convergence gates and atomic run status.

Version 0.6.0 extends this local copy of `v8/self_collision.py` with optional
gesture-capped and contact-coupled window recovery; the sibling official V8
checkout is untouched. The original default repair path is still covered by
the copied geometry regressions. `graduated_collision.py` calls the extension
only after a failed window, limits total changes from the original trajectory,
and independently checks denser interpolation before committing a candidate.
Source hashes identify these changes separately from upstream V8.

Version 0.2.0 adds a separate fast pipeline in `contact_transition.py` and
`local_refine.py`, wired through `v8/convert.py`; the formal sibling V8 checkout
is unchanged by this improvement. The default `frax` mode eases only adjacent
accepted contact-type orientation transitions exceeding 3 degrees, over up to
0.1 s, keeping anchors/weights fixed and recomputing ankle XYZ using the same
convex hull. Foot pose targets therefore intentionally differ from the baseline.
IK adds a weight-3 one-sided soft knee preference toward at least 1 degree of
flexion; it does not change physical joint bounds or guarantee positive knees.
Torso error above 1 degree or upper-body acceleration above 400 rad/s2 triggers
0.4 s-padded offline windows, limited to 4 s each. The acceleration threshold is
a repair heuristic, not a hardware rating. The sparse pose SQP fixes root/legs
and two boundary samples at both ends, jointly bounds all upper joints and
adjacent velocities, and includes second differences. Active upper-body support
or rejected windows are preserved. Existing local collision repair runs last.
`hybrid` retains the original initial IK/full V8 route. `--legacy-frax` retains
the unrefined old frax route for reproducible ablations.

Version 0.3.0 supersedes the hybrid detail above: hybrid now enables the same
knee preference/contact easing before full V8 refinement. The fast route adds
`adaptive_refine.py` after local pose fitting and before collision recovery.
It measures full-clip geometry/arm errors, slices already-prepared targets for
V8 repair windows (without re-estimating contacts), restores window joins, and
checks the entire candidate for angle, speed, floor, contact and knee-branch
regressions. Existing V8 numerical modules stay unchanged. Failing local
geometry/pose results could escalate to the corresponding full-clip stage in 0.3.0; 0.4.0 disables this by default.
An active body-contact window bypasses arm-only branch repair and is routed
to the existing body solver; unresolved arm errors remain review failures.
This does not add force/friction constraints, nor does V8's existing body
solver guarantee normal/XY anchoring of every inferred body contact.
Continuous 0.1 s arm/hand/elbow outliers are independently gated after export.

`kinematics.py` loads the unchanged Dex URDF through frax with the asset's
explicit 29-joint ordering. Fixed MuJoCo body frames are mapped to their
nearest articulated parent plus a fixed transform. A randomized FK comparison
checks these transforms against the original MuJoCo geometry on each cold model key; identical verified models are reused within a process. The solver has 32 coordinates (root XYZ and hinges), while
the source-calibrated pelvis rotation remains prescribed per frame.

`ik.py` constructs the V8 pose residuals using differentiable frax FK:
root translation, torso axes, limb directions, foot targets, source-derived
plant anchors, actual convex mesh floor clearance, hand axes, elbow anatomy,
wrist preferences, uncertain extra-body height cues, and temporal terms.
The residual cost and its gradient are checked against V8's analytic MuJoCo
objective. This is objective compatibility, not identical optimizer output.
That parity test applies to the shared base objective; the optional 0.2.0 knee
term has its own cost/gradient difference check.

frax `ResidualTask` and `DiffIKSolver` solve velocity-level elastic QPs.
The full trajectory is compiled as a JAX scan. First-frame initialization
compares three arm branches, then each frame takes a fixed number of damped
half steps. Bounds combine anatomical position limits and an adjacent-frame
speed box referenced to the previous output pose throughout all inner steps.
Root rotation has no optimization degrees of freedom. Elastic constraints
are not treated as guaranteed: nonconvergence and final errors are audited.

The frax stage's floor and contact terms are soft residuals, not certified
nonpenetration constraints. The hybrid path additionally runs the original
V8 support/trajectory/body-contact/collision modules. This retains their
failure and rollback behavior. Neither mode solves dynamic balance or forces.

Version 0.4.0 adds sequential same-process `batch` conversion with bounded model
and compiled-kernel caches. Asset/config/geometry/shape/fps keys invalidate reuse;
first pose and body-contact tolerance are dynamic inputs. A changed-initial-pose
cached solve is tested against an uncached solve. No motion outputs are cached.
Long pose windows are split with overlap; only internal joins pin boundary frames.

Default fast work budget: at most four pose windows with six iterations each;
at most three V8 repair windows with total attempted frames bounded by
max(100, round(N*0.25)). Failed attempts consume budget. Full-clip escalation is
disabled; explicit hybrid retains complete V8 postprocessing. Existing acceptance
checks and final collision recovery remain. This is not a hard wall-clock cap.
Budget exhaustion preserves review failures rather than relaxing quality gates.

The upstream multi-device scheduler and automatic batch resume are not implemented.
Private data/results live under ignored regression_local/. See FRAX_040_REGRESSION.md.

Version 0.4.1 distinguishes foot and body-floor residuals before routing. Serious
body-floor defects take priority within the same frame/window budget. Padding
is reduced before dropping a long window; an oversized defect span is cropped
to its largest residual mass. Full-clip fallback remains disabled.

Fast body clearance injects a minimum-change objective into the existing contact
solver. All existing geometry, planted-foot, joint and speed checks still apply;
a per-frame lower corridor implements the already-required knee-branch check
inside the solve. Initial IK retains source/body-height cues. This cleanup does
not refit those cues or establish force/XY body-contact constraints. The optional
objective factory defaults to the original V8 Objective for hybrid/v8; the formal
sibling repository was not edited in this round.

`fast_collision.py` preserves successful normal repairs. Only primal-infeasible
windows receive incremental displacement recovery (8 mm demanded per step),
limited to two extra attempts / 100 attempted frames total. The existing 24-step
window limit and final exact-geometry/float32/interpolation/velocity checks stay.
Supported-arm failures are not retried. Experimental support-window splitting
and increased torso weights were not adopted; see FRAX_041_VALIDATION.md.

Version 0.5.0 adds explicit `batch --mode auto`, separate from fixed-mode batch.
A spawned persistent worker preserves warm JAX kernels across clips. The parent
runs all frax cases first, then only unresolved cases through hybrid and V8;
it owns per-stage/total deadlines and kills only its worker on timeout.
Successful artifacts are hash-checked before entering accepted.json. Incomplete
exports are removed on timeout/crash/interruption; previews/logs remain.
The manifest fingerprint pins the exact bytes loaded at startup.

All modes now use the continuous upper-arm/hand/elbow gate, including V8 and
legacy ablations. Historical raw V8 grades therefore need common-gate comparison.
Hybrid reuse validates source/code/assets/runtime/solver identity, saved artifact
hashes, prepared-target digest, joint ordering, root rotations and array schema.
Only pre-refinement x and solver records are reused. Any mismatch computes a
fresh frax initial trajectory; final V8 remains independent. Existing fixed-mode
batch keeps its soft stop-before-next-clip budget. Automatic resume is not added.
