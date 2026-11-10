"""CLI and independently identified, quality-gated motion outputs."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(tmp, path)


_FROZEN_IDENTITY=None

def static_identity():
    from .paths import ASSETS
    from . import __version__
    root = Path(__file__).parent
    files = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in (".py", ".json", ".bvh")}
    model = {str(p.relative_to(ASSETS)): sha(p) for p in sorted(ASSETS.rglob("*")) if p.is_file()}
    import frax
    frax_root = Path(frax.__file__).parent
    frax_hashes = {str(p.relative_to(frax_root)): sha(p) for p in sorted(frax_root.rglob("*.py"))}
    return dict(converter="dex_frax", version=__version__, implementation=files,
                frax_implementation=frax_hashes, assets=model,
                runtime={k: importlib.metadata.version(k) for k in ("frax", "jax", "jaxlib", "elastiqp", "numpy", "scipy", "mujoco", "mink", "osqp", "qpsolvers")})


def identity(args):
    import copy
    info=copy.deepcopy(_FROZEN_IDENTITY) if _FROZEN_IDENTITY is not None else static_identity()
    info.update(mode=args.mode,source=args.source,fps=50.,iterations=args.iterations,
                initial_iterations=args.initial_iterations,max_frames=args.max_frames,
                input_sha256=sha(args.input),legacy_frax=getattr(args,'legacy_frax',False),
                ik_block_frames=int(os.environ.get('DEX_FRAX_IK_BLOCK_FRAMES','128')))
    from .contract import conversion_contract
    info['conversion_contract']=conversion_contract(args)
    info['ik_block_frames']=info['conversion_contract']['ik_block_frames']
    if args.source == 'smplx':
        from .smplx_cli import preparation_identity
        info['smplx_input'] = preparation_identity(args)
    info["fingerprint"] = hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()
    return info


def export_training(*args):
    from .acceptance import export_training as export
    return export(*args)


def convert(args):
    if args.mode not in ('frax','v8'):
        raise ValueError('Conversion mode must be frax or v8; hybrid has been removed')
    if getattr(args,'reuse_initial_from',None) is not None or getattr(args,'require_initial_reuse',False):
        raise ValueError('Stages must solve independently; initial trajectory reuse is not supported')
    if getattr(args,'source',None)=='smplx' and (getattr(args,'bvh_units','cm')!='cm' or getattr(args,'bvh_translations','absolute')!='absolute'):
        raise ValueError('BVH conventions do not apply to SMPL-X')
    start = time.perf_counter()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "run.json", dict(status="running", input=str(Path(args.input).resolve()), mode=args.mode))
    try:
        from .v8.convert import convert as run
        if getattr(args,"ik_block_frames",None) is not None:
            os.environ["DEX_FRAX_IK_BLOCK_FRAMES"]=str(args.ik_block_frames)
        if args.source == 'smplx':
            from .smplx_cli import prepare
            prepare(args, out)
        ident = identity(args)
        atomic_json(out / "identity.json", ident)
        solve = None
        if args.mode != "v8":
            from .ik import FraxIK
            solve = FraxIK(args.iterations, args.initial_iterations,
                           improved=False if getattr(args,'profile',None) in ('fast','refine') else not getattr(args,'legacy_frax',False), assets=ident['assets'])
        if getattr(args,'profile',None)=='refine':
            from .pelvis_priority import PelvisPriorityIK
            solve=PelvisPriorityIK(args.iterations,args.initial_iterations)
        continuation=getattr(args,'continue_from',None)
        if continuation is not None:
            from .refine_resume import locate,verify,Initial
            continuation=locate(continuation);state=verify(continuation,ident,recheck=getattr(args,'resume_recheck',False));solve=Initial(continuation)
            if state.get('from_fast'):
                from .pelvis_priority import PelvisPriorityIK
                solve=PelvisPriorityIK(args.iterations,args.initial_iterations,baseline_solver=solve)
        pkl = out / "motion.pkl"
        meta = run(args.input, pkl, source=args.source, fps=50., max_frames=args.max_frames,
                   initial_solver=solve, refine=args.mode != "frax",
                   prepared_input=getattr(args,"smplx_prepared",None),
                   unified_repair=getattr(args,"unified_repair",False),repair_similarity=getattr(args,"repair_similarity","soft"), profile=getattr(args,"profile",None), repair_budget=getattr(args,"repair_budget",None),continue_from=continuation,resume_recheck=getattr(args,'resume_recheck',False),allow_local_slowdown=getattr(args,'allow_local_slowdown',False),bvh_units=getattr(args,'bvh_units','cm'),bvh_translations=getattr(args,'bvh_translations','absolute'))
        from .acceptance import evaluate
        from .resource_events import phase
        result = meta.pop('_unified_quality',None)
        if result is None:result = evaluate(pkl, meta, solve, args.mode)
        result.update(schema='dex_motion_quality_v1', backend='dex-retarget', solver_mode=args.mode,
                      scope='kinematic_pose_contact_geometry', model_sha256=meta['model_sha256'],
                      input_sha256=meta['input_sha256'], output_sha256=sha(pkl))
        if args.source == 'smplx':
            result['input_report'] = ident['smplx_input']['parameters']
        result['profile']=getattr(args,'profile',None)
        flags = result["flags"]
        atomic_json(out / "quality.json", result)
        phase('export')
        if not flags:
            export_training(pkl, out / "training.npz", ident)
        from .pipeline_explain import write_stage_report
        write_stage_report(out, meta, result, ident, dict(status='complete', mode=args.mode, fps=50.,
            pipeline='unified' if getattr(args,'unified_repair',False) else 'reference',
            training_npz='training.npz' if not flags else None))
        files = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "run.json"}
        record = dict(status="complete", quality=result["status"], acceptance_policy="common_pose_geometry_v1", mode=args.mode, frames=meta["output_frames"],
                      fps=50., total_wall_s=time.perf_counter() - start, stage_times_s=meta["stage_times_s"],
                      frax=meta["frax"], flags=result["flags"], fingerprint=ident["fingerprint"], files=files,
                      training_npz="training.npz" if not flags else None)
        record['profile']=getattr(args,'profile',None)
        record['pipeline']='unified' if getattr(args,'unified_repair',False) else 'reference'
        record['unified_repair']=meta.get('unified_repair')
        recovery=meta.get('self_collision',{}).get('graduated_repair',{})
        record['collision_repair_exhausted']=bool('arm_self_collision_unresolved' in flags and recovery.get('all_unresolved_tried_all_tiers',False))
        record['collision_repair_outcome']=recovery.get('outcome','not_attempted')
        atomic_json(out / "run.json", record)
        print(json.dumps({k: record[k] for k in ("status", "quality", "frames", "total_wall_s", "flags")}, ensure_ascii=False, indent=2))
        return record
    except BaseException as exc:
        atomic_json(out / "run.json", dict(status="failed", error=repr(exc), error_type=type(exc).__name__, elapsed_s=time.perf_counter() - start))
        raise


def single_auto(args,worker_factory=None):
    """Default unified backbone, or explicitly selected two-stage reference."""
    if args.source == 'smplx':
        from .smplx_cli import single_auto as smplx_auto
        return smplx_auto(args, worker_factory)
    import tempfile
    from . import cascade
    from .resources import output_frames
    source=Path(args.input).resolve();frames=output_frames(source)
    contents=dict(cases=[dict(id='motion',input=str(source),input_sha256=sha(source),source=args.source)])
    with tempfile.TemporaryDirectory(prefix='dex-single-') as scratch:
        manifest=Path(scratch)/'manifest.json';atomic_json(manifest,contents)
        options=argparse.Namespace(**vars(args));options.manifest=manifest;options.single_pipeline=True
        options.budget=getattr(args,'budget',86400.)
        options.frax_timeout=args.budget if getattr(args,'profile',None)=='refine' else max(900.,300.+.5*frames)
        options.v8_timeout=max(1800.,600.+1.5*frames)
        records=cascade.run(options,worker_factory or cascade.Worker)
    out=Path(args.output).resolve();batch=json.loads((out/'batch.json').read_text());record=records[0]
    atomic_json(out/'run.json',dict(status=batch['status'],quality=record['status'],mode='auto',
        selected_mode=record['selected_mode'],selected_output=record['selected_output'],
        training_npz=record['training_npz'],total_wall_s=batch['elapsed_s']))
    return record


def batch(args):
    """One process, separate checked outputs; cached IK never caches motions."""
    import contextlib
    manifest_path=Path(args.manifest).resolve()
    from .cascade import load_manifest
    cases=load_manifest(manifest_path)
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();records=[]
    for row in cases:
        record=dict(case=row['id'],status='budget_not_run')
        if time.perf_counter()-started < args.budget:
            case=argparse.Namespace(input=manifest_path.parent/row['input'],output=out/row['id'],
                source=row['source'],mode=args.mode,iterations=args.iterations,
                initial_iterations=args.initial_iterations,max_frames=None,legacy_frax=False,bvh_units=getattr(args,'bvh_units','cm'),bvh_translations=getattr(args,'bvh_translations','absolute'))
            try:
                with (out/(row['id']+'.log')).open('w') as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                    result=convert(case)
                record.update(status='complete',quality=result['quality'],flags=result['flags'],
                    total_wall_s=result['total_wall_s'],cache=result['frax'].get('runtime_cache') if result['frax'] else None)
            except Exception as exc:
                record.update(status='failed',error=repr(exc))
                from .ik import clear_runtime_cache
                clear_runtime_cache()
        records.append(record)
        atomic_json(out/'batch.json',dict(manifest_sha256=sha(manifest_path),mode=args.mode,
            elapsed_s=time.perf_counter()-started,records=records))
        print(row['id'],record['status'],record.get('quality'),flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description="Dex retargeting: complete fast conversion or window-based refine")
    sub = parser.add_subparsers(dest="command", required=True)
    conv = sub.add_parser("convert")
    conv.add_argument("input", type=Path)
    conv.add_argument("output", type=Path, help="New output directory; existing directories are never overwritten")
    conv.add_argument("--source", choices=["auto", "lafan", "soma", "mocap", "smplx"], default="auto")
    conv.add_argument('--bvh-units',choices=['cm','m','mm'],default='cm',help='BVH length units; default cm, never inferred')
    conv.add_argument('--bvh-translations',choices=['absolute','offset'],default='absolute',help='BVH position channels: absolute values or deltas added to OFFSET')
    conv.add_argument("--mode", choices=["fast", "refine"], default="fast")
    conv.set_defaults(repair_similarity='soft')
    conv.add_argument('--continue-from',type=Path,help='Refine only: continue a checked checkpoint into a new directory without rerunning IK')
    conv.add_argument('--recheck-from',type=Path,help='Refine only: after code updates, verify source/model and artifacts, freshly audit, preserve motion/reference, rebuild repair queue')
    conv.add_argument('--allow-local-slowdown',action='store_true',help='Refine only: slow over-fast root intervals and save original source-time mapping')
    conv.add_argument('--repair-budget',type=float,help='Optional refine wall-time limit; checkpoint pending windows instead of declaring them unsolvable')
    conv.add_argument('--budget',type=float,default=86400.,help='Total seconds including preparation, IK, repair and final audit')
    conv.add_argument("--iterations", type=int, default=8)
    conv.add_argument("--initial-iterations", type=int, default=100)
    conv.add_argument("--max-frames", type=int)
    from .smplx_cli import add_options
    add_options(conv)
    checking=sub.add_parser('check-input',help='Read-only validation of SMPL-X parameters; no geometry or IK')
    checking.add_argument('input',type=Path)
    add_options(checking)
    batching=sub.add_parser('batch',help='Run a hash-pinned manifest with resource-guarded CPU workers and independent solvers')
    batching.add_argument('manifest',type=Path)
    batching.add_argument('output',type=Path)
    batching.add_argument('--bvh-units',choices=['cm','m','mm'],default='cm',help='BVH length units; default cm, never inferred')
    batching.add_argument('--bvh-translations',choices=['absolute','offset'],default='absolute',help='BVH position channels: absolute values or deltas added to OFFSET')
    batching.add_argument('--mode',choices=['fast','refine'],default='fast')
    batching.set_defaults(repair_similarity='soft')
    batching.add_argument('--iterations',type=int,default=8)
    batching.add_argument('--initial-iterations',type=int,default=100)
    batching.add_argument('--allow-local-slowdown',action='store_true',help='Refine only: allow necessary local root slowdown with a source-time map')
    batching.add_argument('--repair-budget',type=float,help='Optional per-motion refine budget; otherwise visit every problem window')
    batching.add_argument('--budget',type=float,default=1800.,help='Total seconds; stop owned workers when exhausted in any batch mode')
    batching.add_argument('--workers',default='auto',help='Maximum worker slots; memory admission may use fewer')
    batching.add_argument('--resource-mode',choices=['dynamic','manual'],default='dynamic',help='Dynamic host-load admission by default; manual retains fixed resource settings')
    batching.add_argument('--memory-target-percent',type=float,default=90.,help='Dynamic mode host used-memory ceiling, including other programs and the system')
    batching.add_argument('--memory-reserve-gib',type=float,help='Optional additional minimum host memory reserve; manual default is 10 GiB')
    batching.add_argument('--memory-limit-gib',type=float,help='Optional lower Linux cgroup hard limit, also enables containment for one worker')
    batching.add_argument('--disk-reserve-gib',type=float,default=80.)
    batching.add_argument('--worker-peak-gib',type=float,default=3.75,help='Frax cold baseline; excess over resident budget is reserved once for the compiler')
    batching.add_argument('--worker-resident-gib',type=float,default=1.,help='Frax resident baseline, plus per-frame memory')
    batching.add_argument('--frame-memory-kib',type=float,default=32.,help='Initial per-output-frame memory budget, hot-reloadable per stage')
    batching.add_argument('--resource-policy',type=Path,help='Live resource-only JSON; defaults to output/resource_policy.json, checked every 2 seconds')
    batching.add_argument('--no-memory-learning',action='store_true',help='Use configured resident estimates without learned completed-stage profiles')
    batching.add_argument('--ik-block-frames',type=int,default=128,help='Fixed IK execution block length, with carried state across blocks')
    batching.add_argument('--resume',action='store_true',help='Verify and retain completed attempts in an interrupted batch; retry unfinished stages')
    batching.add_argument('--resource-wait',type=float,default=60.,help='Maximum seconds with no active work and insufficient resources')
    batching.add_argument('--frax-timeout',type=float,help='Per-motion worker deadline; partial refined output is retained')
    batching.set_defaults(v8_timeout=None)
    explaining = sub.add_parser('explain', help='Explain existing retarget records without IK, re-audit or changing old outputs')
    explaining.add_argument('result', type=Path, help='One motion result directory or robot PKL/NPZ with JSON sidecars')
    explaining.add_argument('--output', type=Path, help='New directory for pipeline.json and pipeline.md; default prints only')
    explaining.add_argument('--format', choices=['markdown','json'], default='markdown', help='Console format')
    preview = sub.add_parser("preview")
    preview.add_argument("motions", nargs="+", type=Path)
    preview.add_argument("--output", type=Path, required=True)
    preview.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()
    if args.command in ('convert','batch'):
        import math
        if getattr(args,'source',None)=='smplx' and (args.bvh_units!='cm' or args.bvh_translations!='absolute'):parser.error('BVH unit/translation options do not apply to SMPL-X; use --units for SMPL-X')
        if args.repair_budget is not None and (not math.isfinite(args.repair_budget) or args.repair_budget<=0):parser.error('repair-budget must be finite and positive')
        # Normalize only after public parsing: no public V8/auto/two-stage entry.
        if getattr(args,'recheck_from',None) is not None:
            if args.mode!='refine' or getattr(args,'continue_from',None) is not None:parser.error('recheck-from requires refine and excludes continue-from')
            args.continue_from=args.recheck_from;args.resume_recheck=True
        if getattr(args,'continue_from',None) is not None and args.mode!='refine':parser.error('continue-from requires refine')
        if args.allow_local_slowdown and args.mode!='refine':parser.error('allow-local-slowdown requires refine')
        args.profile=args.mode;args.mode='auto';args.pipeline='unified'
        args.legacy_frax=True;args.v8_worker_peak_gib=1.
        args.unified_repair=args.profile=='refine'

    if args.command == 'explain':
        from .pipeline_explain import explain, render, write_report
        try:
            report = explain(args.result)
            if args.output is not None:
                args.output.mkdir(parents=True, exist_ok=False)
                write_report(report, args.output)
                print(str(args.output.resolve()))
            else:
                print(json.dumps(report, ensure_ascii=False, indent=2) if args.format=='json' else render(report))
        except (OSError, ValueError) as exc:
            parser.exit(2, 'Cannot explain result: '+str(exc)+'\n')
        return
    if args.command == 'check-input':
        from .smplx_cli import check_input
        parser.exit(check_input(args))
    if args.command == "convert":
        import math
        if args.source == 'auto' and args.input.suffix.lower() in ('.npz','.pt','.pkl'):
            args.source = 'smplx'
        if args.source != 'smplx' and args.grounding != 'off':
            parser.error('--grounding applies only to SMPL-X input')
        if min(args.iterations, args.initial_iterations) < 1 or (args.max_frames is not None and args.max_frames < 3):
            parser.error("Iteration counts must be positive; max-frames must be >= 3")
        if not math.isfinite(args.budget) or args.budget<=0:parser.error('Budget must be finite and positive')
        from .input_errors import MotionInputError
        try:
            if args.mode=='auto':single_auto(args)
            else:convert(args)
        except MotionInputError as exc:
            parser.exit(2,'Motion input error: '+str(exc)+'\n')
    elif args.command == 'batch':
        import math
        values=[args.iterations,args.initial_iterations,args.budget,args.memory_target_percent,args.disk_reserve_gib,args.worker_peak_gib,args.worker_resident_gib,args.frame_memory_kib,args.ik_block_frames,args.v8_worker_peak_gib,args.resource_wait]+[getattr(args,m+'_timeout') for m in ('frax','v8') if getattr(args,m+'_timeout') is not None]+([args.memory_limit_gib] if args.memory_limit_gib is not None else [])+([args.memory_reserve_gib] if args.memory_reserve_gib is not None else [])
        if any(not math.isfinite(v) or v<=0 for v in values):
            parser.error('Batch iteration counts and budgets must be finite and positive')
        if args.worker_resident_gib>args.worker_peak_gib:parser.error('Resident memory cannot exceed cold-worker peak')
        if not 50<=args.memory_target_percent<=95:parser.error('Memory target percent must be between 50 and 95')
        from .resources import contain_linux
        if contain_linux(args) is not None:return
        from .managed_batch import run
        run(args)
    else:
        from .preview import render
        render(args.motions, args.output, args.fps)


if __name__ == "__main__":
    main()
