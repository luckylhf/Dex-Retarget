"""Two-stage batch cascade with live resource admission and one compiler."""
import gc
import heapq
from collections import Counter
import copy
import json
import multiprocessing
import os
from pathlib import Path
import signal
import time

from . import resources as res
from .cascade import Worker,_worker,load_manifest,accepted_artifact,STAGES
from .cli import atomic_json,sha,static_identity
from .resource_policy import ResourcePolicy
from .dispatch import duration_seconds,route,priority,deduplicate
from .memory_learning import MemoryLearning
from .memory_queue import MemoryQueue,pressure,victim as memory_victim
from .stage_memory import StageMemory
from .contract import DEFAULT_FOOT_CONTACT, conversion_contract
from .dispatch import pipeline_stages,use_unified

def child(connection,options):
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[key]='1'
    os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_ENABLE_X64='1',
                      DEX_FRAX_KERNEL_CACHE_SIZE='2',DEX_FRAX_COMPILE_LOCK_DIR=options['compile_dir'],
                      JAX_COMPILATION_CACHE_DIR=options['persistent_cache_dir'],JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS='0',
                      DEX_FRAX_IK_BLOCK_FRAMES=str(options['block_frames']))
    if options.get('cpu') is not None and hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,{options['cpu']})
    if options.get('dynamic'):res.lower_priority()
    from . import cli,model_cache
    cli._FROZEN_IDENTITY=options['identity']
    _worker(connection,initialize=lambda:model_cache.enable(options['model']))

class Slot(Worker):
    supports_stage_memory=True
    def __init__(self,options):
        super().__init__();self.options=options;self.jobs=0;self.started=0.;self.timeout=0.;self.telemetry={};self.phase_peaks={}
        self.memory_request=None;self.wait_started=None;self.waited_s=0.
    @property
    def pid(self):return self.process.pid if self.process is not None else None
    def start(self,job,timeout):
        if self.process is None or not self.process.is_alive():
            self.close();ctx=multiprocessing.get_context('spawn');self.connection,child_conn=ctx.Pipe()
            self.process=ctx.Process(target=child,args=(child_conn,self.options),daemon=True)
            self.process.start();child_conn.close()
        self.started=time.monotonic();self.timeout=timeout;self.connection.send(job)
        self.memory_request=None;self.wait_started=None;self.waited_s=0.
        self.phase_peaks={};self.telemetry=dict(phase='starting',stage='loading',compile_possible=job['arguments']['mode']!='v8' and not job['arguments'].get('require_initial_reuse',False),cache_state=self.telemetry.get('cache_state','pure'))
    def close(self):
        super().close()
        self.jobs=0;self.telemetry={};self.phase_peaks={}
        self.memory_request=None;self.wait_started=None;self.waited_s=0.

    def memory_wait_s(self):
        return self.waited_s+(time.monotonic()-self.wait_started if self.wait_started is not None else 0.)

    def grant_memory(self):
        self.connection.send(dict(memory_grant=True))
        if self.wait_started is not None:self.waited_s+=time.monotonic()-self.wait_started
        self.memory_request=None;self.wait_started=None

    def poll(self):
        while self.connection.poll(0):
            try:
                result=self.connection.recv()
                if 'memory_request' in result:
                    self.memory_request=result['memory_request'];self.wait_started=time.monotonic();continue
                if 'resource_event' in result:
                    event=result['resource_event'];key=(event['phase'] if event['phase'] in ('ik_prepare','ik_compile','waiting_compile') else event.get('stage',event['phase']))+'|'+event['cache_state']
                    self.phase_peaks[key]=max(self.phase_peaks.get(key,0),event['pss_bytes']);self.telemetry=event;continue
                result['worker_wall_s']=time.monotonic()-self.started;result['memory_wait_s']=self.memory_wait_s();self.jobs+=1;return result
            except (EOFError,OSError):
                self.process.join(.1);code=self.process.exitcode;self.close()
                return dict(status='failed',error_type='WorkerExit',exit_code=code,error='Owned worker exited before completing the export')
        if time.monotonic()-self.started-self.memory_wait_s()>=self.timeout:
            self.close();return dict(status='timeout',elapsed_s=time.monotonic()-self.started)
        return None

def timeout_for(args,mode,frames):
    explicit=getattr(args,mode+'_timeout',None)
    if explicit is None and getattr(args,'profile',None)=='refine':return args.budget
    return explicit if explicit is not None else (max(900.,300.+.5*frames) if mode=='frax' else max(1800.,600.+1.5*frames))

def finish(record,response,mode,folder,out,allow_collision_v8=False):
    attempt={**response,'mode':mode,'output':str(folder.relative_to(out))}
    try:selection=accepted_artifact(folder,attempt)
    except (OSError,ValueError,KeyError) as error:
        attempt.update(status='failed',error_type='InvalidExport',error=str(error));selection=None
    if attempt['status']!='complete':
        folder.mkdir(parents=True,exist_ok=True);(folder/'training.npz').unlink(missing_ok=True)
        atomic_json(folder/'run.json',{k:v for k,v in attempt.items() if k!='result'})
    if selection:
        record.update(status='accepted',selected_mode=mode,selected_output=str(folder.relative_to(out)),training_npz=str((folder/'training.npz').relative_to(out)),**selection)
    elif attempt['status']=='input_error':record['status']='input_error'
    else:record['status']='pending'
    result=attempt.pop('result',None)
    if result:
        attempt['conversion_wall_s']=result['total_wall_s'];attempt['initial_reuse']=(result.get('frax') or {}).get('initial_reuse')
        attempt['collision_repair_exhausted']=result.get('collision_repair_exhausted',False)
    record['attempts'].append(attempt)
    return attempt

def run(args,slot_factory=Slot,monitor=res.memory):
    # Shared lease is held across prepare, execution and final identity verification.
    with res.batch_lock():return _run(args,slot_factory,monitor)

def scheduler_update(original,snapshot):
    """An explicit resume may change only these resource/scheduling modules.

    Never ignore solver, acceptance, exporter, CLI, asset or dependency changes.
    Preserve original per-artifact identities instead of relabelling outputs.
    """
    if {k:v for k,v in original.items() if k!='implementation'}!={k:v for k,v in snapshot.items() if k!='implementation'}:
        raise ValueError('Resume runtime, assets or solver implementation changed')
    before=original.get('implementation',{});after=snapshot.get('implementation',{})
    changed={k for k in before.keys()|after.keys() if before.get(k)!=after.get(k)}
    if not changed.issubset({'managed_batch.py','resources.py'}):
        raise ValueError('Resume runtime, assets or solver implementation changed')
    return {k:dict(before=before.get(k),after=after.get(k)) for k in sorted(changed)}


def enable_scheduler_initial_reuse():
    """Worker-local adapter for an explicitly authorized scheduler-only resume.

    The conversion/export identity remains current. Only the reader's comparison
    identity is adapted after the same two-file check used by restore; all source,
    model, solver, parameter, target and artifact checks still run unchanged.
    """
    from . import initial_reuse
    base=initial_reuse.ReusedInitial
    if getattr(base,'scheduler_resume_adapter',False):return
    class SchedulerInitial(base):
        scheduler_resume_adapter=True
        def __init__(self,directory,identity,fallback,strict=False):
            super().__init__(directory,identity,fallback,strict)
            self.scheduler_changes={}
            try:
                previous=json.loads((Path(directory)/'identity.json').read_text())
                keys=('converter','version','implementation','frax_implementation','assets','runtime')
                changes=scheduler_update({k:previous[k] for k in keys},{k:identity[k] for k in keys})
            except (OSError,ValueError,KeyError,TypeError):
                return  # Original strict reader rejects the mismatch.
            if changes:
                self.identity=copy.deepcopy(identity)
                self.identity['implementation']=copy.deepcopy(previous['implementation'])
                self.scheduler_changes=changes
        def __call__(self,*args,**kwargs):
            result=super().__call__(*args,**kwargs)
            if self.scheduler_changes and self.report.get('initial_reuse',{}).get('used'):
                self.report['initial_reuse']['scheduler_compatibility']=self.scheduler_changes
            return result
    initial_reuse.ReusedInitial=SchedulerInitial

def restore(args,out,content,cases,snapshot):
    """Resume only verified attempts; solver or asset changes require a new run."""
    previous=json.loads((out/'batch.json').read_text())
    if previous['manifest_sha256']!=sha_bytes(content) or previous['mode']!=args.mode:
        raise ValueError('Resume manifest or mode changed')
    original=json.loads((out/'batch_identity.json').read_text())
    changes={}
    if original!=snapshot:
        if os.environ.get('DEX_FRAX_ALLOW_SCHEDULER_RESUME')!='1':
            raise ValueError('Resume runtime, assets or solver implementation changed')
        changes=scheduler_update(original,snapshot)
    contract_path=out/'conversion_contract.json'
    if not contract_path.exists() or json.loads(contract_path.read_text())!=conversion_contract(args):
        raise ValueError('Resume conversion contract changed or missing; start a new batch')
    records=copy.deepcopy(previous['records'])
    if [r['case'] for r in records]!=[c['id'] for c in cases]:raise ValueError('Resume case order changed')
    for record in records:
        for attempt in record['attempts']:
            if attempt['status']!='complete':continue
            folder=(out/attempt['output']).resolve()
            if not folder.is_relative_to(out):raise ValueError('Resume output escaped batch directory')
            run=json.loads((folder/'run.json').read_text())
            if run.get('status')!='complete':raise ValueError('Resume completed attempt is inconsistent')
            for name,h in run['files'].items():
                path=(folder/name).resolve()
                if not path.is_relative_to(folder) or sha(path)!=h:raise ValueError('Resume artifact changed: '+name)
            ident=json.loads((folder/'identity.json').read_text())
            if changes:
                scheduler_update({k:ident[k] for k in snapshot},snapshot)
            if ident.get('conversion_contract')!=conversion_contract(args):
                raise ValueError('Resume conversion parameters changed')
        if record['status']=='accepted':
            attempt=next(a for a in reversed(record['attempts']) if a['mode']==record['selected_mode'])
            if not accepted_artifact(out/attempt['output'],dict(attempt,result=json.loads((out/attempt['output']/'run.json').read_text()))):
                raise ValueError('Resume accepted artifact did not validate')
        elif record['status'] not in ('input_error','review','failed'):record['status']='pending'
    atomic_json(out/('batch.before_resume.'+str(time.time_ns())+'.json'),previous)
    if changes:
        atomic_json(out/('scheduler_resume.'+str(time.time_ns())+'.json'),
                    dict(before=original,after=snapshot,changed_files=changes,
                         completed_artifacts_verified=True,original_artifacts_unchanged=True))
    return records,previous

def _run(args,slot_factory,monitor):
    priority_lowered=res.lower_priority() if res.dynamic_mode(args) else None
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
    os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_ENABLE_X64='1')
    content=Path(args.manifest).read_bytes();cases,aliases=deduplicate(load_manifest(args.manifest,content))
    nworkers=res.workers(args.workers);out=Path(args.output).resolve();resume=getattr(args,'resume',False)
    if resume and not (out/'batch.json').is_file():raise ValueError('Resume requires an existing batch.json')
    out.mkdir(parents=True,exist_ok=resume)
    atomic_json(out/'source_aliases.json',dict(policy='full_bvh_sha256',duplicates=aliases))
    admission=ResourcePolicy(args,out,nworkers)
    reserve=res.host_reserve(args,monitor());disk_floor=int(args.disk_reserve_gib*res.GIB)
    dynamic=res.dynamic_mode(args)
    from .dynamic_resources import CpuAdmission
    cpu_admission=CpuAdmission(os.sched_getaffinity(0) if hasattr(os,'sched_getaffinity') else range(res.cpu_count()),nworkers) if dynamic else None
    started=time.monotonic();last_checkpoint=started;last_heartbeat=started;loop_seconds=0.;usage={};state='running';phase='prepare';active={};slots=[];last=0.;stopped=False;wait_since=None;last_reason=None
    records=[dict(case=c['id'],status='pending',attempts=[],selected_mode=None,selected_output=None,training_npz=None) for c in cases]
    snapshot=static_identity();previous=None
    learning=MemoryLearning(res.state_root(),dict(identity=snapshot,execution=dict(
        conversion=conversion_contract(args))),not getattr(args,'no_memory_learning',False))
    if resume:
        records,previous=restore(args,out,content,cases,snapshot)
        if not learning.profiles:
            learning.profiles=copy.deepcopy(previous.get('resources',{}).get('memory_learning',{}).get('profiles',{}))
    frames={};durations={};routes={};snap=monitor();priority_wait=None;observed=dict(max_active_workers=0,worker_pss_peak_bytes=0,total_worker_pss_peak_bytes=0,scope_usage_peak_bytes=0)
    queue=[]
    memory_queue=MemoryQueue(nworkers,(previous or {}).get('resources',{}).get('memory_queue'))
    stage_memory=StageMemory() if getattr(args,'profile',None) in ('fast','refine') else None
    oom_kills=snap.get('cgroup_oom_kills',0)
    policy=dict(workers=nworkers,jit_compilers=1,memory_reserve_gib=reserve/res.GIB,
        resource_mode='dynamic' if dynamic else 'manual',memory_target_percent=getattr(args,'memory_target_percent',None),priority_lowered=priority_lowered,
        disk_reserve_gib=args.disk_reserve_gib,worker_peak_base_gib=args.worker_peak_gib,
        v8_worker_peak_base_gib=getattr(args,'v8_worker_peak_gib',args.worker_peak_gib),
        frax_resident_base_gib=getattr(args,'worker_resident_gib',args.worker_peak_gib),
        compile_extra_gib=max(0,args.worker_peak_gib-getattr(args,'worker_resident_gib',args.worker_peak_gib)),
        peak_bytes_per_output_frame=admission.value['stages']['frax']['bytes_per_output_frame'],start_method='spawn',cgroup_limit_bytes=snap['cgroup_limit'],
        frax_peak_bytes_per_output_frame=admission.value['stages']['frax']['bytes_per_output_frame'],
        scheduling='frax_checked_local_repairs_v1' if use_unified(args) else 'frax_all_then_independent_v8_v6',stage_barriers=args.mode=='auto' and not use_unified(args),long_seconds=None,
        identity='frozen before workers, recomputed after workers',model='verified MJB, reused per spawn worker; no cross-process fork sharing')
    pool_stats=Counter()
    stop_path=out/'STOP'
    previous_handlers={}
    def on_signal(*_):
        nonlocal stopped
        stopped=True
    for sig in (signal.SIGINT,signal.SIGTERM):previous_handlers[sig]=signal.signal(sig,on_signal)

    def save(full=True):
        nonlocal last,last_checkpoint,last_heartbeat
        policy['worker_pool']=dict(policy='retain_warm_until_actual_pressure_v1',fixed_job_recycle=False,
            counters=dict(pool_stats),resident_idle=sum(s not in active and bool(s.pid) for s in slots),
            maximum_jobs_in_resident=max((getattr(s,'jobs',0) for s in slots),default=0))
        policy['resource_policy']=admission.report()
        policy['workers']=admission.value['workers']
        policy['memory_reserve_gib']=reserve/res.GIB
        policy['memory_learning']=learning.report()
        policy['memory_queue']=memory_queue.report()
        if stage_memory:policy['stage_memory']=stage_memory.report(active)
        if cpu_admission:
            measured=cpu_admission.report()
            policy['cpu_admission']=dict(measured,load_based_target_ignored=measured.get('target_workers'),
                target_workers=min(admission.value['workers'],nworkers),reason='full_available_cpu',
                external_load_control=False)
        policy.update(frax_resident_base_gib=admission.value['stages']['frax']['base_gib'],
            v8_worker_peak_base_gib=admission.value['stages']['v8']['base_gib'],
            compile_extra_gib=admission.value['compile_extra_gib'],
            worker_peak_base_gib=admission.value['stages']['frax']['base_gib']+admission.value['compile_extra_gib'],
            peak_bytes_per_output_frame=admission.value['stages']['frax']['bytes_per_output_frame'],
            frax_peak_bytes_per_output_frame=admission.value['stages']['frax']['bytes_per_output_frame'])
        measured=[usage.get(s,0) for s in slots]
        observed['max_active_workers']=max(observed['max_active_workers'],len(active))
        observed['worker_pss_peak_bytes']=max(observed['worker_pss_peak_bytes'],max(measured,default=0))
        observed['total_worker_pss_peak_bytes']=max(observed['total_worker_pss_peak_bytes'],sum(measured))
        if snap['cgroup_limit'] is not None:observed['scope_usage_peak_bytes']=max(observed['scope_usage_peak_bytes'],snap['cgroup_limit']-snap['cgroup_headroom'])
        now=time.monotonic()
        live=dict(schema=2,mode=args.mode,status=state,phase=phase,
            updated_at=time.time(),elapsed_s=now-started,
            accepted=sum(r['status']=='accepted' for r in records),
            counts=dict(Counter(r['status'] for r in records)),
            resources=policy,memory=snap,resource_wait_reason=last_reason,
            priority_wait=priority_wait,sampled_resource_peaks=observed,
            worker_pss_bytes=sum(measured),scheduler_loop_s=loop_seconds,
            checkpoint_interval_s=30.,checkpoint_age_s=now-last_checkpoint,
            queued=len(queue),
            runnable=sum(getattr(s,'memory_request',None) is None for s in active),
            waiting_memory=sum(getattr(s,'memory_request',None) is not None for s in active),
            active=[dict(case=records[v['index']]['case'],mode=v['mode'],pid=s.pid,
                estimated_peak_bytes=v['peak'],pss_bytes=usage.get(s,0),
                memory_state='waiting' if getattr(s,'memory_request',None) is not None else 'running',
                granted_peak_bytes=v.get('grant',v['peak']),completion_peak_bytes=v.get('completion_peak',v['peak']),
                requested_peak_bytes=v.get('requested_peak'),
                memory_wait_s=s.memory_wait_s() if hasattr(s,'memory_wait_s') else 0.,
                telemetry=getattr(s,'telemetry',{})) for s,v in active.items()])
        if full:
            accepted=[]
            for c,r in zip(cases,records):
                if r['status']=='accepted':accepted.append(dict(case=r['case'],input=c['input'],input_sha256=c['input_sha256'],mode=r['selected_mode'],output=r['selected_output'],training_npz=r['training_npz'],training_sha256=r['training_sha256'],fingerprint=r['fingerprint']))
            atomic_json(out/'accepted.json',dict(schema=1,path_base='batch output directory',motions=accepted))
            atomic_json(out/'batch.json',dict(live,
                stage_order=list(pipeline_stages(args)) if args.mode=='auto' else [args.mode],pipeline=getattr(args,'pipeline','unified'),
                unique_sources=len(cases),duplicate_aliases=len(aliases),
                acceptance_policy='common_pose_geometry_v1',manifest_sha256=sha_bytes(content),budget_s=args.budget,
                records=records,resumed=resume,previous_elapsed_s=previous.get('elapsed_s') if previous else None))
            last_checkpoint=time.monotonic()
            live['checkpoint_age_s']=0.
        atomic_json(out/'batch.running.json',live)
        if now-last_heartbeat>=30:
            done=sum(r['status'] in ('accepted','review') for r in records)
            print(f"[batch] running={live['runnable']} waiting_memory={live['waiting_memory']} queued={live['queued']} done={done}/{len(records)}",flush=True)
            last_heartbeat=now
        last=time.monotonic()

    def abort(reason):
        for slot,item in list(active.items()):
            slot.close();finish(records[item['index']],dict(status=reason),item['mode'],item['folder'],out)
        active.clear()

    save()
    try:
        for i,c in enumerate(cases):
            try:
                frames[i]=res.output_frames(c['input']);durations[i]=duration_seconds(c['input'])
                routes[i]=route(args.mode,durations[i],getattr(args,'single_pipeline',False),getattr(args,'pipeline','unified'))
                records[i].update(duration_s=float(durations[i]),route=list(routes[i]))
            except (OSError,ValueError) as error:records[i].update(status='input_error',error=str(error))
        if resume:atomic_json(out/('resume_identity.'+str(time.time_ns())+'.json'),snapshot)
        else:
            atomic_json(out/'batch_identity.json',snapshot)
            atomic_json(out/'conversion_contract.json',conversion_contract(args))
        from .model_cache import prepare
        model=prepare(snapshot['assets'],res.state_root()/'models');res.release_memory()
        cpus=res.cpu_order() if hasattr(os,'sched_getaffinity') else [None]*nworkers
        cache_dir=res.state_root()/'jax_cache';cache_dir.mkdir(parents=True,exist_ok=True,mode=0o700)
        options=dict(identity=snapshot,model=model,compile_dir=str(res.state_root()/'compiler'),persistent_cache_dir=str(cache_dir),block_frames=getattr(args,'ik_block_frames',128),dynamic=dynamic,
            scheduler_resume=bool(resume and os.environ.get('DEX_FRAX_ALLOW_SCHEDULER_RESUME')=='1'))
        queue=[];sequence=0
        def enqueue(index,mode):
            nonlocal sequence
            heapq.heappush(queue,(priority((index,mode),durations),sequence,index,mode));sequence+=1
        def defer(slot,item,reason):
            i=item['index'];mode=item['mode'];record=records[i]
            observed_peak=max(item.get('observed_peak',0),usage.get(slot,0)) or item['peak']
            per=admission.value['stages'][mode]['bytes_per_output_frame']
            memory_queue.defer(mode,cases[i]['source'],frames[i],per,observed_peak,len(active))
            slot.close()
            finish(record,dict(status='memory_retry',reason=reason),mode,item['folder'],out)
            record['memory_retries']=record.get('memory_retries',0)+1
            del active[slot];enqueue(i,mode)
            print(cases[i]['id'],mode,'memory_retry',reason,flush=True)
            save()
        completion_lengths={}
        for i,route_modes in routes.items():
            for mode in route_modes:
                key=(mode,cases[i]['source'])
                completion_lengths[key]=max(completion_lengths.get(key,0),frames[i])
        for i,r in enumerate(records):
            if r['status'] in ('accepted','input_error','review','failed') or r.get('escalation',{}).get('stop'):continue
            for mode in routes[i]:
                if not any(a['mode']==mode and a['status'] in ('complete','input_error') for a in r['attempts']):
                    enqueue(i,mode);break
        while queue or active:
            loop_started=time.monotonic()
            admission.reload()
            learning.enabled=admission.value.get('memory_learning',not getattr(args,'no_memory_learning',False))
            if stage_memory:
                for (mode,source),length in completion_lengths.items():
                    predicted=max(admission.completion_peak(mode,length),learning.predict(mode,source,'jax_hot',length,admission))
                    predicted=memory_queue.peak(mode,source,length,admission.value['stages'][mode]['bytes_per_output_frame'],predicted)
                    stage_memory.bound=max(stage_memory.bound,predicted)
                for item in active.values():item['completion_peak']=max(item['completion_peak'],stage_memory.bound)
            modes={v['mode'] for v in active.values()} or {entry[3] for entry in queue}
            phase=next(iter(modes)) if len(modes)==1 else 'mixed'
            snap=monitor()
            reserve=res.host_reserve(args,snap)
            # CPU load is telemetry only. Fill every permitted slot; the OS
            # schedules competing work. Host/cgroup memory admission still gates starts.
            if cpu_admission:cpu_admission.sample([os.getpid()]+[s.pid for s in slots])
            dispatch_limit=min(admission.value['workers'],nworkers,memory_queue.limit)
            if stopped or stop_path.exists():state='interrupted';abort('interrupted');break
            if time.monotonic()-started>=args.budget:state='budget_exhausted';abort('budget_exhausted');break
            free=res.shutil.disk_usage(out).free
            if free<disk_floor:state='resource_exhausted';last_reason='disk_reserve';abort('resource_exhausted');break
            for slot,item in list(active.items()):
                response=slot.poll()
                if response is not None:
                    if (response.get('error_type')=='WorkerExit' and response.get('exit_code')==-9
                            and snap.get('cgroup_oom_kills',0)>oom_kills):
                        defer(slot,item,'cgroup_oom_retry');continue
                    mode=item['mode'];i=item['index']
                    learning.observe(slot,item,usage.get(slot,0))
                    attempt=finish(records[i],response,mode,item['folder'],out,getattr(args,'v8_for_collision_only',False));del active[slot]
                    if attempt['status']=='complete':
                        learning.complete(slot,item,response,cases[i],frames[i],admission)
                        memory_queue.complete()
                    print(cases[i]['id'],mode,attempt['status'],attempt.get('quality'),flush=True)
                    if records[i]['status']=='pending':
                        next_stage=routes[i].index(mode)+1
                        if next_stage<len(routes[i]):enqueue(i,routes[i][next_stage])
                        else:records[i]['status']='review' if any(a.get('quality')=='review' for a in records[i]['attempts']) else 'failed'
                    atomic_json(item['folder'].parent/'selection.json',records[i])
            oom_kills=snap.get('cgroup_oom_kills',oom_kills)
            # Drain completions before the one O(workers) PSS sweep. Reuse this
            # sweep for every admission and report; never rescan all workers
            # once per newly filled slot (the old O(workers squared) hot path).
            usage={s:res.process_memory(s.pid) for s in slots}
            if stage_memory:stage_memory.elect(active,usage)
            for slot,item in active.items():
                learning.observe(slot,item,usage.get(slot,0))
                i=item['index'];mode=item['mode']
                item['observed_peak']=max(item.get('observed_peak',0),usage.get(slot,0),getattr(slot,'telemetry',{}).get('pss_bytes',0))
                # Refine can revisit audit after repair; reserve the complete
                # lifecycle rather than release a supposedly finished stage.
                estimate=learning.predict(mode,cases[i]['source'],getattr(slot,'telemetry',{}).get('cache_state','pure'),frames[i],admission,reuse=item.get('reuse',False))
                estimate=max(estimate,int(item['observed_peak']*1.1)+128*res.MIB)
                if stage_memory:
                    item['completion_peak']=max(item['completion_peak'],admission.completion_peak(mode,frames[i]),estimate)
                    item['peak']=item['grant']
                else:item['peak']=memory_queue.peak(mode,cases[i]['source'],frames[i],admission.value['stages'][mode]['bytes_per_output_frame'],estimate)
            snap=monitor()
            victim=res.reclaim_idle(slots,active,usage,snap,reserve)
            if victim is not None:
                victim.close();pool_stats['pressure_recycles']+=1
                # Both snapshot and PSS must be refreshed together next loop.
                continue
            critical,waiting=pressure(snap,reserve)
            if critical and active:
                last_reason='memory_pressure_requeue';priority_wait=None
                slot=memory_victim(active,usage,stage_memory.protected if stage_memory else None)
                defer(slot,active[slot],last_reason)
                # Never credit released pages until a fresh snapshot/PSS sweep.
                continue
            if waiting:
                last_reason='memory_pressure_wait';priority_wait=None
            # Workers request before allocating the next stage and before
            # acquiring shared locks. Existing growth has priority over starts.
            if stage_memory:
                shared=admission.shared([v['mode'] for v in active.values()])
                for slot,item in active.items():
                    request=getattr(slot,'memory_request',None)
                    if request is None:continue
                    i=item['index'];mode=item['mode'];stage=request.get('stage','loading')
                    estimate=learning.stage_peak(mode,cases[i]['source'],request.get('cache_state','pure'),frames[i],admission,stage)
                    estimate=max(estimate,learning.predict(mode,cases[i]['source'],request.get('cache_state','pure'),frames[i],admission),request.get('required_peak_bytes',0))
                    estimate=max(estimate,int(max(usage.get(slot,0),request.get('pss_bytes',0))*1.1)+128*res.MIB)
                    estimate=memory_queue.peak(mode,cases[i]['source'],frames[i],admission.value['stages'][mode]['bytes_per_output_frame'],estimate)
                    stage_memory.request(slot,item,estimate,stage,retained=usage.get(slot,request.get('pss_bytes',0)))
                for item in active.values():item['completion_peak']=max(item['completion_peak'],stage_memory.bound)
                for slot in stage_memory.ordered(active):
                    item=active[slot]
                    if not stage_memory.can_grant(slot,item,snap,reserve,active,usage,shared):continue
                    stage_memory.grant(slot,item)
                    measured=monitor()
                    for key in ('available','cgroup_headroom'):snap[key]=min(snap[key],measured[key])
                blocked=stage_memory.ordered(active)
                if blocked:
                    last_reason='stage_memory_wait';priority_wait=None
                    # A finished hot process still owns RAM. Reclaim it when
                    # it prevents the protected worker from receiving its grant.
                    owner=stage_memory.protected
                    if owner in blocked:
                        others=sum(max(0,p-u) for p,u in stage_memory.running(active,usage,owner))
                        idle=res.reclaim_idle(slots,active,usage,snap,reserve,owner,active[owner]['completion_peak'],shared+others)
                        if idle is not None:
                            idle.close();pool_stats['pressure_recycles']+=1;continue
                    # If estimates were violated and everyone is blocked,
                    # release one non-protected attempt instead of deadlocking.
                    if len(blocked)==len(active) and len(active)>1:
                        slot=memory_victim(active,usage,owner)
                        defer(slot,active[slot],'stage_budget_underestimate');continue
                waiting=waiting or bool(blocked)
            # Do not credit reclaimed pages again while using this sweep's PSS.
            burst_headroom={k:snap[k] for k in ('available','cgroup_headroom')}
            dispatch_limit=min(admission.value['workers'],nworkers,memory_queue.limit)
            if queue and not waiting and len(active)>=dispatch_limit:last_reason='cpu_capacity';priority_wait=None
            while queue and not waiting and len(active)<dispatch_limit:
                # Complete the entire Frax pass before any independent V8 work.
                if args.mode=='auto' and queue[0][3]=='v8' and any(v['mode']=='frax' for v in active.values()):
                    last_reason='frax_stage_barrier';priority_wait=None
                    break
                idle=res.select_idle(slots,active,usage)
                running=[(v['peak'],usage.get(s,0)) for s,v in active.items()]
                chosen=None;last_reason='memory_reserve'
                # Never bypass the highest-priority queued job for a cheaper V8.
                # Existing work continues; its completion builds the head job's
                # required reserve instead of immediately consuming it again.
                for _,_,index,mode in queue[:1]:
                    cache=getattr(idle,'telemetry',{}).get('cache_state','pure') if idle and idle.pid else 'pure'
                    reuse_possible=False
                    peak=learning.predict(mode,cases[index]['source'],cache,frames[index],admission,reuse=reuse_possible)
                    peak=memory_queue.peak(mode,cases[index]['source'],frames[index],admission.value['stages'][mode]['bytes_per_output_frame'],peak)
                    completion=max(peak,admission.completion_peak(mode,frames[index]),learning.predict(mode,cases[index]['source'],cache,frames[index],admission))
                    if stage_memory:completion=max(completion,stage_memory.bound)
                    compiling=[(v['mode'],cases[v['index']]['source']) for s,v in active.items()
                        if getattr(s,'telemetry',{}).get('compile_possible',v['mode']!='v8')]
                    if mode!='v8' and not reuse_possible:compiling.append((mode,cases[index]['source']))
                    shared=max((learning.compiler_extra(m,source,admission) for m,source in compiling),default=0)
                    if stage_memory:shared=max(shared,admission.shared([mode]+[v['mode'] for v in active.values()]))
                    if stage_memory:
                        allowed=stage_memory.can_start(snap,reserve,active,usage,peak,completion,usage.get(idle,0),shared)
                        reason=None if allowed else 'memory_reserve'
                    else:allowed,reason=res.can_admit(snap,reserve,free,disk_floor,running,peak,usage.get(idle,0),shared)
                    if allowed:chosen=(index,mode,peak);break
                    last_reason='priority_head_wait:'+reason
                    priority_wait=dict(case=cases[index]['id'],mode=mode,duration_s=float(durations[index]),
                        candidate_peak_bytes=completion if stage_memory and stage_memory.protected is None else peak,
                        normal_stage_peak_bytes=peak,completion_peak_bytes=completion,compiler_reserve_bytes=shared,
                        inflight_growth_bytes=sum(max(0,p-u) for p,u in running),
                        available_headroom_bytes=min(snap['available']-reserve,snap['cgroup_headroom']))
                if chosen is None:
                    victim=res.reclaim_idle(slots,active,usage,snap,reserve,idle,peak,shared)
                    if victim is not None:
                        victim.close();pool_stats['pressure_recycles']+=1
                    else:pool_stats['reservation_waits_preserving_warm']+=1
                    break
                i,mode,peak=chosen;heapq.heappop(queue);c=cases[i];r=records[i]
                if idle is None:
                    idle=slot_factory(dict(options,cpu=cpus[len(slots)%len(cpus)]));slots.append(idle)
                folder=out/c['id']/mode if args.mode=='auto' else out/c['id'];folder.parent.mkdir(parents=True,exist_ok=True)
                if folder.exists():
                    if not resume and not r.get('memory_retries'):raise FileExistsError(folder)
                    archived=folder.with_name(folder.name+'.interrupted.'+str(time.time_ns()))
                    folder.rename(archived)
                    for attempt in r['attempts']:
                        if attempt['mode']==mode and attempt['status']!='complete' and attempt['output']==str(folder.relative_to(out)):
                            attempt['output']=str(archived.relative_to(out))
                    old_log=folder.parent/(mode+'.log') if args.mode=='auto' else out/(c['id']+'.log')
                    if old_log.exists():old_log.rename(old_log.with_name(old_log.name+'.interrupted.'+str(time.time_ns())))
                reuse=None
                arguments=dict(bvh_units=getattr(args,'bvh_units','cm'),bvh_translations=getattr(args,'bvh_translations','absolute'),foot_contact=getattr(args,'foot_contact',DEFAULT_FOOT_CONTACT),input=c['input'],output=str(folder),source=c['source'],mode=mode,iterations=args.iterations,initial_iterations=args.initial_iterations,max_frames=None,legacy_frax=False,reuse_initial_from=reuse,require_initial_reuse=reuse is not None,pipeline=getattr(args,'pipeline','unified'),unified_repair=use_unified(args),repair_similarity=getattr(args,'repair_similarity','soft'),profile=getattr(args,'profile',None),repair_budget=getattr(args,'repair_budget',None),allow_local_slowdown=getattr(args,'allow_local_slowdown',False),resume_recheck=getattr(args,'resume_recheck',False))
                job=dict(input=c['input'],input_sha256=c['input_sha256'],log=str(folder.parent/(mode+'.log') if args.mode=='auto' else out/(c['id']+'.log')),arguments=arguments,
                    stage_memory=bool(stage_memory and getattr(idle,'supports_stage_memory',False)))
                active[idle]=dict(index=i,case=c['id'],mode=mode,folder=folder,peak=peak,grant=peak,completion_peak=completion,reuse=reuse is not None)
                if stage_memory:stage_memory.register(idle,active[idle])
                pool_stats['resident_reuses' if idle.pid else 'process_starts']+=1
                idle.start(job,min(timeout_for(args,mode,frames[i]),max(.01,args.budget-(time.monotonic()-started))))
                snap=monitor()
                for key in burst_headroom:
                    burst_headroom[key]=min(burst_headroom[key],snap[key])
                    snap[key]=burst_headroom[key]
                _,waiting=pressure(snap,reserve)
                last_reason=None;wait_since=None;priority_wait=None
            if queue and not active:
                wait_since=wait_since or time.monotonic()
                if not dynamic and args.resource_wait>0 and time.monotonic()-wait_since>=args.resource_wait:state='resource_exhausted';break
            else:wait_since=None
            loop_seconds=time.monotonic()-loop_started
            if time.monotonic()-last>=2:save(full=time.monotonic()-last_checkpoint>=30)
            if queue or active:time.sleep(.1)
        for slot in slots:slot.close()
        if static_identity()!=snapshot:
            state='identity_changed'
            for r in records:
                if r['status']=='accepted':r.update(status='identity_changed',selected_mode=None,selected_output=None,training_npz=None)
        if state=='running':state='complete'
        for r in records:
            if r['status']=='pending':r['status']=state if state!='complete' else ('review' if any(a.get('quality')=='review' for a in r['attempts']) else 'failed')
        phase='finished';save()
    except BaseException:
        state='failed';abort('interrupted')
        for r in records:
            if r['status']=='pending':r['status']='interrupted'
        save();raise
    finally:
        for slot in slots:slot.close()
        for sig,handler in previous_handlers.items():signal.signal(sig,handler)
        for r in records:
            d=out/r['case'];d.mkdir(exist_ok=True);atomic_json(d/'selection.json',r)
    return records

def sha_bytes(content):
    import hashlib
    return hashlib.sha256(content).hexdigest()
