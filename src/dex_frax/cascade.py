"""Time-bounded single-file complete Frax -> independent complete V8 orchestration."""
import argparse
import contextlib
import hashlib
import json
import multiprocessing
from pathlib import Path
import re
import sys
import time
from .cli import atomic_json, sha

from .dispatch import STAGES


def load_manifest(path, contents=None):
    path = Path(path).resolve()
    manifest = json.loads(path.read_bytes() if contents is None else contents)
    cases = manifest.get('cases')
    if not isinstance(cases,list) or not cases:
        raise ValueError('Manifest needs a nonempty cases list')
    seen=set();validated=[]
    for row in cases:
        name=row['id']
        if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',name) or name in seen:
            raise ValueError('Case IDs must be unique safe directory names: '+str(name))
        seen.add(name)
        if row['source'] not in ('auto','lafan','soma','mocap'):
            raise ValueError('Unknown source for '+name)
        source=(path.parent/row['input']).resolve()
        if not source.is_file() or sha(source)!=row['input_sha256']:
            raise ValueError('Missing or changed source: '+name)
        validated.append({**row,'input':str(source)})
    return validated


def _worker(connection):
    from .cli import convert
    from .input_errors import MotionInputError
    from .resource_events import set_sink,phase
    set_sink(lambda event:connection.send(dict(resource_event=event)))
    while True:
        try:job=connection.recv()
        except EOFError:break
        if job is None:break
        begin=time.perf_counter()
        try:
            phase('loading',compile_possible=job['arguments'].get('mode','v8')!='v8' and not job['arguments'].get('require_initial_reuse',False))
            if sha(job['input'])!=job['input_sha256']:
                raise MotionInputError('Input changed after manifest validation')
            with Path(job['log']).open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                result=convert(argparse.Namespace(**job['arguments']))
            response=dict(status='complete',quality=result['quality'],flags=result['flags'],
                          result=result,elapsed_s=time.perf_counter()-begin)
        except Exception as error:
            response=dict(status='input_error' if isinstance(error,(MotionInputError,FileNotFoundError)) else 'failed',
                          error=repr(error),error_type=type(error).__name__,elapsed_s=time.perf_counter()-begin)
            # Drop compiled state after a numerical/runtime failure.
            if response['status']=='failed':
                module=sys.modules.get('dex_frax.ik')
                if module is not None:module.clear_runtime_cache()
        from .resources import release_memory
        release_memory()
        phase('idle')
        connection.send(response)
    connection.close()


class Worker:
    """Keep JAX warm between clips, but kill this owned process on timeout."""
    def __init__(self):self.process=None;self.connection=None

    def close(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate();self.process.join(2.)
                if self.process.is_alive():self.process.kill();self.process.join(2.)
            else:self.process.join()
            self.connection.close();self.process.close()
        self.process=self.connection=None

    def run(self,job,timeout):
        start=time.perf_counter()
        if self.process is None or not self.process.is_alive():
            self.close();ctx=multiprocessing.get_context('spawn')
            self.connection,child=ctx.Pipe()
            self.process=ctx.Process(target=_worker,args=(child,),daemon=True)
            self.process.start();child.close()
        try:
            self.connection.send(job)
            remaining=max(0.,timeout-(time.perf_counter()-start))
            while True:
                remaining=max(0.,timeout-(time.perf_counter()-start))
                if not self.connection.poll(remaining):
                    self.close();return dict(status='timeout',elapsed_s=time.perf_counter()-start)
                response=self.connection.recv()
                if 'resource_event' not in response:break
            response['worker_wall_s']=time.perf_counter()-start
            return response
        except (EOFError,BrokenPipeError,OSError) as error:
            self.close()
            return dict(status='failed',error_type='WorkerExit',error=repr(error),elapsed_s=time.perf_counter()-start)


def accepted_artifact(folder,attempt):
    """Never select a partial or inconsistent training export."""
    if attempt.get('status')!='complete' or attempt.get('quality')!='accepted' or attempt.get('flags'):
        return None
    result=attempt['result']
    if result.get('acceptance_policy')!='common_pose_geometry_v1':
        raise ValueError('Missing common acceptance policy')
    quality=json.loads((folder/'quality.json').read_text())
    if quality['status']!='accepted' or quality['flags']:
        raise ValueError('Quality/export mismatch')
    path=folder/'training.npz';digest=sha(path)
    if digest!=result['files']['training.npz']:
        raise ValueError('Training export hash mismatch')
    return dict(training_sha256=digest,fingerprint=result['fingerprint'])


def run(args,worker_factory=Worker):
    manifest_bytes=Path(args.manifest).read_bytes()
    manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest()
    cases=load_manifest(args.manifest,manifest_bytes)
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'source_manifest.json',dict(cases=cases))
    started=time.perf_counter();records=[dict(case=c['id'],status='pending',attempts=[],
        selected_mode=None,selected_output=None,training_npz=None) for c in cases]
    worker=worker_factory();state='running';active=None
    timeouts={mode:getattr(args,mode+'_timeout') for mode in STAGES}

    def save():
        accepted=[]
        for source,record in zip(cases,records):
            if record['status']=='accepted':
                accepted.append(dict(case=record['case'],input=source['input'],input_sha256=source['input_sha256'],
                    mode=record['selected_mode'],output=record['selected_output'],training_npz=record['training_npz'],
                    training_sha256=record['training_sha256'],fingerprint=record['fingerprint']))
        atomic_json(out/'accepted.json',dict(schema=1,path_base='batch output directory',motions=accepted))
        atomic_json(out/'batch.json',dict(schema=1,mode='auto',status=state,stage_order=STAGES,
            acceptance_policy='common_pose_geometry_v1',manifest_sha256=manifest_sha256,
            budget_s=args.budget,stage_timeout_s=timeouts,elapsed_s=time.perf_counter()-started,
            accepted=len(accepted),records=records))

    save()
    try:
        for mode in STAGES:
            for source,record in zip(cases,records):
                if record['status'] in ('accepted','input_error'):continue
                remaining=args.budget-(time.perf_counter()-started)
                if remaining<=0:
                    state='budget_exhausted';break
                case_dir=out if getattr(args,'single_pipeline',False) else out/source['id'];case_dir.mkdir(exist_ok=True)
                folder=case_dir/mode
                arguments=dict(input=source['input'],output=str(folder),source=source['source'],mode=mode,
                    iterations=args.iterations,initial_iterations=args.initial_iterations,max_frames=getattr(args,'max_frames',None),
                    legacy_frax=False,reuse_initial_from=None,require_initial_reuse=False)
                job=dict(input=source['input'],input_sha256=source['input_sha256'],log=str(case_dir/(mode+'.log')),arguments=arguments)
                active=(record,folder,mode)
                response=worker.run(job,min(timeouts[mode],remaining))
                attempt={**response,'mode':mode,'output':str(folder.relative_to(out))}
                try:selection=accepted_artifact(folder,attempt)
                except (OSError,ValueError,KeyError) as error:
                    attempt.update(status='failed',error_type='InvalidExport',error=str(error));selection=None
                # A killed/incomplete attempt cannot leave an adoptable export.
                if attempt['status']!='complete':
                    folder.mkdir(exist_ok=True)
                    (folder/'training.npz').unlink(missing_ok=True)
                    atomic_json(folder/'run.json',{k:v for k,v in attempt.items() if k!='result'})
                if selection is not None:
                    record.update(status='accepted',selected_mode=mode,selected_output=str(folder.relative_to(out)),
                                  training_npz=str((folder/'training.npz').relative_to(out)),**selection)
                elif attempt['status']=='input_error':record['status']='input_error'
                else:record['status']='pending'
                # Keep reports compact; full conversion records stay in stage/run.json.
                result=attempt.pop('result',None)
                if result:
                    attempt['conversion_wall_s']=result['total_wall_s']
                    attempt['initial_reuse']=(result.get('frax') or {}).get('initial_reuse')
                    attempt['runtime_cache']=(result.get('frax') or {}).get('runtime_cache')
                record['attempts'].append(attempt)
                active=None
                atomic_json(case_dir/'selection.json',record);save()
                print(source['id'],mode,attempt['status'],attempt.get('quality'),flush=True)
            if state=='budget_exhausted':break
        for record in records:
            if record['status']=='pending':
                record['status']='budget_exhausted' if state=='budget_exhausted' else (
                    'review' if any(a.get('quality')=='review' for a in record['attempts']) else 'failed')
        if state=='running':state='complete'
    except BaseException:
        state='interrupted'
        for record in records:
            if record['status']=='pending':record['status']='interrupted'
        raise
    finally:
        worker.close()
        if active is not None:
            current,folder,mode=active
            folder.mkdir(exist_ok=True)
            (folder/'training.npz').unlink(missing_ok=True)
            interrupted=dict(status='interrupted',mode=mode,output=str(folder.relative_to(out)))
            current['attempts'].append(interrupted)
            atomic_json(folder/'run.json',interrupted)
        for record in records:
            case_dir=out if getattr(args,'single_pipeline',False) else out/record['case'];case_dir.mkdir(exist_ok=True)
            atomic_json(case_dir/'selection.json',record)
        save()
    return records
