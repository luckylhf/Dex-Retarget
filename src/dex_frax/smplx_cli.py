"""Optional SMPL-X entry: separate geometry environment, existing solver modes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def add_options(parser):
    parser.add_argument('--body-models',type=Path,default=os.environ.get('SMPLX_MODEL_DIR'))
    parser.add_argument('--smplx-python',default=os.environ.get('SMPLX_PYTHON',sys.executable),
                        help='Python with torch/smplx/numpy/scipy; can differ from the solver environment')
    parser.add_argument('--input-fps',type=float,help='Required when the human input has no fps metadata')
    parser.add_argument('--up-axis',choices=['y','z'],help='Required for generic inputs without up_axis metadata')
    parser.add_argument('--units',choices=['m'],help='Explicitly confirm translation units when metadata is absent')
    parser.add_argument('--grounding',choices=['off','contact'],default='off')


def prepare(args, directory, timeout=None):
    from .input_errors import MotionInputError
    from .cli import sha
    prepared = getattr(args,'smplx_prepared',None)
    if prepared is not None:
        if sha(prepared)!=args.smplx_prepared_sha256:
            raise MotionInputError('Prepared SMPL-X input changed')
        return Path(prepared)
    if not getattr(args,'body_models',None):
        raise MotionInputError('SMPL-X requires --body-models (parent of smplx/SMPLX_GENDER.npz)')
    dest = Path(directory)/'input'/'human.npz'
    cmd = [str(args.smplx_python),'-m','dex_frax.smplx_input',str(args.input),
           '--output',str(dest),'--body-models',str(args.body_models),'--grounding',args.grounding]
    for attr in ('input_fps','up_axis','units','max_frames'):
        value=getattr(args,attr,None)
        if value is not None: cmd.extend(['--'+attr.replace('_','-'),str(value)])
    env=os.environ.copy();env['PYTHONNOUSERSITE']='1'
    env['PYTHONPATH']=str(Path(__file__).resolve().parents[1])+os.pathsep+env.get('PYTHONPATH','')
    try:
        subprocess.run(cmd,env=env,check=True,timeout=timeout)
    except subprocess.CalledProcessError as exc:
        raise MotionInputError('SMPL-X input preparation failed; see the field-specific error above') from exc
    args.smplx_prepared=str(dest);args.smplx_prepared_sha256=sha(dest)
    return dest


def preparation_identity(args):
    import numpy as np
    with np.load(args.smplx_prepared,allow_pickle=False) as data:
        meta=json.loads(str(data['metadata_json']))
    from .cli import sha
    if meta['input_sha256']!=sha(args.input):
        raise ValueError('SMPL-X source changed after preparation')
    return dict(prepared_sha256=sha(args.smplx_prepared),parameters=meta)


def check_input(args):
    cmd=[str(args.smplx_python),'-m','dex_frax.smplx_input',str(args.input),'--grounding',args.grounding]
    for attr in ('input_fps','up_axis','units'):
        value=getattr(args,attr,None)
        if value is not None:cmd.extend(['--'+attr.replace('_','-'),str(value)])
    env=os.environ.copy();env['PYTHONNOUSERSITE']='1'
    env['PYTHONPATH']=str(Path(__file__).resolve().parents[1])+os.pathsep+env.get('PYTHONPATH','')
    return subprocess.run(cmd,env=env).returncode


def single_auto(args,worker_factory=None):
    """Reuse the geometry once; bound owned solver workers with the same budget."""
    from .cli import atomic_json,sha
    from .cascade import Worker,accepted_artifact
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();attempts=[];selected=None;status='running';preview=None
    atomic_json(out/'run.json',dict(status=status,mode='auto',source='smplx'))
    worker=None
    try:
        prepare(args,out,timeout=args.budget)
        worker=(worker_factory or Worker)()
        for mode in ('frax','hybrid','v8'):
            remaining=args.budget-(time.monotonic()-started)
            if remaining<=0: status='budget_exhausted';break
            stage=out/mode
            options=vars(args).copy();options.update(output=str(stage),input=str(Path(args.input).resolve()),mode=mode)
            if mode=='hybrid' and attempts and attempts[0]['status']=='complete':
                options.update(reuse_initial_from=str(out/'frax'),require_initial_reuse=True)
            result=worker.run(dict(input=str(args.input),input_sha256=sha(args.input),log=str(out/(mode+'.log')),arguments=options),remaining)
            attempts.append(dict(mode=mode,**result))
            artifact=accepted_artifact(stage,result)
            if artifact:
                selected=mode;status='accepted';preview=mode;break
            if result['status']=='complete': preview=mode
            if result['status']=='input_error': status='input_error';break
            if result['status']=='timeout': status='budget_exhausted';break
        if status=='running': status='review' if preview else 'failed'
    except BaseException as exc:
        atomic_json(out/'run.json',dict(status='failed',error=repr(exc),mode='auto',source='smplx'))
        raise
    finally:
        if worker is not None: worker.close()
    record=dict(status=status,selected_mode=selected,selected_output=selected,
                preview_output=preview,training_npz=f'{selected}/training.npz' if selected else None,attempts=attempts)
    atomic_json(out/'selection.json',record)
    atomic_json(out/'accepted.json',dict(motions=[] if not selected else [dict(id='motion',training_npz=record['training_npz'],
                sha256=sha(out/record['training_npz']))]))
    atomic_json(out/'run.json',dict(status='complete' if status in ('accepted','review') else status,quality=status,mode='auto',
                selected_output=selected,preview_output=preview,training_npz=record['training_npz'],total_wall_s=time.monotonic()-started))
    print(json.dumps(record,ensure_ascii=False,indent=2))
    return record
