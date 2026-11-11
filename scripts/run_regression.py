"""Bounded private regression runner with immutable baseline source snapshots."""
import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from dex_frax.cascade import load_manifest
from dex_frax.cli import atomic_json, sha
from compare import summarize


def main():
    p=argparse.ArgumentParser();p.add_argument('suite',type=Path)
    p.add_argument('--label',required=True);p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--modes',nargs='+',choices=['fast','refine'],default=['fast','refine'])
    p.add_argument('--timeout',type=float,default=90.);p.add_argument('--budget',type=float,default=900.)
    args=p.parse_args();root=args.suite.resolve()
    manifest = root / 'manifest.json'
    cases = load_manifest(manifest)
    if Path(args.label).name != args.label or args.label in ('', '.', '..'):
        p.error('--label must be a single directory name')
    if len(set(args.modes)) != len(args.modes):
        p.error('--modes must not repeat')
    if not all(math.isfinite(v) and v > 0 for v in (args.timeout, args.budget)):
        p.error('--timeout and --budget must be finite and positive')
    manifest_sha256 = sha(manifest)
    dest=root/'results'/args.label;dest.mkdir(parents=True,exist_ok=False)
    env={**os.environ,'PYTHONPATH':str(args.source_root.resolve()),'DEX_FRAX_ASSETS':str(args.assets.resolve())}
    started=time.perf_counter();records=[]
    for row in cases:
        src=Path(row['input'])
        if sha(src) != row['input_sha256']:
            raise ValueError('Fixture changed: ' + row['id'])
        for mode in args.modes:
            out=dest/(row['id']+'_'+mode);log=dest/(row['id']+'_'+mode+'.log')
            remaining=args.budget-(time.perf_counter()-started)
            record=dict(case=row['id'],mode=mode,output=out.name,status='budget_not_run')
            if remaining>0:
                seconds = min(remaining, args.timeout)
                cmd=[sys.executable,'-m','dex_frax.cli','convert',str(src),str(out),'--source',row['source'],
                     '--mode',mode,'--budget',str(seconds)]
                begin=time.perf_counter()
                try:
                    with log.open('w') as stream:
                        result=subprocess.run(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=seconds+10.)
                    record.update(status='complete' if result.returncode==0 else 'failed',returncode=result.returncode)
                except subprocess.TimeoutExpired:
                    record['status']='timeout'
                record['process_wall_s']=time.perf_counter()-begin
                if record['status']=='complete':
                    try:
                        record.update(summarize(out))
                    except (OSError, ValueError, KeyError) as exc:
                        record.update(status='invalid_result', error=str(exc))
            records.append(record)
            atomic_json(dest/'summary.json', dict(label=args.label,manifest_sha256=manifest_sha256,
                        elapsed_s=time.perf_counter()-started,records=records))
            print(row['id'],mode,record['status'],record.get('quality'),record.get('flags'),flush=True)
    print(dest/'summary.json',flush=True)
    return int(any(record['status'] != 'complete' for record in records))


if __name__=='__main__':raise SystemExit(main())
