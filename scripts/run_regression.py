"""Bounded private regression runner with immutable baseline source snapshots."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('suite',type=Path)
    p.add_argument('--label',required=True);p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--modes',nargs='+',default=['frax','v8'])
    p.add_argument('--timeout',type=float,default=90.);p.add_argument('--budget',type=float,default=900.)
    args=p.parse_args();root=args.suite.resolve();manifest=json.loads((root/'manifest.json').read_text())
    dest=root/'results'/args.label;dest.mkdir(parents=True,exist_ok=False)
    env={**os.environ,'PYTHONPATH':str(args.source_root.resolve()),'DEX_FRAX_ASSETS':str(args.assets.resolve())}
    started=time.perf_counter();records=[]
    for row in manifest['cases']:
        src=root/row['input']
        if hashlib.sha256(src.read_bytes()).hexdigest()!=row['input_sha256']:raise ValueError('fixture changed')
        for mode in args.modes:
            out=dest/(row['id']+'_'+mode);log=dest/(row['id']+'_'+mode+'.log')
            remaining=args.budget-(time.perf_counter()-started)
            record=dict(case=row['id'],mode=mode,output=str(out),status='budget_not_run')
            if remaining>5:
                cmd=[sys.executable,'-m','dex_frax.cli','convert',str(src),str(out),'--source',row['source'],'--mode',mode]
                begin=time.perf_counter()
                try:
                    with log.open('w') as stream:
                        result=subprocess.run(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,
                            timeout=min(remaining,180. if row['id']=='nobody' and mode=='v8' else args.timeout))
                    record.update(status='complete' if result.returncode==0 else 'failed',returncode=result.returncode)
                except subprocess.TimeoutExpired:
                    record['status']='timeout'
                    # Replace the interrupted worker's stale running marker.
                    if out.exists():(out/'run.json').write_text(json.dumps(dict(status='timeout',reason='regression per-case budget'),indent=2))
                record['process_wall_s']=time.perf_counter()-begin
                if record['status']=='complete':
                    run=json.loads((out/'run.json').read_text());quality=json.loads((out/'quality.json').read_text())
                    record.update(quality=run['quality'],flags=run['flags'],frames=run['frames'],
                        total_wall_s=run['total_wall_s'],torso=quality['source_pose_audit']['torso_orientation_deg'],
                        elbow_gate=quality.get('adaptive_pose_gate'))
            records.append(record)
            (dest/'summary.json').write_text(json.dumps(dict(label=args.label,elapsed_s=time.perf_counter()-started,records=records),ensure_ascii=False,indent=2))
            print(row['id'],mode,record['status'],record.get('quality'),record.get('flags'),flush=True)
    print(dest/'summary.json',flush=True)


if __name__=='__main__':main()
