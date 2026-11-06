"""Compare two same-manifest batch runs; retain common gates and raw metrics."""
import argparse
import json
from pathlib import Path
import numpy as np
from summarize_regression import row


def main():
    p=argparse.ArgumentParser()
    p.add_argument('suite',type=Path)
    p.add_argument('--baseline',required=True)
    p.add_argument('--candidate',required=True)
    p.add_argument('--output',default='paired_comparison.json')
    p.add_argument('--dense',action='store_true')
    args=p.parse_args();root=args.suite.resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    old=root/'results'/args.baseline;new=root/'results'/args.candidate
    old_batch=json.loads((old/'batch.json').read_text());new_batch=json.loads((new/'batch.json').read_text())
    if old_batch['manifest_sha256']!=new_batch['manifest_sha256']:
        raise ValueError('Cannot compare different manifests')
    records=[]
    for case in manifest['cases']:
        name=case['id'];before=row(old/name,args.dense);after=row(new/name,args.dense)
        with np.load(old/name/'motion.diagnostics.npz') as a,np.load(new/name/'motion.diagnostics.npz') as b:
            keys=[k for k in a.files if k in b.files and k!='solver_records' and not k.startswith('support_')]
            changed=[k for k in keys if not np.array_equal(a[k],b[k])]
        records.append(dict(case=name,baseline=before,candidate=after,changed_target_arrays=changed,
                            byte_identical=before['motion_sha256']==after['motion_sha256']))
        print(name,before['quality'],'->',after['quality'],after['common_flags'],flush=True)
    result=dict(scope=manifest['scope'],baseline=args.baseline,candidate=args.candidate,
                elapsed_s=dict(baseline=old_batch['elapsed_s'],candidate=new_batch['elapsed_s']),
                manifest_sha256=old_batch['manifest_sha256'],cases=records,
                common_pass={k:sum(not r[k]['common_flags'] for r in records) for k in ('baseline','candidate')},
                timing_scope='sequential same-machine batch runs; single measurement, no parallel benchmark/test work')
    (root/args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
