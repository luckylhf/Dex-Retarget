"""Compare complete results from two regression suites or managed batches."""
import argparse
import json
from pathlib import Path

from compare import compare_motions, summarize
from dex_frax.cli import atomic_json, sha


def read_results(root, manifest_sha256):
    summary = root / 'summary.json'
    batch = not summary.is_file()
    path = root / 'batch.json' if batch else summary
    report = json.loads(path.read_text())
    if report['manifest_sha256'] != manifest_sha256:
        raise ValueError(f'Manifest mismatch in {path}')
    if batch and report['status'] != 'complete':
        raise ValueError(f'Incomplete batch: {root}')
    rows = {}
    for record in report['records']:
        expected = ('accepted', 'review') if batch else ('complete',)
        if record['status'] not in expected:
            raise ValueError(f"Incomplete case {record['case']} in {root}")
        name = record['case'] if batch else f"{record['case']}:{record['mode']}"
        folder = root / (record['case'] if batch else record['output'])
        if not folder.resolve().is_relative_to(root.resolve()):
            raise ValueError(f'Case output is outside {root}')
        rows[name] = summarize(folder)
    return report, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('suite', type=Path)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path('paired_comparison.json'))
    args = parser.parse_args()
    root = args.suite.resolve()
    manifest_sha256 = sha(root / 'manifest.json')
    before, old = read_results(root / 'results' / args.baseline, manifest_sha256)
    after, new = read_results(root / 'results' / args.candidate, manifest_sha256)
    if old.keys() != new.keys():
        raise ValueError('Regression cases or modes differ')
    records = [
        dict(case=name, baseline=old[name], candidate=new[name],
             **compare_motions(old[name], new[name]))
        for name in old
    ]
    atomic_json(root / args.output, dict(
        baseline=str(args.baseline), candidate=str(args.candidate),
        manifest_sha256=manifest_sha256, cases=records,
        elapsed_s=dict(baseline=before['elapsed_s'], candidate=after['elapsed_s']),
        passed={name: sum(row['quality'] == 'accepted' for row in rows.values())
                for name, rows in [('baseline', old), ('candidate', new)]},
    ))


if __name__ == '__main__':
    main()
