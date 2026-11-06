import argparse
import json
from pathlib import Path
import time
import pytest
from dex_frax import cascade
from dex_frax.cli import sha


def arguments(tmp_path,names=('easy','hybrid_ok','v8_ok','bad_input','unresolved')):
    source=tmp_path/'input.bvh';source.write_text('fixture')
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'cases':[
        dict(id=name,input=source.name,input_sha256=sha(source),source='lafan') for name in names]}))
    return argparse.Namespace(manifest=manifest,output=tmp_path/'out',budget=30.,
        frax_timeout=2.,hybrid_timeout=3.,v8_timeout=4.,iterations=8,initial_iterations=100)


class FakeWorker:
    def __init__(self,plan):self.plan=plan;self.calls=[];self.closed=False
    def close(self):self.closed=True
    def run(self,job,timeout):
        args=job['arguments'];folder=Path(args['output']);name=folder.parent.name;mode=args['mode']
        self.calls.append((name,mode,args.get('reuse_initial_from'),timeout))
        status=self.plan.get((name,mode),'review')
        if status in ('failed','timeout','input_error'):return dict(status=status,elapsed_s=.1)
        folder.mkdir();quality=dict(status=status,flags=[] if status=='accepted' else ['test_quality_failure'])
        (folder/'quality.json').write_text(json.dumps(quality))
        files={}
        if status=='accepted':
            (folder/'training.npz').write_bytes(b'test export');files['training.npz']=sha(folder/'training.npz')
        result=dict(total_wall_s=.1,frax=None,acceptance_policy='common_pose_geometry_v1',
                    fingerprint='identity',files=files)
        return dict(status='complete',quality=status,flags=quality['flags'],result=result,elapsed_s=.1)


def test_wave_order_early_accept_reuse_and_input_error(tmp_path):
    args=arguments(tmp_path)
    worker=FakeWorker({('easy','frax'):'accepted',('hybrid_ok','hybrid'):'accepted',
        ('v8_ok','frax'):'timeout',('v8_ok','hybrid'):'failed',('v8_ok','v8'):'accepted',
        ('bad_input','frax'):'input_error'})
    records=cascade.run(args,lambda:worker)
    assert [r['status'] for r in records]==['accepted','accepted','accepted','input_error','review']
    assert [(a,b) for a,b,_,_ in worker.calls]==[
        ('easy','frax'),('hybrid_ok','frax'),('v8_ok','frax'),('bad_input','frax'),('unresolved','frax'),
        ('hybrid_ok','hybrid'),('v8_ok','hybrid'),('unresolved','hybrid'),('v8_ok','v8'),('unresolved','v8')]
    assert worker.calls[5][2].endswith('hybrid_ok/frax')
    assert worker.calls[6][2] is None
    assert worker.calls[-2][2] is None
    assert worker.closed
    accepted=json.loads((args.output/'accepted.json').read_text())['motions']
    assert [r['mode'] for r in accepted]==['frax','hybrid','v8']
    assert all((args.output/r['training_npz']).exists() for r in accepted)
    assert json.loads((args.output/'v8_ok/frax/run.json').read_text())['status']=='timeout'


def test_total_budget_keeps_accepted_and_marks_unrun(tmp_path):
    args=arguments(tmp_path,('easy','later'));args.budget=.1
    worker=FakeWorker({('easy','frax'):'accepted'})
    original=worker.run
    def slow(*args):
        result=original(*args);time.sleep(.15);return result
    worker.run=slow
    records=cascade.run(args,lambda:worker)
    assert [r['status'] for r in records]==['accepted','budget_exhausted']
    assert len(worker.calls)==1
    assert json.loads((args.output/'batch.json').read_text())['status']=='budget_exhausted'


@pytest.mark.parametrize('ids',[['../escape'],['a','a'],['a/b'],['.'],['a\\b']])
def test_manifest_rejects_unsafe_or_duplicate_ids(tmp_path,ids):
    args=arguments(tmp_path,ids)
    with pytest.raises(ValueError):cascade.load_manifest(args.manifest)
    assert not args.output.exists()


def test_manifest_hash_checked_before_any_worker(tmp_path):
    args=arguments(tmp_path);(tmp_path/'input.bvh').write_text('changed')
    with pytest.raises(ValueError,match='changed source'):cascade.run(args,lambda:pytest.fail('must not start'))
    assert not args.output.exists()


def test_invalid_export_is_not_adopted(tmp_path):
    args=arguments(tmp_path,('one',));worker=FakeWorker({('one','frax'):'accepted'})
    original=worker.run
    def corrupt(job,timeout):
        result=original(job,timeout)
        if job['arguments']['mode']=='frax':(Path(job['arguments']['output'])/'training.npz').write_bytes(b'corrupt')
        return result
    worker.run=corrupt
    record=cascade.run(args,lambda:worker)[0]
    assert record['status']=='review' and record['selected_mode'] is None
    assert record['attempts'][0]['error_type']=='InvalidExport'
    assert not (args.output/'one/frax/training.npz').exists()
    assert json.loads((args.output/'accepted.json').read_text())['motions']==[]


def test_worker_timeout_is_killed_and_next_job_gets_fresh_worker(tmp_path):
    worker=cascade.Worker()
    job=dict(input=str(tmp_path/'nonexistent'),input_sha256='x',arguments={},log=str(tmp_path/'log'))
    try:
        assert worker.run(job,1e-9)['status']=='timeout'
        assert worker.process is None
        assert worker.run(job,10.)['status']=='input_error'
    finally:worker.close()


def test_interrupt_cleans_partial_export_and_preserves_status(tmp_path):
    args=arguments(tmp_path,('one',));worker=FakeWorker({})
    def interrupt(job,timeout):
        folder=Path(job['arguments']['output']);folder.mkdir()
        (folder/'training.npz').write_bytes(b'incomplete')
        raise KeyboardInterrupt
    worker.run=interrupt
    with pytest.raises(KeyboardInterrupt):cascade.run(args,lambda:worker)
    assert worker.closed
    assert not (args.output/'one/frax/training.npz').exists()
    assert json.loads((args.output/'one/frax/run.json').read_text())['status']=='interrupted'
    result=json.loads((args.output/'batch.json').read_text())
    assert result['status']=='interrupted' and result['records'][0]['status']=='interrupted'


def test_manifest_fingerprint_pins_loaded_snapshot(tmp_path):
    args=arguments(tmp_path,('easy',));original_hash=sha(args.manifest)
    worker=FakeWorker({('easy','frax'):'accepted'});original=worker.run
    def modify(job,timeout):
        args.manifest.write_text('{"cases": []}')
        return original(job,timeout)
    worker.run=modify
    cascade.run(args,lambda:worker)
    assert json.loads((args.output/'batch.json').read_text())['manifest_sha256']==original_hash
