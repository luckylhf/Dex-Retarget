import argparse
import json
from pathlib import Path
from dex_frax.cli import single_auto


def test_long_single_file_runs_complete_frax_then_independent_v8(tmp_path):
    source=tmp_path/'long.bvh';source.write_text('Frames: 6001\nFrame Time: 0.02\n')
    args=argparse.Namespace(pipeline='two-stage',input=source,output=tmp_path/'out',source='lafan',mode='auto',
        iterations=8,initial_iterations=100,max_frames=None,budget=30.)
    class Worker:
        calls=[]
        def close(self):pass
        def run(self,job,timeout):
            assert job['arguments'].get('reuse_initial_from') is None
            assert not job['arguments'].get('require_initial_reuse',False)
            mode=job['arguments']['mode'];self.calls.append(mode);p=Path(job['arguments']['output']);p.mkdir()
            flags=['test'];(p/'quality.json').write_text(json.dumps(dict(status='review',flags=flags)))
            return dict(status='complete',quality='review',flags=flags,result=dict(total_wall_s=.1,frax=None,files={}))
    worker=Worker();record=single_auto(args,lambda:worker)
    assert worker.calls==['frax','v8'] and record['status']=='review'
    assert all((args.output/m/'quality.json').exists() for m in worker.calls)
    assert (args.output/'selection.json').exists()


def test_removed_mode_and_trajectory_reuse_fail_before_output(tmp_path):
    import pytest
    from dex_frax.cli import convert
    for mode,reuse,match in [('hybrid',None,'hybrid has been removed'),('v8','old/frax','independently')]:
        args=argparse.Namespace(mode=mode,output=tmp_path/mode,reuse_initial_from=reuse)
        with pytest.raises(ValueError,match=match):convert(args)
        assert not args.output.exists()
