import json
from types import SimpleNamespace
import numpy as np
import pytest
from dex_frax.initial_reuse import ReusedInitial,target_digest
from dex_frax.cli import sha


def checkpoint(tmp_path):
    n=5;names=[str(i) for i in range(29)]
    t=dict(hips=np.zeros((n,3)),root_r=np.tile(np.eye(3),(n,1,1)),fps=50.)
    config={'weight':1.};ident=dict(converter='dex_frax',version='test',source='lafan',fps=50.,iterations=8,
        initial_iterations=100,max_frames=None,input_sha256='input',implementation={'a':'hash'},
        frax_implementation={},assets={},runtime={},mode='frax',fingerprint='old',legacy_frax=False,conversion_contract={'schema':1})
    (tmp_path/'identity.json').write_text(json.dumps(ident))
    np.savez(tmp_path/'motion.initial.npz',x=np.zeros((n,32)),solver_records=np.zeros((n,3)),
             root_r=t['root_r'],fps=50.,joint_names=names)
    meta=dict(prepared_targets_sha256=target_digest(t),configuration=dict(solver=config),
              frax=dict(qp_nonconverged_steps=0,local_refinement={'stale':True},adaptive_refinement={'stale':True}))
    (tmp_path/'motion.json').write_text(json.dumps(meta))
    (tmp_path/'run.json').write_text(json.dumps(dict(status='complete',files={k:sha(tmp_path/k) for k in ('motion.initial.npz','motion.json')})))
    class Fallback:
        def __init__(self):self.calls=0;self.report={}
        def __call__(self,*args):self.calls+=1;return np.ones((n,32)),np.ones((n,3)).tolist()
    fallback=Fallback()
    solve=ReusedInitial(tmp_path,{**ident,'mode':'hybrid'},fallback)
    return solve,fallback,SimpleNamespace(names=names),t,config


def test_valid_initial_reused_without_stale_postprocessing(tmp_path):
    solve,fallback,r,t,cfg=checkpoint(tmp_path)
    x,records=solve(r,t,cfg)
    assert fallback.calls==0 and solve.report['initial_reuse']['used']
    assert 'local_refinement' not in solve.report and 'adaptive_refinement' not in solve.report
    assert x.shape==(5,32) and np.all(x==0.)


@pytest.mark.parametrize('change',['input','implementation','artifact','targets','solver'])
def test_stale_initial_recomputed_instead_of_used(tmp_path,change):
    solve,fallback,r,t,cfg=checkpoint(tmp_path)
    if change=='input':solve.identity['input_sha256']='changed'
    elif change=='implementation':solve.identity['implementation']={'different':'code'}
    elif change=='artifact':(tmp_path/'motion.initial.npz').write_bytes(b'bad')
    elif change=='targets':t['hips'][0,0]=1.
    else:cfg['weight']=2.
    x,_=solve(r,t,cfg)
    assert fallback.calls==1 and not solve.report['initial_reuse']['used']
    assert np.all(x==1.)


def test_digest_tracks_dtype_shape_values_and_metadata():
    a=dict(array=np.arange(6.).reshape(2,3),static={'body':1})
    assert target_digest(a)==target_digest(dict(static={'body':1},array=a['array'].copy()))
    assert target_digest(a)!=target_digest({**a,'array':a['array'].astype(np.float32)})
    assert target_digest(a)!=target_digest({**a,'array':a['array'].reshape(3,2)})
    assert target_digest(a)!=target_digest({**a,'static':{'body':2}})
