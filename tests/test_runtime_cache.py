import copy
import json
import os
from pathlib import Path
import numpy as np
import pytest
from dex_frax.ik import FraxIK, runtime_keys, clear_runtime_cache
from dex_frax.v8.robot import Robot
from dex_frax.v8.anatomy import configure
from dex_frax.v8.bvh import load
from dex_frax.v8.source import adapt
from dex_frax.v8.contacts import prepare
from dex_frax.v8 import convert
from dex_frax.adaptive_refine import window_targets


@pytest.fixture(scope='module')
def scene():
    path=os.environ.get('DEX_FRAX_TEST_BVH')
    if not path:pytest.skip('Set DEX_FRAX_TEST_BVH for the authorized fixture')
    cfg=json.loads((Path(convert.__file__).parent/'config.json').read_text())
    r=Robot();configure(r,cfg['anatomy'])
    t=prepare(adapt(load(path,fps=50.),'mocap'),r,cfg['contact'])
    return r,window_targets(t,0,145,len(t['hips'])),cfg['solver']


def test_cache_identity_tracks_static_constraints_and_geometry(scene):
    r,t,cfg=scene;assets={'fixture':'test-only-key'}
    def key(t=t,cfg=cfg,assets=assets):return runtime_keys(r,t,cfg,8,100,True,assets)
    original=key()
    shorter=window_targets(t,0,33,len(t['hips']))
    assert key(t=shorter)==original
    assert key(cfg={**cfg,'elbow_axis':cfg['elbow_axis']+.1})[1]!=original[1]
    assert key(assets={'fixture':'changed'})[0]!=original[0]
    t2=copy.deepcopy(t);t2['body_contact_height_tolerance_m']*=.9
    assert key(t=t2)==original  # This value is a dynamic input, never captured.
    old=r.lower.copy();r.lower[3]+=.01
    try:assert key()[0]!=original[0]
    finally:r.lower[:]=old
    oldpos=r.m.body_pos.copy();r.m.body_pos[1,0]+=.01
    try:assert key()[0]!=original[0]
    finally:r.m.body_pos[:]=oldpos


def test_reused_kernel_uses_new_initial_pose_and_body_tolerance(scene,monkeypatch,tmp_path):
    r,t,cfg=scene;clear_runtime_cache()
    first=FraxIK(iterations=2,initial_iterations=12)
    first(r,t,cfg)
    t2=copy.deepcopy(t);shift=np.array([.037,0.,0.])
    t2['root_p']+=shift;t2['body_contact_height_tolerance_m']*=.9
    for v in t2['sides'].values():
        v['foot_target_p']+=shift;v['anchors']+=shift
    hot=FraxIK(iterations=2,initial_iterations=12)
    import subprocess,sys
    monkeypatch.setenv('DEX_FRAX_COMPILE_LOCK_DIR',str(tmp_path))
    code='import fcntl,sys,time; f=open(sys.argv[1],"a"); fcntl.flock(f,fcntl.LOCK_EX); print("held",flush=True); time.sleep(2)'
    blocker=subprocess.Popen([sys.executable,'-c',code,str(tmp_path/'compile.lock')],stdout=subprocess.PIPE,text=True)
    try:
        assert blocker.stdout.readline().strip()=='held'
        q_hot,records_hot=hot(r,t2,cfg)
        assert hot.report['compile_queue_wait_s']==0.
    finally:blocker.communicate(timeout=5)
    assert hot.report['runtime_cache']['kernel_hit']
    assert hot.report['runtime_cache']['model_hit']
    prefix=window_targets(t2,0,129,len(t2['hips']))
    shorter=FraxIK(iterations=2,initial_iterations=12)
    short_q,short_records=shorter(r,prefix,cfg)
    assert shorter.report['runtime_cache']['kernel_hit']
    np.testing.assert_array_equal(short_q,q_hot[:129])
    np.testing.assert_array_equal(short_records,np.asarray(records_hot)[:129])
    cold=FraxIK(iterations=2,initial_iterations=12,use_cache=False)
    q_cold,records_cold=cold(r,t2,cfg)
    np.testing.assert_allclose(q_hot,q_cold,rtol=0.,atol=1e-12)
    np.testing.assert_allclose(records_hot,records_cold,rtol=0.,atol=1e-10)
    clear_runtime_cache()
