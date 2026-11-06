"""Optional real-data regression; fixtures remain private and hash pinned."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pytest
from dex_frax.v8.robot import Robot
from dex_frax.v8.anatomy import configure
from dex_frax.v8.bvh import load
from dex_frax.v8.source import adapt
from dex_frax.v8.contacts import prepare
from dex_frax.v8 import convert
from dex_frax.v8.self_collision import CollisionModel,scan
from dex_frax.contact_transition import refine as transitions
from dex_frax.local_refine import refine as local
from dex_frax.adaptive_refine import refine,inspect,check_candidate,policy


def test_get_up_clearance_without_added_collision():
    suite=os.environ.get('DEX_FRAX_REGRESSION_SUITE')
    if not suite:pytest.skip('private regression suite not supplied')
    root=Path(suite);manifest=json.loads((root/'manifest.json').read_text())
    row=next(r for r in manifest['cases'] if r['id']=='get_up');source=root/row['input']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==row['input_sha256']
    cfg=json.loads((Path(convert.__file__).parent/'config.json').read_text())
    robot=Robot();configure(robot,cfg['anatomy'])
    t=prepare(adapt(load(source,fps=50.),row['source']),robot,cfg['contact']);transitions(t,robot)
    with np.load(root/'results/final_fast_v040/get_up/motion.initial.npz') as d:x=d['x']
    x,_=local(robot,t,x,cfg['solver']);before=inspect(robot,t,x)
    assert before['body_floor'].min()<-.005
    out,report=refine(robot,t,x,cfg['solver'])
    after=check_candidate(robot,t,x,out,cfg['solver'],policy(),before)
    assert after['floor'].min()>-.001
    assert report['work_budget']['used_frames']<=100
    model=CollisionModel(robot)
    initial=scan(model,x,t['root_r'])[0];final=scan(model,out,t['root_r'])[0]
    assert final['native_colliding_frames']<=initial['native_colliding_frames']
    assert final['minimum_distance_m']>=initial['minimum_distance_m']-1e-6
