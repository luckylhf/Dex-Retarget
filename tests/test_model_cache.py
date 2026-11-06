import hashlib
import json
import numpy as np
import pytest
from dex_frax import model_cache
from dex_frax.paths import ASSETS
from dex_frax.v8.robot import Robot

def test_binary_model_reuse_has_independent_state_and_identical_fk(tmp_path,monkeypatch):
    assets={str(p.relative_to(ASSETS)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ASSETS.rglob('*') if p.is_file()}
    reference=Robot();info=model_cache.prepare(assets,tmp_path)
    monkeypatch.setattr(model_cache,'_model',None);monkeypatch.setattr(model_cache,'_xml',None)
    model_cache.enable(info);a=Robot();b=Robot()
    assert a.m is b.m and a.d is not b.d
    rng=np.random.default_rng(17)
    for _ in range(4):
        q=np.r_[np.array([.02,.03,.8]),rng.uniform(-.1,.1,29)]
        reference.set(q,np.eye(3));a.set(q,np.eye(3))
        np.testing.assert_array_equal(a.d.xpos,reference.d.xpos)
        np.testing.assert_array_equal(a.d.xmat,reference.d.xmat)
    assert not np.array_equal(a.d.qpos,b.d.qpos)
    assert model_cache.prepare(assets,tmp_path)==info
    path=tmp_path/next(tmp_path.iterdir()).name/'model.mjb';path.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='hash mismatch'):model_cache.prepare(assets,tmp_path)
