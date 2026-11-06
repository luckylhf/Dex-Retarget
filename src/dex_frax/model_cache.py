"""Verified binary model cache; spawn workers reuse models, never fork JAX."""
import hashlib
import json
from pathlib import Path

_model=None
_xml=None

def prepare(assets,cache):
    import mujoco
    import numpy as np
    from .paths import ASSETS
    key=hashlib.sha256(json.dumps(dict(assets=assets,mujoco=mujoco.__version__),sort_keys=True).encode()).hexdigest()
    folder=Path(cache)/key;folder.mkdir(parents=True,exist_ok=True)
    binary=folder/'model.mjb';receipt=folder/'receipt.json';xml=ASSETS/'mjcf/dex_29.xml'
    if receipt.exists():
        info=json.loads(receipt.read_text())
        if hashlib.sha256(binary.read_bytes()).hexdigest()==info['sha256']:
            return dict(binary=str(binary),xml=str(xml),sha256=info['sha256'])
        raise ValueError('Cached MJB hash mismatch')
    original=mujoco.MjModel.from_xml_path(str(xml))
    temporary=folder/'model.tmp.mjb';mujoco.mj_saveModel(original,str(temporary))
    restored=mujoco.MjModel.from_binary_path(str(temporary));compared=[]
    for name in dir(original):
        a=getattr(original,name)
        if isinstance(a,np.ndarray):
            if not np.array_equal(a,getattr(restored,name)):raise ValueError('XML/MJB mismatch: '+name)
            compared.append(name)
    temporary.replace(binary)
    info=dict(sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),arrays_compared=compared,assets=assets,mujoco=mujoco.__version__)
    receipt.write_text(json.dumps(info,indent=2))
    return dict(binary=str(binary),xml=str(xml),sha256=info['sha256'])

def enable(info):
    import mujoco
    global _model,_xml
    if hashlib.sha256(Path(info['binary']).read_bytes()).hexdigest()!=info['sha256']:raise ValueError('MJB changed before worker load')
    _model=mujoco.MjModel.from_binary_path(info['binary']);_xml=str(Path(info['xml']).resolve())

def load(path):
    import mujoco
    if _model is not None and str(Path(path).resolve())==_xml:return _model
    return mujoco.MjModel.from_xml_path(str(path))
