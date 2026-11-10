import numpy as np
from dex_frax.v8.self_collision import audit_export


def test_final_collision_cannot_sample_less_than_candidate_guard(monkeypatch):
    import dex_frax.v8.self_collision as m
    from types import SimpleNamespace
    calls=[]
    monkeypatch.setattr(m,'CollisionModel',lambda robot,cfg:SimpleNamespace(cfg=cfg))
    monkeypatch.setattr(m,'scan',lambda model,x,r,steps:(calls.append(steps) or {},None,None))
    payload=dict(root_pos=np.zeros((2,3)),dof_pos=np.zeros((2,29)),root_rot=np.tile([0.,0.,0.,1.],(2,1)))
    audit_export(None,payload,{'interpolation_steps':4})
    audit_export(None,payload,{'interpolation_steps':16})
    assert calls==[8,16]
