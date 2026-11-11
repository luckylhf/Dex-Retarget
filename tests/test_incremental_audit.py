import copy,json,pickle
from pathlib import Path
import numpy as np
import pytest
from dex_frax.audit_cache import fork


def export_neutral(tmp_path,n=20):
    from dex_frax.v8.robot import Robot
    from dex_frax.v8.anatomy import configure
    from dex_frax.v8 import convert
    r=Robot();cfg=json.loads(Path(convert.__file__).with_name('config.json').read_text());configure(r,cfg['anatomy'])
    x=np.r_[0.,0.,1.,np.zeros(29)];r.set(x,np.eye(3))
    repeat=lambda v:np.repeat(np.asarray(v)[None],n,axis=0)
    dg=dict(root_target=repeat(x[:3]),root_r=repeat(np.eye(3)),torso_target=repeat(r.rot(r.torso)),source_expected_contact=np.zeros(n,bool))
    for side,g in r.sides.items():
        v=dict(foot_target_p=r.point(g['foot']),phase=0,contact=0,
            hand_forward=-r.rot(g['tcp'])[:,2],hand_width=r.rot(g['tcp'])[:,0],forearm_width=r.rot(g['forearm'])[:,0],
            elbow_axis=-r.rot(g['shoulder'])[:,1],signed_elbow_bend=g['elbow_geometric_zero'])
        for a,b,k in [('shoulder','elbow','upper'),('elbow','wrist','lower'),('hip','knee','thigh'),('knee','foot','shin')]:
            delta=r.point(g[b])-r.point(g[a]);v[k]=delta/np.linalg.norm(delta)
        dg.update({side+'_'+k:repeat(val) for k,val in v.items()})
    data=dict(root_pos=repeat(x[:3]).astype(np.float32),root_rot=repeat([0.,0.,0.,1.]).astype(np.float32),dof_pos=repeat(x[3:]).astype(np.float32),joint_names=r.names,quat_order='xyzw',fps=50.)
    meta=dict(model=r.path,configuration=cfg,estimated_contact_segments={s:[] for s in r.sides},body_contact_models=[])
    p=tmp_path/'motion.pkl';p.write_bytes(pickle.dumps(data,protocol=4));p.with_suffix('.json').write_text(json.dumps(meta));np.savez_compressed(p.with_suffix('.diagnostics.npz'),**dg)
    return p,data,meta,dg


@pytest.mark.parametrize('bounds',[(0,2),(8,12),(18,20)])
def test_cached_audit_equals_fresh_including_boundary_intervals(tmp_path,bounds):
    from dex_frax.v8.audit import audit
    p,data,meta,dg=export_neutral(tmp_path);cache={};baseline=audit(p,cache=cache)
    before=copy.deepcopy(cache);a,b=bounds;data['dof_pos'][a:b,15]+=.005
    p.write_bytes(pickle.dumps(data,protocol=4));trial=fork(cache)
    actual=audit(p,cache=trial,changed=bounds)
    assert actual==audit(p)
    assert trial['work']==dict(native_frames_recomputed=b-a,cache_hit=True)
    assert np.array_equal(cache['data']['dof_pos'],before['data']['dof_pos'])
    assert cache['frames']['torso_orientation']==before['frames']['torso_orientation']
    # Dishonest/incorrect range cannot reuse measurements outside it.
    wrong=fork(cache);assert audit(p,cache=wrong,changed=(5,6))==actual
    assert not wrong['work']['cache_hit']
    dg['Left_upper'][:,0]+=.001;np.savez_compressed(p.with_suffix('.diagnostics.npz'),**dg)
    audit(p,cache=trial,changed=bounds);assert not trial['work']['cache_hit']


def test_relocated_audit_uses_verified_runtime_assets(tmp_path):
    from dex_frax.cli import sha
    from dex_frax.paths import asset_fingerprints
    from dex_frax.v8.audit import audit

    path, payload, meta, _ = export_neutral(tmp_path, n=5)
    original = audit(path, write=False)
    meta['model_sha256'] = sha(meta['model'])
    meta['model'] = str(tmp_path / 'previous-host/dex_29.xml')
    path.with_suffix('.json').write_text(json.dumps(meta))
    assets = asset_fingerprints()
    assets['.DS_Store'] = 'legacy-finder-metadata'
    identity = tmp_path / 'identity.json'
    identity.write_text(json.dumps(dict(assets=assets)))
    assert audit(path, write=False) == original
    assets['model.json'] = 'different-asset'
    identity.write_text(json.dumps(dict(assets=assets)))
    with pytest.raises(ValueError, match='asset hashes differ'):
        audit(path, write=False)


def test_asset_identity_excludes_metadata_but_tracks_model_changes(tmp_path):
    from dex_frax.paths import asset_fingerprints

    xml = tmp_path / 'dex.xml'
    xml.write_text('<mujoco/>')
    original = asset_fingerprints(tmp_path)
    for name in ('.DS_Store', '._dex.xml', 'README.md', 'LICENSE'):
        (tmp_path / name).write_text('metadata')
    assert asset_fingerprints(tmp_path) == original
    xml.write_text('<mujoco model="changed"/>')
    assert asset_fingerprints(tmp_path) != original


def test_final_gate_reads_support_jitter_full_collision_and_separates_history(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from dex_frax import acceptance
    p,data,meta,dg=export_neutral(tmp_path)
    np.savez_compressed(p.with_suffix('.pose_errors.npz'),**{s+'_'+k+'_deg':np.zeros(20) for s in ('Left','Right') for k in ('upper','lower','hand_forward','hand_width','elbow_axis')})
    check=dict(minimum_body_visual_cm=0.,interpolated_floor={'minimum_body_visual_cm':0.},sides={},
        body_support={'violating_frames':[5]},temporal_quality={'violating_frames':[8]},full_self_collision={'collision_samples':1})
    monkeypatch.setattr(acceptance,'audit',lambda _:copy.deepcopy(check))
    monkeypatch.setattr(acceptance,'quality',lambda *args:{'flags':[]})
    result=acceptance.evaluate(p,meta,SimpleNamespace(report={'qp_nonconverged_steps':1}),'frax')
    assert set(result['flags'])=={'trusted_body_support_mismatch','conversion_added_jitter','full_self_collision_unresolved','frax_qp_nonconvergence'}
    assert 'frax_qp_nonconvergence' not in result['final_residual_flags']
    assert result['initial_solver_history']['qp_nonconverged_steps']==1
