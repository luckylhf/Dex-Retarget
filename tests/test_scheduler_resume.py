import copy
import json
from types import SimpleNamespace
import pytest
from dex_frax import managed_batch as b,resources as r


def test_explicit_scheduler_update_preserves_original_identity(tmp_path,monkeypatch):
    args=SimpleNamespace(mode='auto',iterations=8,initial_iterations=100,ik_block_frames=128)
    before=dict(implementation={'resources.py':'old','managed_batch.py':'old','ik.py':'same'},assets={'robot':'same'},runtime={'jax':'same'})
    after=copy.deepcopy(before);after['implementation']['resources.py']='new'
    (tmp_path/'batch_identity.json').write_text(json.dumps(before))
    (tmp_path/'batch.json').write_text(json.dumps(dict(manifest_sha256=b.sha_bytes(b'input'),mode='auto',records=[])))
    (tmp_path/'conversion_contract.json').write_text(json.dumps(b.conversion_contract(args)))
    monkeypatch.delenv('DEX_FRAX_ALLOW_SCHEDULER_RESUME',raising=False)
    with pytest.raises(ValueError,match='changed'):b.restore(args,tmp_path,b'input',[],after)
    monkeypatch.setenv('DEX_FRAX_ALLOW_SCHEDULER_RESUME','1')
    b.restore(args,tmp_path,b'input',[],after)
    assert json.loads((tmp_path/'batch_identity.json').read_text())==before
    assert len(list(tmp_path.glob('scheduler_resume.*.json')))==1


@pytest.mark.parametrize('field',['cli.py','ik.py','acceptance.py','initial_reuse.py','v8/convert.py','assets','runtime'])
def test_scheduler_resume_never_allows_solver_or_asset_changes(field):
    before=dict(implementation={'resources.py':'a',field:'a'},assets={'robot':'a'},runtime={'jax':'a'})
    after=copy.deepcopy(before)
    if field in ('assets','runtime'):after[field]={'changed':'b'}
    else:after['implementation'][field]='b'
    with pytest.raises(ValueError,match='changed'):b.scheduler_update(before,after)


def test_explicit_56_gib_backstop_keeps_host_admission_reserve(monkeypatch):
    monkeypatch.setattr(r.sys,'platform','linux');monkeypatch.setattr(r,'workers',lambda _:28)
    monkeypatch.setattr(r,'memory',lambda:dict(total=int(62.4*r.GIB),available=40*r.GIB,cgroup_headroom=62*r.GIB,cgroup_limit=None))
    monkeypatch.delenv('DEX_FRAX_RESOURCE_SCOPE',raising=False)
    monkeypatch.setattr(r.shutil,'which',lambda _:'/usr/bin/systemd-run')
    calls=[]
    monkeypatch.setattr(r.subprocess,'run',lambda command,**kw:(calls.append(command) or SimpleNamespace(returncode=0)))
    r.contain_linux(SimpleNamespace(workers=28,memory_reserve_gib=12,worker_peak_gib=3.75,memory_limit_gib=56))
    assert 'MemoryMax='+str(56*r.GIB) in calls[0]
    assert 'MemorySwapMax=0' in calls[0]
    snap=dict(available=13*r.GIB,cgroup_headroom=40*r.GIB)
    assert not r.can_admit(snap,12*r.GIB,100*r.GIB,80*r.GIB,[],2*r.GIB)[0]


@pytest.mark.parametrize('tamper',[False,True])
def test_scheduler_update_preserves_verified_completed_artifacts(tmp_path,monkeypatch,tamper):
    args=SimpleNamespace(mode='auto',iterations=8,initial_iterations=100,ik_block_frames=128)
    before=dict(converter='dex_frax',version='0.8.0',implementation={'resources.py':'old','ik.py':'same'},assets={'robot':'same'},runtime={'jax':'same'})
    after=copy.deepcopy(before);after['implementation']['resources.py']='new'
    contract=b.conversion_contract(args);folder=tmp_path/'one/frax';folder.mkdir(parents=True)
    (folder/'identity.json').write_text(json.dumps(dict(before,conversion_contract=contract)))
    (folder/'quality.json').write_text(json.dumps(dict(status='accepted',flags=[])))
    (folder/'training.npz').write_bytes(b'unchanged original motion')
    files={p.name:b.sha(p) for p in folder.iterdir()}
    run=dict(status='complete',quality='accepted',flags=[],files=files,acceptance_policy='common_pose_geometry_v1',fingerprint='original')
    (folder/'run.json').write_text(json.dumps(run))
    preserved={p.name:p.read_bytes() for p in folder.iterdir()}
    record=dict(case='one',status='accepted',selected_mode='frax',attempts=[dict(status='complete',quality='accepted',flags=[],output='one/frax',mode='frax')])
    (tmp_path/'batch.json').write_text(json.dumps(dict(manifest_sha256=b.sha_bytes(b'input'),mode='auto',records=[record])))
    (tmp_path/'batch_identity.json').write_text(json.dumps(before))
    (tmp_path/'conversion_contract.json').write_text(json.dumps(contract))
    monkeypatch.setenv('DEX_FRAX_ALLOW_SCHEDULER_RESUME','1')
    if tamper:
        (folder/'training.npz').write_bytes(b'changed')
        with pytest.raises(ValueError,match='artifact changed'):b.restore(args,tmp_path,b'input',[{'id':'one'}],after)
    else:
        restored,_=b.restore(args,tmp_path,b'input',[{'id':'one'}],after)
        assert restored==[record]
        assert {p.name:p.read_bytes() for p in folder.iterdir()}==preserved
