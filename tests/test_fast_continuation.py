import copy,json
from pathlib import Path
import pytest
from dex_frax.refine_resume import verify,locate
from dex_frax.cli import sha


def completed_fast(tmp_path):
    identity=dict(input_sha256='source',assets={'model':'a'},implementation={'solver':'b'},frax_implementation={},runtime={},
        conversion_contract=dict(profile='fast',unified_repair=False,repair_budget=None,fps=50.,max_frames=None))
    (tmp_path/'identity.json').write_text(json.dumps(identity))
    names=['motion.pkl','motion.initial.npz','motion.json','motion.diagnostics.npz','quality.json']
    for name in names:(tmp_path/name).write_text(name)
    (tmp_path/'run.json').write_text(json.dumps(dict(status='complete',files={name:sha(tmp_path/name) for name in names})))
    target=copy.deepcopy(identity);target['conversion_contract'].update(profile='refine',unified_repair=True,repair_budget=20.)
    return target


def test_completed_fast_can_initialize_refine_without_repeating_ik(tmp_path):
    target=completed_fast(tmp_path)
    assert locate(tmp_path)==tmp_path and verify(tmp_path,target)=={'from_fast':True}


@pytest.mark.parametrize('file',['motion.pkl','motion.initial.npz','motion.diagnostics.npz'])
def test_fast_continuation_rejects_changed_artifact(tmp_path,file):
    target=completed_fast(tmp_path);(tmp_path/file).write_text('changed')
    with pytest.raises(ValueError,match='Fast artifact hash changed'):verify(tmp_path,target)


def test_fast_continuation_keeps_preparation_and_code_identity_strict(tmp_path):
    target=completed_fast(tmp_path);target['conversion_contract']['max_frames']=500
    with pytest.raises(ValueError,match='contract changed'):verify(tmp_path,target)
    target=completed_fast(tmp_path);target['implementation']['solver']='new'
    with pytest.raises(ValueError,match='implementation'):verify(tmp_path,target)


def test_smplx_continuation_carries_exact_prepared_geometry_through_multiple_hops(tmp_path):
    from types import SimpleNamespace
    from dex_frax.smplx_cli import prepare
    first=tmp_path/'first';first.mkdir();completed_fast(first)
    (first/'input').mkdir();source=first/'input/human.npz';source.write_bytes(b'unchanged prepared geometry')
    for name in ('second','third'):
        folder=tmp_path/name
        args=SimpleNamespace(continue_from=str(first))
        result=prepare(args,folder)
        assert result==folder/'input/human.npz' and result.read_bytes()==source.read_bytes()
        completed_fast(folder)
        first=folder


def test_explicit_recheck_allows_only_project_code_change(tmp_path):
    target=completed_fast(tmp_path);target['implementation']['solver']='new'
    assert verify(tmp_path,target,recheck=True)=={'from_fast':True}
    for key,value in [('input_sha256','other'),('assets',{'model':'other'}),('runtime',{'python':'other'}),('frax_implementation',{'ik':'other'})]:
        changed=copy.deepcopy(target);changed[key]=value
        with pytest.raises(ValueError,match='identity mismatch'):verify(tmp_path,changed,recheck=True)


@pytest.mark.parametrize('name',['motion.json','motion.diagnostics.npz','quality.json'])
def test_recheck_does_not_trust_changed_evidence(tmp_path,name):
    target=completed_fast(tmp_path);target['implementation']['solver']='new'
    (tmp_path/name).write_text('tampered')
    with pytest.raises(ValueError,match='Recheck artifact hash changed'):verify(tmp_path,target,recheck=True)


@pytest.mark.parametrize('name',['quality_pose.json','adaptive_policy.json','acceptance.py','v8/self_collision.py'])
def test_recheck_rejects_changed_safety_policy_even_with_same_version(tmp_path,name):
    target=completed_fast(tmp_path);target['implementation'][name]='changed'
    with pytest.raises(ValueError,match='Recheck safety policy changed'):verify(tmp_path,target,recheck=True)
