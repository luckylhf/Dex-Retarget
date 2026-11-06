import argparse
import json
import pytest
from dex_frax.resource_policy import ResourcePolicy
from dex_frax.resources import GIB


def make(tmp_path):
    args=argparse.Namespace(worker_peak_gib=3.75,worker_resident_gib=1.,
        hybrid_worker_peak_gib=1.5,v8_worker_peak_gib=1.,frame_memory_kib=32.)
    return ResourcePolicy(args,tmp_path,32)


def test_reload_changes_existing_admission_budget(tmp_path):
    policy=make(tmp_path);before=policy.revision
    assert policy.peak('frax',15000)==GIB+15000*32768
    value=json.loads(policy.path.read_text());value['stages']['frax']['bytes_per_output_frame']=16384
    policy.path.write_text(json.dumps(value))
    assert policy.reload(force=True)
    assert policy.revision!=before and policy.peak('frax',15000)==GIB+15000*16384
    assert policy.shared(['frax','hybrid','v8'])==2.75*GIB
    assert policy.shared(['v8'])==0


@pytest.mark.parametrize('invalid',[{'workers':33},{'compile_extra_gib':-1},
    {'stages':{'frax':dict(base_gib=0,bytes_per_output_frame=0)}},{'unknown':True}])
def test_invalid_reload_keeps_last_valid_policy(tmp_path,invalid):
    policy=make(tmp_path);before=policy.report();value=json.loads(policy.path.read_text());value.update(invalid)
    policy.path.write_text(json.dumps(value))
    assert not policy.reload(force=True)
    assert policy.report()==before
    assert json.loads(policy.events.read_text().splitlines()[-1])['status']=='rejected'


def test_half_written_json_does_not_stop_batch(tmp_path):
    policy=make(tmp_path);before=policy.revision;policy.path.write_text('{')
    assert not policy.reload(force=True) and policy.revision==before


def test_invalid_initial_policy_refuses_startup(tmp_path):
    (tmp_path/'resource_policy.json').write_text('{')
    with pytest.raises(ValueError,match='Initial resource policy'):make(tmp_path)
