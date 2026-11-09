import argparse
from decimal import Decimal
from types import SimpleNamespace
import pytest
from dex_frax.dispatch import duration_seconds,route,deduplicate
from dex_frax.memory_learning import MemoryLearning
from dex_frax.resource_policy import ResourcePolicy
from dex_frax.resources import GIB


def test_duration_uses_source_time_not_resampled_frames(tmp_path):
    path=tmp_path/'motion.bvh';path.write_text('Frames: 7200\nFrame Time: 0.008333333333333333333333333334\n')
    assert duration_seconds(path)>=Decimal(60)
    assert route('auto',Decimal('59.999'))==('frax','v8')
    assert route('auto',Decimal(60))==('frax','v8')
    assert route('auto',Decimal(600),single=True)==('frax','v8')
    assert route('v8',Decimal(600))==('v8',)
    with pytest.raises(ValueError,match='Unknown conversion mode'):route('hybrid',Decimal(1))


def test_identical_sources_are_processed_once_with_aliases():
    rows=[dict(id='a',input='one.bvh',input_sha256='same',source='soma'),
        dict(id='b',input='copy.bvh',input_sha256='same',source='soma')]
    unique,aliases=deduplicate(rows)
    assert len(unique)==1 and aliases[0]['canonical']=='a' and aliases[0]['id']=='b'
    rows[1]['source']='lafan'
    with pytest.raises(ValueError,match='conflicting'):deduplicate(rows)


def test_memory_learns_completed_resident_phases_not_compile(tmp_path):
    policy=ResourcePolicy(argparse.Namespace(worker_peak_gib=4.,worker_resident_gib=1.25,
        v8_worker_peak_gib=1.,frame_memory_kib=32),tmp_path,32)
    learn=MemoryLearning(tmp_path,{'version':'test'})
    slot=SimpleNamespace(telemetry={'cache_state':'jax_hot'})
    item=dict(mode='frax',phase_peaks={'ik_compile|jax_imported':int(4*GIB),'fast_refinement|jax_hot':int(.9*GIB)})
    before=learn.predict('frax','lafan','jax_hot',1000,policy)
    learn.complete(slot,item,{'status':'complete'},{'source':'lafan'},1000,policy)
    assert learn.predict('frax','lafan','jax_hot',1000,policy)==before
    learn.complete(slot,item,{'status':'complete'},{'source':'lafan'},1000,policy)
    learned=learn.predict('frax','lafan','jax_hot',1000,policy)
    assert .9*GIB<learned<1.1*GIB
    assert policy.shared([False,False])==0 and policy.shared([False,True])>0
    assert learn.compiler_extra('frax','lafan',policy)>3.1*GIB
    assert learn.compiler_extra('frax','soma',policy)==int(2.75*GIB)
    # Changing the hot per-frame coefficient cannot undercut recorded peaks.
    policy.value['stages']['frax']['bytes_per_output_frame']=0
    assert learn.predict('frax','lafan','jax_hot',1000,policy)>.9*GIB
    assert MemoryLearning(tmp_path,{'version':'test'}).profiles==learn.profiles
    assert not MemoryLearning(tmp_path,{'version':'new'}).profiles


def test_failed_or_unmeasured_jobs_never_lower_budgets(tmp_path):
    learner=MemoryLearning(tmp_path,{});slot=SimpleNamespace(telemetry={})
    policy=SimpleNamespace(value={'stages':{'v8':{'bytes_per_output_frame':32768}}})
    item=dict(mode='v8',phase_peaks={'starting|pure':int(.4*GIB)})
    learner.complete(slot,item,{'status':'complete'},{'source':'soma'},100,policy)
    learner.complete(slot,item,{'status':'failed'},{'source':'soma'},100,policy)
    assert not learner.profiles
