from dex_frax.memory_queue import MemoryQueue, pressure
from dex_frax.resources import GIB, MIB


def test_pressure_floor_applies_with_plenty_of_host_ram():
    assert pressure(dict(available=50*GIB,cgroup_headroom=16*MIB,cgroup_limit=8*GIB),6*GIB)==(True,True)
    assert pressure(dict(available=5*GIB,cgroup_headroom=20*GIB),6*GIB)==(False,True)


def test_pressure_backoff_survives_resume_and_recovers_gradually():
    queue=MemoryQueue(32)
    queue.defer('frax','lafan',1000,32768,3*GIB,24)
    assert queue.limit==23
    guarded=queue.peak('frax','lafan',2000,32768,GIB)
    assert guarded>3.75*GIB and queue.peak('frax','soma',2000,32768,GIB)==GIB
    resumed=MemoryQueue(32,queue.report())
    assert resumed.peak('frax','lafan',2000,32768,GIB)==guarded
    resumed.complete();assert resumed.limit==23
    resumed.complete();assert resumed.limit==24


def test_external_pressure_does_not_compound_the_same_observed_peak():
    queue=MemoryQueue(32)
    for _ in range(10):queue.defer('frax','lafan',1000,32768,3*GIB,8)
    assert queue.peak('frax','lafan',1000,32768,GIB)==int(3*GIB*1.25)+128*MIB
