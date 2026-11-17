from types import SimpleNamespace
from dex_frax.stage_memory import StageMemory
from dex_frax.resources import GIB


class Slot:
    def __init__(self):self.memory_request={'stage':'audit'};self.wait_started=0;self.grants=0
    def grant_memory(self):self.grants+=1;self.memory_request=None


def pool(available):return dict(available=available*GIB,cgroup_headroom=available*GIB)


def item(name,grant=2,completion=8):return dict(case=name,grant=grant*GIB,completion_peak=completion*GIB)


def test_full_pool_still_has_space_for_one_completion():
    broker=StageMemory();owner=Slot();active={owner:item('owner')};usage={owner:2*GIB}
    broker.register(owner,active[owner])
    # 48 GiB pool: one 8 GiB entitlement and twenty normal 2 GiB grants.
    for i in range(20):
        slot=Slot()
        assert broker.can_start(pool(48-sum(usage.values())/GIB),0,active,usage,2*GIB,8*GIB,0,0)
        active[slot]=item(str(i));usage[slot]=2*GIB;broker.register(slot,active[slot])
    snapshot=pool(48-sum(usage.values())/GIB)
    assert not broker.can_start(snapshot,0,active,usage,2*GIB,8*GIB,0,0)
    normal=next(s for s in active if s is not owner)
    broker.request(normal,active[normal],4*GIB,'audit')
    assert not broker.can_grant(normal,active[normal],snapshot,0,active,usage,0)
    assert normal.grants==0
    broker.request(owner,active[owner],8*GIB,'audit')
    assert broker.can_grant(owner,active[owner],snapshot,0,active,usage,0)
    broker.grant(owner,active[owner]);assert owner.grants==1
    # Completion frees RAM; a waiting task gets the next completion entitlement.
    del active[owner];del usage[owner];broker.elect(active,usage)
    assert broker.protected is normal
    assert broker.can_grant(normal,active[normal],pool(8),0,active,usage,0)


def test_length_increases_stage_and_completion_budgets(tmp_path):
    from dex_frax.resource_policy import ResourcePolicy
    policy=ResourcePolicy(SimpleNamespace(worker_peak_gib=4,worker_resident_gib=1,
        completion_worker_gib=8,frame_memory_kib=32),tmp_path,32)
    assert policy.stage_peak('frax',20000,'initial')>policy.stage_peak('frax',1000,'initial')
    assert policy.completion_peak('frax',20000)>policy.completion_peak('frax',1000)
    assert policy.completion_peak('frax',1000)>policy.stage_peak('frax',1000,'initial')
    policy.value['phase_bases_gib']={'audit':12.}
    assert policy.completion_peak('frax',1000)>12*GIB


def test_notifications_under_compiler_lock_do_not_block(monkeypatch):
    from dex_frax import resource_events as events
    calls=[];monkeypatch.setattr(events,'_gate',calls.append)
    monkeypatch.setattr(events,'_sink',None);monkeypatch.setattr(events,'process_memory',lambda _:0)
    events.phase('waiting_compile');events.phase('ik_prepare');events.phase('ik_compile');events.phase('ik_solve')
    assert [c['phase'] for c in calls]==['waiting_compile']


def test_warm_cache_must_be_reclaimed_before_budget_is_reissued():
    broker=StageMemory();owner=Slot();active={owner:item('next')};usage={owner:2*GIB}
    broker.elect(active,usage);broker.request(owner,active[owner],8*GIB,'initial')
    # A completed process still has 4 GiB cached; no imaginary release credit.
    assert not broker.can_grant(owner,active[owner],pool(4),0,active,usage,0)
    assert broker.can_grant(owner,active[owner],pool(8),0,active,usage,0)


def test_first_completed_sample_can_raise_but_not_lower_estimates(tmp_path):
    from dex_frax.memory_learning import MemoryLearning
    from dex_frax.resource_policy import ResourcePolicy
    policy=ResourcePolicy(SimpleNamespace(worker_peak_gib=2,worker_resident_gib=1,
        frame_memory_kib=32),tmp_path,32)
    learner=MemoryLearning(tmp_path,{})
    slot=SimpleNamespace(telemetry={})
    response={'status':'complete'};case={'source':'lafan'}
    learner.complete(slot,dict(mode='frax',phase_peaks={'audit|jax_hot':4*GIB}),response,case,1000,policy)
    assert learner.predict('frax','lafan','jax_hot',1000,policy)>4*GIB
    assert learner.stage_peak('frax','lafan','jax_hot',1000,policy,'audit')>4*GIB
    learner.profiles={}
    learner.complete(slot,dict(mode='frax',phase_peaks={'audit|jax_hot':GIB//2}),response,case,1000,policy)
    assert learner.predict('frax','lafan','jax_hot',1000,policy)==policy.peak('frax',1000)


def test_blocked_stage_returns_unused_credits_without_losing_completion_path():
    broker=StageMemory();owner=Slot();a=Slot();b=Slot()
    active={owner:item('owner'),a:item('a',4),b:item('b',4)}
    usage={owner:2*GIB,a:GIB,b:GIB}
    for slot in active:broker.register(slot,active[slot])
    broker.request(a,active[a],5*GIB,'audit',retained=GIB)
    assert not broker.can_grant(a,active[a],pool(12),0,active,usage,0)
    broker.request(b,active[b],5*GIB,'audit',retained=GIB)
    assert active[b]['grant']<2*GIB
    assert broker.can_grant(a,active[a],pool(12),0,active,usage,0)
    broker.grant(a,active[a])
    assert not broker.can_grant(b,active[b],pool(12),0,active,usage,0)
    broker.request(owner,active[owner],8*GIB,'audit',retained=2*GIB)
    assert broker.can_grant(owner,active[owner],pool(12),0,active,usage,0)
