from types import SimpleNamespace
import pytest
from dex_frax import resources as r
from dex_frax.dynamic_resources import CpuAdmission,process_ticks


def sampler(cpus=4):
    state=dict(now=0.,cpu={i:(1000,500) for i in range(cpus)},own={7:(1,100)})
    obj=CpuAdmission(range(cpus),cpus,clock=lambda:state['now'],read_cpu=lambda _:state['cpu'],read_process=lambda pid:state['own'].get(pid))
    assert obj.sample([7])==0
    def step(busy,own):
        state['now']+=1
        state['cpu']={i:(state['cpu'][i][0]+100,state['cpu'][i][1]+100-busy) for i in range(cpus)}
        state['own'][7]=(1,state['own'][7][1]+own)
        return obj.sample([7])
    return obj,step,state


def test_only_converter_can_fill_all_cpus_and_refill_at_full_load():
    obj,step,_=sampler()
    assert step(0,0)==4
    assert step(100,400)==4  # our own full CPU usage must not block replenishment
    assert obj.report()['external_cpu_equivalents']==0


def test_external_load_reduces_target_and_disappearance_restores_capacity():
    obj,step,_=sampler()
    assert step(100,200)==2
    assert step(100,0)==0
    for _ in range(5):step(50,200)
    assert obj.target==4


def test_process_identity_change_does_not_count_old_cpu_time():
    obj,step,state=sampler()
    state['own'][7]=(2,100000)
    state['now']=1;state['cpu']={i:(1100,500) for i in range(4)}
    assert obj.sample([7])==0


def test_cpu_probe_failure_is_explicit_and_memory_ceiling_still_exists():
    def fail(_):raise OSError('unsupported')
    obj=CpuAdmission(range(8),3,read_cpu=fail)
    assert obj.sample([])==3
    assert obj.report()['reason']=='cpu_sampling_unavailable'


def test_proc_stat_names_with_spaces_and_parentheses(tmp_path):
    p=tmp_path/'7';p.mkdir();fields=['0']*20;fields[0]='S';fields[11]='20';fields[12]='5';fields[19]='123'
    (p/'stat').write_text('7 (a tricky ) name) '+' '.join(fields))
    assert process_ticks(7,tmp_path)==(123,25)


@pytest.mark.parametrize('total',[32,64,128])
def test_memory_target_scales_with_host_and_other_programs(total):
    args=SimpleNamespace(resource_mode='dynamic',memory_target_percent=90,memory_reserve_gib=None)
    snap=dict(total=total*r.GIB,available=int(total*.1*r.GIB),cgroup_headroom=total*r.GIB)
    reserve=r.host_reserve(args,snap)
    assert reserve==pytest.approx(total*.1*r.GIB,abs=1)
    assert not r.can_admit(snap,reserve,100*r.GIB,80*r.GIB,[],r.GIB)[0]
    snap['available']+=2*r.GIB
    assert r.can_admit(snap,reserve,100*r.GIB,80*r.GIB,[],r.GIB)[0]


def test_memory_explicit_minimum_and_inflight_growth_remain_protected():
    args=SimpleNamespace(resource_mode='dynamic',memory_target_percent=90,memory_reserve_gib=12)
    reserve=r.host_reserve(args,{'total':64*r.GIB});assert reserve==12*r.GIB
    snap=dict(available=16*r.GIB,cgroup_headroom=56*r.GIB)
    assert not r.can_admit(snap,reserve,100*r.GIB,80*r.GIB,[(5*r.GIB,r.GIB)],r.GIB)[0]


def test_dynamic_hard_limit_scales_and_does_not_freeze_startup_free_ram(monkeypatch):
    monkeypatch.setattr(r.sys,'platform','linux');monkeypatch.setattr(r,'workers',lambda _:32)
    monkeypatch.setattr(r,'memory',lambda:dict(total=64*r.GIB,available=8*r.GIB,cgroup_headroom=64*r.GIB,cgroup_limit=None))
    monkeypatch.delenv('DEX_FRAX_RESOURCE_SCOPE',raising=False)
    monkeypatch.setattr(r.shutil,'which',lambda _:'/usr/bin/systemd-run')
    calls=[];monkeypatch.setattr(r.subprocess,'run',lambda cmd,**kw:(calls.append(cmd) or SimpleNamespace(returncode=0)))
    args=SimpleNamespace(resource_mode='dynamic',memory_target_percent=90,memory_reserve_gib=None,memory_limit_gib=None,workers=32,worker_peak_gib=3.75)
    r.contain_linux(args)
    assert 'MemoryMax='+str(64*r.GIB-r.host_reserve(args,{'total':64*r.GIB})) in calls[0]
    assert 'MemorySwapMax=0' in calls[0]


def test_batch_cli_defaults_to_dynamic_without_fixed_gib_or_core_count(monkeypatch):
    from dex_frax import cli,managed_batch
    calls=[]
    monkeypatch.setattr('sys.argv',['dex-frax','batch','manifest.json','output'])
    monkeypatch.setattr(r,'contain_linux',lambda args:None)
    monkeypatch.setattr(managed_batch,'run',lambda args:calls.append(args))
    cli.main();args=calls[0]
    assert args.resource_mode=='dynamic' and args.memory_target_percent==90
    assert args.workers=='auto' and args.memory_reserve_gib is None and args.memory_limit_gib is None


def test_priority_restrictions_do_not_break_resource_admission(monkeypatch):
    def denied(_):raise PermissionError('sandbox')
    monkeypatch.setattr(r.os,'nice',denied)
    assert r.lower_priority() is False
