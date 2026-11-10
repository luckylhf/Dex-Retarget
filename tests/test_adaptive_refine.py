from types import SimpleNamespace
import numpy as np
import pytest
from dex_frax import adaptive_refine as adaptive


def scene():
    n=80
    r=SimpleNamespace(lower=np.full(32,-3.), upper=np.full(32,3.),
                      sides={'Left':dict(leg=[3,4,5,6,7,8]),'Right':dict(leg=[9,10,11,12,13,14])},
                      speed_limits=np.full(29,5.),joint_step=lambda fps,cfg:np.full(29,.1))
    t=dict(fps=50.,hips=np.zeros((n,3)),root_r=np.tile(np.eye(3),(n,1,1)),
           sides={'Left':dict(segments=[]),'Right':dict(segments=[])},body_contacts=[])
    return r,t,np.zeros((n,32))


def fake_metrics(r,t,x):
    n=len(x)
    errors={s:{k:np.zeros(n) for k in ('upper','lower','hand_forward','hand_width','elbow_axis')}
            for s in ('Left','Right')}
    return dict(contact=abs(x[:,0]),floor=np.zeros(n),body_floor=np.zeros(n),errors=errors)


def test_clean_clip_bypasses_all_stages(monkeypatch):
    r,t,x=scene();monkeypatch.setattr(adaptive,'inspect',fake_metrics)
    def fail(*args):raise AssertionError('clean clip must not run V8 refiners')
    monkeypatch.setattr(adaptive,'_stage',fail)
    out,report=adaptive.refine(r,t,x,{})
    np.testing.assert_array_equal(out,x)
    assert report['repairs']==[]


def test_local_support_changes_only_checked_window(monkeypatch):
    r,t,x=scene();x[35,0]=.0006
    monkeypatch.setattr(adaptive,'inspect',fake_metrics)
    calls=[]
    def stage(r,t,q,w,kind):
        calls.append((kind,len(q)));v=q.copy();v[:,0]=0.;return v,[]
    monkeypatch.setattr(adaptive,'_stage',stage)
    out,report=adaptive.refine(r,t,x,{})
    assert calls==[('support',41)]
    assert out[35,0]==0.
    np.testing.assert_array_equal(out[:15],x[:15]);np.testing.assert_array_equal(out[56:],x[56:])
    assert report['repairs'][0]['status']=='accepted'


def test_bad_local_join_rolls_back(monkeypatch):
    r,t,x=scene();x[35,0]=.0006
    monkeypatch.setattr(adaptive,'inspect',fake_metrics)
    def stage(r,t,q,w,kind):
        v=q.copy();v[:,0]=0.;v[len(v)//2,3]=1.;return v,[]
    monkeypatch.setattr(adaptive,'_stage',stage)
    out,report=adaptive.refine(r,t,x,{})
    np.testing.assert_array_equal(out,x)
    assert report['repairs'][0]['reason']=='join_or_velocity_failure'


def test_hard_failure_escalates_to_full_clip(monkeypatch):
    r,t,x=scene();x[35,0]=.002
    monkeypatch.setattr(adaptive,'inspect',fake_metrics)
    calls=[]
    def stage(r,t,q,w,kind):
        calls.append(len(q))
        if len(q)<len(x):raise ValueError('local_infeasible')
        v=q.copy();v[:,0]=0.;return v,[]
    monkeypatch.setattr(adaptive,'_stage',stage)
    out,report=adaptive.refine(r,t,x,{},config={**adaptive.policy(),'allow_full_clip_fallback':True})
    assert calls==[41,80]
    assert report['repairs'][-1]['scope']=='full_clip'
    assert report['after']['max_contact_mm']==0.


def test_window_preserves_static_geometry_and_source_contacts():
    r,t,x=scene()
    t['body_contacts']=[dict(robot_hull=np.ones((80,3)),mode=np.arange(80))]
    t['sides']['Left']['segments']=[dict(start=5,end=40)]
    result=adaptive.window_targets(t,10,30,80)
    assert result['hips'].shape==(20,3)
    assert result['body_contacts'][0]['robot_hull'].shape==(80,3)
    np.testing.assert_array_equal(result['body_contacts'][0]['mode'],np.arange(10,30))
    assert result['sides']['Left']['segments']==[dict(start=0,end=20)]
    result['hips'][:]=1.
    assert np.all(t['hips']==0.)


def test_stage_restores_bounds_and_diagnostics_even_on_failure(monkeypatch):
    r,t,x=scene();before=r.lower.copy();r.support_fast_report={'old':True}
    def fail(robot,*args):
        assert robot.lower[6]==0.
        robot.support_fast_report={'window':True}
        robot.support_branch_relaxations=['local']
        raise ValueError('intentional')
    monkeypatch.setattr(adaptive,'constrain_support',fail)
    with pytest.raises(ValueError,match='intentional'):
        adaptive._stage(r,t,x,{},'support')
    np.testing.assert_array_equal(r.lower,before)
    assert r.support_fast_report=={'old':True}
    assert not hasattr(r,'support_branch_relaxations')


def test_elbow_gate_requires_sustained_error():
    r,t,x=scene();errors=fake_metrics(r,t,x)['errors'];cfg=adaptive.policy()
    errors['Left']['elbow_axis'][10:14]=41.
    assert not adaptive.pose_mask(errors,50.,cfg).any()
    errors['Left']['elbow_axis'][14]=41.
    assert adaptive.pose_mask(errors,50.,cfg).sum()==5


def test_new_knee_hyperextension_rejected():
    r,t,x=scene();candidate=x.copy();candidate[35,6]=-.01
    with pytest.raises(ValueError,match='new_knee_hyperextension'):
        adaptive.check_candidate(r,t,x,candidate,{},adaptive.policy(),fake_metrics(r,t,x))


def test_active_body_contact_dispatches_body_stage(monkeypatch):
    r,t,x=scene();mode=np.zeros(80);mode[35]=1
    t['body_contacts']=[dict(mode=mode)]
    monkeypatch.setattr(adaptive,'inspect',fake_metrics)
    calls=[]
    def stage(r,t,q,w,kind):calls.append(kind);return q.copy(),[]
    monkeypatch.setattr(adaptive,'_stage',stage)
    _,report=adaptive.refine(r,t,x,{})
    assert calls==['body']
    assert report['repairs'][0]['stage']=='body'


def test_arm_repair_does_not_release_body_contact(monkeypatch):
    r,t,x=scene();mode=np.zeros(80);mode[35]=1
    t['body_contacts']=[dict(mode=mode)]
    def metrics(r,t,q):
        v=fake_metrics(r,t,q);v['errors']['Left']['elbow_axis'][30:40]=41.;return v
    monkeypatch.setattr(adaptive,'inspect',metrics)
    calls=[]
    def stage(r,t,q,w,kind):calls.append(kind);return q.copy(),[]
    monkeypatch.setattr(adaptive,'_stage',stage)
    out,report=adaptive.refine(r,t,x,{})
    assert calls==['body']
    assert report['after']['sustained_arm_bad_frames']==10
    assert any(v.get('reason') in ('active_body_contact_requires_coupled_pose_search','fast_mode_full_clip_disabled_use_v8') for v in report['repairs'])
    np.testing.assert_array_equal(out,x)


def test_default_fast_route_never_runs_full_clip_fallback(monkeypatch):
    r,t,x=scene();x[35,0]=.002
    monkeypatch.setattr(adaptive,'inspect',fake_metrics)
    calls=[]
    def fail(r,t,q,w,stage):calls.append(len(q));raise ValueError('not_repairable')
    monkeypatch.setattr(adaptive,'_stage',fail)
    out,report=adaptive.refine(r,t,x,{})
    assert calls==[41]
    np.testing.assert_array_equal(out,x)
    assert report['repairs'][-1]['reason']=='fast_mode_full_clip_disabled_use_v8'


def test_v8_work_budget_does_not_relax_acceptance(monkeypatch):
    r,t,x=scene();x[35,0]=.002
    monkeypatch.setattr(adaptive,'inspect',fake_metrics)
    def fail(*args):raise AssertionError('budget must stop solver dispatch')
    monkeypatch.setattr(adaptive,'_stage',fail)
    cfg={**adaptive.policy(),'maximum_v8_windows':0}
    out,report=adaptive.refine(r,t,x,{},config=cfg)
    np.testing.assert_array_equal(out,x)
    assert report['after']['max_contact_mm']==2.
    assert report['repairs'][0]['reason']=='fast_v8_work_budget'


def test_bounded_window_shrinks_padding_before_dropping_defects():
    score=np.zeros(300);score[70:145]=1.
    a,b=adaptive.bounded_window(50,165,score,100,20)
    assert b-a<=100 and a<=68 and b>=147
    assert np.count_nonzero(score[a+2:b-2])==75


def test_bounded_window_targets_largest_residual_mass():
    score=np.zeros(300);score[20:30]=1.;score[200:220]=5.
    a,b=adaptive.bounded_window(0,260,score,70,20)
    assert b-a==70 and a+2<=200 and b-2>=220


def test_body_clearance_takes_priority_over_subthreshold_foot_error(monkeypatch):
    r,t,x=scene();x[35,1]=.004
    def metrics(r,t,q):
        m=fake_metrics(r,t,q)
        m['body_floor']=-abs(q[:,1]);m['floor']=m['body_floor'].copy()
        m['foot_floor']=np.zeros(len(q));return m
    monkeypatch.setattr(adaptive,'inspect',metrics)
    calls=[]
    def stage(r,t,q,w,kind):
        calls.append(kind);out=q.copy();out[:,1]=0.;return out,[]
    monkeypatch.setattr(adaptive,'_stage',stage)
    out,report=adaptive.refine(r,t,x,{})
    assert calls==['body'] and out[35,1]==0.
    assert report['work_budget']['used_frames']<=report['work_budget']['frame_budget']


def test_mixed_knee_signs_get_per_frame_existing_branch_corridor(monkeypatch):
    r,t,x=scene();j=r.sides['Left']['leg'][3];x[10,j]=-.1
    captured={}
    def body(r,target,q,w,**kwargs):
        captured.update(target);return q.copy(),[]
    monkeypatch.setattr(adaptive,'refine_body',body)
    adaptive._stage(r,t,x,{},'body')
    assert captured['repair_lower_bounds'][0,j]==0.
    assert captured['repair_lower_bounds'][10,j]==r.lower[j]
    assert 'repair_lower_bounds' not in t


def test_final_candidate_uses_motor_contract_not_numerical_solver_cap(monkeypatch):
    r,t,x=scene();r.speed_limits[:]=10.
    after=x.copy();after[35,3]=.15  # 7.5 rad/s is above the numeric5 but below motor10.
    monkeypatch.setattr(adaptive,'inspect',fake_metrics)
    adaptive.check_candidate(r,t,x,after,{},adaptive.policy(),fake_metrics(r,t,x))
    after[35,3]=.21
    with pytest.raises(ValueError,match='join_or_velocity_failure'):
        adaptive.check_candidate(r,t,x,after,{},adaptive.policy(),fake_metrics(r,t,x))
