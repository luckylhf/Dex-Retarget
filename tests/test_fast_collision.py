import numpy as np
from dex_frax import fast_collision as fast


def test_progressive_step_preserves_positive_approach_bound(monkeypatch):
    matrix=np.eye(3);bounds=np.array([-.07,-.002,.03])
    monkeypatch.setattr(fast.CollisionModel,'linear',lambda *args:(matrix,bounds))
    model=fast.RecoveryModel.__new__(fast.RecoveryModel)
    C,u=model.linear([1,2,3])
    np.testing.assert_array_equal(C,matrix)
    np.testing.assert_array_equal(u,[-.008,-.002,.03])


def test_successful_baseline_does_not_start_extra_solver(monkeypatch):
    x=np.zeros((10,32));report={'windows':[{'status':'repaired'}]}
    monkeypatch.setattr(fast,'baseline_refine',lambda *args:(x,report))
    def fail(*args):raise AssertionError('clean baseline must bypass recovery')
    monkeypatch.setattr(fast,'RecoveryModel',fail)
    out,result=fast.refine(None,{},x,{})
    assert out is x and result is report


def test_active_support_failure_is_not_released(monkeypatch):
    x=np.zeros((10,32));report={'windows':[dict(status='unresolved',reason='active_arm_support_requires_coupled_repair')]}
    monkeypatch.setattr(fast,'baseline_refine',lambda *args:(x,report))
    def fail(*args):raise AssertionError('supported arms stay protected')
    monkeypatch.setattr(fast,'RecoveryModel',fail)
    out,result=fast.refine(None,{},x,{})
    assert out is x and 'recovery' not in result


def test_failed_attempt_consumes_budget_without_committing(monkeypatch):
    x=np.zeros((140,32));t={'root_r':np.tile(np.eye(3),(140,1,1))}
    report={'windows':[dict(status='unresolved',reason='collision_qp_primal infeasible',start=a,end=a+60,sides=['Right']) for a in (0,70)]}
    monkeypatch.setattr(fast,'baseline_refine',lambda *args:(x.copy(),report))
    monkeypatch.setattr(fast,'RecoveryModel',lambda *args:None)
    monkeypatch.setattr(fast,'scan',lambda *args:({'collision_samples':2},None,None))
    calls=[]
    def fail(*args):calls.append(1);raise ValueError('nonlinear_collision_or_floor_clearance')
    monkeypatch.setattr(fast,'_repair_window',fail)
    out,result=fast.refine(None,t,x,{})
    np.testing.assert_array_equal(out,x)
    assert len(calls)==1
    assert result['recovery']['used_frames']==60
    assert result['recovery']['attempts'][1]['reason']=='fast_collision_recovery_budget'
