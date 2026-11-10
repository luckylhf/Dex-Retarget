import copy
from types import SimpleNamespace
import numpy as np
import pytest
from dex_frax.arm_policy import POLICY,collision_options
from dex_frax.similarity import WholeArmObjective,score
from dex_frax.adaptive_refine import pose_mask,policy


def test_elbow_40_gate_preserves_duration_and_arm_20_gate():
    errors={'Left':{k:np.zeros(10) for k in ('upper','lower','hand_forward','hand_width','elbow_axis')}}
    cfg=policy();assert cfg['elbow_axis_deg']==40.
    errors['Left']['elbow_axis'][:5]=39.9
    assert not pose_mask(errors,50,cfg).any()
    errors['Left']['elbow_axis'][:5]=40.1
    assert pose_mask(errors,50,cfg).sum()==5
    errors['Left']['elbow_axis'][:]=0
    errors['Left']['upper'][:5]=20.1
    assert pose_mask(errors,50,cfg).sum()==5


def test_whole_arm_score_rejects_palm_polish_that_creates_large_elbow_error():
    old={k:np.full(100,v) for k,v in dict(upper=10.,lower=10.,hand_forward=2.,hand_width=2.,elbow_axis=38.).items()}
    new=copy.deepcopy(old);new['hand_forward'][:]=.1;new['hand_width'][:]=.1;new['elbow_axis'][:]=46.
    assert score(new)>score(old)


def test_worst_tail_matters_even_when_forearm_mean_improves():
    old={k:np.full(100,v) for k,v in dict(upper=10.,lower=15.,hand_forward=2.,hand_width=2.,elbow_axis=20.).items()}
    new=copy.deepcopy(old);new['lower'][:]=5.;new['lower'][:10]=35.
    assert new['lower'].mean()<old['lower'].mean()
    assert score(new)>score(old)


@pytest.mark.parametrize('degrees',[.1,4.,6.,30.,45.])
def test_whole_arm_residual_jacobian(degrees):
    class Objective:
        w=dict(arm_direction=5.,hand_direction=5.,hand_width=5.,elbow_axis=1.6)
        t={'sides':{'Left':{}}};side='Left';i=0
        def evaluate(self,q):
            r=[];j=[]
            for w in [5.,5.,5.,5.,1.,1.6]:
                r.extend(w*np.array([np.cos(q[0])-1,np.sin(q[0]),0.]))
                j.extend(w*np.array([-np.sin(q[0]),np.cos(q[0]),0.]))
            return np.array(r),np.array(j)[:,None]
    objective=WholeArmObjective(Objective());q=np.array([np.deg2rad(degrees)])
    values,J=objective.evaluate(q);eps=1e-6
    numerical=(objective.fun(q+eps)-objective.fun(q-eps))/(2*eps)
    assert np.allclose(J[:,0],numerical,atol=1e-6)
    assert (values[-1]>0)==(degrees>40.)


def test_collision_tiers_keep_waist_fixed_and_distal_joints_available():
    for tier in (10.,20.,30.):
        options=collision_options(tier)
        assert options['joint_deg']==tier and options['free_distal']
        assert not options['coupled'] and options['whole_arm']
    with pytest.raises(ValueError):collision_options(35.)


@pytest.mark.parametrize('supported',[False,True])
def test_graduated_recovery_routes_support_separately(monkeypatch,supported):
    from dex_frax import graduated_collision as g,arm_policy
    x=np.zeros((12,32));t={'fps':50.,'root_r':np.tile(np.eye(3),(12,1,1))}
    report={'after':{'collision_samples':1},'windows':[dict(start=0,end=12,sides=['Left'],status='unresolved')]}
    monkeypatch.setattr(g,'IncrementalRecovery',lambda *_:SimpleNamespace())
    monkeypatch.setattr(g,'scan',lambda *a:({'collision_samples':1,'minimum_distance_m':-.01},[],set()))
    monkeypatch.setattr(arm_policy,'active_arm_support',lambda *a:supported)
    calls=[]
    def repair(*a,options):
        calls.append(options);raise ValueError('test_no_feasible_candidate')
    monkeypatch.setattr(g,'_repair_window',repair)
    out,result=g.recover(None,t,x,dict(arm_repair_policy=POLICY),None,report)
    assert [c['joint_deg'] for c in calls]==([10.,20.,35.] if supported else [10.,20.,30.])
    assert calls[-1]['coupled']==supported
    np.testing.assert_array_equal(out,x)
