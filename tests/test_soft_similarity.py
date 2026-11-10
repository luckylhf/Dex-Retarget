import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dex_frax import experimental_arm_refine as e
from dex_frax.similarity import score,palm_envelope
from dex_frax.unified_repair import quality_guard


def errors(upper=26.,lower=23.,palm=1.):
    return {k:np.full(20,v,dtype=float) for k,v in dict(upper=upper,lower=lower,
        hand_forward=palm,hand_width=palm,elbow_axis=20.).items()}


def test_soft_score_allows_small_palm_trade_for_large_shape_improvement():
    assert score(errors(18,17,2))<score(errors())
    assert score(errors(40,35,0))>score(errors())
    before={'Left':errors()};after={'Left':errors(18,17,2)}
    assert palm_envelope(before,after) is None
    after['Left']['hand_forward'][10]=25.1
    assert palm_envelope(before,after).startswith('palm_absolute_envelope')


def test_preexisting_bad_palm_is_not_permission_to_make_it_worse():
    before={'Left':errors(palm=30)};after=copy.deepcopy(before)
    after['Left']['hand_forward'][0]=31
    assert palm_envelope(before,after) is not None


def quality():
    return dict(flags=['Left_upper_deg_source_mismatch'],adaptive_pose_gate=dict(violating_frames=5),
        source_pose_audit={k:dict(p95=4.,max=10.) for k in ('root_orientation_deg','torso_orientation_deg','torso_up_deg')},
        self_collision=dict(collision_samples=0,native_colliding_frames=0,interpolated_colliding_intervals=0,minimum_distance_m=.003))


def test_candidate_similarity_trade_does_not_change_export_gate():
    before=quality();after=copy.deepcopy(before)
    after['flags'].append('Left_lower_deg_source_mismatch')
    after['adaptive_pose_gate']['violating_frames']=6
    assert quality_guard(before,after) is not None
    assert quality_guard(before,after,soft_similarity=True) is None
    assert after['flags']  # The final quality remains review; never stripped.


@pytest.mark.parametrize('field',['collision','waist','speed'])
def test_soft_policy_does_not_bypass_safety_or_waist(field):
    before=quality();after=copy.deepcopy(before)
    if field=='collision':after['self_collision']['collision_samples']=1
    elif field=='waist':after['source_pose_audit']['torso_orientation_deg']['max']=11.
    else:after['flags'].append('joint_velocity_exceeds_model_limit')
    assert quality_guard(before,after,soft_similarity=True) is not None


def test_solver_soft_guard_releases_direction_envelope_but_keeps_dense_collision(monkeypatch):
    robot=SimpleNamespace(sides={'Left':dict(arm=list(range(3,10)))},lower=np.full(32,-2.),upper=np.full(32,2.),speed_limits=np.full(29,10.))
    robot.set=lambda *a:None
    baseline=np.zeros((3,32));target=dict(fps=50.,root_r=np.tile(np.eye(3),(3,1,1)))
    values=np.zeros((3,6));values[:,0]=5.;values[:,2:4]=2.
    monkeypatch.setattr(e,'features',lambda *a:values)
    monkeypatch.setattr(e,'scan',lambda *a:(dict(collision_samples=0,minimum_distance_m=.004),None,None))
    monkeypatch.setattr(e,'_arm_bodies',lambda *a:[])
    args=(robot,target,baseline,baseline.copy(),'Left',np.zeros((3,6)),None,.003)
    assert not e.regression_guard(*args)[0]
    assert e.regression_guard(*args,soft_similarity=True)[0]
    monkeypatch.setattr(e,'scan',lambda *a:(dict(collision_samples=1,minimum_distance_m=-.001),None,None))
    assert e.regression_guard(*args,soft_similarity=True)[1]=='dense_collision_regression'


def test_similarity_mode_is_part_of_resume_identity():
    from dex_frax.contract import conversion_contract
    args=SimpleNamespace(mode='auto',iterations=8,initial_iterations=100,repair_similarity='soft')
    soft=conversion_contract(args);args.repair_similarity='conservative'
    assert soft!=conversion_contract(args)


@pytest.mark.parametrize('tier',[10.,20.,30.])
def test_resemblance_bounds_do_not_escalate_by_collision_tier(tier):
    r=SimpleNamespace(lower=np.full(32,-2.),upper=np.full(32,2.))
    ids=np.arange(3,10);ref=np.zeros((4,7))
    lo,hi=e.limits(r,ids,ref,tier,soft_similarity=True)
    assert np.allclose(hi,2.)
    assert np.allclose(lo,-2.)
    r.upper[7]=.15;r.lower[8]=-.05
    lo,hi=e.limits(r,ids,ref,tier,soft_similarity=True)
    assert np.all(hi[:,4]<=.15) and np.all(lo[:,5]>=-.05)


@pytest.mark.parametrize('angle',[10.,30.,60.])
def test_difficult_shape_soft_penalty_jacobian_matches_finite_difference(angle):
    from dex_frax.similarity import DifficultShapeObjective
    weight=6.
    class Objective:
        def evaluate(self,q):
            a=q[0]
            r=weight*np.array([np.cos(a)-1,np.sin(a),0.,0.,0.,0.])
            J=weight*np.array([-np.sin(a),np.cos(a),0.,0.,0.,0.])[:,None]
            return r,J
    wrapped=DifficultShapeObjective(Objective(),weight);q=np.array([np.deg2rad(angle)])
    residual,jac=wrapped.evaluate(q);eps=1e-6
    numeric=(wrapped.evaluate(q+eps)[0]-wrapped.evaluate(q-eps)[0])/(2*eps)
    assert np.allclose(jac[:,0],numeric,atol=1e-6)
    assert (residual[-2]>0)==(angle>20.)
    assert residual[-1]==0.
