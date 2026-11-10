"""Geometry, objective parity, bypass and repaired-history speed safeguards."""
import json
from pathlib import Path
import numpy as np
import pytest
from dex_frax.v8.support_fast import two_link_knee,SupportObjective,constrain_support
from dex_frax.v8.solver import Objective,constrain_support as baseline
from dex_frax.v8.robot import Robot
from dex_frax.v8.anatomy import configure


@pytest.mark.parametrize('target',[[.2,0,-.7],[0,0,-1.5],[0,0,0],[0,0,-1.]])
def test_geometric_hint_is_finite_and_preserves_proximal_length(target):
    a=np.zeros(3);b=np.array([.3,0,-.4]);c=np.array([0,0,-.8])
    knee=two_link_knee(a,b,c,target)
    assert np.isfinite(knee).all()
    assert abs(np.linalg.norm(knee-a)-np.linalg.norm(b-a))<1e-8
    assert knee[0]>=-1e-10


@pytest.fixture(scope='module')
def sample():
    import dex_frax.v8.solver as module
    cfg=json.loads(Path(module.__file__).with_name('config.json').read_text())
    # Generate deterministic targets from the shipped real Dex geometry. No
    # private BONES motion, pickle download or preceding benchmark is needed.
    from dex_frax.paths import ASSETS
    assert (ASSETS/'mjcf/dex_29.xml').is_file(), (
        'Real-model tests require shipped assets/dex_evt; set DEX_ASSETS or '
        'DEX_SUPERMIMIC_ROOT to the package root. These tests must not skip.')
    r=Robot();configure(r,cfg['anatomy'])
    pose=np.r_[0.,0.,1.,np.zeros(29)]
    for g in r.sides.values():
        pose[np.array(g['leg'])[[0,3,4]]]=[-.2,.4,-.2]
        pose[g['arm'][3]]=-.4
    r.set(pose,np.eye(3))
    pose[2]-=min(r.point(g['foot'],r.lowest(s))[2] for s,g in r.sides.items())
    r.set(pose,np.eye(3));n=8
    def repeat(v):return np.repeat(np.asarray(v)[None],n,axis=0)
    def direction(a,b):
        v=r.point(b)-r.point(a);return repeat(v/np.linalg.norm(v))
    t=dict(fps=50.,root_p=repeat(pose[:3]),root_r=repeat(np.eye(3)),
           torso_r=repeat(r.rot(r.torso)),sides={},body_contacts=[])
    for side,g in r.sides.items():
        t['sides'][side]=dict(thigh=direction(g['hip'],g['knee']),shin=direction(g['knee'],g['foot']),
            foot_target_p=repeat(r.point(g['foot'])),foot_target_r=repeat(r.rot(g['foot'])),
            contact=np.ones(n),pivots=repeat(g['sole']),anchors=repeat(r.point(g['foot'],g['sole'])),
            upper=direction(g['shoulder'],g['elbow']),lower=direction(g['elbow'],g['wrist']),
            hand_forward=repeat(-r.rot(g['tcp'])[:,2]),hand_width=repeat(r.rot(g['tcp'])[:,0]),
            forearm_width=repeat(r.rot(g['forearm'])[:,0]),elbow_axis=repeat(-r.rot(g['shoulder'])[:,1]),
            signed_elbow_bend=np.full(n,g['elbow_geometric_zero']-pose[g['arm'][3]]))
    x=repeat(pose);x[:,2]-=.005
    return r,t,x,cfg['solver']


def test_restricted_cost_and_gradient_match_full_objective(sample):
    r,t,x,cfg=sample;ids=np.array([0,1,2]+sum([g['leg'] for g in r.sides.values()],[]));i=4
    full=Objective(r,t,i,cfg,x[i-1],x[i-2]);small=SupportObjective(r,t,i,cfg,x[i],ids,x[i-1],x[i-2])
    rng=np.random.default_rng(4)
    for _ in range(12):
        at=x[i].copy();at[ids]+=rng.normal(0,.02,len(ids));res,J=full.evaluate(at);cost,grad=small.fun(at[ids])
        np.testing.assert_allclose(cost,.5*(res@res),atol=1e-8,rtol=1e-10)
        np.testing.assert_allclose(grad,J[:,ids].T@res,atol=1e-8,rtol=1e-10)


def test_feasible_full_hull_input_is_preserved(sample,monkeypatch):
    r,t,x,cfg=sample;y,reports=baseline(r,t,x[:5],cfg)
    assert not reports
    import dex_frax.v8.support_fast as module
    def forbidden(*args,**kwargs):raise AssertionError('Bypass must not construct objectives or Jacobians')
    monkeypatch.setattr(module,'SupportObjective',forbidden)
    monkeypatch.setattr(r,'jac',forbidden)
    actual,reports=constrain_support(r,t,y,cfg)
    assert not reports
    np.testing.assert_array_equal(actual,y)
    assert r.support_fast_report['bypass']==5
    assert r.support_fast_report['objective_builds']==0
    assert r.support_fast_report['objective_evaluations']==0


def test_zero_light_budget_keeps_checked_sqp_fallback(sample):
    r,t,x,cfg=sample;actual,reports=constrain_support(r,t,x[:5],dict(cfg,support_light_iterations=0,support_fast_bypass=False))
    assert r.support_fast_report['fallback']==5
    assert r.support_fast_report['fallback_successes']==5
    assert r.support_fast_report['fallback_failures']==0
    assert all(f['light_stop_reason']=='iteration_budget' and f['fallback_triggers'] for f in r.support_fast_report['frames'])
    assert not reports and np.isfinite(actual).all()
    assert all(f['minimum_z']>=-1e-5 and f['contact_error_m']<=1e-4 for f in r.support_fast_report['frames'])


def test_speed_uses_previous_repaired_output(sample):
    r,t,x,cfg=sample;data=x[:8].copy();leg=np.array(sum([g['leg'] for g in r.sides.values()],[]))
    data[1:,leg[0]]+=.4
    actual,_=constrain_support(r,t,data,cfg)
    assert np.all(abs(np.diff(actual[:,leg],axis=0))<=r.joint_step(t['fps'],cfg)[leg-3]+1e-8)
    assert r.support_fast_report['qp_steps']>0


def test_qp_failure_is_diagnosed_separately_from_successful_fallback(sample,monkeypatch):
    import dex_frax.v8.support_fast as module
    from types import SimpleNamespace
    r,t,x,cfg=sample
    def failed(*args,**kwargs):
        return SimpleNamespace(x=None,info=SimpleNamespace(status='maximum iterations reached',
            status_val=7,iter=4000,setup_time=0.,solve_time=0.))
    monkeypatch.setattr(module.osqp.OSQP,'solve',failed)
    _,reports=constrain_support(r,t,x[:2],cfg)
    assert not reports
    assert r.support_fast_report['fallback_successes']==2
    assert all(f['light_stop_reason']=='qp_maximum_iterations_reached' and f['final_feasible']
               for f in r.support_fast_report['frames'])


def test_height_only_support_keeps_horizontal_path_and_upper_pose(sample):
    r,t,x,cfg=sample
    actual,reports=constrain_support(r,t,x[:5],dict(cfg,_pelvis_height_only=True))
    ids=np.array([2]+sum([g['leg'] for g in r.sides.values()],[]))
    fixed=np.setdiff1d(np.arange(32),ids)
    np.testing.assert_array_equal(actual[:,fixed],x[:5,fixed])
    assert np.max(abs(actual[:,2]-x[:5,2]))>1e-5
    assert not reports
    assert all(f['minimum_z']>=-1e-5 and f['contact_error_m']<=1e-4 for f in r.support_fast_report['frames'])


def test_height_only_cost_and_gradient_match_full_objective(sample):
    r,t,x,cfg=sample;ids=np.array([2]+sum([g['leg'] for g in r.sides.values()],[]));i=4
    full=Objective(r,t,i,cfg,x[i-1],x[i-2]);small=SupportObjective(r,t,i,cfg,x[i],ids,x[i-1],x[i-2])
    rng=np.random.default_rng(6)
    for _ in range(6):
        at=x[i].copy();at[ids]+=rng.normal(0,.01,len(ids));res,J=full.evaluate(at);cost,grad=small.fun(at[ids])
        np.testing.assert_allclose(cost,.5*(res@res),atol=1e-8,rtol=1e-10)
        np.testing.assert_allclose(grad,J[:,ids].T@res,atol=1e-8,rtol=1e-10)


def test_frax_support_rows_measure_true_sole_and_allow_only_height(sample):
    import jax.numpy as jnp
    from dex_frax.kinematics import DexKinematics
    from dex_frax.frax_foot_constraints import FootSupport
    r,t,x,cfg=sample;model=DexKinematics(r);constraint=FootSupport(model,r,50.)
    ids=np.array([2]+sum([g['leg'] for g in r.sides.values()],[]));mask=np.ones(32,bool);mask[ids]=False
    ref=dict(root_r=jnp.asarray(t['root_r'][0]),priority_mask=jnp.asarray(mask),priority_pose=jnp.asarray(x[0]),
        sides={s:{k:jnp.asarray(v[k][0]) for k in ('contact','pivots','anchors')} for s,v in t['sides'].items()})
    G,h=constraint.rows(jnp.asarray(x[0]),ref);G=np.asarray(G);h=np.asarray(h)
    np.testing.assert_array_equal(G[:,mask],0.)
    r.set(x[0],t['root_r'][0]);height=np.array([r.point(g['foot'],r.lowest(s))[2] for s,g in r.sides.items()])
    np.testing.assert_allclose(h[:2],50*(height-.0001),atol=1e-10)
    shifted=x[0].copy();shifted[2]+=.01
    _,h2=constraint.rows(jnp.asarray(shifted),ref)
    np.testing.assert_allclose(np.asarray(h2)[:2]-h[:2],.5,atol=1e-10)
