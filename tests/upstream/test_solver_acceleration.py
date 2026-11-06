import numpy as np
import pytest
from scipy.sparse import csc_matrix
from dex_frax.v8.support_fast import scale_qp
from test_support_fast import sample


@pytest.mark.parametrize('mode',['none','objective'])
def test_qp_scaling_preserves_cost_order_and_feasible_set(mode):
    rng=np.random.default_rng(9);B=rng.normal(size=(12,7))*np.geomspace(.01,100,7)
    H=B.T@B+np.eye(7)*.01;g=rng.normal(size=7);A=csc_matrix(rng.normal(size=(16,7)))
    l=np.full(16,-.3);u=np.full(16,.4)
    P,q,C,lo,hi,D=scale_qp(H,g,A,l,u,mode)
    factor=np.median(np.diag(H)) if mode=='objective' else 1.
    for x in [np.zeros(7),*rng.normal(size=(12,7))]:
        z=x/D
        np.testing.assert_allclose((.5*z@P@z+q@z)*factor,.5*x@H@x+g@x,rtol=1e-12,atol=1e-10)
        np.testing.assert_array_equal((A@x>=l)&(A@x<=u),(C@z>=lo)&(C@z<=hi))


def test_kinematics_cache_is_exact_and_invalidates_on_pose_change(sample):
    import mujoco
    r,t,x,cfg=sample;b=r.sides['Left']['foot'];local=np.array([.1,0.,0.])
    for i in range(5):
        at=x[0].copy();at[3:]+=i*.003;r.set(at,t['root_r'][0])
        expected=r.d.xpos.copy();jp,jr=r.jac(b,local)
        count=r.cache_stats['fk_calls'];r.set(at,t['root_r'][0]);a,c=r.jac(b,local)
        assert r.cache_stats['fk_calls']==count
        np.testing.assert_array_equal(jp,a);np.testing.assert_array_equal(jr,c)
        # Force un-cached MuJoCo geometry/Jacobian for the same pose.
        mujoco.mj_kinematics(r.m,r.d);mujoco.mj_comPos(r.m,r.d)
        p=np.zeros((3,r.m.nv));q=np.zeros_like(p)
        mujoco.mj_jac(r.m,r.d,p,q,r.point(b,local),b)
        np.testing.assert_array_equal(r.d.xpos,expected)
        np.testing.assert_array_equal(jp,p[:,r.cols]);np.testing.assert_array_equal(jr,q[:,r.cols])
        a[:]=100
        np.testing.assert_array_equal(r.jac(b,local)[0],jp)
    assert r.cache_stats['jac_hits']>0


def test_adaptive_search_preserves_synthetic_standing_arm_branch(sample):
    from dex_frax.v8.arms import plan_and_refine
    r,t,x,cfg=sample
    # Preserve the original V8 adaptive-search parity test independently of
    # the new guarded event-tracking algorithm, which has its own tests.
    full,report_full=plan_and_refine(r,t,x,dict(cfg,arm_adaptive_candidates=False,arm_event_tracking=False))
    adaptive,report_adaptive=plan_and_refine(r,t,x,dict(cfg,arm_adaptive_candidates=True,arm_event_tracking=False))
    np.testing.assert_allclose(adaptive,full,rtol=0,atol=1e-7)
    for s,g in r.sides.items():
        assert report_adaptive[s]['candidate_solver_calls']<=report_full[s]['candidate_solver_calls']
        q=adaptive[:,g['arm']]
        assert np.all(q>=r.lower[g['arm']]-1e-8) and np.all(q<=r.upper[g['arm']]+1e-8)
        assert np.all(abs(np.diff(q,axis=0))<=r.joint_step(t['fps'],cfg)[np.array(g['arm'])-3]+1e-8)
