import numpy as np
from types import SimpleNamespace
from dex_frax.refine_queue import WorkQueue,jobs_from_masks,PHASE
from dex_frax.refine_routing import dependencies,collision_groups,movable_submodel


def test_strict_phase_order_overrides_old_visit_counts_and_exhausted_blockers():
    q=WorkQueue();rows=np.zeros((200,32));modules=['arms','collision','leg_collision','torso','jitter','support']
    jobs=q.configure(jobs_from_masks([(name,'shoulder_pitch_l_joint' if name=='jitter' else None,np.arange(200)==80) for name in modules],200),rows)
    for j in jobs:q.state(j)['visits']=100 if PHASE[j['module']]==0 else 0
    assert PHASE[q.next(jobs)['module']]==0
    active=[j for j in jobs if j['module']!='support']
    ready,blocked=dependencies(active,np.zeros(200),np.zeros(200,bool),pending=jobs)
    assert {j['module'] for j in ready}=={'jitter'}
    assert {'torso','leg_collision','collision','arms'} <= {j['module'] for j in blocked}


def test_leg_collision_blocks_overlapping_arm_repair_even_without_solver_routes():
    pending=[dict(module='leg_collision',start=20,end=80),dict(module='arms',start=50,end=100),dict(module='arms',start=110,end=150)]
    ready,blocked=dependencies(pending[1:],np.zeros(200),np.zeros(200,bool),pending=pending)
    assert ready==[pending[2]] and blocked[0]['reason']=='waiting_for_leg_collision'


def test_initial_solves_feet_before_arms_and_keeps_upstream_columns(monkeypatch):
    import dex_frax.pelvis_priority as p
    import dex_frax.ik as ik
    import dex_frax.local_refine as local
    robot=SimpleNamespace(sides={'Left':{'leg':list(range(3,9)),'arm':list(range(18,25))},'Right':{'leg':list(range(9,15)),'arm':list(range(25,32))}})
    baseline=np.zeros((5,32));pose=baseline.copy();pose[:,15:18]=.2
    def waist(r,t,seed,*a,**k):
        out=seed.copy();out[:,15:18]=.2;return out,np.array([15,16,17])
    monkeypatch.setattr(p,'waist_trajectory',waist)
    monkeypatch.setattr(local,'torso_errors',lambda *a:np.zeros(5));calls=[]
    class Fake:
        def __init__(self,*a,**kw):self.report={'qp_nonconverged_steps':0}
        def __call__(self,r,t,cfg):
            x=t['priority_pose'].copy();mask=t['priority_mask'];calls.append((x.copy(),mask.copy()));x[:,~mask]=len(calls)*.1
            return x,np.zeros((5,1))
    monkeypatch.setattr(ik,'FraxIK',Fake)
    out,records,report=p.solve_protected(robot,{}, {},baseline,SimpleNamespace(iterations=8,initial_iterations=100,assets={}),root_mode='feasible')
    assert np.flatnonzero(~calls[0][1]).tolist()==[2]+list(range(3,15))
    assert np.flatnonzero(~calls[1][1]).tolist()==list(range(3,15))
    assert np.flatnonzero(~calls[2][1]).tolist()==list(range(18,32))
    np.testing.assert_array_equal(out[:,3:15],.2)
    np.testing.assert_array_equal(out[:,15:18],.2)
    np.testing.assert_array_equal(out[:,:2],0.)
    np.testing.assert_array_equal(out[:,2],.1)
    assert report['stage_order']==['pelvis','waist','feet','arms']


def test_full_feet_heel_shin_pairs_survive_routing_and_preserve_original_model():
    from dex_frax.v8.robot import Robot
    from dex_frax.v8.self_collision import CollisionModel,scan
    r=Robot();old_groups=r.m.geom_group.copy();old_ngeom=r.m.ngeom;m=CollisionModel(r,{'coverage':'full'})
    names=[frozenset(v) for v in m.names];owners=collision_groups(m);leg=movable_submodel(m,sum([g['leg'] for g in r.sides.values()],[]))
    expected=[{'ankle_roll_l_link_visual','ankle_roll_r_link_visual'},
              {'ankle_roll_l_link_visual','knee_pitch_r_link_collision'},
              {'retarget_foot_l_heel','retarget_shin_l_shaft'},
              {'retarget_foot_r_heel','retarget_shin_r_shaft'}]
    for pair in expected:
        assert frozenset(pair) in names
        assert names.index(frozenset(pair)) in owners['leg_collision']
        assert frozenset(pair) in [frozenset(v) for v in leg.names]
    rows=np.tile(np.r_[0.,0.,1.,np.zeros(29)],(3,1));rr=np.tile(np.eye(3),(3,1,1))
    assert scan(m,rows,rr,8)[0]['collision_samples']==0
    assert r.m.ngeom==old_ngeom;np.testing.assert_array_equal(r.m.geom_group,old_groups)


def test_leg_preferences_keep_assist_axes_free_and_prefer_swing():
    from dex_frax.leg_policy import preferences
    cross=preferences({'cross_leg'},False);own=preferences({'own_foot_shin'},False)
    assert np.all(cross>0) and np.all(own>0)
    assert max(cross[1:3])<min(cross[[0,3,4,5]])
    assert max(own[4:6])<min(own[:4])
    assert np.all(preferences({'cross_leg'},True)>cross)
