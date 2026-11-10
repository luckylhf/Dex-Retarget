import numpy as np
from dex_frax.refine_queue import jobs_from_masks,WorkQueue,RECIPES
from dex_frax.trajectory_escape import envelope


def test_dense_floor_windows_fit_solver_at_every_context():
    mask=np.zeros(300,bool);mask[70:90]=True
    raw=jobs_from_masks([('dense_floor',None,mask)],len(mask));queue=WorkQueue();rows=np.zeros((len(mask),32))
    for level in range(3):
        for job in raw:queue.state(job)['level']=level
        assert all(5<=j['end']-j['start']<=31 for j in queue.configure(raw,rows))


def test_escape_envelope_preserves_joins_and_reaches_collision():
    v=envelope(100,20,70,True,True)
    np.testing.assert_array_equal(v[:2],0.)
    np.testing.assert_array_equal(v[-2:],0.)
    np.testing.assert_array_equal(v[20:71],1.)
    assert np.all((v>=0)&(v<=1))
    assert RECIPES['collision'].index('sweep10')<RECIPES['collision'].index('direct10')
    assert RECIPES['collision'].index('sweep20')>RECIPES['collision'].index('recover10')


def test_collision_episode_does_not_freeze_penetrating_grid_cut():
    mask=np.zeros(300,bool);mask[25:169]=True
    jobs=jobs_from_masks([('collision',None,mask)],len(mask));assert len(jobs)==1
    job=WorkQueue().configure(jobs,np.zeros((len(mask),32)))[0]
    assert job['start']<25 and job['end']>169
    assert not mask[job['start']:job['start']+2].any()
    assert not mask[job['end']-2:job['end']].any()


def test_own_improvement_advances_routes_and_external_change_reactivates():
    rows=np.zeros((250,32));raw=jobs_from_masks([('arms','Left',np.arange(250)==150)],250)
    queue=WorkQueue();job=queue.configure(raw,rows)[0];first=queue.recipe(job)
    rows[150,20]=.01;queue.complete(job,True,recipe=first,result_rows=rows)
    job=queue.configure(raw,rows)[0]
    assert queue.recipe(job)!=first
    assert queue.state(job)['reactivations']==0
    rows[151,21]=.01;job=queue.configure(raw,rows)[0]
    assert first in queue.choices(job)  # Re-enabled, behind unvisited families.
    assert queue.recipe(job)!=first
    assert queue.state(job)['reactivations']==1


def test_bounded_escape_keeps_limiting_frames_and_correct_jacobian():
    from dex_frax.trajectory_escape import bounded_offsets
    base=np.array([[0.,0.],[.99,-.99],[-.99,.99],[0.,0.]])
    shape=np.array([0.,1.,1.,0.]);q=np.array([.3,-.4]);lo=np.full_like(base,-1.);hi=-lo
    rows,D=bounded_offsets(base,shape,q,lo,hi)
    assert np.all(rows>=lo) and np.all(rows<=hi)
    np.testing.assert_array_equal(rows[[0,-1]],base[[0,-1]])
    for j in range(2):
        step=np.zeros(2);step[j]=1e-6
        numeric=(bounded_offsets(base,shape,q+step,lo,hi)[0]-bounded_offsets(base,shape,q-step,lo,hi)[0])/2e-6
        expected=np.zeros_like(base);expected[:,j]=D[:,j]
        np.testing.assert_allclose(numeric,expected,atol=1e-9)
    # Reaching opposite joint limits at different frames does not eliminate
    # a legal coherent seed, unlike the old constant-offset intersection.
    assert rows[1,0]==1. and rows[2,1]==.59


def test_waist_preflight_clears_combined_axis_collision_before_arms():
    from dex_frax.v8.robot import Robot
    from dex_frax.v8.self_collision import CollisionModel,scan
    from dex_frax.refine_routing import collision_groups,submodel
    from dex_frax.pelvis_priority import waist_trajectory
    r=Robot();ids=np.array([3+r.names.index('waist_'+a+'_joint') for a in ('yaw','roll','pitch')])
    x=np.zeros((5,32));x[:,2]=1.;x[:,ids]=[-.2935449183,.3643248677,.8656424284]
    rr=np.tile(np.eye(3),(5,1,1));r.set(x[0],rr[0]);torso=r.rot(r.torso).copy()
    t=dict(fps=50.,root_r=rr,torso_r=np.tile(torso,(5,1,1)))
    full=CollisionModel(r,{'coverage':'full'});model=submodel(full,collision_groups(full)['body_collision'])
    assert scan(model,x,rr,8)[0]['collision_samples']>0
    out,_=waist_trajectory(r,t,x)
    assert scan(model,out,rr,8)[0]['collision_samples']==0
    fixed=np.setdiff1d(np.arange(32),ids);np.testing.assert_array_equal(out[:,fixed],x[:,fixed])
    assert np.all(abs(np.diff(out[:,ids],axis=0))*50<=r.speed_limits[ids-3]+1e-5)


def test_leg_solver_protects_fixed_hands_as_well_as_owned_leg_pairs():
    from dex_frax.v8.robot import Robot
    from dex_frax.v8.self_collision import CollisionModel
    from dex_frax.refine_routing import collision_groups,movable_submodel
    r=Robot();m=CollisionModel(r,{'coverage':'full'});groups=collision_groups(m)
    leg=movable_submodel(m,sum([g['leg'] for g in r.sides.values()],[]))
    names=[set(v) for v in m.names]
    k=names.index({'hip_pitch_l_link_collision','left_tcp_link_collision'})
    assert k in groups['collision'] and k not in groups['leg_collision']
    assert set(m.names[k]) in [set(v) for v in leg.names]
    assert {'waist_yaw_link_collision','waist_pitch_link_collision'} not in [set(v) for v in leg.names]


def test_unsafe_body_preflight_blocks_overlapping_shoulders_and_legs():
    from dex_frax.refine_routing import dependencies
    jobs=[dict(module=k,start=0,end=10) for k in ['arms','collision','leg_collision']]
    body=np.zeros(10,bool);body[5]=True
    ready,blocked=dependencies(jobs,np.zeros(10),np.zeros(10,bool),body)
    assert ready==[]
    assert {j['reason'] for j in blocked}=={'waiting_for_body_geometry'}


def test_priority_branch_order_still_visits_all_families():
    q=WorkQueue();job=dict(module='arms',side='Left',core_start=0,core_end=10,start=0,end=10,candidate_order=['current_best','original_frax','fit'])
    seen=[]
    while q.choices(job):
        seen.append(q.recipe(job));q.complete(job,False)
    assert seen==['current_best','original_frax','fit']


def test_pose_budget_rejection_is_not_a_time_slice():
    from dex_frax.refine_queue import time_sliced
    assert not time_sliced(ValueError('repair_hand_pose_budget'))
    assert not time_sliced(ValueError('dense_floor_window_budget'))
    assert not time_sliced(ValueError('window_exceeds_local_repair_budget'))
    assert time_sliced(ValueError('shoulder_recovery_budget'))
    assert time_sliced(ValueError('continuous_shoulder_seed_budget'))
    assert time_sliced(TimeoutError('candidate_done_later'))
    assert time_sliced(ValueError('collision_repair_interrupted_watchdog'))


def test_coherent_escape_preserves_numerical_start_after_timeout(monkeypatch):
    import pytest
    from types import SimpleNamespace
    from dex_frax import trajectory_escape as module
    monkeypatch.setattr(module,'active_arm_support',lambda *a:False)
    ids=list(range(3,10));r=SimpleNamespace(sides={'Left':{'arm':ids}},lower=np.full(32,-1.),upper=np.full(32,1.),joint_step=lambda *a:np.ones(29))
    current=[None];model=SimpleNamespace(r=r,pairs=[(0,1)],arms={'Left':{0}},update=lambda x,rot:current.__setitem__(0,x),distances=lambda:np.array([current[0][3]-.02]))
    class Objective:
        def __init__(self,*a):pass
        def evaluate(self,q):return q-np.r_[.15,np.zeros(6)],np.eye(7),np.array([q[0]-.02])
    monkeypatch.setattr(module,'FrameRecovery',Objective);starts=[]
    def fit(fun,q,**kwargs):
        starts.append(q.copy());fun(q+.001)
        if len(starts)==1:raise TimeoutError('test_deadline')
        return SimpleNamespace(x=q)
    monkeypatch.setattr(module,'least_squares',fit)
    validations=[]
    def validate(*args):validations.append(1);return args[2].copy(),{}
    monkeypatch.setattr(module,'_repair_window',validate)
    x=np.zeros((5,32));t=dict(fps=50.,root_r=np.tile(np.eye(3),(5,1,1)));progress={};options={'_reference':x,'joint_deg':10.,'_progress':progress}
    with pytest.raises(TimeoutError):module.propose(model,t,x,{},0,5,options,float('inf'))
    saved=np.array(progress['coherent_escape']['Left']['trial_q']);assert not validations
    module.propose(model,t,x,{},0,5,options,float('inf'))
    np.testing.assert_array_equal(starts[1],saved)
    assert validations==[1] and progress['coherent_escape']['Left']['complete']
    np.testing.assert_array_equal(x,0.)
