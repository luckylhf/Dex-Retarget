import json
import pytest
import numpy as np
from dex_frax.refine_routing import collision_groups,collision_masks,submodel,dependencies
from dex_frax.refine_queue import WorkQueue,jobs_from_masks,RECIPES


def test_waist_dependency_uses_context_not_just_core_and_keeps_legs_independent():
    masks=[('torso',None,np.arange(400)==140),('collision',None,np.arange(400)==150),('leg_collision',None,np.arange(400)==350)]
    jobs=jobs_from_masks(masks,400);torso=np.zeros(400);torso[140]=21
    ready,blocked=dependencies(jobs,torso,np.zeros(400,bool))
    assert {j['module'] for j in ready}=={'torso','leg_collision'}
    assert blocked[0]['reason']=='waiting_for_torso'
    # An exhausted unrelated waist window must not starve clear regions.
    free=jobs_from_masks([('collision',None,np.arange(400)==350)],400)
    assert dependencies(free,torso,np.zeros(400,bool))[0]==free


def test_arm_similarity_waits_for_clearance_and_unknown_body_never_uses_shoulders():
    jobs=jobs_from_masks([('arms','Left',np.ones(30,bool)),('body_collision',None,np.ones(30,bool))],30)
    ready,blocked=dependencies(jobs,np.zeros(30),np.ones(30,bool))
    assert not ready
    assert {j['reason'] for j in blocked}=={'waiting_for_arm_clearance','no_certified_body_collision_module'}
    assert WorkQueue().next([j for j in jobs if j['module']=='body_collision']) is None


def test_real_collision_ownership_has_leg_leg_and_arm_leg_routes():
    from dex_frax.v8.robot import Robot
    from dex_frax.v8.self_collision import CollisionModel
    model=CollisionModel(Robot(),{'coverage':'full'});groups=collision_groups(model)
    names=[set(p) for p in model.names]
    leg=names.index({'knee_pitch_l_link_collision','knee_pitch_r_link_collision'})
    arm=names.index({'wrist_roll_l_link_collision','knee_pitch_r_link_collision'})
    assert leg in groups['leg_collision'] and arm in groups['collision']
    assert sorted(sum(groups.values(),[]))==list(range(len(model.pairs)))
    samples=[(.0,0,[]) for _ in range(17)];samples[7]=(-.01,leg,[leg]);samples[8]=(-.01,arm,[arm])
    masks=collision_masks(samples,groups,3)
    assert masks['leg_collision'].tolist()==[True,True,False]
    assert masks['collision'].tolist()==[False,True,False]
    subset=submodel(model,groups['leg_collision'])
    assert set(map(tuple,subset.pairs))==set(map(tuple,model.pairs[groups['leg_collision']]))


def test_context_expands_and_neighbor_change_reactivates_without_two_retry_cap():
    q=WorkQueue();rows=np.zeros((1000,32));raw=jobs_from_masks([('torso',None,np.arange(1000)==450)],1000)
    jobs=q.configure(raw,rows);assert jobs[0]['start']==384
    for _ in RECIPES['torso']:q.complete(jobs[0],False)
    jobs=q.configure(raw,rows);assert jobs[0]['start']==354 and q.recipe(jobs[0])=='source_fit'
    for k in range(4):
        q.complete(jobs[0],False);rows[450,4]+=.01;jobs=q.configure(raw,rows)
        assert q.recipe(jobs[0])==RECIPES['torso'][(k+1)%len(RECIPES['torso'])]
    assert q.state(jobs[0])['reactivations']==4


def test_unchanged_context_keeps_progress_but_changed_context_discards_it():
    q=WorkQueue();rows=np.zeros((250,32));raw=jobs_from_masks([('collision',None,np.arange(250)==150)],250)
    jobs=q.configure(raw,rows);q.progress(jobs[0])['window_qp']={'iteration':3}
    q.complete(jobs[0],False,timed=True);saved=json.loads(json.dumps(q.states))
    q=WorkQueue(saved,retry_timed=True);q.configure(raw,rows)
    assert q.state(jobs[0])['progress'][RECIPES['collision'][0]]['window_qp']['iteration']==3
    rows[150,4]=.01;q.configure(raw,rows)
    assert not q.state(jobs[0])['progress']


@pytest.mark.parametrize('recipe,qp_limit',[('legs10',20000),('legs10_elastic',80000),('geometry10',20000),('geometry10_elastic',80000)])
def test_leg_proposal_moves_only_legs_and_pelvis_height(monkeypatch,recipe,qp_limit):
    from dex_frax import refine_pipeline as p
    from types import SimpleNamespace
    robot=SimpleNamespace(sides={'Left':{'leg':list(range(3,9))},'Right':{'leg':list(range(9,15))}})
    rows=np.zeros((12,32));job=dict(start=0,end=12,module='support' if recipe.startswith('geometry') else 'leg_collision',side=None)
    t={'fps':50.,'sides':{},'root_r':np.tile(np.eye(3),(12,1,1))}
    monkeypatch.setattr(p,'collision_seams',lambda *a:{})
    monkeypatch.setattr(p,'movable_submodel',lambda model,ids:model)
    monkeypatch.setattr(p,'collision_groups',lambda *a:{'collision':[]})
    def solve(model,t,x,cfg,a,b,sides,options):
        assert options['coupled'] and options['_moving_ids']==[2]+list(range(3,15))
        assert np.isinf(options['root_position_m']) and options['body_joint_deg']==10
        assert options['qp_iterations']==qp_limit
        assert options['_elastic_recovery']==recipe.endswith('_elastic')
        out=x.copy();out[:,options['_moving_ids']]=.01;return out,{}
    monkeypatch.setattr(p,'_repair_window',solve)
    import dex_frax.leg_similarity as ls
    monkeypatch.setattr(ls,'errors',lambda *a:np.zeros((12,4)))
    import dex_frax.leg_policy as lp
    monkeypatch.setattr(lp,'plan',lambda *a:(np.ones((12,13))*.12,{}))
    out,_=p.propose(robot,t,rows,rows,{'leg_direction':1.,'arm_direction':4.},None,job,recipe,float('inf'),{})
    assert np.array_equal(out[:,15:],rows[:,15:]) and np.array_equal(out[:,:2],rows[:,:2])
    assert np.all(out[:,2]==.01)


def test_correct_neighbor_reference_cannot_accumulate_avoidance_across_overlaps():
    from dex_frax.refine_routing import alignment_reference
    reference=np.zeros((10,32));aligned=np.ones((10,32));unfixed=np.zeros(10,bool);unfixed[4:6]=True
    updated=alignment_reference(reference,aligned,unfixed,2,8)
    assert np.array_equal(updated[:4],reference[:4]) and np.array_equal(updated[6:],reference[6:])
    assert np.all(updated[4:6]==1)
    unfixed[4:6]=False
    again=alignment_reference(updated,aligned*2,unfixed,0,10)
    np.testing.assert_array_equal(again,updated)


def test_collision_retries_precede_unvisited_similarity_but_rotate_safety_windows():
    queue=WorkQueue();rows=np.zeros((300,32))
    masks=[('collision',None,np.arange(300)==150),('leg_collision',None,np.arange(300)==50),('arms','Left',np.arange(300)==250)]
    jobs=queue.configure(jobs_from_masks(masks,300),rows)
    collision=next(j for j in jobs if j['module']=='collision')
    leg=next(j for j in jobs if j['module']=='leg_collision')
    queue.complete(collision,False)
    assert queue.next(jobs)['module']=='leg_collision'
    queue.complete(leg,False);queue.complete(leg,False)
    assert queue.next(jobs)['module']=='leg_collision'  # Legs precede arms regardless of visits.


def test_small_adoption_does_not_prevent_expanding_a_still_failing_window():
    queue=WorkQueue();rows=np.zeros((600,32))
    raw=jobs_from_masks([('arms','Left',np.arange(600)==250)],600)
    job=queue.configure(raw,rows)[0]
    for index,recipe in enumerate(RECIPES['arms']):
        queue.complete(job,index==0,recipe=recipe,result_rows=rows if index==0 else None)
    expanded=queue.configure(raw,rows)[0]
    assert expanded['context_level']==1
    assert expanded['start']==expanded['core_start']-50
    assert queue.choices(expanded)


def test_neighbor_updates_try_unvisited_families_before_repeating_first():
    queue=WorkQueue();rows=np.zeros((600,32))
    raw=jobs_from_masks([('arms','Left',np.arange(600)==250)],600)
    visited=[]
    for k in range(len(RECIPES['arms'])):
        job=queue.configure(raw,rows)[0];recipe=queue.recipe(job);visited.append(recipe)
        queue.complete(job,False)
        rows[250,25]+=.001  # Neighbor change invalidates numerical progress.
    assert visited==RECIPES['arms']
