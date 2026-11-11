import time
import numpy as np
import pytest

from dex_frax import hand_leg_repair as joint
from dex_frax.refine_routing import collision_groups
from dex_frax.v8.robot import Robot
from dex_frax.v8.self_collision import CollisionModel


@pytest.fixture(scope='module')
def model():
    return CollisionModel(Robot(), {'coverage': 'full'})


def pair(model, a='left_tcp_link_collision', b='hip_pitch_l_link_collision'):
    return next(k for k, names in enumerate(model.names) if set(names) == {a, b})


def test_routing_uses_real_geometry_and_keeps_legacy_fallbacks(model):
    mixed=pair(model)
    mapping=joint.pair_sides(model)
    assert mapping[mixed]==dict(arms=['Left'],legs=['Left'])
    pure_leg=pair(model,'knee_pitch_l_link_collision','knee_pitch_r_link_collision')
    assert pure_leg not in mapping
    samples=[(0.,0,[]) for _ in range(81)];samples[40]=(-.01,mixed,[mixed])
    jobs=[dict(module='leg_collision',side=None,start=0,end=10),
          dict(module='collision',side=None,start=20,end=30)]
    routed=joint.attach(jobs,samples,model)
    assert routed[0]['candidate_order'][:3]==['legs5',*joint.RECIPES]
    assert 'legs5' in routed[0]['candidate_order']
    assert routed[0]['hand_leg_pair_ids']==[mixed]
    assert routed[1]==jobs[1] and 'candidate_order' not in jobs[0]


def test_joint_opens_only_involved_limbs_and_preserves_all_other_pairs(model):
    mixed=pair(model);r=model.r
    ids,arms,legs=joint.columns(model,[mixed],True)
    assert set(ids)=={2,*r.sides['Left']['arm'],*r.sides['Left']['leg']}
    assert not set(ids)&{0,1,*r.sides['Right']['arm'],*r.sides['Right']['leg']}
    owned=joint.owned_pairs(model,[mixed],True)
    assert mixed in owned
    assert set(owned)-{mixed}<=set(collision_groups(model)['leg_collision'])
    with pytest.raises(ValueError):joint.columns(model,[],True)


def test_one_solver_tries_hand_before_leg_and_keeps_fixed_reference(monkeypatch,model):
    import dex_frax.leg_policy as leg
    import dex_frax.unified_repair as repair
    monkeypatch.setattr(leg,'plan',lambda m,t,x,a,b,ids:(np.ones((b-a,len(ids)))*.1,{}))
    monkeypatch.setattr(repair,'collision_seams',lambda *args:dict(freeze_start=False,freeze_end=False))
    monkeypatch.setattr(joint,'geometric_seed',lambda m,t,x,*args:x)
    rows=np.zeros((5,32));ref=rows.copy();calls=[]
    def solve(m,t,x,w,a,b,sides,options):
        calls.append(options)
        assert options['_reference'] is ref
        assert options['_geometry_reference'] is rows
        assert options['_source_palm_constraints']
        mixed_local=pair(m)
        assert mixed_local not in options['_passive_collision_indices']
        if len(calls)<=2:
            assert options['_moving_ids']==model.r.sides['Left']['arm']
            options['_progress']['window_qp']={'iteration':2}
            raise ValueError('collision_qp_primal infeasible')
        assert options['_moving_ids']==[2]+model.r.sides['Left']['leg']+model.r.sides['Left']['arm']
        np.testing.assert_allclose(options['_joint_prior'][:,:7],1.)
        result=x.copy();result[:,options['_moving_ids']]+=.001
        return result,{'status':'repaired'}
    monkeypatch.setattr(joint,'_repair_window',solve)
    job=dict(start=0,end=5,hand_leg_pair_ids=[pair(model)])
    result,detail=joint.propose(model,{},rows,ref,dict(leg_direction=1.,arm_direction=6.),job,
                               'hand_leg5',time.monotonic()+3.,{})
    assert detail['route']=='cooperate' and len(calls)==3
    np.testing.assert_array_equal(rows,0.)
    fixed=np.setdiff1d(np.arange(32),detail['moving_ids'])
    np.testing.assert_array_equal(result[:,fixed],rows[:,fixed])


def test_arm_only_success_never_opens_legs(monkeypatch,model):
    import dex_frax.unified_repair as repair
    monkeypatch.setattr(repair,'collision_seams',lambda *args:{})
    calls=[]
    def solve(*args):
        calls.append(args[-1]);return args[2].copy(),{}
    monkeypatch.setattr(joint,'_repair_window',solve)
    rows=np.zeros((5,32));job=dict(start=0,end=5,hand_leg_pair_ids=[pair(model)])
    _,detail=joint.propose(model,{},rows,rows.copy(),dict(leg_direction=1.,arm_direction=6.),
                          job,'hand_leg10',time.monotonic()+3.,{})
    assert detail['route']=='arms_only' and detail['leg_sides']==[] and len(calls)==1


def test_guard_rejects_waist_changes_in_joint_proposal(model):
    from dex_frax.refine_pipeline import guard
    r=model.r;row=np.r_[0.,0.,1.,np.zeros(29)]
    row[3:]=np.clip(row[3:],r.lower[3:]+1e-4,r.upper[3:]-1e-4)
    waist=3+r.names.index('waist_yaw_joint')
    before=np.tile(row,(5,1));after=before.copy();after[:,waist]+=.00001
    assert waist not in joint.columns(model,[pair(model)],True)[0]
    job=dict(module='leg_collision',side=None,start=0,end=5,
             hand_leg_pair_ids=[pair(model)],hand_leg_route='cooperate')
    assert guard(r,{'fps':50.},before,after,before,{}, {},{}, {},{}, {},model,job)=='hand_leg_changed_protected_joint'


@pytest.mark.parametrize('stage', ['leg_collision', 'collision'])
@pytest.mark.parametrize('recipe', joint.RECIPES)
def test_pipeline_dispatches_hand_leg_before_regular_geometry(monkeypatch, model, stage, recipe):
    from dex_frax import refine_pipeline

    rows = np.zeros((7, 32))
    job = dict(module=stage, side=None, start=1, end=6, hand_leg_pair_ids=[pair(model)])
    progress = {}
    monkeypatch.setattr(refine_pipeline, 'window_targets', lambda *args: {})

    def dispatch(selected_model, targets, original, reference, weights, task, tier, deadline, state):
        assert selected_model is model and original is rows
        assert task is job and tier == recipe and state is progress
        candidate = rows.copy()
        candidate[1, model.r.sides['Left']['arm'][0]] = .01
        return candidate, dict(route='cooperate')

    monkeypatch.setattr(joint, 'propose', dispatch)
    result, detail = refine_pipeline.propose(model.r, {}, rows, rows, {}, model, job, recipe,
                                             time.monotonic() + 1, progress)
    assert detail['route'] == 'cooperate'
    assert result[1, model.r.sides['Left']['arm'][0]] == .01


def test_resume_keeps_each_start_separate_and_does_not_retry_completed_failures(monkeypatch,model):
    import dex_frax.leg_policy as leg
    import dex_frax.unified_repair as repair
    monkeypatch.setattr(leg,'plan',lambda m,t,x,a,b,ids:(np.ones((b-a,len(ids)))*.1,{}))
    monkeypatch.setattr(repair,'collision_seams',lambda *args:{})
    monkeypatch.setattr(joint,'geometric_seed',lambda m,t,x,*args:x)
    rows=np.zeros((5,32));reference=rows.copy();progress={};first=[]
    def interrupted(*args):
        state=args[-1]['_progress'];first.append(state)
        if len(first)==1:raise ValueError('collision_qp_primal infeasible')
        state['window_qp']={'iteration':3,'q':[[.01]]}
        raise ValueError('collision_repair_interrupted_watchdog')
    monkeypatch.setattr(joint,'_repair_window',interrupted)
    job=dict(start=0,end=5,hand_leg_pair_ids=[pair(model)])
    with pytest.raises(ValueError,match='watchdog'):
        joint.propose(model,{},rows,reference,dict(leg_direction=1.,arm_direction=6.),job,
                      'hand_leg5',time.monotonic()+3.,progress)
    assert progress['routes']['arms_only']['failed']
    assert progress['routes']['arms_seed']['window_qp']['iteration']==3
    resumed=[]
    def complete(*args):
        state=args[-1]['_progress'];resumed.append(state)
        assert state is progress['routes']['arms_seed']
        return args[2].copy(),{}
    monkeypatch.setattr(joint,'_repair_window',complete)
    _,detail=joint.propose(model,{},rows,reference,dict(leg_direction=1.,arm_direction=6.),job,
                          'hand_leg5',time.monotonic()+3.,progress)
    assert len(resumed)==1 and detail['route']=='arms_seed'
    np.testing.assert_array_equal(rows,reference)
