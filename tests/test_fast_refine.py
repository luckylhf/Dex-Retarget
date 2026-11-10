import argparse,copy,json,subprocess,sys
from pathlib import Path
import numpy as np
from dex_frax.refine_queue import WorkQueue,jobs_from_masks,RECIPES
from dex_frax.support_quality import evaluate_contacts
from dex_frax.temporal_quality import assess
from dex_frax.dispatch import use_unified
from dex_frax.contract import conversion_contract


def test_more_than_six_windows_get_fair_visits_and_no_small_improvement_stop():
    jobs=jobs_from_masks([('collision',None,np.ones(1200,bool))],1200)
    queue=WorkQueue();seen=[]
    for _ in jobs:
        job=queue.next(jobs);seen.append(queue.key(job));queue.complete(job,False)
    assert len(set(seen))==len(jobs)>6
    assert all(v['cursor']==1 for v in queue.states.values())
    while (job:=queue.next(jobs)) is not None:queue.complete(job,False)
    assert all(v['next_recipe'] is None for v in queue.pending(jobs))


def test_queue_checkpoint_keeps_untried_tiers_and_fixed_window_identity():
    jobs=jobs_from_masks([('collision',None,np.ones(101,bool))],101);queue=WorkQueue()
    first=queue.next(jobs);assert queue.recipe(first)==RECIPES['collision'][0];queue.complete(first,True)
    resumed=WorkQueue(json.loads(json.dumps(queue.states)))
    assert resumed.recipe(resumed.next(jobs))==RECIPES['collision'][1]
    assert resumed.pending(jobs)[0]['reason']=='untried_candidates'


def test_modes_have_distinct_resume_contracts_but_match_workers():
    parent=argparse.Namespace(mode='auto',profile='fast',iterations=8,initial_iterations=100)
    worker=copy.copy(parent);worker.mode='frax';worker.unified_repair=False
    assert not use_unified(parent) and conversion_contract(parent)==conversion_contract(worker)
    parent.profile='refine';assert use_unified(parent)
    assert conversion_contract(parent)!=conversion_contract(worker)


def test_only_two_public_conversion_modes():
    result=subprocess.run([sys.executable,'-B','-m','dex_frax.cli','convert','--help'],capture_output=True,text=True,timeout=20)
    assert result.returncode==0
    assert '--mode {fast,refine}' in result.stdout
    assert '--pipeline' not in result.stdout and '--legacy-frax' not in result.stdout
    bad=subprocess.run([sys.executable,'-B','-m','dex_frax.cli','convert','x','y','--mode','v8'],capture_output=True,text=True,timeout=20)
    assert bad.returncode!=0 and 'invalid choice' in bad.stderr


def body_case(mode=1,confidence=1.,geometry=1.):
    n=10;c=dict(body_name='hand',source_geometry_confidence=geometry)
    meta=dict(configuration={'contact':{'height_tolerance_m':.035,'max_anchor_excursion_m':.035}},leg_scale=1.,body_contact_models=[c])
    dg=dict(body_hand_mode=np.full(n,mode),body_hand_confidence=np.full(n,confidence))
    errors={'hand':{'normal_cm':[10.]*n,'fixed_xy_cm':[20.]*n}}
    return meta,dg,errors,n,50.


def test_trusted_body_support_enters_gate_and_uncertain_hints_do_not_lock():
    q=evaluate_contacts(*body_case());assert len(q['violating_frames'])==10
    q=evaluate_contacts(*body_case(geometry=.25));assert q['violating_frames']==[] and q['contacts']['hand']['hint_frames']==10
    args=list(body_case(mode=2));args[2]['hand']['normal_cm']=[0.]*10
    q=evaluate_contacts(*args);assert q['violating_frames']==[]  # Sliding cannot freeze XY.


def test_single_joint_jitter_survives_all_joint_percentile_dilution():
    n=50;q=np.zeros((n,29));q[:,0]=np.deg2rad(2.)*(-1.)**np.arange(n)
    names=['shoulder_pitch_l_joint']+['other_'+str(i) for i in range(28)]
    dg={'Left_upper':np.tile([0.,0.,1.],(n,1))}
    result=assess(q,50.,names,dg)
    assert result['violating_frames']
    assert result['joints'][names[0]]['acceleration_p95_rad_s2']>300
    theta=q[:,0];dg['Left_upper']=np.c_[np.sin(theta),np.zeros(n),np.cos(theta)]
    assert not assess(q,50.,names,dg)['violating_frames']  # Same oscillation exists in the human.
    assert not assess(q,50.,names,{})['violating_frames'] # Unknown source is not proof.


def test_full_collision_includes_legs_and_same_arm_without_modifying_model():
    from dex_frax.v8.robot import Robot
    from dex_frax.v8.self_collision import CollisionModel,scan
    r=Robot();original=r.m.geom_contype.copy();model=CollisionModel(r,{'coverage':'full'})
    pairs=[set(p) for p in model.names]
    assert {'knee_pitch_l_link_collision','knee_pitch_r_link_collision'} in pairs
    assert {'shoulder_yaw_l_link_collision','wrist_roll_l_link_collision'} in pairs
    assert np.array_equal(original,r.m.geom_contype)
    report=scan(model,np.tile(np.r_[0.,0.,1.,np.zeros(29)],(2,1)),np.tile(np.eye(3),(2,1,1)),8)[0]
    assert report['pair_count']>240 and report['collision_samples']==0
    assert report['structural_exclusions']


def test_time_sliced_candidate_preserves_iterate_and_retries_after_fresh_routes():
    jobs=jobs_from_masks([('torso',None,np.ones(101,bool))],101);queue=WorkQueue()
    first=queue.next(jobs);recipe=queue.recipe(first)
    queue.progress(first)['window_qp']={'q':[[.1]],'iteration':2}
    queue.complete(first,False,timed=True)
    while queue.recipe(queue.next(jobs))!=recipe:queue.complete(queue.next(jobs),False)
    resumed=WorkQueue(json.loads(json.dumps(queue.states)),retry_timed=True)
    assert resumed.recipe(resumed.next(jobs))==recipe
    assert resumed.progress(first)['window_qp']['iteration']==2
    # A timeout is not a failed numerical branch; resume its latest iterate.
    resumed.complete(first,False)
    assert resumed.next(jobs) is None


def test_resume_prioritizes_untried_branches_over_previous_time_slices():
    jobs=jobs_from_masks([('collision',None,np.ones(101,bool))],101);queue=WorkQueue()
    job=queue.next(jobs);queue.complete(job,False,timed=True)
    resumed=WorkQueue(json.loads(json.dumps(queue.states)),retry_timed=True)
    assert resumed.recipe(resumed.next(jobs))==RECIPES['collision'][1]
    while resumed.recipe(resumed.next(jobs))!=RECIPES['collision'][0]:resumed.complete(job,False)
    assert resumed.recipe(resumed.next(jobs))==RECIPES['collision'][0]
