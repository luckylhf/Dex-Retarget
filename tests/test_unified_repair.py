import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from dex_frax.dispatch import pipeline_stages, use_unified, route
from dex_frax.contract import conversion_contract
from dex_frax.unified_repair import plan, quality_guard, preservation_guard
from dex_frax.unified_repair import arm_window_pending, palm_guard, propose_arm_branch


def arm_metrics(n=20):
    return {'errors':{s:{k:np.zeros(n) for k in ('upper','lower','hand_forward','hand_width','elbow_axis')}
                      for s in ('Left','Right')}}


def test_small_improvement_does_not_stop_pending_branch_search():
    metrics=arm_metrics();metrics['errors']['Left']['upper'][:]=26.
    assert arm_window_pending(metrics,'Left',0,20,50.)
    metrics['errors']['Left']['upper'][:]=21.  # Improved, still fails 20 degrees.
    assert arm_window_pending(metrics,'Left',0,20,50.)
    metrics['errors']['Left']['upper'][:]=20.
    assert not arm_window_pending(metrics,'Left',0,20,50.)


def test_branch_retry_is_side_local_and_retains_full_clip_duration_context():
    metrics=arm_metrics();metrics['errors']['Left']['elbow_axis'][2:7]=45.
    assert arm_window_pending(metrics,'Left',4,6,50.)  # Five-frame run crosses window cuts.
    assert not arm_window_pending(metrics,'Right',4,6,50.)
    assert not arm_window_pending(metrics,'Left',10,20,50.)


def test_isolated_palm_spike_cannot_hide_behind_unchanged_p95():
    before=arm_metrics(100)['errors'];after=copy.deepcopy(before)
    after['Left']['hand_width'][50]=.3
    assert palm_guard(before,after)=='palm_peak_regression:Left:hand_width'
    assert palm_guard(before,copy.deepcopy(before)) is None


def test_branch_solver_uses_one_side_and_keeps_current_body(monkeypatch):
    from types import SimpleNamespace
    from dex_frax.v8 import arms
    robot=SimpleNamespace(sides={'Left':dict(arm=list(range(3,10)))})
    current=np.ones((6,32));original=np.zeros_like(current);seen=[]
    def propose(r,t,seed,cfg,**kwargs):
        seen.append(seed.copy())
        assert kwargs['_only_sides']=={'Left'} and cfg['arm_event_tracking'] is False
        return seed,{}
    monkeypatch.setattr(arms,'plan_and_refine',propose)
    for ref in ('current_best','original_frax'):
        propose_arm_branch(robot,{},current,original,{},'Left',ref)
    assert np.array_equal(seen[0],current)
    assert np.all(seen[1][:,3:10]==0) and np.all(seen[1][:,10:]==1) and np.all(seen[1][:,:3]==1)
    assert np.all(current==1)  # Proposing a new branch must not overwrite the best.


def quality():
    return dict(flags=['Left_upper_deg_source_mismatch'],adaptive_pose_gate=dict(violating_frames=5),
        source_pose_audit={k:dict(p95=4.,max=10.) for k in ('root_orientation_deg','torso_orientation_deg','torso_up_deg')},
        self_collision=dict(collision_samples=0,native_colliding_frames=0,interpolated_colliding_intervals=0,minimum_distance_m=.003))


@pytest.mark.parametrize('change,reason',[
    (lambda q:q['flags'].append('arm_self_collision_unresolved'),'new_quality_flags'),
    (lambda q:q['adaptive_pose_gate'].update(violating_frames=6),'more_sustained_pose'),
    (lambda q:q['source_pose_audit']['torso_orientation_deg'].update(max=11.),'source_orientation'),
    (lambda q:q['self_collision'].update(collision_samples=1),'collision_count'),
    (lambda q:q['self_collision'].update(minimum_distance_m=-.001),'collision_depth'),
])
def test_rejects_worsening_even_without_new_flags(change,reason):
    before=quality();after=copy.deepcopy(before);change(after)
    assert quality_guard(before,after).startswith(reason)


def test_quality_improvement_is_eligible_not_automatically_adopted():
    before=quality();after=copy.deepcopy(before);after['flags']=[]
    after['adaptive_pose_gate']['violating_frames']=0
    assert quality_guard(before,after) is None


def test_default_is_one_backbone_and_control_remains_available():
    args=argparse.Namespace(mode='auto',iterations=8,initial_iterations=100)
    assert pipeline_stages(args)==('frax',) and use_unified(args)
    unified=conversion_contract(args)
    args.pipeline='two-stage'
    assert pipeline_stages(args)==('frax','v8') and not use_unified(args)
    assert conversion_contract(args)!=unified
    assert route('auto',1000,pipeline='unified')==('frax',)
    args.mode='frax';assert not use_unified(args)


def test_auto_and_its_worker_share_resume_contract():
    args=argparse.Namespace(mode='auto',iterations=8,initial_iterations=100,pipeline='unified')
    worker=copy.deepcopy(args);worker.mode='frax';worker.unified_repair=True
    assert conversion_contract(args)==conversion_contract(worker)


def test_clean_motion_skips_modules_and_arms_route_per_side():
    errors={s:{k:np.zeros(10) for k in ('upper','lower','hand_forward','hand_width','elbow_axis')} for s in ('Left','Right')}
    metrics=dict(contact=np.zeros(10),foot_floor=np.zeros(10),body_floor=np.ones(10),errors=errors)
    q=quality();q['flags']=[]
    assert plan(q,metrics,np.zeros(10),50.)==[]
    q['flags']=['Left_upper_deg_source_mismatch'];errors['Left']['upper'][:]=25.
    jobs=plan(q,metrics,np.zeros(10),50.)
    assert [(a,b) for a,b,_ in jobs]==[('arms','Left')]


def test_window_guard_preserves_body_and_seams_before_any_geometry_work():
    from types import SimpleNamespace
    robot=SimpleNamespace(sides={'Left':dict(arm=list(range(3,10)))})
    baseline=np.zeros((10,32));candidate=baseline.copy();candidate[4,12]=.1
    assert preservation_guard(robot,{},baseline,candidate,{},None,baseline,'arms','Left',2,8)=='protected_body_changed'
    candidate=baseline.copy();candidate[2,3]=.1
    assert preservation_guard(robot,{},baseline,candidate,{},None,baseline,'arms','Left',2,8)=='left_join_changed'
    candidate=baseline.copy();candidate[0,3]=.1
    assert preservation_guard(robot,{},baseline,candidate,{},None,baseline,'arms','Left',2,8)=='outside_window_changed'
    candidate=baseline.copy();candidate[4,3]=np.deg2rad(31)
    assert preservation_guard(robot,{},baseline,candidate,{},None,baseline,'arms','Left',2,8)=='original_frax_angle_cap'


def test_bvh_auto_review_does_not_restart_v8(tmp_path):
    from dex_frax.cli import single_auto
    source=tmp_path/'input.bvh';source.write_text('Frames: 6001\nFrame Time: 0.02\n')
    args=argparse.Namespace(mode='auto',input=source,output=tmp_path/'out',source='lafan',
        iterations=8,initial_iterations=100,max_frames=None,budget=30.)
    class Worker:
        calls=[]
        def close(self):pass
        def run(self,job,timeout):
            a=job['arguments'];self.calls.append(a)
            assert a['unified_repair'] and a['mode']=='frax'
            folder=Path(a['output']);folder.mkdir()
            return dict(status='complete',quality='review',flags=['pose'],result=dict(total_wall_s=.1,frax=None))
    worker=Worker();record=single_auto(args,lambda:worker)
    assert len(worker.calls)==1 and record['status']=='review'
    assert not (args.output/'v8').exists()


def test_smplx_auto_passes_unified_flag_and_stops_after_review(tmp_path,monkeypatch):
    from dex_frax import smplx_cli
    source=tmp_path/'input.pt';source.write_bytes(b'input')
    monkeypatch.setattr(smplx_cli,'prepare',lambda *a,**k:None)
    args=argparse.Namespace(mode='auto',input=source,output=tmp_path/'out',budget=30.)
    class Worker:
        calls=[]
        def close(self):pass
        def run(self,job,timeout):
            self.calls.append(job);assert job['arguments']['unified_repair']
            return dict(status='complete',quality='review',flags=['pose'])
    worker=Worker();result=smplx_cli.single_auto(args,lambda:worker)
    assert len(worker.calls)==1 and result['status']=='review'
    assert not (args.output/'accepted.json').read_text().count('training.npz')


def test_collision_seams_do_not_freeze_known_penetration(monkeypatch):
    import dex_frax.unified_repair as m
    rows=np.zeros((12,32));rows[7,3]=-1.
    target={'root_r':np.tile(np.eye(3),(12,1,1))}
    monkeypatch.setattr(m,'scan',lambda model,x,r,steps:({'collision_samples':int((x[:,3]<0).sum())},None,None))
    assert m.collision_seams(None,target,rows,2,8)==dict(freeze_start=True,freeze_end=False)
    assert m.collision_seams(None,target,rows,0,12)==dict(freeze_start=False,freeze_end=False)
    rows[7,3]=0.
    assert m.collision_seams(None,target,rows,2,8)==dict(freeze_start=True,freeze_end=True)


def test_subframe_floor_defects_are_routed_when_native_floor_is_clear():
    q=quality();q['flags']=['interpolated_full_body_mesh_penetration_over_1mm'];q['interpolated_floor']={'violating_intervals':[7,19]}
    metrics=arm_metrics();metrics.update(contact=np.zeros(20),foot_floor=np.zeros(20),body_floor=np.zeros(20))
    jobs=plan(q,metrics,np.zeros(20),50.)
    assert len(jobs)==1 and jobs[0][0]=='dense_floor'
    assert np.flatnonzero(jobs[0][2]).tolist()==[7,8,19]


def test_clearance_reduction_is_not_penetration_regression():
    before=quality();after=copy.deepcopy(before)
    before['self_collision']['minimum_distance_m']=.007
    after['self_collision']['minimum_distance_m']=.005
    assert quality_guard(before,after) is None
    before['self_collision']['minimum_distance_m']=-.005
    after['self_collision']['minimum_distance_m']=-.006
    assert quality_guard(before,after)=='collision_depth_regression'
    after['self_collision']['minimum_distance_m']=-.004
    assert quality_guard(before,after) is None


def test_collision_cap_is_window_local_but_outside_must_be_unchanged(monkeypatch):
    from types import SimpleNamespace
    import dex_frax.unified_repair as module
    n=10;r=SimpleNamespace(sides={'Left':dict(arm=list(range(3,10)),leg=list(range(10,16)))},
        lower=np.full(32,-3.),upper=np.full(32,3.),speed_limits=np.full(29,100.))
    original=np.zeros((n,32));before=original.copy();before[0,3]=np.deg2rad(75.)
    metrics=dict(errors=arm_metrics(n)['errors'],contact=np.zeros(n),floor=np.ones(n))
    monkeypatch.setattr(module,'inspect',lambda *args:metrics)
    monkeypatch.setattr(module,'window_targets',lambda *args:{})
    monkeypatch.setattr(module,'torso_errors',lambda r,t,x:np.zeros(len(x)))
    candidate=before.copy();candidate[4,3]=np.deg2rad(29.)
    args=(r,{'fps':50.},before,candidate,{},metrics,original,'collision','Left',2,8,True,metrics)
    assert preservation_guard(*args) is None
    candidate[4,3]=np.deg2rad(31.)
    assert preservation_guard(*args)=='original_frax_shoulder_cap'
    candidate[4,3]=0.;candidate[0,3]+=1e-3
    assert preservation_guard(*args)=='outside_window_changed'
