import argparse,copy,json,pickle
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest


def test_mocap_rest_width_accumulates_separate_parent_offsets():
    from dex_frax.v8.source import rest_forearm_width
    motion=SimpleNamespace(names=['Root','LeftForeArm','LeftHand','LeftInHandIndex','LeftHandIndex1','LeftInHandPinky','LeftHandPinky1'],
        parents=[-1,0,1,2,3,2,5],offsets=np.array([[0,0,0],[0,1,0],[0,1,0],[.2,.1,0],[.1,.3,0],[-.2,-.1,0],[-.05,.2,0]]))
    expected=np.array([.55,.3,0])
    np.testing.assert_allclose(rest_forearm_width(motion,'Left'),expected)
    assert not np.allclose(expected,motion.offsets[4]-motion.offsets[6])


def test_bvh_units_and_delta_semantics_are_explicit(tmp_path):
    from dex_frax.v8.bvh import load
    text='''HIERARCHY
ROOT Hips
{
 OFFSET 100 0 0
 CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation
 End Site
 {
 OFFSET 0 100 0
 }
}
MOTION
Frames: 3
Frame Time: 0.02
200 0 0 0 0 0
200 0 0 0 0 0
200 0 0 0 0 0
'''
    p=tmp_path/'a.bvh';p.write_text(text)
    absolute=load(p,units='cm',translations='absolute');delta=load(p,units='cm',translations='offset')
    assert absolute.local_p[0,0,0]==2 and delta.local_p[0,0,0]==3
    np.testing.assert_allclose(load(p,units='m').local_p,absolute.local_p*100)
    with pytest.raises(ValueError,match='translation'):load(p,translations='guess')
    from dex_frax.contract import conversion_contract
    args=argparse.Namespace(iterations=8,initial_iterations=100)
    old=conversion_contract(args);args.bvh_units='m'
    assert conversion_contract(args)!=old


def test_single_cli_carries_bvh_conventions_to_owned_worker(tmp_path):
    from dex_frax.cli import single_auto
    src=tmp_path/'x.bvh';src.write_text('Frames: 80\nFrame Time: 0.02\n')
    args=argparse.Namespace(input=src,output=tmp_path/'out',source='mocap',mode='auto',profile='fast',
        iterations=8,initial_iterations=100,max_frames=40,budget=30.,bvh_units='m',bvh_translations='offset')
    class Worker:
        def close(self):pass
        def run(self,job,timeout):
            assert job['arguments']['bvh_units']=='m' and job['arguments']['bvh_translations']=='offset'
            p=Path(job['arguments']['output']);p.mkdir();(p/'quality.json').write_text(json.dumps(dict(status='review',flags=['test'])))
            return dict(status='complete',quality='review',flags=['test'],result=dict(total_wall_s=.1,frax=None,files={}))
    single_auto(args,Worker)


def test_speed_gate_rejects_corrupted_export_even_when_generator_uses_same_limit():
    from dex_frax.quality_v8 import adjacent_speed_report
    q=np.zeros((10,1));q[5:,0]=.2
    assert adjacent_speed_report(q,50.,['joint'],[1.])['violating_intervals']==1


def test_root_integrity_check_rejects_corrupted_quaternion(tmp_path):
    from test_incremental_audit import export_neutral
    from dex_frax.v8.audit import audit
    from dex_frax.quality_v8 import quality
    from scipy.spatial.transform import Rotation
    p,data,meta,dg=export_neutral(tmp_path,n=40);meta['support_constraint_failures']=[]
    data['root_rot']=np.tile(Rotation.from_euler('z',10,degrees=True).as_quat(),(40,1)).astype(np.float32)
    p.write_bytes(pickle.dumps(data));before=p.read_bytes();check=audit(p)
    assert 'source_pelvis_orientation_mismatch' in quality(p,meta,check)['flags']
    assert p.read_bytes()==before and check['full_self_collision']['pair_set_sha256']
    assert set(check['root_translation_error_cm'])=={'xy','z','reference'}


def test_nonconvergence_counts_are_per_stage_not_four_times_probability():
    from dex_frax.quality_scope import solver_history
    stage=dict(qp_nonconverged_steps=1,qp_nonconverged_output_frames=[3],qp_selected_iteration_count=100)
    report=dict(qp_nonconverged_steps=4,raw_frax_report=stage,body_priority=dict(pelvis_feasibility_solver=stage,ordered_limb_solvers={'feet':stage,'arms':stage}))
    actual=solver_history(report)
    assert actual['qp_nonconverged_steps']==4 and actual['unique_affected_output_frames']==[3]
    assert len(actual['stages'])==4 and not actual['cleared_by_repair']
    assert solver_history({'qp_nonconverged_steps':1})['unique_affected_output_frames'] is None


def test_velocity_impulse_routes_to_quality_without_inventing_accel_limit():
    from dex_frax.temporal_quality import assess
    q=np.zeros((40,1));q[20:]=np.deg2rad(2.)
    dg={'Left_upper':np.tile([1.,0,0],(40,1))}
    result=assess(q,50.,['shoulder_pitch_l_joint'],dg)
    assert result['joints']['shoulder_pitch_l_joint']['added_velocity_impulse_frames']
    assert result['violating_frames'] and not result['policy']['hardware_acceleration_limit']
    assert not assess(q,50.,['shoulder_pitch_l_joint'],{})['violating_frames']
    a=np.zeros(40);a[20:]=np.deg2rad(2.);dg['Left_upper']=np.c_[np.cos(a),np.sin(a),np.zeros(40)]
    assert not assess(q,50.,['shoulder_pitch_l_joint'],dg)['violating_frames']


def test_leg_guard_allows_necessary_safety_tradeoff_but_rejects_dominated_change():
    from dex_frax.leg_similarity import tradeoff
    old=np.ones((20,4))*5;new=old+3
    assert tradeoff(old,new,.02,.02)['dominated']
    better=tradeoff(old,new,.02,.01)
    assert not better['dominated'] and better['shape_cost_for_safety']


def test_quality_scope_does_not_turn_missing_thresholds_or_empty_reports_into_certificates():
    from dex_frax.quality_scope import describe
    check=dict(sides={},full_self_collision={'pair_count':1235,'pair_set_sha256':'a'},temporal_quality={})
    result=describe({'support_solver_execution':{'status':'not_run','executed':False}},check,{'criteria':{}})
    assert result['support_solver_execution']['executed'] is False
    assert set(result['acceptance_scope']['unthresholded_similarity'])=={'leg_direction','forearm_axial_direction','root_xy','root_z'}
    fingerprint=result['audit_contract']['fingerprint'];check['full_self_collision']['pair_set_sha256']='b'
    assert describe({},check,{'criteria':{}})['audit_contract']['fingerprint']!=fingerprint


def test_initial_branch_reduction_only_when_seed_axes_are_fixed():
    from dex_frax.ik import initial_branch_count,runtime_keys
    assert initial_branch_count({})==3
    mask=np.ones(32,bool);mask[:15]=False
    assert initial_branch_count({'priority_mask':mask})==1
    mask[20]=False
    assert initial_branch_count({'priority_mask':mask})==3


def test_optional_similarity_thresholds_are_actual_gates(tmp_path,monkeypatch):
    from test_incremental_audit import export_neutral
    from dex_frax.v8.audit import audit
    from dex_frax import quality_pose
    p,data,meta,dg=export_neutral(tmp_path,n=40);meta['support_constraint_failures']=[]
    check=audit(p);check['sides']['Left']['thigh_deg']['p95']=25.;check['sides']['Left']['forearm_width_deg']['p95']=32.
    check['root_translation_error_cm']['xy']['max']=8.
    limits=json.loads(Path(quality_pose.__file__).with_suffix('.json').read_text())
    limits.update(leg_direction_p95_deg=20.,forearm_width_p95_deg=30.,root_xy_max_m=.05)
    (tmp_path/'quality_pose.json').write_text(json.dumps(limits));monkeypatch.setattr(quality_pose,'__file__',str(tmp_path/'quality_pose.py'))
    flags=quality_pose.quality(p,meta,check)['flags']
    assert {'Left_thigh_deg_source_mismatch','Left_forearm_width_deg_source_mismatch','root_xy_source_mismatch'}<=set(flags)


def test_same_pose_can_pass_upper_body_but_fail_expanded_collision_scope():
    from dex_frax.v8.robot import Robot
    from dex_frax.v8.self_collision import CollisionModel
    r=Robot();old=CollisionModel(r);full=CollisionModel(r,{'coverage':'full'});found=False
    for angle in (-.3,.3,-.6,.6):
        x=np.r_[0.,0.,1.,np.zeros(29)];x[r.sides['Left']['leg'][1]]=angle;x[r.sides['Right']['leg'][1]]=-angle
        saved=x.copy();old.update(x,np.eye(3));full.update(x,np.eye(3))
        if old.distances().min()>=0 and full.distances().min()<-.001:found=True
        np.testing.assert_array_equal(x,saved)
    assert found
