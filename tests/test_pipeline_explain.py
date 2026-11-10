import argparse
import hashlib
import json
from pathlib import Path
import pytest
from dex_frax.pipeline_explain import build_stage, explain, reason_info, render, write_report


def put(folder, name, data):
    folder.mkdir(parents=True, exist_ok=True)
    (folder/name).write_text(json.dumps(data))


def quality(flags=None):
    return dict(status='review' if flags else 'accepted', flags=flags or [],
        criteria={'arm_direction_p95_deg':17.}, adaptive_pose_gate={'violating_frames':3,'policy':{'elbow_axis_deg':37.}},
        interpolated_floor={'samples_below_minus_1mm':0})


def test_missing_and_contradictory_quality_never_become_success():
    assert build_stage()['status']=='not_recorded'
    report=build_stage(quality={'status':'accepted','flags':['unknown_new_flag']})
    assert report['status']=='review' and report['unresolved'][0]['category']=='unclassified'
    assert report['warnings'] and report['authoritative'] is False
    assert build_stage(quality={'status':'accepted','flags':'invalid'})['status']=='not_recorded'
    assert build_stage(quality=quality(),run={'status':'failed'})['status']=='failed'


def test_historical_thresholds_and_evidence_are_not_replaced_by_defaults():
    report=build_stage(quality=quality(),meta={'output_fps':30,'stage_times_s':{'initial_ik':1}})
    assert report['acceptance']['criteria']['arm_direction_p95_deg']==17.
    assert report['acceptance']['sustained_pose']['elbow_axis_deg']==37.
    assert [s['id'] for s in report['stages']]==['source','targets','initial_ik','audit','export']
    assert report['stages'][2]['status']=='recorded'
    assert report['stages'][0]['status']=='not_recorded'


@pytest.mark.parametrize('reason,kind',[
    ('repair_budget','budget'),('collision_repair_interrupted_watchdog','budget'),
    ('collision_qp_primal infeasible','solver'),('original_frax_shoulder_cap','search_envelope'),
    ('joint_limit_regression','limits_or_seam'),('source_orientation_regression','protected_motion'),
    ('new_or_deeper_collision_sample','geometry'),('no_target_metric_improvement','objective'),
    ('active_upper_body_support','unsupported_route'),('new_future_reason','unclassified')])
def test_failure_categories_do_not_imply_global_unreachability(reason,kind):
    info=reason_info(reason)
    assert info['raw']==reason and info['category']==kind


def test_tiers_windows_and_source_palms_are_preserved():
    record={'module':'collision','start':50,'end':100,'status':'kept','detail':{'tiers':[
        {'degrees':10,'kept':False,'reason':'repair_budget'},
        {'degrees':20,'kept':True,'reason':'improved_and_checked','recovery':{'validation_attempts':[
            {'route':'continuous_multistart_with_source_palms','status':'passed','initial_shoulder_offset':[-1,0,0]}]}}]}}
    report=build_stage(meta={'output_fps':50},quality=quality(),repair={'attempts':[record]})
    event=report['attempts'][0]
    assert event['window']['start_s']==1. and event['window']['end_exclusive_s']==2.
    assert event['tiers'][0]['reason']['category']=='budget'
    assert event['tiers'][1]['validation_routes'][0]['initial_shoulder_offset']==[-1,0,0]
    assert event['tiers'][0]['evidence']=='repair.json:attempts/0/detail/tiers/0'
    assert '10°' in render(report) and report['attempt_status_counts']=={'kept':1}


def test_read_only_custom_research_pkl_never_unpickles(tmp_path):
    motion=tmp_path/'custom_REVIEW.pkl';motion.write_bytes(b'not a pickle')
    put(tmp_path,'custom_REVIEW.json',{'research_checkpoint':True})
    put(tmp_path,'custom_REVIEW.quality.json',quality(['sustained_arm_pose_mismatch']))
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    report=explain(motion)
    assert report['research_checkpoint'] and report['status']=='review'
    assert before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    assert '研究检查点' in render(report)


def test_selection_reports_only_observed_attempts_and_rejects_escape(tmp_path):
    put(tmp_path,'selection.json',{'status':'review','pipeline':'unified','attempts':[{'mode':'frax'},{'mode':'../secret'}]})
    put(tmp_path/'frax','quality.json',quality(['unknown']))
    report=explain(tmp_path)
    assert [a['mode'] for a in report['attempts']]==['frax']
    assert report['warnings'] and 'V8' not in render(report)


def test_selection_self_symlink_does_not_recurse(tmp_path):
    put(tmp_path,'selection.json',{'status':'failed','attempts':[{'mode':'frax'}]})
    (tmp_path/'frax').symlink_to(tmp_path,target_is_directory=True)
    report=explain(tmp_path)
    assert not report['attempts'] and report['warnings']


def test_incomplete_or_malformed_evidence_is_explicit(tmp_path):
    put(tmp_path,'run.json',{'status':'timeout'})
    (tmp_path/'quality.json').write_text('{broken')
    report=explain(tmp_path)
    assert report['status']=='timeout' and any('无法读取' in w for w in report['warnings'])
    assert report['unresolved']==[]


def test_raw_input_and_batch_root_are_not_certified(tmp_path):
    source=tmp_path/'human.npz';source.write_bytes(b'raw')
    with pytest.raises(ValueError,match='没有找到'):explain(source)
    put(tmp_path,'batch.json',{'status':'complete'})
    with pytest.raises(ValueError,match='具体动作'):explain(tmp_path)


def test_write_report_refuses_overwrite(tmp_path):
    report=build_stage(quality=quality())
    write_report(report,tmp_path)
    with pytest.raises(FileExistsError):write_report(report,tmp_path)


def test_cli_convert_generates_explanation_without_changing_quality(tmp_path,monkeypatch):
    import dex_frax.cli as cli
    import dex_frax.v8.convert as converter
    import dex_frax.acceptance as acceptance
    monkeypatch.setattr(cli,'identity',lambda args:{'fingerprint':'test'})
    def convert(*args,**kwargs):
        path=Path(args[1]);path.write_bytes(b'not-unpickled-by-reporter')
        return dict(model_sha256='model',input_sha256='input',output_frames=40,output_fps=50,
            stage_times_s={'initial_ik':.1},frax=None)
    monkeypatch.setattr(converter,'convert',convert)
    monkeypatch.setattr(acceptance,'evaluate',lambda *args:quality(['unknown_new_flag']))
    args=argparse.Namespace(mode='v8',output=tmp_path/'out',input=tmp_path/'input',source='mocap',max_frames=None)
    result=cli.convert(args)
    report=json.loads((args.output/'pipeline.json').read_text())
    assert report['unresolved'][0]['flag']=='unknown_new_flag'
    assert result['flags']==['unknown_new_flag'] and result['quality']=='review'
    assert 'pipeline.json' in result['files'] and 'pipeline.md' in result['files']
    assert not (args.output/'training.npz').exists()


def test_body_priority_explanation_uses_recorded_preflight_only():
    priority=dict(policy='pelvis_then_safe_waist_then_limbs_v2',root_mode='feasible',body_geometry_preflight=dict(pair_count=22,unresolved_frames=[]))
    report=build_stage(meta={'stage_times_s':{'initial_ik':3},'frax':{'body_priority':priority}},quality=quality(['full_self_collision_unresolved']))
    body=next(s for s in report['stages'] if s['id']=='body_priority')
    assert body['included_in']=='initial_ik' and body['status']=='recorded'
    assert report['body_priority']==priority and report['status']=='review'
    assert build_stage()['body_priority']=={}


def test_pose_budget_is_explained_as_quality_not_time():
    assert reason_info('repair_hand_pose_budget')['category']=='quality'
    assert reason_info('dense_floor_window_budget')['category']=='unsupported_route'
