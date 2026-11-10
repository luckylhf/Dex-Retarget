"""Evidence-only pipeline reports. No IK, motion loading, gate changes or certification.

One reporter serves BVH/SMPL-X, Frax/V8 and selected auto attempts. The current
solver/robot adapter remains Dex-specific; this schema is not a robot driver.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

SCHEMA = 'dex_retarget_pipeline_explanation_v1'
STAGES = {
    'initial_ik': ('初解', '按目标姿态与机器人模型求初解；求解成功不等于验收通过'),
    'local_pose': ('局部姿态拟合', '只处理已标定目标中的难窗，保持根部、腿及衔接'),
    'torso_branch': ('躯干与肩部分支', '先检查腰姿和初始分支，避免靠手臂避碰补偿歪腰'),
    'adaptive_v8': ('按问题调用局部模块', '接触、身体支撑和手臂按残差路由，不重跑整条V8'),
    'self_collision': ('碰撞修复', '固定参考下逐级搜索，随后检查真实几何与连续性'),
    'unified_repair': ('统一检查与候选回退', '只保留通过保护且目标改善的候选，失败保留原结果'),
    'support': ('足部支撑修复', '维持已推断的足部接触和地面约束'),
    'arms': ('V8手臂分支拟合', '独立V8路线的手臂规划与连续拟合'),
    'body_contacts': ('身体接触修复', '处理手、膝等额外支撑，不能套用自由手臂策略'),
}


def _dict(value):
    return value if isinstance(value, dict) else {}


def _list(value):
    return value if isinstance(value, list) else []


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def reason_info(reason):
    """A local failure explains an attempt; it never proves physical unreachability."""
    raw = str(reason or '')
    low = raw.lower()
    rules = [
        (('repair_hand_pose_budget',), 'quality', '手部姿态包络未通过；不是计算超时，应继续其他分支'),
        (('dense_floor_window_budget','window_exceeds_local'), 'unsupported_route', '窗口超出当前求解器支持范围；不是计算时间耗尽'),
        (('budget', 'deadline', 'timeout', 'watchdog', 'time limit'), 'budget', '预算或时间耗尽；不能据此判断无解'),
        (('primal infeasible', 'infeasible', 'maximum iterations', 'nonconvergence'), 'solver', '当前求解或起点未收敛/不可行；不是全局不可达证明'),
        (('no_target_metric', 'no_similarity', 'no_global_similarity', 'no_torso_improvement'), 'objective', '候选未改善当前比较目标'),
        (('shoulder_cap', 'angle_cap'), 'search_envelope', '超过本次固定参考的搜索范围；不等同于机械限位'),
        (('joint_limit', 'velocity', 'speed_or_join'), 'limits_or_seam', '限位、实际速度或拼接检查未通过'),
        (('collision', 'clearance'), 'geometry', '碰撞/间隙检查未通过或产生几何退步'),
        (('floor', 'ground', 'penetration'), 'ground', '地面或帧间穿地检查未通过'),
        (('active_body_support', 'active_upper_body_support', 'active_arm_support'), 'unsupported_route', '存在活动支撑，此自由肢体修复路线不适用'),
        (('contact', 'support'), 'contact', '接触或支撑保护未通过'),
        (('torso', 'source_orientation', 'protected_body', 'outside', 'join_changed'), 'protected_motion', '已正确的躯干、窗口外数据或固定衔接受到破坏'),
        (('palm', 'hand', 'pose', 'quality_flags'), 'quality', '方向包络或整体质量比较未通过'),
        (('passed', 'improved_and_checked', 'safe'), 'accepted_candidate', '候选通过对应检查；仍须看最终整片验收'),
    ]
    for tokens, category, explanation in rules:
        if any(token in low for token in tokens):
            return dict(raw=raw, category=category, explanation=explanation)
    return dict(raw=raw, category='unclassified' if raw else 'not_recorded',
                explanation='保留原始原因；当前解释器不推断未记录的结论')


def flag_info(flag):
    raw = str(flag)
    if raw.startswith('root_') and 'speed' in raw:
        return dict(flag=raw, category='root_speed', action='对照原始人体、预处理与初解；输入过快和求解新增跳变需要分开查，不能仅凭此标记认定来源')
    if raw == 'conversion_added_jitter':
        return dict(flag=raw,category='temporal_quality',action='逐关节和窗口对照人体源动作，修复转换新增的快速反复换向；不把加速度统计当作电机硬限')
    if raw == 'frax_qp_nonconvergence':
        return dict(flag=raw, category='solver', action='检查初解收敛记录及难帧起点；不要把求解失败直接叫作不可达')
    if 'not_checked' in raw:
        return dict(flag=raw, category='missing_check', action='补足独立检查；缺少检查不等于无问题')
    if any(v in raw for v in ('collision', 'penetration', 'joint_limit', 'velocity', 'nonfinite')):
        return dict(flag=raw, category='safety', action='检查对应几何、模型限位或速度；候选必须通过原安全验收')
    if any(v in raw for v in ('contact', 'support', 'floating')):
        return dict(flag=raw, category='contact', action='核对接触证据、地面与支撑；使用接触修复模块')
    if 'torso' in raw or 'pelvis' in raw:
        return dict(flag=raw, category='body_shape', action='先查预处理/躯干/初始分支，不能靠手臂避碰补偿错误腰姿')
    if 'mismatch' in raw:
        return dict(flag=raw, category='resemblance', action='比较避碰前后臂形、肘窝和手心，再尝试受保护的局部拟合与分支')
    if raw == 'too_short_for_training_window':
        return dict(flag=raw, category='input_contract', action='动作长度不满足训练窗口要求；不是IK失败')
    return dict(flag=raw, category='unclassified', action='保留未知标记，不自动降级或删除；查看原质量报告')


def _event(row, module, pointer, fps):
    row = _dict(row)
    state = row.get('status')
    if state is None and isinstance(row.get('kept'), bool):
        state = 'kept' if row['kept'] else 'rejected'
    start, end = row.get('start', row.get('a')), row.get('end', row.get('b'))
    window = None
    if _number(start) and _number(end):
        window = dict(start=start, end_exclusive=end, frame_space='output_zero_based')
        if _number(fps) and fps > 0:
            window.update(start_s=start/fps, end_exclusive_s=end/fps)
    detail = _dict(row.get('detail'))
    tiers = row.get('tiers', detail.get('tiers'))
    children = []
    for i, tier in enumerate(_list(tiers)):
        tier = _dict(tier)
        child = _event(tier, module, pointer+('/tiers/' if isinstance(row.get('tiers'), list) else '/detail/tiers/')+str(i), fps)
        child['degrees'] = tier.get('degrees', tier.get('tier'))
        recovery = _dict(tier.get('recovery'))
        child['validation_routes'] = [dict(route=v.get('route'), status=v.get('status'),
            initial_shoulder_offset=v.get('initial_shoulder_offset'), reason=reason_info(v.get('reason')))
            for v in _list(recovery.get('validation_attempts')) if isinstance(v, dict)]
        children.append(child)
    return dict(module=row.get('module', row.get('stage', module)), side=row.get('side'),
        status=state or 'not_recorded', window=window, evidence=pointer,
        reason=reason_info(row.get('reason')), elapsed_s=row.get('elapsed_s'),
        recipe=row.get('recipe'),degrees=detail.get('tier_deg'),
        comparison=detail.get('candidate_similarity_score'),
        pose_before_after=dict(before=detail.get('before'), after=detail.get('after')), tiers=children)


def _metrics(quality):
    q = _dict(quality)
    return dict(source_pose=q.get('source_pose_audit'), collision=q.get('self_collision'),
        full_collision=q.get('full_self_collision'), body_support=q.get('body_support'),
        temporal_quality=q.get('temporal_quality'), initial_solver_history=q.get('initial_solver_history'),
        interpolated_floor=q.get('interpolated_floor'), step_velocity=q.get('step_velocity'),
        sustained_pose_frames=_dict(q.get('adaptive_pose_gate')).get('violating_frames'),
        root_angular_speed_rad_s=q.get('root_angular_speed_max_rad_s'))


def build_stage(meta=None, quality=None, repair=None, run=None, identity=None, check=None):
    """Normalize saved evidence; never fill historical thresholds from live defaults."""
    meta, quality, repair, run, identity, check = map(_dict, (meta, quality, repair, run, identity, check))
    flags_valid = isinstance(quality.get('flags'), list) and all(isinstance(v, str) for v in quality['flags'])
    flags = quality.get('flags', []) if flags_valid else []
    recorded = quality.get('status', 'not_recorded')
    state = ('review' if flags else recorded) if flags_valid else 'not_recorded'
    execution = run.get('status', 'not_recorded')
    if execution in ('failed', 'input_error', 'timeout', 'interrupted', 'budget_exhausted', 'running'):
        state = execution
    research = bool(meta.get('research_checkpoint') or meta.get('research_merge') or quality.get('backend') == 'research_checkpoint')
    stages = [dict(id='source', name='输入与源适配', status='recorded' if meta.get('input_sha256') or meta.get('source_type') else 'not_recorded',
                   evidence='motion.json:input_sha256/source_type/input_adapter'),
              dict(id='targets', name='整片目标、比例与接触标定', status='recorded' if meta.get('prepared_targets_sha256') else 'not_recorded',
                   evidence='motion.json:prepared_targets_sha256/ground_calibration')]
    priority = _dict(_dict(meta.get('frax')).get('body_priority'))
    times = _dict(meta.get('stage_times_s')) or _dict(run.get('stage_times_s'))
    for key, duration in times.items():
        name, purpose = STAGES.get(key, (key, '未登记阶段，保留原记录'))
        stages.append(dict(id=key, name=name, purpose=purpose, status='recorded', elapsed_s=duration,
                           evidence='motion.json:stage_times_s/'+key))
        if key=='initial_ik' and priority:
            stages.append(dict(id='body_priority',name='骨盆、腰部优先（初解内）',status='recorded',
                evidence='motion.json:frax/body_priority',included_in='initial_ik',policy=priority.get('policy'),
                body_geometry_preflight=priority.get('body_geometry_preflight')))
    stages.append(dict(id='audit', name='独立整片验收', status=state, evidence='quality.json', elapsed_s=quality.get('audit_s')))
    stages.append(dict(id='export', name='训练导出', status='recorded' if run.get('training_npz') else 'not_recorded',
                       evidence='run.json:training_npz', path=run.get('training_npz')))
    events = []
    frax = _dict(meta.get('frax'))
    fps = meta.get('output_fps', run.get('fps'))
    for name, key in [('local_refinement','windows'), ('torso_branch_recovery','attempts'), ('adaptive_refinement','repairs')]:
        for i, row in enumerate(_list(_dict(frax.get(name)).get(key))):
            events.append(_event(row, name, f'motion.json:frax/{name}/{key}/{i}', fps))
    collision = _dict(meta.get('self_collision'))
    for i, row in enumerate(_list(collision.get('windows'))):
        events.append(_event(row, 'self_collision', f'motion.json:self_collision/windows/{i}', fps))
    graduated = _dict(collision.get('graduated_repair'))
    for i, row in enumerate(_list(graduated.get('attempts'))):
        events.append(_event(row, 'graduated_collision', f'motion.json:self_collision/graduated_repair/attempts/{i}', fps))
    for i, row in enumerate(_list(repair.get('attempts'))):
        events.append(_event(row, 'unified_repair', f'repair.json:attempts/{i}', fps))
    warnings = []
    if not flags_valid: warnings.append('没有可解释的最终质量标记列表，不能判定合格')
    if flags and recorded == 'accepted': warnings.append('记录矛盾：accepted同时带有失败标记，按review展示')
    if not quality.get('interpolated_floor'): warnings.append('没有帧间地面审计证据；不推断历史产物已经做过此检查')
    if research: warnings.append('研究检查点，不代表默认一次完整转换或训练通过')
    if not times: warnings.append('未记录阶段顺序/耗时；不能据此推断哪些阶段已运行或跳过')
    return dict(schema=SCHEMA, kind='stage', status=state, recorded_quality=recorded,
        execution_status=execution, research_checkpoint=research, authoritative=False,
        profile=meta.get('profile',run.get('profile')),
        workflow=run.get('pipeline', 'unified' if meta.get('unified_repair') else 'not_recorded'),
        source=dict(type=meta.get('source_type'), input=meta.get('input'), sha256=meta.get('input_sha256'),
                    frames=meta.get('input_frames'), fps=meta.get('input_fps'), adapter=meta.get('input_adapter')),
        target=dict(model=meta.get('model'), sha256=meta.get('model_sha256'), joint_names=meta.get('joint_names'),
                    output_fps=fps, frames=meta.get('output_frames', quality.get('frames'))),
        recorded_identity=dict(solver_policy=meta.get('solver_policy'), converter_release=meta.get('converter_release'),
            quality_policy=quality.get('quality_policy'), fingerprint=identity.get('fingerprint'),
            prepared_targets_sha256=meta.get('prepared_targets_sha256'), conversion_contract=identity.get('conversion_contract')),
        acceptance=dict(criteria=quality.get('criteria'), sustained_pose=_dict(quality.get('adaptive_pose_gate')).get('policy'),
                        geometry_m=quality.get('support_acceptance_m')),
        retiming=meta.get('retiming'),
        body_priority={k:priority[k] for k in ('policy','root_mode','body_geometry_preflight') if k in priority},
        repair_budget=repair.get('policy'), stages=stages, attempts=events,
        attempt_status_counts=dict(Counter(v['status'] for v in events)),
        unresolved=[flag_info(v) for v in flags], final_metrics=_metrics(quality),
        arm_axis_metrics={s:{k:v for k,v in _dict(d).items() if k in ('upper_deg','lower_deg','hand_forward_deg','hand_width_deg','forearm_width_deg','elbow_axis_deg','signed_elbow_goal_error_deg')} for s,d in _dict(check.get('sides')).items()},
        unified_before_after=dict(before=_metrics(repair.get('before')), after=_metrics(repair.get('after'))),
        warnings=warnings, limitations=['只解释已有JSON记录，不重新求解或重新验收',
          '预算耗尽、局部不可行与分支用尽都不证明全局无解',
          '当前转换实现仍绑定Dex 29关节；报告协议可复用不代表任意机器人已适配',
          '运动学/采样几何验收不代表动力学跟踪或真机安全'])


def _read(path, evidence, warnings):
    if not path.exists(): return {}
    try:
        data = path.read_bytes()
        value = json.loads(data)
        if not isinstance(value, dict): raise ValueError('expected JSON object')
        evidence.append(dict(path=str(path), sha256=hashlib.sha256(data).hexdigest()))
        return value
    except (ValueError, OSError) as exc:
        warnings.append(f'{path.name}: 无法读取JSON证据 ({exc})')
        return {}


def explain(path):
    """Read JSON sidecars only. Never unpickle an input or change an old output."""
    path = Path(path).resolve()
    if not path.exists(): raise FileNotFoundError(path)
    evidence, warnings = [], []
    if path.is_dir() and (path/'selection.json').exists():
        selection = _read(path/'selection.json', evidence, warnings)
        attempts, seen = [], set()
        for item in _list(selection.get('attempts')):
            if not isinstance(item, dict): continue
            mode = item.get('mode')
            if mode not in ('frax','v8'):
                warnings.append(f'未识别的阶段路径 {mode!r}，未读取'); continue
            if mode in seen: continue
            seen.add(mode); child = path/mode
            if child.resolve() == path or not child.resolve().is_relative_to(path):
                warnings.append(f'{mode}: 阶段路径越出结果目录，未读取'); continue
            child_report = None
            if child.is_dir():
                try:
                    child_report = explain(child)
                except (ValueError, OSError) as exc:
                    warnings.append(f'{mode}: 阶段记录不完整 ({exc})')
            attempts.append(dict(mode=mode, recorded_attempt=item, report=child_report))
        return dict(schema=SCHEMA, kind='selection', status=selection.get('status','not_recorded'),
            workflow=selection.get('pipeline','not_recorded'), authoritative=False,
            selected_output=selection.get('selected_output'), preview_output=selection.get('preview_output'),
            attempts=attempts, evidence=evidence, warnings=warnings,
            limitations=['这是已有选择记录的说明；不会重选结果、重新验收或批准训练导出'])
    if path.is_dir() and (path/'batch.json').exists() and not (path/'quality.json').exists():
        raise ValueError('批次目录请指定具体动作子目录；本命令不扫描或重跑整个批次')
    stem = path/'motion' if path.is_dir() else path.with_suffix('')
    folder = stem.parent
    if not path.is_dir() and path.suffix.lower() not in ('.pkl','.npz'):
        raise ValueError('请指定结果目录或机器人PKL/NPZ；只读取其JSON旁文件')
    meta = _read(stem.with_suffix('.json'), evidence, warnings)
    qp = folder/'quality.json' if stem.name == 'motion' else stem.with_suffix('.quality.json')
    quality = _read(qp, evidence, warnings)
    run = _read(folder/'run.json', evidence, warnings)
    report = build_stage(meta, quality, _read(folder/'repair.json', evidence, warnings), run,
        _read(folder/'identity.json', evidence, warnings), _read(stem.with_suffix('.check.json'), evidence, warnings))
    if not evidence: raise ValueError('没有找到转换JSON记录；不能把原始人体文件当作已转换结果')
    report['evidence'] = evidence; report['warnings'].extend(warnings)
    return report


def _cell(value):
    return str(value if value is not None else '未记录').replace('|','\\|').replace('\n',' ')


def render(report):
    lines = ['# Retarget pipeline 解释', '', f"记录状态：**{_cell(report['status'])}**。此报告不重新验收、不授权训练导出。", '']
    if report['kind'] == 'selection':
        lines += [f"流程：{_cell(report['workflow'])}；选中：{_cell(report.get('selected_output'))}。", '']
        for item in report['attempts']:
            lines += [f"## 阶段 {_cell(item['mode'])}", '']
            lines += render(item['report']).splitlines()[2:] if item.get('report') else ['没有阶段JSON记录；仅保留调度记录。', '']
    else:
        lines += [f"模式：{_cell(report.get('profile'))}。",'',f"输入：{_cell(report['source']['type'])}；输出：{_cell(report['target']['frames'])}帧 / {_cell(report['target']['output_fps'])} FPS。", '',
            '|阶段|作用|记录状态|秒|', '|---|---|---|---:|']
        for stage in report['stages']:
            lines.append(f"|{_cell(stage['name'])}|{_cell(stage.get('purpose'))}|{_cell(stage['status'])}|{_cell(stage.get('elapsed_s'))}|")
        lines += ['', '## 未通过的检查', '']
        lines += [f"- `{v['flag']}`：{v['action']}" for v in report['unresolved']] or ['没有记录到失败标记；是否完成/通过仍以原始质量与执行状态为准。']
        lines += ['', '## 候选处理记录', '', '|模块 / 侧|输出帧窗口（0基，右端不含）|结果|原因|', '|---|---|---|---|']
        for e in report['attempts']:
            w = e['window']; window = f"[{w['start']}, {w['end_exclusive']})" if w else '未记录'
            reason = (('候选 '+str(e.get('recipe'))+'；') if e.get('recipe') else '')+e['reason']['explanation'] + (': '+e['reason']['raw'] if e['reason']['raw'] else '')
            if e['tiers']:
                reason += '; ' + '; '.join(f"{t['degrees']}°: {t['status']} / {t['reason']['raw']}" for t in e['tiers'])
            lines.append(f"|{_cell(e['module'])} / {_cell(e['side'])}|{window}|{_cell(e['status'])}|{_cell(reason)}|")
        lines += ['', '## 已记录的验收参数与最终指标', '', '```json', json.dumps(dict(
            acceptance=report['acceptance'], final_metrics=report['final_metrics'], arm_axis_metrics=report['arm_axis_metrics']), ensure_ascii=False, indent=2), '```', '']
    lines += ['', '## 记录边界', ''] + ['- '+v for v in report.get('warnings',[])+report['limitations']]
    return '\n'.join(lines)+'\n'


def write_report(report, folder):
    """Caller owns a new directory; refuse report overwrite even there."""
    folder = Path(folder)
    for name in ('pipeline.json','pipeline.md'):
        if (folder/name).exists(): raise FileExistsError(folder/name)
    (folder/'pipeline.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (folder/'pipeline.md').write_text(render(report))


def write_stage_report(folder, meta, quality, identity, context):
    folder = Path(folder); evidence, warnings = [], []
    repair = _read(folder/'repair.json', evidence, warnings)
    check = _read(folder/'motion.check.json', evidence, warnings)
    for name in ('motion.json','quality.json','identity.json'):
        _read(folder/name, evidence, warnings)
    report = build_stage(meta, quality, repair, context, identity, check)
    report['evidence'] = evidence; report['warnings'].extend(warnings)
    write_report(report, folder)
