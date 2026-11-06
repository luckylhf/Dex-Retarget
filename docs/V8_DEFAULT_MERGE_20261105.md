# V8 自动联合手臂修复已进入默认流程

Dex-Frax 0.1.1 同步正式 V8 的 `cpu_mink_margin_aware_arm_selection_20261105`。
`hybrid` 和 `v8` 模式都会自动检查碰撞窗口中的间距不足情况，必要时加入
另一侧手臂。`frax` 纯 IK 模式仍不运行 V8 后处理。

两处 `self_collision.py` 字节一致。原局部优化器、支撑保护、限位、限速、
碰撞目标和失败回滚未改。单臂情况不强制双臂，干净动作不启动修复。

正式 V8 111 项测试、Dex-Frax 67 项测试通过。正式 V8 从原始 nobody BVH
完整重跑默认入口后，自动修复两个窗口，全片最终碰撞帧为零；32 等分帧间
检查也未检出碰撞。根位姿、非手臂关节及片尾修复接缝前数据与旧 V8 逐值一致。
这只覆盖本条完整动作的运动学审核，未做全库、训练或真机验证。

- [正式合入说明](../../Dex-SuperMimic/docs/V8_COUPLED_ARMS_20261105.md)
- [默认入口完整动作 PKL](../outputs/v8_default_merge_20261105/formal_default/motion.pkl)
- [默认入口验证记录](../outputs/v8_default_merge_20261105/formal_default/validation.json)
- [质量审核](../outputs/v8_default_merge_20261105/formal_default/quality.json)

旧三路对照、独立重试产物和初版验证记录继续保留，不能把它们的版本和
耗时当作本版结果。默认流程现在无需再调用 `retry_v8_window.py`。
