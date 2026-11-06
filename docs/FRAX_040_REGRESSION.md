# 0.4.0 多动作回归（2026-11-05）

本轮优先限制计算成本、复用编译，没有增加物理约束或收紧验收门槛。默认 frax 禁止自动整片 V8 升级；显式 hybrid 仍提供完整精修。最多 4 个姿态窗口、每窗口 6 次迭代；V8 最多 3 个窗口、累计尝试帧数 max(100, round(N×25%))。最终避碰和审核仍执行，因此这是工作量预算，不是严格秒数上限。

## 数据与复现

私有数据在 `regression_local/suite_20261105/`，整个目录已加入 `.gitignore`，没有上传。manifest 固定原始和片段 SHA256、源帧范围与筛选指标。包含 nobody 全片，以及 walk_turn、run、sprint、dance、jump、obstacle、aiming、fight、ground、get_up、stumble 共 11 个约 6 秒 LAFAN 独立片段。类别来自源名称及运动学筛选，未声称人工确认每一帧语义。尚无运动中的 SOMA/BONES 样本；不能代表全库。

独立片段有自己的初始化及接触预处理，不等价于整片截取。`scripts/build_regression_set.py` 构造样本；`scripts/run_regression.py` 运行冻结基线；`dex-frax batch` 运行新版本；`scripts/summarize_regression.py --candidate final_fast_v040 --output final_comparison.json --dense` 汇总同一套审核及插值碰撞检查。

## 结果

| 指标 | 0.3.0 frax | 0.4.0 frax |
|---|---:|---:|
| 当前审核通过/待审 | 7 / 5 | 8 / 4 |
| sprint 躯干最大方向误差 | 12.885° | 5.571° |
| dance 声明碰撞对异常帧 | 4 | 0 |
| dance 躯干最大方向误差 | 34.323° | 34.552° |
| nobody 躯干最大方向误差 | 1.103° | 1.103° |

nobody 输出与冻结基线逐字节相同。新预算前后通过集合一致。dance 的碰撞改善，但姿态最大误差略增，不能宣称全面提升。obstacle、ground、get_up、stumble 仍待审；get_up 相比旧快速版新增全身穿地超限，必须保留失败，不作为训练参考。V8 原生报告为 9 通过 / 3 待审；补齐相同的持续手臂门槛后为 7 / 5，因此不能直接混用原始通过率。

最终同进程 12 项总耗时 71.05 秒，其中 nobody 20.02 秒，其余单项约 2.98–9.85 秒。冻结基线计时与开发工作有重叠，不据此宣称严格倍数提升。独立、同一 walk_turn 输入的冷/热测试分别为 10.07 / 3.43 秒，PKL SHA256 完全相同；收益主要来自模型和编译复用。新 batch 中模型命中 11/12，内核命中 10/12；不同长度/配置仍可能重新编译。

肘轴校准另有风险：aiming 的源轴与几何弯曲平面残差中位数约 31–32°，因此本轮没有盲目增加肘窝权重。应先核查源轴可靠性，再决定局部修复。

90 项测试通过（42 条依赖弃用警告）；包含缓存失效、不同初始姿态复用与无缓存对照、窗口连接、禁止隐式全片回退以及预算不放宽验收。逐帧和插值审核是运动学验证，不包含控制跟踪、动力学或真机结论。

详细本地证据：`final_comparison.json`、`results/final_fast_v040/batch.json`、`isolated_cache_validation.json`、`tests_final_v040.log`。
