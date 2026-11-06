# V8 nobody 片尾避碰重试 — 2026-11-05

本次修复成功。原 V8 输出仍保存在 `outputs/nobody_v8`；新结果另存于
`outputs/nobody_v8_retry_20261105/both_arms`。

## 失败原因

原窗口为零基帧 `[1081, 1158)`，即 21.62–23.14 秒。原算法根据已经穿透的
碰撞对选择活动手臂，因此只选中了左臂；但是 QP 对全部候选碰撞对都要求
达到 3.5 mm 安全间距。

帧 1138–1157 的右腕/右髋没有穿透，但间距小于 3.5 mm。只优化左臂时，
这些右侧距离约束的雅可比在所有可调变量上均为零，同时右端项为负，
即出现 `0 <= negative` 的不可行约束。本次逐帧检查发现 20 行此类冲突。
证据保存在 `outputs/nobody_v8_retry_20261105/zero_rows.json`。

这说明上次的不可行至少包含了活动关节选择与全局距离约束不一致的问题，
不代表该动作不存在无碰撞的可行解。

## 本次操作

在同一窗口内将活动关节改为左右双臂，继续调用原 V8 的 `_repair_window`。
修复器文件与最新 `Dex-SuperMimic` 的 `self_collision.py` 字节一致。
未修改碰撞模型、3.5 mm 求解目标、限位、速度限制、窗口或迭代预算。
上游代码和默认转换器未改，只新增了独立重试脚本。

```sh
.venv/bin/python scripts/retry_v8_window.py outputs/nobody_v8 outputs/new_retry \
  --start 1081 --end 1158 --sides Left Right
```

## 结果

| 项目 | 原 V8 | 局部重试后 |
|---|---:|---:|
| 全片原生帧碰撞数 | 60 | 0 |
| 最深穿透 | 19.64 mm | 无检出 |
| 相邻关节速度违规 | 0 | 0 |
| 本次修复迭代 | 失败 | 3 |
| 左臂最大改动 | — | 2.5373°，左肩 roll，22.58 s |
| 右臂最大改动 | — | 0.2360°，右肩 yaw，21.80 s |

新片尾窗口原生帧最小距离约 3.5000 mm；修复器的窗口插值检查最小约
3.4964 mm。全片再以每个原始帧间隔 32 等分扫描，240 组既有声明碰撞对
均无检出碰撞，最小距离约 **2.7143 mm**，出现在先前已修好的帧 309 附近，
不是本次片尾窗口。采样检查不等于连续碰撞数学证明。

根位置、根四元数、腰腿、窗口外动作及窗口开头两帧均逐值不变。
脚底和接触指标逐值保持；最终 50 Hz、float32 输出的速度检查通过。
双手前向误差 P95 保持不变；左手宽度方向误差 P95 由 1.0668° 变为
1.0918°。通过当前全片运动学审核，标为 `accepted_kinematic_candidate`。
本次未训练、未做动力学/控制器/真机验证，也未改另外两条 frax/hybrid 结果。

## 交付文件

- [修复后的完整 PKL](../outputs/nobody_v8_retry_20261105/both_arms/motion.pkl)
- [修复后的完整 NPZ](../outputs/nobody_v8_retry_20261105/both_arms/motion.npz)
- [修复记录](../outputs/nobody_v8_retry_20261105/both_arms/retry.json)
- [完整动作质量审核](../outputs/nobody_v8_retry_20261105/both_arms/quality.json)
- [32 等分帧间检查](../outputs/nobody_v8_retry_20261105/dense_validation.json)
- [片尾修复前后 GIF](../outputs/nobody_v8_retry_20261105/tail_before_after.gif)
- [片尾修复前后视频](../outputs/nobody_v8_retry_20261105/tail_before_after.mp4)

预览截取原动作 21.00 秒至结束，左为修复前、右为修复后，画面时间从截取
片段的 0 秒起算。预览截取文件只用于观看；正式修复输出保留全片 1,158 帧。

原始 PKL SHA-256：`4480e18eee01d2d12759a7651afde13d3f9ede09df1705b1811648240c55ceee`。
修复后 PKL SHA-256：`01ec1ed16cd8f07299a9931eda6f8b40aa9e74d3079f0c502c9e35297db43ce7`。
