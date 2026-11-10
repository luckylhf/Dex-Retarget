# Dex-Retarget

把人体动作转换为 **Dex 29 关节、50 Hz 的机器人运动学参考**，用于动作预览和后续训练数据准备。支持 LAFAN1、BONES-SEED SOMA Uniform、已适配骨架的 BVH，以及 SMPL-X 人体参数（`.pt` / `.npz` / `.pkl`）。

如果想使用视频中人体动作，建议参考Dex-Video2Robot(https://github.com/luckylhf/Dex-Video2Robot)的Dex-GMR部分，使用生成的.pt文件当做输入进行转换。

提供 `fast`（默认）与 `refine` 两种模式。两者都会生成完整可播放动作和检查报告；只有质量通过才生成训练导出。结果仍需下游动力学跟踪与真机验证。

## 1. 安装

使用 Python **3.11 或 3.12**，在项目根目录执行：

```sh
git clone https://github.com/StanfordASL/frax.git Dex-StanfordASL-frax
git -C Dex-StanfordASL-frax checkout f5e87ab59a0b82ba43095e882c1b88c9b0dacf07
python3.12 -m venv .venv
.venv/bin/python -m pip install -e './Dex-StanfordASL-frax[ik]'
.venv/bin/python -m pip install -e '.[preview]'
.venv/bin/dex-frax --help
```

已有匹配的 frax 目录时跳过下载。目录搬迁后应重新建立虚拟环境；不要分发 `.venv`。安装包名和命令保留 `dex-frax`，Python 包名为 `dex_frax`。

以下步骤面向完整 Git 源码目录。单独安装 wheel 或 Python 源码分发包时，机器人资产和示例需另备；资产路径通过 `DEX_FRAX_ASSETS` 指定，详见 [机器人资产](docs/INPUTS.md#机器人资产)。

仓库包含 Dex 机器人资产和少量示例。**LAFAN/SOMA 另需参考几何表，SMPL-X 另需人体模型及可选依赖**，准备方法见 [输入规范](docs/INPUTS.md)。依赖版本快照见 [requirements-lock.txt](requirements-lock.txt)。

## 2. 转换一条动作

先用附带的 BVH 示例运行，输出必须是新目录：

```sh
.venv/bin/dex-frax convert \
  examples/inputs/bvh/nobody/nobody_without_calibration.bvh \
  outputs/nobody --source mocap
```

| 模式 | 完整流程 |
|---|---|
| `--mode fast`（默认） | 读取、源适配和整片标定 → Frax IK → PKL／NPZ 导出 → 质量报告。没有局部修复、V8 精修或自动升级。 |
| `--mode refine` | Frax 初解 → 固定骨盆参考 → 拟合腰部 → 固定躯干先解脚腿 → 固定脚腿再解手臂 → 整片检查 → 问题窗口队列 → 局部修复、复查与回退 → 最终整片验收与导出。 |

`fast` 是 **本项目 Dex 适配下的纯 Frax 路线**，保留输入转换、目标构造及完整导出；不加旧版膝部偏好、接触缓变和姿态后处理。上游示例面向 G1，机器人、目标及约束不同，不能宣称输出数值等同于官方 G1 示例。这里的“快”也不代表省略检查。

`refine` 复用 V8 的局部修复能力；已移除公开的 `v8`、`frax`、`auto`、`hybrid` 和 `--pipeline two-stage` 入口。历史代码/产物保留作对照，不再作为新的调用方式。

### 验收范围

`accepted` 表示通过当前配置的输出检查，不等于所有部位与人体完全一致，也不是动力学或真机通过。`quality.json` 的 `acceptance_scope`、`optional_similarity_checks` 和 `metric_semantics` 明确列出范围：

- 关节速度由导出相邻帧重新计算；使用与求解器相同的机器人限速是有意的，后处理或导出错误仍会被拦截。
- 根朝向检查验证指定人体目标朝向是否被保留，不是独立拟合成绩；肘屈伸目标误差属于同目标诊断，不是单独最终门槛。
- 腿部方向、前臂轴向和根位置默认记录但不设任意硬限。可在 `quality_pose.json` 配置 `leg_direction_p95_deg`、`forearm_width_p95_deg`、`root_xy_max_m`、`root_z_max_m`；默认 `null` 明确表示未设门槛。根高度包含允许的脚腿补偿，单独报告，不与水平漂移混算。
- 空的 `support_constraint_failures` 不代表专用后处理已经运行；`support_solver_execution` 明确记录是否执行，最终足部／身体支撑仍由导出审计检查。
- 加速度、jerk 没有伪造硬件上限；按源动作对照检查快速反复换向和孤立速度脉冲，问题进入抖动修复队列。

比较历史通过率时需匹配 `audit_contract.fingerprint`、机器人和输入适配身份。碰撞检查范围扩大，即使动作数值不变，通过率也可能改变。旧数据缺少这些标识时应复审，不能直接混算。


```sh
.venv/bin/dex-frax convert input.bvh outputs/fast --source mocap --mode fast
.venv/bin/dex-frax convert input.bvh outputs/refined --source mocap --mode refine
```

支撑候选若与碰撞约束冲突，会转入同一套脚腿联合约束修复，不通过则回退。

精修保留 Frax 的骨盆水平轨迹和源骨盆朝向；先用脚部几何、地面与支撑约束联合求解腿部和骨盆高度，脚腿调整仅可向上传递 Z 向补偿，再在机械限位与限速内拟合腰部并预查腰部自身及帧间碰撞，再固定骨盆、腰部和手臂，先求解脚腿；然后固定骨盆、腰部及脚腿，求解手臂。手臂必须适应已确定的脚部。问题窗口严格按 **支撑/抖动 → 腰部 → 腿部碰撞 → 手臂碰撞 → 手臂相似度** 推进；前序问题未解决（包括候选用尽）时，阻塞重叠的后序窗口，其他区域仍可继续。

碰撞检查和腿部修复覆盖双脚之间、脚对另一条腿，以及同侧脚跟/脚尖/脚板对小腿。保留原碰撞对，额外使用整脚体积和按踝部装配几何划分的外部区域，避免只检查薄脚底或把正常踝关节连接误报为穿透。腿部候选仅允许骨盆高度随脚腿补偿，骨盆水平位置、朝向、腰部及手臂关节保持不变；高度变化连同全身一起平移，并检查支撑、速度、接缝和新增抖动。在支撑、限位、限速及全身碰撞保护内，以源大腿/小腿方向为软目标。腿部仅使用固定参考的 ±5°、±10° 两档，六轴共同参与：左右腿/双脚互撞优先摆动腿 J2/J3，J1/J4配合；同侧脚—小腿碰撞优先 J5/J6，髋膝补偿。主力轴用软偏好表达，不锁死配合轴。支撑的相位、接触点与目标足姿沿用源预处理，踮脚/脚跟目标不改成平脚；大腿、小腿形状优先于脚朝向和关节回正。候选会测量源腿形变化；没有腿部碰撞收益却损伤腿形的候选被拒绝，必要安全避让的腿形代价写入报告。这不是腿部绝对误差保证。

refine 的初始脚腿 Frax 求解已加入同一套声明脚部几何约束，避免先生成超出两档修复范围的脚姿。这里尚未建立修复参考，新初解允许相对旧版改变超过10°；初解结束后参考固定，后处理不能重置或累计放宽。初解仍需整片验收。

难窗的腿部弹性恢复将支撑行保持为硬约束，并允许更多内部迭代，但仍受单次时间片保护，并保存有效数值进度继续；不增加整片强制计算，也不放宽最终检查。具体覆盖与关节连接排除见 [流程说明](docs/PIPELINE.md#碰撞覆盖)。

精修不再整条共用“6个窗口、120秒”上限。只搜索仍有问题的窗口，单个候选有时间片；通过、候选族用尽或用户指定预算结束后停止。可用 `--repair-budget 240` 限制精修时间，剩余窗口、阻塞原因及数值进度记录为待处理；继续时从有效进度重建求解器，保持固定避碰参考。`--continue-from` 也接受同版本完整 `fast` 结果，严格校验后复用原始 Frax 初解，并补上骨盆、腰部、脚腿、手臂的有序求解。候选用尽可扩大上下文，但不重做整片标定。`--budget` 是包含准备/初解/验收的外层总时限（默认86400秒）；硬超时会中断在途候选。

精修可从已保存检查点继续，输入、代码、资产、依赖及转换规则必须相同；允许更改精修时间预算。不会重跑初始 IK，也不会把上一次候选当成新的三级角度参考：

```sh
.venv/bin/dex-frax convert input.bvh outputs/refined-next --source mocap \
  --mode refine --continue-from outputs/refined --repair-budget 240
```

代码升级后的检查点不能直接 `--continue-from`。若仅本项目修复实现改变，可显式使用 `--recheck-from outputs/refined`：源文件、资产、上游 Frax、依赖与转换规则仍须一致，检查原产物哈希；保留动作和固定三级参考，重新验收并重建候选队列，记录迁移来源。它不允许换源、改时间轴或清空质量门槛。

局部修复按完整碰撞片段组织，优先尝试连续肩臂避让；10°档清除本窗口肩臂碰撞后，不因其他腿部碰撞继续放宽。每次采用小幅改进后仍会轮到其他候选方法；一整轮有进展才继续下一轮，避免同一个方法反复占用预算。最终是否合格始终由整片输出复查决定。

如果源骨盆转动或平移快到无法通过现有限速，可显式加 `--allow-local-slowdown`（仅 `refine`）。它保持整片标定与源姿态，只延长必要区间；输出仍为50Hz，`motion.time_map.npz`、预览NPZ及训练NPZ保留 `source_time_s` / `source_frame_float`，用于对照原输入。默认不改时间轴；续跑必须保持此选项一致。

替换输入时，LAFAN1 用 `--source lafan`，SOMA Uniform 用 `--source soma`。`mocap` 只适用于已验证的骨架，不能直接接收任意 BVH。

SMPL-X 示例需先按 [输入规范](docs/INPUTS.md) 配好 `SMPLX_MODEL_DIR` 和 `SMPLX_PYTHON`：

```sh
.venv/bin/dex-frax check-input examples/inputs/smplx/gbtc/gbtc.pt \
  --smplx-python "$SMPLX_PYTHON" --input-fps 30
.venv/bin/dex-frax convert examples/inputs/smplx/gbtc/gbtc.pt outputs/gbtc \
  --source smplx --body-models "$SMPLX_MODEL_DIR" \
  --smplx-python "$SMPLX_PYTHON" --input-fps 30 --grounding contact
```

文件后缀不代表内容：示例中的 `gbtc.pt` 是人体输入，`gbtc.npz` / `gbtc.pkl` 是已有机器人动作。

## 3. 查看结果

两种模式都读取输出根目录的 `selection.json` 和 `accepted.json`；具体动作、检查报告位于 `frax/`。目录名表示求解主干，实际模式记录在 `profile` 字段。

| 结果 | 含义与处理 |
|---|---|
| `accepted` | 通过当前运动学质量检查，生成 `training.npz` |
| `review` | 有完整动作但未通过检查；查看 `quality.json` 的 `flags`，复核预览 |
| `failed` / `input_error` / 超时或中断 | 转换未完成或输入需修正；查看 `run.json`、阶段日志和输入说明 |

每个已完成阶段包含 `motion.pkl`、`motion.npz`、`quality.json`、`identity.json` 和 `run.json`。预览动作是米、弧度、根四元数 **XYZW**；`training.npz` 的根四元数为 **WXYZ**，采用 `root_joint_lossless_v1`，下游读取器须匹配此格式和模型身份。

预览实际生成的阶段，例如：

```sh
.venv/bin/dex-frax preview outputs/nobody/frax/motion.pkl \
  --output outputs/nobody/preview.mp4
```

生成 MP4、GIF 和关键帧；可传入多个等长 PKL 并排对照。播放成功或 `run.json` 显示 `complete`，都不代表质量通过。

## 4. 批量转换

按 [批处理说明](docs/BATCH.md) 创建带源文件 SHA-256 的 BVH 清单，然后运行：

```sh
.venv/bin/dex-frax batch manifest.json outputs/batch --budget 604800
```

批处理默认使用全部可用 CPU，受内存和磁盘准入约束；内存目标为系统与程序合计约 90%。默认 `--mode fast` 按短动作优先转换；加 `--mode refine` 才为有问题的窗口执行精修。不再排第二遍完整 V8。输入错误停止该动作；未通过的完整动作保留待审，不导出训练数据。默认总预算只有 1800 秒，长批次需显式调整。SMPL-X 使用单文件入口。

默认磁盘保留量为 **80 GiB**。小批次试跑且空余空间不足时，可加 `--disk-reserve-gib 5`；按实际输出规模设置余量。低于预留量会停止批次，释放空间后手动 `--resume`。

**下游只读取 `accepted.json` 的 `motions` 列表**，其中路径相对于批次输出目录。停止、续跑、资源限制与实时状态见 [批处理说明](docs/BATCH.md)。

## 5. 可解释的 pipeline

**共用：源适配与整片标定 → Frax 求解 → 完整导出与独立检查。精修追加：问题队列 → 局部候选 → 受影响窗口及接缝复查 → 保留改进或回退 → 最终整片验收。**

默认转换在每个完成的求解阶段目录自动保存 `pipeline.json` 和 `pipeline.md`，解释实际阶段、窗口、三级尝试、拒绝原因、剩余标记与采用的门槛。它们是说明文件，训练准入仍以原来的 `quality.json` / `accepted.json` 为准。

解释已有结果不会重新求解或改写原文件：

```sh
.venv/bin/dex-frax explain outputs/nobody
.venv/bin/dex-frax explain outputs/nobody --output outputs/nobody-explanation
.venv/bin/dex-frax explain outputs/nobody/frax --format json
```

`--output` 必须是新目录；批次请指定具体动作子目录。研究检查点和历史报告缺少的证据会明确标注，不根据当前代码补写成“已执行/已通过”。

[完整流程、硬约束与软目标、错误分流、机器人适配边界](docs/PIPELINE.md)。组织方式可扩展，但当前执行器仍面向 **Dex 29关节**；任意BVH、其他机器人或复杂地形不能只换路径就当作已支持。

## 进一步说明

- [可解释 pipeline 与扩展规范](docs/PIPELINE.md)
- [输入规范与示例](docs/INPUTS.md)
- [批处理、资源与续跑](docs/BATCH.md)
- [第三方来源和许可](THIRD_PARTY.md)

本项目使用 Stanford ASL 的 frax IK 和源自 Dex-SuperMimic 的 V8 转换组件。代码、机器人资产、人体参考表及示例数据各自遵循其许可。
