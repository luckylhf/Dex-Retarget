# Dex-Retarget

将 LAFAN1、BONES-SEED SOMA Uniform、已验证骨架的 BVH，以及 SMPL-X 的 `.pt`、`.npz`、`.pkl` 转为 Dex 29 关节、50 Hz 运动学参考。可直接运行的示例见 `examples/inputs/`。

**0.8.1：批处理默认按主机实时负载动态补位，内存目标为程序与系统合计约 90%，CPU 上限取可用核心数，不固定 56 GiB、28/32 槽或预留 2 核。** 单文件仍为 `frax → hybrid → v8`，批处理保留 60 秒分流和严格短动作优先。继承0.8.0的修复：长碰撞窗口覆盖、验收身份、剩余阶段内存预测、手臂局部回退、求解工作预算和参数一致性六项问题。安全与源动作姿态门槛不降低，三级放宽只扩大搜索范围。验证边界见 [0.8.0 本地记录](docs/PIPELINE_080_VALIDATION.md)；历史记录见 [0.7.0](docs/PIPELINE_070_VALIDATION.md) 与 [0.6.1](docs/RESOURCE_061_VALIDATION.md)。

## Frax 来源与本项目的关系

快速 IK 基于斯坦福 ASL 的开源项目 **[StanfordASL/frax](https://github.com/StanfordASL/frax)**，使用其 JAX 运动学和 `frax.ik` 差分逆运动学能力。[上游文档](https://stanfordasl.github.io/frax/ik/)介绍了原始 IK 接口与示例。

本地依赖版本为 **frax 0.2.0**，源码保存在本项目内的 `Dex-StanfordASL-frax/` 目录，checkout 基准提交为 `f5e87ab59a0b82ba43095e882c1b88c9b0dacf07`。该目录已加入 `.gitignore`，作为本地依赖使用，不随本项目提交到 GitHub。上游采用 **MIT License**，版权声明为 © 2026 Daniel Morton，完整许可保留在 [FRAX_LICENSE](docs/FRAX_LICENSE)。转换身份记录会保存实际导入的 frax Python 源文件哈希，以核验具体使用的实现。

**Dex-Retarget 是面向 Dex 机器人的独立转换流水线。** 本项目在 frax 基础上实现 Dex 29 关节适配、任务与约束、局部修复、分级批处理和资源保护；人体标定、目标、接触处理、手臂规划、Mink 避碰和审核来自本地 `Dex-SuperMimic` V8 转换组件（`cpu_mink_margin_aware_arm_selection_20261105`，含 1 mm 支撑验收）。上游 frax、V8 组件及本项目改造的来源分别记录，其他依赖与资产许可见 [THIRD_PARTY.md](THIRD_PARTY.md)。

项目目录由 `Dex-Frax` 更名为 `Dex-Retarget`；当前 Python 包名 `dex_frax`、安装包名与命令 `dex-frax`、转换身份字段保持兼容。历史版本报告保留当时的项目名称。

## 安装与运行

已建立本目录 `.venv`，可直接运行：

```sh
cd /path/to/Dex-Retarget
.venv/bin/dex-frax convert /path/to/input.bvh outputs/example --source mocap
```

仓库自带可直接跑通的示例输入（说明见[示例输入](#示例输入)）：

```sh
.venv/bin/dex-frax convert examples/inputs/bvh/nobody/nobody_without_calibration.bvh \
  outputs/example_nobody --source mocap
```

输出目录必须是新目录；已有结果拒绝覆盖。`--source mocap` 只用于逐关节验证过的 Y-up、带手指和脚趾标记骨架，不表示支持任意 BVH；`nobody` 只是该来源的一个示例，不是特殊路径。LAFAN1 用 `--source lafan`；BONES-SEED SOMA Uniform 用 `--source soma`；SMPL-X 参数用 `--source smplx`，`.npz`/`.pt`/`.pkl` 在 `--source auto` 下也会按 SMPL-X 处理。

新环境使用 Python 3.11 或 3.12。此目录的资产和本地参考数据需要一并保留：

如果尚无本地上游目录，先在 `Dex-Retarget` 根目录获取指定版本：

```sh
git clone https://github.com/StanfordASL/frax.git Dex-StanfordASL-frax
git -C Dex-StanfordASL-frax checkout f5e87ab59a0b82ba43095e882c1b88c9b0dacf07
```

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e './Dex-StanfordASL-frax[ik]'
.venv/bin/python -m pip install -e '.[test,preview]'
```

本次环境的精确版本见 `requirements-lock.txt`。JAX 的 NumPy 2 环境与旧训练环境独立，未修改上游依赖。CPU 双精度、BLAS 单线程；首次 JIT 编译单独计时。

## 示例输入

`examples/inputs/` 只保留少量可直接运行的示例，不代表完整数据集。`bvh/nobody/` 的来源、帧数和逐文件 SHA-256 见该目录的 `说明.txt` 与 `manifest.json`。

### BVH（`examples/inputs/bvh/nobody/`）

`nobody` 只是此前验证用的一个人体骨架示例，本项目不对该骨架做特殊处理。

| 文件 | 说明 |
|---|---|
| `1013_nobody_chr01.bvh` | 原始人体 BVH，2086 帧，`Frame Time 0.011111`，含开头校准帧 |
| `nobody_without_calibration.bvh` | 去掉开头校准帧的实际转换输入，2085 帧；与历史文档中的 `00_nobody_原始人体.bvh` 逐字节相同（SHA-256 `668caeef…`） |
| `nobody.pkl`、`nobody_robot.npz`、`reference.npz` | 既有 WBC 对比使用的机器人动作和参考特征；不是人体参数，也不是本次转换结果 |
| `preprocessing.json`、`input_manifest.json`、`nobody.ik.json` | 校准帧移除记录（去掉第 0 帧 T-pose）、原始路径与哈希、当年 IK 参数 |

```sh
# 单文件自动流程：frax → hybrid → v8，通过即停止
.venv/bin/dex-frax convert examples/inputs/bvh/nobody/nobody_without_calibration.bvh \
  outputs/example_nobody --source mocap
```

### SMPL-X（`examples/inputs/smplx/gbtc/`）

`gbtc` 是同一段约 14 秒动作的一组示例：`.pt` 是人体缓存，`.npz`/`.pkl` 是转换后的机器人动作，用来说明容器后缀不等于内容：

| 文件 | 内容 | 可作为 SMPL-X 输入 |
|---|---|---|
| `gbtc.pt` | GVHMR `hmr4d_results.pt` 原始缓存，420 帧、30 fps、Y-up、米，含 `smpl_params_global`、`smpl_params_incam` 和 `net_outputs` | 是 |
| `gbtc.npz` | Dex 29 关节机器人动作：`root_pos`/`root_rot`/`dof_pos`，699 帧、50 Hz，`robot=tienkung_dex_29` | 否，属于转换结果 |
| `gbtc.pkl` | 与 `gbtc.npz` 数组相同的字典版本，另带 `retarget_backend`、`quality_status=review`、`quality_flags=[sustained_arm_pose_mismatch]` 等元数据 | 否 |

```sh
# 只读校验；GVHMR 缓存没有帧率元数据，必须显式给 --input-fps 30
.venv/bin/dex-frax check-input examples/inputs/smplx/gbtc/gbtc.pt \
  --smplx-python "$SMPLX_PYTHON" --input-fps 30

# 视频人体参数：有静止置信度时启用接地高度修正
.venv/bin/dex-frax convert examples/inputs/smplx/gbtc/gbtc.pt outputs/example_gbtc \
  --source smplx --body-models "$SMPLX_MODEL_DIR" \
  --smplx-python "$SMPLX_PYTHON" --input-fps 30 --grounding contact
```

`gbtc.npz`、`gbtc.pkl` 是机器人动作，读取器会直接报 `This is robot motion, not SMPL-X human parameters`；`.npz`/`.pkl` 后缀不代表人体参数。

## SMPL-X 输入（视频人体参数也可独立使用）

`convert` 现在支持 `--source smplx`（`--source auto` 遇到 `.npz`/`.pt`/`.pkl` 也按 SMPL-X 处理）。`.pt`、`.npz`、`.pkl` 是容器后缀，**不代表其中一定是 SMPL-X**。原有 LAFAN/SOMA/骨架 BVH 路径保持原求解方式；本节扩展单文件 `convert` 和 `check-input`，现有批处理 `batch` 清单仍只接收 BVH。

### 安装与模型

人体前向几何需要 `torch`、`smplx==0.1.28`、NumPy 和 SciPy。可以选择：

```bash
# 在已安装本项目的环境补充可选人体依赖
.venv/bin/python -m pip install -e '.[smplx]'

# 或复用已有 SMPL-X 环境，避免改变求解器环境
export SMPLX_PYTHON=/path/to/smplx/env/bin/python
export SMPLX_MODEL_DIR=/path/to/body_models
```

`SMPLX_MODEL_DIR` 必须指向包含 `smplx/` 的父目录，例如：

```text
body_models/
└── smplx/
    ├── SMPLX_NEUTRAL.npz
    ├── SMPLX_MALE.npz
    └── SMPLX_FEMALE.npz
```

只需要本次输入性别对应的文件。模型从 [SMPL-X 官方网站](https://smpl-x.is.tue.mpg.de/) 按许可获取；代码仓库不分发这些参数。可以复用 Dex-GMR 的 `assets/body_models`，无需复制。模型与输入参数、预处理实现和运行依赖版本都会进入转换身份记录。

### 支持的内容和格式

输入是**单个人体的一段连续动作**。局部旋转必须是轴角、单位弧度；平移必须是米。数组须有限、帧数一致，至少 3 帧；输出统一为 50 Hz。旋转在人体前向几何之前做局部 SLERP，平移与形状参数线性插值，保留原时间尺度。

| 字段 | 形状与要求 |
|---|---|
| `global_orient` 或 `root_orient` | `(T,3)` 根朝向 |
| `body_pose` 或 `pose_body` | `(T,63)`，21 个 SMPL-X 身体关节 |
| `transl`、`trans` 或 `translations` | `(T,3)` 世界平移，必需 |
| `betas` 或 `shape` | `(B,)`、`(1,B)`、`(T,B)`，`1≤B≤16`；缺少时采用零体型并告警。时变 betas 保留并告警，足底分区用中位体型标定 |
| `fps`、`mocap_framerate`、`mocap_frame_rate` 或 `mocap_fps` | 正的有限标量；缺少时必须给 `--input-fps` |
| `up_axis` | 标量字符串 `y` 或 `z`；缺少时给 `--up-axis`，不通过某一帧站姿猜坐标系 |
| `units` | 标量字符串 `m`；缺少时确认后给 `--units m` |
| `gender` | `neutral`、`male` 或 `female`，默认 `neutral` |
| `model_type` / `surface_model_type` | 若提供，只允许 `smplx`；调用者必须确认源模型确实为 SMPL-X |
| `left_hand_pose` / `right_hand_pose` | 可选，各 `(T,45)`，绝对局部轴角；不接受 PCA 手部系数 |
| `pose_hand` | 上述两只手的另一种写法：`(T,90)`，左手在前 |
| `jaw_pose` / `pose_jaw` | 可选 `(T,3)` |
| `leye_pose` / `reye_pose`，或 `pose_eye` | 可选各 `(T,3)`，或合并 `(T,6)`，左眼在前 |
| `expression` | 可选 `(T,10)`，默认零 |
| `static_confidence` | 可选 `(T,4)`，范围 `[0,1]`，顺序为左踝、左脚、右踝、右脚的静止概率 |

也接受 `poses`（别名 `pose`、`smpl_poses`、`thetas`）：

- `(T,66)`：根 3 + 身体 63。
- `(T,165)`：根 3 + 身体 63 + 下颌 3 + 左眼 3 + 右眼 3 + 左手 45 + 右手 45。
- 不接受 72 维 SMPL 或 156 维 SMPL-H 直接冒充 SMPL-X。分字段与合并字段同时存在时必须一致；别名冲突会报错。

缺失手部姿态时，使用 SMPL-X 模型的放松手均值，并报告 `unobserved_hands_use_model_relaxed_mean`；这不代表视频识别出了手指。显式手部轴角按绝对局部旋转解释，不重复叠加模型手部均值。

容器读取约定：

- **NPZ**：普通数值/字符串数组，`allow_pickle=False`；拒绝 object 数组。
- **PKL**：普通字典及数值 NumPy 数组；使用受限读取器，仅允许 NumPy 数组重建。含 Torch 对象、自定义类或外部引用的 PKL 不支持，请由源工具导出普通 NPZ。
- **PT**：使用 `torch.load(..., weights_only=True)`，失败不退回不安全加载。支持普通张量字典及 GVHMR 的 `smpl_params_global` 嵌套字典；不接受只有 `smpl_params_incam` 的相机坐标动作。
- **GVHMR 特例**：已知世界约定为 Y-up、米；仍须提供真实帧率。Dex-Video2Robot 先规范为 30 fps，所以桥接显式传 `--input-fps 30`。网络的 `static_conf_logits` 若存在，读取前四个脚部通道；它表达关节静止概率，不是测力数据。

AITViewer 可用于查看这些人体动作，但其显示容错不等于重定向输入合格。例如查看器可以对缺失字段取默认值或显示部分 SMPL 关节，本转换器会拒绝会改变骨架意义的歧义。机器人 `root_pos/root_rot/dof_pos` PKL 和 `joint_pos/body_pos_w` NPZ **不是人体 SMPL-X 参数**，不能作为本入口输入。

### 先校验，再转换

```bash
# 只读校验字段/帧率/坐标系，不执行人体模型或机器人 IK
.venv/bin/dex-frax check-input /path/to/human.npz \
  --smplx-python "$SMPLX_PYTHON"

# 文件缺少元数据时，明确补充；错误或字段冲突会给出说明并返回非零退出码
.venv/bin/dex-frax check-input /path/to/human.pkl \
  --smplx-python "$SMPLX_PYTHON" --input-fps 30 --up-axis y --units m

# 已可靠接地的 SMPL-X 动捕：不开视频高度漂移修正
.venv/bin/dex-frax convert /path/to/human.npz outputs/human \
  --source smplx --body-models "$SMPLX_MODEL_DIR" \
  --smplx-python "$SMPLX_PYTHON" --grounding off

# 视频人体参数：有静止置信度时，显式启用高度修正
.venv/bin/dex-frax convert /path/to/hmr4d_results.pt outputs/video_motion \
  --source smplx --body-models "$SMPLX_MODEL_DIR" \
  --smplx-python "$SMPLX_PYTHON" --input-fps 30 --grounding contact
```

`examples/inputs/smplx/gbtc/gbtc.pt` 是附带的一组 30 fps GVHMR 缓存，可直接替换上面示例中的 `.pt` 路径，运行方式见[示例输入](#示例输入)。

`check-input` 只检查内容结构；实际模型文件、人体几何和支撑证据充分性在 `convert` 时检查。`--max-frames` 仅用于短输入接口测试，不能作为全片质量验收。SMPL-X 单文件默认同样按 `frax → hybrid → v8`，完整通过即停止；`--mode hybrid` 等固定模式仍可用。`--budget` 限制自动流程（包含人体准备），固定模式不使用该总预算。

人体准备在 `--smplx-python` 的独立子进程中执行，不要求旧 GMR 环境安装 JAX。准备后的 `input/human.npz` 是内部几何缓存，不是可重新当作 SMPL-X 参数提交的公共输入格式。

### 可选接地修正的含义

`--grounding contact` 利用静止概率大于 0.8、且接近较低脚底的观测，拟合一条平滑的公共垂直漂移曲线；对所有人体点减去同一高度偏移，不改局部旋转和肢体长度。没有支撑观测的区间只延续/插值漂移，**不会每帧把最低脚压到地面**。缺少置信度或有效支撑证据不足时直接提示失败，不静默切换算法。

这种修正假设平地，静止也不等于真实承重。长时间腾空、静态抬腿、台阶和平台可能使判断出错；应查看起跳/落地及长无观测区间。它没有自动证明所有跳跃均被正确保留。可靠动捕通常用 `off`。

### 输出与质量说明

每个求解阶段保留 `motion.pkl`、`motion.npz`、`quality.json`、`identity.json` 和 `run.json`。机器人 PKL/NPZ 显式带 `model_sha256`、关节名、帧率和 `xyzw` 根四元数约定，便于 Dex-GMR/Dex-Mimic 验证。质量报告新增共同标识 `schema=dex_motion_quality_v1`，包含来源、模型和输出哈希；SMPL-X 输入报告会附在 `input_report`，包含默认值、接地方法和限制。

- `accepted`：通过当前运动学质量门槛，才生成原有 `training.npz`。
- `review`：完整结果可查看，报告保留具体失败原因；不生成已通过的训练导出。
- 自动模式全部未通过时，`selection.json.preview_output` 指向最后一个完整结果，**不是历史最佳、也不是已通过结果**；`selected_output` 仍为空，`accepted.json` 不收录它。
- 输出存在、进程退出成功、FK 可播放，都不代表 WBC 动力学跟踪或真机验证完成。

输入适配沿用了已做短片试验的 SMPL-X 解剖构造；仍需结合报告复核上肢方向、缺失手部信息、支撑和碰撞。加入正式输入接口不等于这些动作质量问题已经全部解决。

## 调用方式与三种求解模式

**`convert` 和 `batch` 默认均为 `auto`，两者的分流规则不同。显式指定 `--mode frax/hybrid/v8` 则只运行该模式。**

| 调用 | 行为 | 是否自动升级 |
|---|---|---|
| `convert input.bvh output` | 单条动作，不论长度都按 frax → hybrid → V8，通过即停止 | 是 |
| `convert input.bvh output --mode frax` | 单条动作，仅运行快速版 | 否 |
| `convert input.bvh output --mode hybrid` | 单条动作，frax 初解后执行完整 V8 后处理 | 否 |
| `convert input.bvh output --mode v8` | 单条动作，从原始 BVH 执行 V8 初解及完整后处理 | 否 |
| `batch manifest.json output --mode frax` | 整批仅运行快速版，复用模型与编译 | 否 |
| `batch manifest.json output --mode hybrid` / `v8` | 整批强制使用指定模式 | 否 |
| **`batch manifest.json output`** | **<60秒先 frax → hybrid；失败项与≥60秒动作进入低优先级V8队列** | **是** |

单条转换示例（输出必须是不同的新目录）：

```sh
.venv/bin/dex-frax convert /path/to/walk.bvh outputs/walk_auto --source lafan
.venv/bin/dex-frax convert /path/to/walk.bvh outputs/walk_fast --source lafan --mode frax
.venv/bin/dex-frax convert /path/to/walk.bvh outputs/walk_hybrid --source lafan --mode hybrid
.venv/bin/dex-frax convert /path/to/walk.bvh outputs/walk_v8 --source lafan --mode v8
```

独立调用 `convert --mode hybrid` 会重新计算 frax 初解；单文件及批处理的自动模式会核验并复用同一动作在 frax 阶段保存的初解。单文件自动模式的阶段目录是 `<output>/frax/`、`hybrid/`、`v8/`，并生成根目录 `selection.json`、`accepted.json` 和汇总 `run.json`；仅运行到需要的阶段。固定模式直接输出到指定目录。

| 模式 | 过程 | 用途 |
|---|---|---|
| `frax` | 接触过渡 → 带膝偏好的 frax IK → 局部拟合 → 按需 V8 精修 → Mink 避碰 → 最终审核 | 快速候选主流程 |
| `hybrid` | 相同接触过渡及带膝偏好的 frax IK → 完整 V8 足部支撑 → 手臂规划 → 身体接触 → Mink 避碰 | 主动执行完整后处理 |
| `v8` | V8 TRF → 完整 V8 后处理 → 相同审核 | 同机、同输入基线 |

`--iterations` 和 `--initial-iterations` 只影响 frax 初解，在 `v8` 模式不调整 V8 的迭代预算。`--iterations 8` 控制 frax 每帧内部迭代，`--initial-iterations 100` 控制首帧三个分支的初始化。内部半步迭代始终受上一输出帧的速度区间约束，不通过增加迭代绕过限速。整片升级由 `auto` 管理；固定 `frax` 只做有预算的局部修复。单文件自动模式总预算默认86400秒，可用 `convert --budget` 调整。

`convert --mode frax --legacy-frax` 保留旧版原始 frax 消融流程，关闭后续改良。新版的局部窗口使用 CPU 稀疏 SQP，同时拟合姿态、速度和二阶差分。新的路由在足部残差超过 0.5 mm 时尝试 V8 支撑精修；上臂/前臂方向、手掌轴、肘弯曲轴持续异常时尝试 V8 分支搜索；额外身体接触或身体穿地触发 V8 全身修复。0.4.0 默认禁止升级至整片处理：姿态精修最多 4 个窗口、每窗口 6 次迭代；V8 局部精修最多 3 个窗口，累计尝试帧数不超过 max(100, round(片长 × 25%))。失败尝试也消耗预算。需要整片精修时显式使用 `--mode hybrid`。0.4.1 优先将同一预算用于实际穿地残差，必要时缩减窗口留白；身体清障使用最小改动目标并守住逐帧膝分支。原有避碰判定为 QP 不可行时，先按旧预算追加最多 2 个窗口、累计 100 帧的分步恢复；最多 24 次迭代/窗口。0.6.0 再对未解决窗口执行下文的三级修复，支撑臂只在保持接触的耦合档允许调整。最终原生帧、插值、地面和速度检查保持不变，仍未修好的结果保留 `review`。

阈值在 `src/dex_frax/adaptive_policy.json`。局部窗口两端与外部连接、关节限位、速度、地面、接触和膝分支都会复查。已有非负膝轨迹在修复中暂时限制到同一弯曲分支，退出后恢复原机器人限位；不是修改机器人模型。`hybrid` 自 0.3.0 起也启用膝偏好和接触过渡，但整片后处理仍可能改变初始分支，需看最终膝指标。

`convert --max-frames 50` 仅用于冒烟检查；它在接触准备前截短动作，不能当作完整动作的等价窗口。

## 自动分级批处理（批量生产入口）

```sh
# 自动分级：短动作先快速版，长动作与失败项进入V8队列
.venv/bin/dex-frax batch regression_local/suite_20261105/manifest.json outputs/regression_auto --mode auto

# 默认动态资源模式；使用全部可用CPU作为上限，按实时负载连续补位
.venv/bin/dex-frax batch manifest.json outputs/full_auto \
  --mode auto --workers auto --budget 604800

# 中断后恢复同一清单：校验完整结果，仅重试未完成的阶段
.venv/bin/dex-frax batch manifest.json outputs/full_auto \
  --mode auto --workers auto --budget 604800 --resume

# 自定义转换总预算和每条动作的各级超时，单位均为秒
.venv/bin/dex-frax batch regression_local/suite_20261105/manifest.json outputs/regression_auto_long \
  --mode auto --budget 7200 --frax-timeout 120 --hybrid-timeout 300 --v8-timeout 300

# 对照实验：整批固定一种模式，不自动升级
.venv/bin/dex-frax batch regression_local/suite_20261105/manifest.json outputs/regression_frax --mode frax
.venv/bin/dex-frax batch regression_local/suite_20261105/manifest.json outputs/regression_v8 --mode v8
```

自动模式仅读取BVH头部的 `Frames × Frame Time` 确定名义时长，不做探测转换。**严格小于60秒**的动作进入frax、hybrid；其余直接进入V8。各队列优先级为frax、hybrid、V8；前两者短动作优先，V8长动作优先以减少尾部等待。**最高优先级的待派发动作预算不足时，保留空槽和预算，不能绕过它启动更便宜的 V8。** 在途任务继续执行，完成释放的资源优先供给队首；没有更高优先级的排队任务时才可派发 V8，无需中断在途短动作。通过即停止；槽位空闲且内存允许就补下一条，不插入固定8条小样或全批阶段等待。即使批清单只有一条，也遵循批处理时长规则；需要单文件三级流程请用 `convert`。

- **统一验收：**frax、hybrid、v8，包括单次调用，都执行持续上肢姿态、接触、穿地、速度和碰撞审核；V8 不会因为少检查一项而“兜底通过”。0.5.0 对 V8 和 legacy 消融补齐了持续上肢门槛，历史结果的通过率需按同一口径重算。
- **复用初解：**hybrid 复用的是 frax 的 `motion.initial.npz`，不是已经局部修补后的最终轨迹。核验输入、实现、模型、依赖版本、参数、文件哈希和预处理目标。已承诺复用却核验失败时记录该阶段失败并升级V8，避免突然启动未预留的冷编译；frax首轮未完成时，hybrid才按冷求解计账。V8始终从原始BVH独立求解。
- **故障隔离：**输入哈希在开始前校验，损坏 BVH/不支持的骨架单独标记，停止该动作的无效升级；数值失败、工作进程异常退出和超时则可以进入下一阶段。
- **完整兜底：**短动作前两阶段未通过统一质量审核，均进入V8队列，包括仅剩碰撞、动作过短或来源根速度异常的情况；这些不可变的来源异常仍可能在V8最终审核中失败。旧 `--v8-for-collision-only` 参数仅保留命令兼容，现已无需指定。只有输入错误会提前停止无效升级。
- **超时：**按完整动作的50Hz输出帧数 N 估算，frax 为 `max(900, 300+0.5×N)` 秒，hybrid/V8 为 `max(1800, 600+1.5×N)` 秒；显式 `--*-timeout` 优先。包含排队等待编译、冷启动、计算与审核。超时终止本批次拥有的工作进程，保留诊断并撤销未完成的训练导出。
- **总预算：**默认 `--budget 1800`。0.5.1 固定模式和自动模式都用剩余预算限制在途任务，耗尽后终止本批次拥有的工作进程，标记未完成项；清理可能略超过该时间。全库运行须显式设置足够大的预算。
- **失败保留：**每个阶段独立保存，不覆盖前一阶段。三条路径都未通过表示“当前路径未通过”，不是不可修复，也不自动挑选一个失败结果作为训练参考。

使用 `--resume` 时，清单、输入哈希、全部 Python 实现（包括 CLI、调度、验收/导出）、配置、资产与运行依赖必须匹配。`conversion_contract.json` 统一记录 FPS、IK 迭代数、初帧迭代数、最大帧数、legacy 模式、IK 分块长度及求解/质量策略；初解复用、转换身份和恢复都核对同一契约。资源配置**数值**仍可热调整，但修改任何实现后须开新批次，不能沿用旧验收结论。所有已完成阶段文件逐项校验哈希；未完成目录和日志先归档再重试。

**0.8.0 必须使用新输出目录，不能续跑混入 0.7.x 或更早结果。** 缺少转换契约的旧输出也会被拒绝。历史数据只作为独立诊断、对照输入。

输出结构：

```text
outputs/regression_auto/
  batch.running.json         # 约每2秒更新的轻量实时状态、实际并发和资源
  batch.json                 # 每30秒及停止时保存的完整检查点、全部尝试与耗时
  accepted.json              # 仅包含最终通过的训练文件与 SHA256
  source_aliases.json        # 按完整BVH SHA256去重；重复输入指向保留的id
  walk_turn/
    selection.json           # 本动作最终采用的模式和路径，或未通过状态
    frax.log
    frax/                    # 第一轮完整输出
  get_up/
    selection.json
    frax.log
    frax/
    hybrid.log
    hybrid/                  # 仅在需要时出现
    v8.log
    v8/                      # 仅在前两轮未通过时出现
```

**下游入库读取 `accepted.json` 的 `motions`，其中输出路径相对于该批次输出目录。** 每条记录包含源文件哈希、采用模式、`training_npz`、训练文件哈希和转换身份。不要收集所有 `motion.pkl` 当作通过结果。

`batch.json` 顶层 `complete` 表示调度完成，进程正常退出也不代表全体通过。逐动作状态：`accepted` 已选定训练文件；`review` 至少有一次质量未通过且没有可采用结果；`failed` 未得到合格的完整转换；`input_error` 输入需修正；`budget_exhausted` 预算耗尽；`interrupted` 被中断；`resource_exhausted` 资源不足，不能视为动作质量不合格；`identity_changed` 执行期间实现或资产改变，结果不能采用。每一级另记录 `timeout` 或具体错误。中断后可显式使用 `--resume`，已有目录不能直接作为新任务覆盖。

固定模式仍将动作直接写入 `<输出目录>/<id>/`；0.5.1 同样生成 `selection.json` 和批次 `accepted.json`，使用相同调度保护。

### CPU与内存调度

默认 `--resource-mode dynamic --memory-target-percent 90 --workers auto`：以 `MemTotal - MemAvailable` 计算程序与系统的实际占用，保留约10%主机内存；每次派发还预留在途动作的后续增长和共享编译峰值。不是等实际碰到90%才忽略未来增长。`--memory-reserve-gib` 可额外指定更严格的最小余量；不指定时不再固定保留10/12 GiB。

CPU每秒采样可用逻辑CPU的忙碌时间，扣除本批次自身CPU时间后估算其他程序的需求。只有转换时，可使用全部可用CPU；有训练或其他程序时，并发上限随其实际CPU需求变化，不按程序名字固定让出两核。动态模式协调进程和工作进程在权限允许时使用较低CPU调度优先级，保留每进程单线程及单核绑定。CPU正在被转换充分利用时，完成一个动作仍立即补下一条，不把自身满载误判为外部压力。编译、I/O和求解串行部分仍可能使CPU达不到100%。

内存或外部CPU负载达到水位时，只暂缓新增，已经在途的任务继续；负载回落后自动补位。动态模式不会因暂时无资源而在60秒后退出，仍受总时间预算、STOP、磁盘底线和紧急内存保护约束。Linux继续建立cgroup，默认硬上限按主机总内存及目标百分比计算、禁用交换；`--memory-limit-gib` 仅用于显式设置额外硬限制，不是默认固定值。`batch.running.json.resources.cpu_admission` 和 `priority_wait` 记录实际限制原因。

需要固定配置对照时使用 `--resource-mode manual`；未指定主机预留时沿用10 GiB。旧的0.8.0批次应继续使用原冻结版本，新版不能直接覆盖运行中的源码。

Linux cgroup 准入会扣除可回收的干净、未映射文件页缓存，包括活跃与非活跃页，避免整库读取缓存阻塞派发；匿名内存、共享内存、映射文件及待写回数据仍计入占用，`MemoryMax` / `MemoryHigh` 硬保护保持。与强化学习并行时，可通过 `taskset` 排除两个物理核及其 SMT 线程，并将 `--workers` 上限设为剩余逻辑 CPU 数；不要只少开两个进程却让所有进程仍共享全部 CPU。

`--workers auto` 按 CPU affinity 和 Linux cgroup CPU 配额确定上限；`--workers 32` 表示最多32个执行槽，不表示无条件同时启动32个编译器。Linux按CPU拓扑优先用不同物理核，再分配同核的第二个逻辑线程；OMP/OpenBLAS/MKL/NUMEXPR单线程，JAX使用CPU双精度。

- 动态模式默认保留 **10%主机内存、80 GiB磁盘**；manual模式默认保留10 GiB主机内存。初始估算：Frax **1 GiB/动作+每输出帧32 KiB**，hybrid **1.5 GiB+每帧32 KiB**，V8 **1 GiB+每帧32 KiB**。只有在途或本次待派发任务确实可能编译时，共享预留初始 **2.75 GiB** 编译增量；核验初解复用和进程内热命中不计这笔预留。`--worker-resident-gib`、`--hybrid-worker-peak-gib`、`--v8-worker-peak-gib` 设置初始基数，`--frame-memory-kib` 设置逐帧项；Frax冷基数 `--worker-peak-gib` 默认3.75 GiB。
- **自动学习：**至少两个完整任务后，按模式、来源、初解是否复用及实际阶段的 JAX 缓存状态学习 PSS 上包络。当前粗阶段及其后续阶段决定剩余峰值；左右臂、多个窗口重复进入的细阶段不会提前释放预留。长度样本不足时保留配置的逐帧项，至少三个不同长度且跨度达到 500 帧后拟合非负斜率，以观测峰值上包络加 64 MiB 余量；未知阶段保守回退。编译增量单独学习并仅在可能编译时计入共享预留。在途增量仍按 `max(0, 剩余峰值预测 − 当前PSS)` 计账，空闲缓存不重复累计。失败任务不用于降低预算。记录位于 `memory_profiles/`，代码、资产、运行依赖和转换参数变化会隔离记录。`--no-memory-learning` 或热配置 `memory_learning:false` 可停用；主机与 cgroup 保护继续生效。
- 冷准备/编译限制为 **1路**，求解阶段释放锁；已命中进程内模型和kernel缓存的任务直接求解，不排队等别的冷编译。每进程保留2个kernel、复用已加载模型，每64条动作回收进程；内存不足时释放空闲进程。
- IK按固定128帧执行块编译，`--ik-block-frames` 可调整；根部增量、前两帧状态和QP热启动连续跨块传递，末块补齐帧不执行求解。共享的JAX磁盘缓存位于状态目录 `jax_cache/`，可在独立进程间复用；缓存目录应仅允许本用户写入。工作进程首次加载仍经过编译租约，进程内热命中绕过租约。
- 默认状态目录为 `~/.cache/dex-frax`，可由 `DEX_FRAX_STATE_DIR` 指定。全局批次锁防止同一状态目录下两个批次重复占用CPU和内存；不要通过换状态目录绕过锁。与旧标准Pipeline共存时，将 `DEX_FRAX_CPU_LEASE_FILE` 指向其 `active_run.lock`，两种调度器可互斥占用CPU。
- Linux多进程CLI还要求可用的 `systemd-run --user`：建立独立scope，设置 `MemoryMax`、`MemoryHigh` 和 `MemorySwapMax=0`，dynamic模式未指定硬上限时按主机总量与目标百分比确定，manual模式按启动时可用内存减预留量确定；显式 `--memory-limit-gib` 是独立的硬后备保护，上限还会裁到总内存减紧急预留量（主机预留的一半），每次派发仍检查完整的主机预留及父级 cgroup 限制。例如 `--memory-limit-gib 56 --memory-reserve-gib 12` 表示硬上限56 GiB、派发时仍保留12 GiB，并不预先占用56 GiB。此参数在单进程Linux批次也启用硬限制。建组失败则拒绝启动，不降级为无保护的并发。macOS没有该cgroup硬限制，依赖内存派发门控和紧急停止。
- manual模式无在途动作但资源持续不足时，默认等待 `--resource-wait 60` 秒后记录 `resource_exhausted`；dynamic模式持续等待资源恢复，但仍受总预算限制；主机可用内存跌破预留值的一半时终止本批次拥有的在途进程。磁盘低于预留量时也停止。
- XML只编译一次并校验MJB模型数组；每个spawn工作进程复用自己的MjModel，各Robot保持独立MjData。**没有将已初始化的JAX进程fork出去，也不声称跨spawn进程共享MuJoCo对象。** 转换身份和资产哈希在父进程冻结，批次结束独立重算。

`batch.running.json` 记录实际在途数量、PID、当前阶段/缓存状态、估算峰值、PSS、学习记录、主机可用内存和cgroup上限。`priority_wait` 记录阻塞队首的动作、阶段、预测峰值、编译预留、在途增长预留与可用余量。创建 `<输出目录>/STOP` 或向调度器发送SIGTERM，会停止本批次并保留结果。0.6.1已有实际32路运行记录；0.7.0的完整效率与质量需看独立全量验收。

本次修复与实测边界见 [0.6.0验证记录](docs/IMPROVEMENT_060_VALIDATION.md)；事故及早期保护记录保留于 [0.5.1资源验证](docs/RESOURCE_051_VALIDATION.md)。

### 运行中调整内存预算

每批启动时生成 `<输出目录>/resource_policy.json`，也可用 `--resource-policy /path/to/policy.json` 指定。已有文件优先于CLI初始预算；调度器每2秒检查修改，重新计算在途和待派发任务的预留，有安全余量就继续补槽，不需要等待动作或整批结束。减少预算本身不会暂停队列或重启工作进程。

```json
{
  "schema": 1,
  "workers": 32,
  "memory_learning": true,
  "compile_extra_gib": 2.75,
  "stages": {
    "frax": {"base_gib": 1.0, "bytes_per_output_frame": 32768},
    "hybrid": {"base_gib": 1.5, "bytes_per_output_frame": 32768},
    "v8": {"base_gib": 1.0, "bytes_per_output_frame": 32768}
  }
}
```

建议先写临时JSON再原子替换该文件。损坏JSON、超出启动槽数上限或非法数值会被拒绝，并继续采用最近有效配置。生效/拒绝原因写入 `resource_policy_events.jsonl`，实际配置和哈希写入 `batch.json`。热加载只调整资源准入；主机预留、磁盘预留、cgroup硬上限、IK参数和质量门槛保持启动时设置。内存不足时暂缓当前派发，下一轮立即重算，已有任务继续运行；原有紧急停止保护保留。

### 手臂分支跟踪

平滑区间优先复用已有分支；拟合时先调整 J1/J2/J3，必要时联合七轴。J4～J7 的肘弯曲、肘轴/肘窝、前臂与手掌方向、腕部软限位目标继续执行。接触切换、接近限位、近伸直状态切换、目标跳变或误差持续上升触发扩大搜索。

困难标记按帧计算，不再因一个接触段而让整条长片的该侧臂失去分支复用。已经在 200 帧以内的难窗口直接使用原 V8；对不超过 600 帧、困难区加 0.4 秒上下文覆盖至少三分之一的短片，也直接保留原 V8，防止密集难段先快解再返工。更长动作不能触发该整段回退。长片局部恢复最多每窗 200 帧、重叠 20 帧，保留原多起点算法。接缝经过带姿态目标、限速、固定边界及支撑检查的再拟合；失败局部回滚并记录，不回到整条长片重搜分支。为保留困难动作的全局连续性，快速时序旁路未通过时，仍执行一次原有全片连续性拟合；最终姿态异常才触发局部原 V8 重搜。全片最终验收仍决定能否导出训练文件。

### 三级碰撞修复

仅对基础修复未解决的窗口逐级尝试，成功即停止。下表是相对修复前轨迹的总改动预算，不是机械限位或最终质量阈值。

| 档位 | 手臂单轴改动 | 手掌旋转变化 | 手掌位移 | 额外自由度 |
|---|---:|---:|---:|---|
| 小幅 | 10° | 10° | 3 cm | 双臂按需联合；接触帧的支撑臂关节固定 |
| 中幅 | 20° | 20° | 8 cm | 同上 |
| 耦合 | 35° | 30° | 15 cm | 联动躯干、腿和根部；其余关节最多5°、根部最多2 cm |

耦合档保持已有固定材料接触点，滑动/滚动只守住法向位置，并守住脚部支撑。窗口衔接处保留原有无穿透间隙，内部以3.5 mm间隙求解；碰撞、地面、关节限位和逐轴速度仍复查。帧间碰撞直接进入优化，并由八倍插值独立复验。碰撞距离基于声明的凸碰撞体，不是连续碰撞或真实网格交叠证明。

0.8.0 将长问题区间切为最多 120 帧、通常重叠 20 帧的窗口。所有未解决区间先尝试小幅档，再对剩余区间尝试中幅、耦合档；同一轮优先较深碰撞。各档总偏移始终相对同一修复前轨迹，不能叠加成 10°+20°+35°。难窗可保留有限的中间迭代作为下一级初值，但不能据此判为通过。内部切口可移动，外部有效衔接固定；接缝不得新增或加深碰撞，并整段八倍插值复验。

三级均保留原源动作姿态权重。完整 hybrid/V8 不再按“全片 600 帧、12 窗、90 秒”跳过余下区间；快速版仍限制最多 600 个尝试帧（每一级重复计数），余项记录 `deferred` 留给后续路径。正常求解按 10/16/24 次外迭代及每次最多 20000 次 QP 迭代控制工作量，不再让 5/8/20 秒墙钟抢占改变质量；任务级超时/总预算仍会中断，不能视为完成了全部修复。

`motion.json` 中记录 `repaired`、`deferred`、`local_solver_failed`、`interrupted` 和 `tiers_exhausted` 等状态；`run.json` 的 `collision_repair_exhausted` 仅在剩余窗口确实尝试过全部三级时成立。三级用尽表示本地搜索尚未找到合格解，不代表数学上无解。任何路径只有最终整段安全、接触、姿态和导出审核通过才得到 `training.npz`。

## 输入清单与私有回归集

固定模式和自动模式使用相同的 JSON 清单。`input` 可为绝对路径，或相对于清单所在目录的路径；`source` 为 `lafan`、`soma`、`mocap` 或 `auto`。推荐已知数据集显式指定 source。`id` 必须唯一，以字母或数字开头，仅含字母、数字、点、下划线或连字符。

批处理先核验全部源文件哈希，再按完整BVH字节SHA256去重，保留清单中第一个id，其他id记录到 `source_aliases.json`。相同内容却指定不同source时拒绝启动，避免不一致解释。不同字节编码的语义重复动作不在此去重范围。

```json
{
  "cases": [
    {
      "id": "walk_001",
      "input": "/path/to/walk.bvh",
      "source": "lafan",
      "input_sha256": "填写该文件的真实SHA256"
    }
  ]
}
```

可用下面的脚本为自己的 BVH 目录生成清单（先修改 `root` 和 `source`）：

```sh
.venv/bin/python - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('/path/to/my_bvh').resolve()
source = 'lafan'  # SOMA Uniform 使用 soma；已验证骨架（如 examples/inputs/bvh/nobody）使用 mocap
cases = [dict(id=f'motion_{i:06d}', input=str(p), source=source,
              input_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
         for i, p in enumerate(sorted(root.rglob('*.bvh')))]
if not cases:
    raise SystemExit('没有找到 BVH，请检查 root')
output = Path('regression_local/custom/manifest.json')
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(dict(cases=cases), ensure_ascii=False, indent=2))
print(output, len(cases))
PY
.venv/bin/dex-frax batch regression_local/custom/manifest.json outputs/custom_auto --mode auto --budget 7200
```

`regression_local/` 中的原始 BVH、片段、清单、基线和结果全部由 `.gitignore` 排除，不随源码发布。当前私有回归集包含 `examples/inputs/bvh/nobody/` 示例全片及 11 个约 6 秒的 LAFAN 独立片段，尚无运动中的 SOMA/BONES 样本。独立片段的接触预处理不等价于完整动作的对应窗口。

## 输出与验收

- `motion.pkl` / `motion.npz`：50 Hz 预览，米、弧度，根四元数 XYZW。
- `motion.initial.npz`：未经 V8 后处理的初始轨迹，`x=[root XYZ, 29 joints]`，另存根旋转矩阵。
- `motion.diagnostics.npz`、`motion.json`：人体目标、接触、求解记录和阶段耗时。
- `motion.check.json`、`motion.pose_errors.npz`、`quality.json`：最终存储精度下的 FK、接触、逐轴速度和自碰撞审核。
- `quality.json` 还记录膝关节最小角度、负角度帧数，以及速度、加速度和 jerk 的统计。现有模型没有加速度/jerk 额定限值，这两项仅作比较，不表示已通过硬件验收。
- `identity.json`：输入、全部转换源码、frax 源码、资产及运行版本指纹。
- `run.json`：`complete` 表示处理结束；还必须看 `quality=accepted/review`。
- `training.npz`：仅通过全部门槛时生成；根 WXYZ，29 关节与预览逐值一致。它有独立的 `dex_frax` 身份，现有严格 V8-only 训练入口需要明确适配后才能消费。

保留 V8 的源姿态、速度、自碰撞审核，另外检查最终足底/全身穿地和接触误差 1 mm。本项目接触误差以 XYZ 向量范数检查，较逐分量 1 mm 更严格。非收敛 QP 会记录并阻止生成训练 NPZ。`review` 结果仍可预览，不能作为通过验收的训练参考。

0.5.0 对所有模式统一执行持续上肢姿态门槛：同一目标误差连续至少 0.1 s 超过上/前臂方向 20°、手掌轴 25°或肘弯曲轴 30°，标记 `sustained_arm_pose_mismatch`。这些是工程筛选阈值，不代表人体或硬件额定限值。它们补充原有全片 P95 检查，并在避碰后的最终导出上执行。

## 预览

```sh
.venv/bin/dex-frax preview \
  outputs/example_nobody/frax/motion.pkl \
  outputs/example_nobody/hybrid/motion.pkl \
  outputs/example_nobody/v8/motion.pkl \
  --output outputs/example_nobody/comparison.mp4
```

上面的路径对应单文件 auto 输出的阶段目录（`frax/`、`hybrid/`、`v8/`），只预览实际生成的阶段；固定 `--mode` 的输出直接位于各自指定目录。

生成并排 MP4、GIF 和三张关键帧。它是运动学回放，不包含动力学、策略跟踪或真机验证。

## 验证与来源

0.7.0短回归采用每轮最多300秒的外层超时，开发中只用已缓存初解和同源目标比较难窗口。最终163项测试约39秒通过，四窗口实际对照约22秒；内存学习与真实CLI记录见 [0.7.0验证记录](docs/PIPELINE_070_VALIDATION.md)。普通窗口有收益，起身/失衡难窗口保留旧算法，不能据此宣称所有动作都加速。

```sh
# 所有开发验证用同一个5分钟上限；日志旁生成超时/退出码/耗时收据
.venv/bin/python scripts/check_bounded.py --log regression_local/check.log -- .venv/bin/python -m pytest -q
```

0.6.1完整测试153项通过，包含在途任务未结束时热加载补槽、跨动作阶段重叠和原有求解质量回归。远端77条从零重跑的状态见 [0.6.1记录](docs/RESOURCE_061_VALIDATION.md)。

0.6.0 本地144项测试通过，双工作进程走路/跑步样例及完整nobody均通过统一审核；真实起身/失衡片段的修复仍有未解决窗口。详见 [0.6.0验证记录](docs/IMPROVEMENT_060_VALIDATION.md)。0.6.1远端77条完整LAFAN已结束：107.36分钟，2条通过、75条待审，最大并发32；完整复盘见 [0.6.1记录](docs/RESOURCE_061_VALIDATION.md)。

0.5.0 已完成 118 项测试及 12 动作、18 次真实分级尝试，最终 9 项通过、3 项待审；见 [自动批处理验证](docs/BATCH_050_VALIDATION.md)。

```sh
DEX_FRAX_TEST_BVH=examples/inputs/bvh/nobody/nobody_without_calibration.bvh .venv/bin/python -m pytest -q
```

设置 `DEX_FRAX_REGRESSION_SUITE=/path/to/regression_local/suite_20261105` 可额外运行私有起身清障回归；未提供时明确跳过。

测试包含 frax/MuJoCo 随机姿态 FK、JAX 任务代价与梯度对照 V8、跨帧速度边界、四元数导出回读，以及复制自最新 V8 的接触、速度、避碰回归。没有指定真实 BVH 时，该项数据测试明确跳过。

见 [实现说明](docs/IMPLEMENTATION.md)、[样例验证](docs/NOBODY_VALIDATION.md)、[第三方来源](THIRD_PARTY.md)。机器人资产保持原始许可；人体参考表和样例数据不因本项目代码而改变授权。

后续对 V8 基线片尾进行了单独重试：同一窗口联合双臂修复后，完整动作的声明碰撞对检查为零碰撞，原三路对照保留。见 [V8 片尾修复记录](docs/V8_TAIL_RETRY_20261105.md)。

V8 默认联合手臂选择已同步至 Dex-Frax 0.1.1，`hybrid` 与 `v8` 模式自动使用；独立重试脚本仅保留用于诊断和旧输出修复。见 [默认流程合入记录](docs/V8_DEFAULT_MERGE_20261105.md)。

### 仅更新调度代码后的续跑

普通 `--resume` 保持严格身份校验。明确只更新了 `managed_batch.py` 和/或 `resources.py` 时，可以用 `DEX_FRAX_ALLOW_SCHEDULER_RESUME=1 .venv/bin/python -m dex_frax batch <原清单> <原输出目录> --resume ...`。程序仍校验完整源文件哈希、每个完成阶段的输出哈希、转换参数、机器人资产、依赖和其他全部实现文件；CLI、求解器、审核或导出变化一律拒绝。新增 `scheduler_resume.*.json` 记录前后身份及精确差异，不覆盖原始 `batch_identity.json` 或已有动作身份。已完成动作及阶段不重算，未完成阶段保留归档后重试，已完成的最终待审/失败记录保持原状态。

编译预留仍采取保守原则：只有真实内核命中事件或已核验的初解复用才免除编译需求；仅有 `jax_hot` 标记不能证明下一条动作的接触拓扑对应内核已存在。

### 远端正在运行旧批次时更新下一轮版本

不要覆盖运行中的 `src/` 或其环境，因为批次结束会复核源码身份。可以将新版部署到包内独立的 `releases/<版本>/`，配备独立环境，以 `next` 链接和 `run-next` 入口启动下一轮。当前批次继续使用旧源码与旧环境；模型和上游frax可只读共享。

```sh
# luckylhf 已部署此入口；下一轮使用新输出目录和完整清单
~/Desktop/Dex-Retarget/run-next batch /path/to/manifest.json /path/to/new_output \
  --mode auto --workers auto --budget 604800
```

这一入口默认启用90%内存目标和动态CPU调度，不要再附加本轮的56 GiB/28槽/固定CPU掩码。新旧版本共享全局批次锁，不能同时启动两个转换批次。

### 0.8.1 调度性能修正

调度器先收取完成结果，每轮只对工作进程做一次PSS扫描，补槽与状态报告复用该轮结果。每次派发仍检查整机可用内存及cgroup余量，并累计在途峰值与共享编译预留；同轮不会重复使用刚释放的内存额度。队列用稳定优先堆维护原有短动作和阶段顺序，不再每轮排序全库。

`batch.running.json` 是约每2秒更新的轻量实时状态；`scheduler_loop_s` 记录最近调度循环耗时。`batch.json` 与 `accepted.json` 是每30秒、正常结束及可处理的中断时同步刷新的完整检查点/导出目录，因此可能落后实时状态。每个动作的 `selection.json` 仍在收尾时立即写入。强杀/断电无法保证最后30秒已写入完整检查点；请优先使用STOP或SIGTERM，等待进程退出后再续跑。

本修正只更改调度实现，不改IK、碰撞、三级放宽、质量门槛或导出格式。对已有批次的调度修复续跑须显式使用 `DEX_FRAX_ALLOW_SCHEDULER_RESUME=1 ... batch ... --resume`；启动时仍校验源文件、已完成产物、转换参数及非调度实现，并保留旧产物身份与变更记录。
