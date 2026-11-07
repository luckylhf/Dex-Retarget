# 输入规范

## 机器人资产

默认使用仓库的 `assets/dex_evt/`。使用外部已核验副本时，指定完整资产目录：

```sh
export DEX_FRAX_ASSETS=/absolute/path/to/assets/dex_evt
```

目录需完整保留 `model.json`、`mjcf/`、`meshes/`、`meshes_visual/`、`meshes_convex/`、`urdf/`、`README.md` 和 `LICENSE`，并核对模型指纹。wheel 和 Python 源码分发包不含机器人资产，安装后须通过该变量指定外部副本。

资产身份包含目录内所有文件，README 也参与校验。已有批次续跑或运行期间，应保持其资产副本的全部文件字节不变。资产目录内的 README 保留原始版本以兼容旧身份，当前项目用法以本节为准；本包不提供其中提到的 `scripts/import_assets.py`，外部资产通过上述变量指定。

## BVH

| `--source` | 输入要求 | 额外文件 |
|---|---|---|
| `lafan` | LAFAN1 骨架与坐标约定 | `lafan1_surfaces.json` |
| `soma` | BONES-SEED SOMA Uniform 骨架 | `soma_surfaces.json` |
| `mocap` | 已逐关节验证的 Y-up 骨架，含手指基点及脚趾标记；附带 nobody 示例使用此项 | 无 |
| `auto` | 按关节名选择上述来源 | 取决于识别的来源 |

推荐显式指定来源。`mocap` 不表示支持任意 BVH；新的关节命名、局部轴或骨架拓扑需要先适配。

LAFAN/SOMA 的参考几何表不随 Git 或 wheel 分发。合法取得后放入
`src/dex_frax/v8/reference/`；来源与许可见 [SOURCE_GEOMETRY.md](../src/dex_frax/v8/reference/SOURCE_GEOMETRY.md)。仓库没有从原始 FBX/USD 生成这些表的工具，未备齐时不能直接运行对应来源。nobody 示例和 SMPL-X 入口不需要这两张表。

```sh
.venv/bin/dex-frax convert /path/to/walk.bvh outputs/walk --source lafan
```

## SMPL-X

`convert` 支持 `--source smplx`（`--source auto` 遇到 `.npz`/`.pt`/`.pkl` 也按 SMPL-X 处理）。`.pt`、`.npz`、`.pkl` 是容器后缀，**不代表其中一定是 SMPL-X**。`batch` 清单只接收 BVH。

### 安装与模型

人体前向几何需要 `torch`、`smplx==0.1.28`、NumPy 和 SciPy。在当前环境安装：

```bash
.venv/bin/python -m pip install -e '.[smplx]'
export SMPLX_PYTHON="$PWD/.venv/bin/python"
```

或指定已有的 SMPL-X 环境：

```sh
export SMPLX_PYTHON=/path/to/smplx/env/bin/python
```

两种方式都需要另备人体模型目录：

```sh
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

只需要输入性别对应的文件。模型从 [SMPL-X 官方网站](https://smpl-x.is.tue.mpg.de/) 按许可获取；代码仓库不分发这些参数。可用已有的合法模型目录，无需复制。模型与输入参数、预处理实现和运行依赖版本都会进入转换身份记录。

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
- **GVHMR 特例**：世界约定为 Y-up、米；缓存没有帧率时须提供真实帧率，附带 GBTC 示例使用 `--input-fps 30`。网络的 `static_conf_logits` 若存在，读取前四个脚部通道作为关节静止概率。

机器人 `root_pos/root_rot/dof_pos` PKL 和 `joint_pos/body_pos_w` NPZ 不能作为人体 SMPL-X 参数输入。

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

`examples/inputs/smplx/gbtc/gbtc.pt` 是附带的一组 30 fps GVHMR 缓存，可直接替换上面示例中的 `.pt` 路径，运行方式见 [README](../README.md)。

`check-input` 只检查内容结构；实际模型文件、人体几何和支撑证据充分性在 `convert` 时检查。`--max-frames` 仅用于短输入接口测试，不能作为全片质量验收。SMPL-X 单文件默认同样按 `frax → hybrid → v8`，完整通过即停止；`--mode hybrid` 等固定模式仍可用。`--budget` 限制自动流程（包含人体准备），固定模式不使用该总预算。

人体准备在 `--smplx-python` 的独立子进程中执行，不要求旧 GMR 环境安装 JAX。准备后的 `input/human.npz` 是内部几何缓存，不是可重新当作 SMPL-X 参数提交的公共输入格式。

### 可选接地修正的含义

`--grounding contact` 利用静止概率大于 0.8、且接近较低脚底的观测，拟合一条平滑的公共垂直漂移曲线；对所有人体点减去同一高度偏移，不改局部旋转和肢体长度。没有支撑观测的区间只延续/插值漂移，**不会每帧把最低脚压到地面**。缺少置信度或有效支撑证据不足时直接提示失败，不静默切换算法。

这种修正假设平地，静止也不等于真实承重。长时间腾空、静态抬腿、台阶和平台可能使判断出错；应查看起跳/落地及长无观测区间。它没有自动证明所有跳跃均被正确保留。可靠动捕通常用 `off`。

## 附带示例

| 路径 | 用途 |
|---|---|
| `examples/inputs/bvh/nobody/nobody_without_calibration.bvh` | 可直接转换的人体 BVH，已移除开头校准帧 |
| `examples/inputs/bvh/nobody/1013_nobody_chr01.bvh` | 含校准帧的原始 BVH，供核对预处理 |
| `examples/inputs/smplx/gbtc/gbtc.pt` | 420 帧、30 fps 的 GVHMR 人体缓存；需另备 SMPL-X 模型和依赖 |
| 两目录中的机器人 PKL/NPZ、`reference.npz` | 既有对照数据，不能作为人体输入，也不表示当前版本的验收结果 |

BVH 示例的文件来源和 SHA-256 见目录内 `manifest.json`；GBTC 的逐文件哈希见其 `说明.txt`。

## 输出格式

每个完成的求解阶段保存以下文件；自动模式按需生成 `frax/`、`hybrid/`、`v8/` 子目录，固定模式直接写入指定目录。

| 文件 | 内容 |
|---|---|
| `motion.pkl` / `motion.npz` | 50 Hz 动作：`root_pos`、`root_rot`（XYZW）、`dof_pos`、关节名和模型身份 |
| `quality.json` | `accepted` 或 `review`，以及未通过原因 `flags` |
| `identity.json` / `run.json` | 输入、实现与资产身份，以及执行状态、耗时和产物哈希 |
| `motion.json` / `motion.diagnostics.npz` | 接触、目标和求解诊断，供排查异常 |
| `training.npz` | 仅通过时生成，格式为 `root_joint_lossless_v1` |

`training.npz` 包含 `joint_pos`、`joint_names`、仅含 pelvis 的 `body_names/body_pos_w/body_quat_w`、帧率和转换身份；根四元数为 WXYZ。关节数据与预览逐值一致。下游须适配 `converter=dex_frax` 和此格式，需要全身刚体 FK、速度或训练特征时另行计算。

自动模式通过 `selection.json` 指定采用的阶段，下游只读取 `accepted.json` 的 `motions`。`accepted` 表示通过当前运动学检查，仍需下游动力学跟踪与真机验证。
