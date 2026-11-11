# 输入规范

## 机器人资产

默认使用仓库的 `assets/dex_evt/`。使用外部已核验副本时，指定完整资产目录：

```sh
export DEX_FRAX_ASSETS=/absolute/path/to/assets/dex_evt
```

目录需完整保留 `model.json`、`mjcf/`、`meshes/`、`meshes_visual/`、`meshes_convex/`、`urdf/`、`README.md` 和 `LICENSE`，并核对模型指纹。wheel 和 Python 源码分发包不含机器人资产，安装后须通过该变量指定外部副本。

资产身份覆盖模型、网格等运行文件，运行与续跑期间须保持其字节一致。目录根部的 README、LICENSE、Finder 的 `.DS_Store` 和 AppleDouble 文件不参与计算。

迁移动作后可用 `python -m dex_frax.v8.audit path/to/motion.pkl` 复核。保留对应的 `motion.json`、`motion.diagnostics.npz` 和 `identity.json`。原模型路径不可用时，复核使用 `DEX_FRAX_ASSETS` 指定的模型，核对导出记录的 XML 哈希和身份记录中的资产哈希。

## BVH

| `--source` | 输入要求 | 额外文件 |
|---|---|---|
| `lafan` | LAFAN1 骨架与坐标约定 | `lafan1_surfaces.json` |
| `soma` | BONES-SEED SOMA Uniform 骨架 | `soma_surfaces.json` |
| `mocap` | 已逐关节验证的 Y-up 骨架，含手指基点及脚趾标记；附带 nobody 示例使用此项 | 无 |
| `auto` | 按关节名选择上述来源 | 取决于识别的来源 |

BVH 默认使用厘米及 `absolute` 位置通道，文件本身通常不声明这些语义，不自动猜测。可显式指定：

```sh
.venv/bin/dex-frax convert input.bvh outputs/motion --source mocap --bvh-units m --bvh-translations offset
```

`--bvh-units` 支持 `cm`、`m`、`mm`；`absolute` 表示位置通道直接替代对应 OFFSET 分量，`offset` 表示通道值加 OFFSET。相同选项也适用于 `batch`，一个批次使用同一约定，不同约定应分批。参数写入转换身份与产物元数据，改变后不能沿用旧续跑身份。SMPL-X 仍使用自己的 `--units m`。

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

输入是**单个人体的一段连续动作**。局部旋转必须是轴角、单位弧度；平移必须是米。数组须有限、帧数一致，至少 3 帧；输出统一为 50 Hz。人体前向几何前，旋转按局部 SLERP 重采样，平移与形状参数线性插值。默认保持时间尺度；局部减速选项及对照方式见 [时间映射](PIPELINE.md#帧率与时间映射)。

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
- 不接受 72 维 SMPL 或 156 维 SMPL-H 直接作为 SMPL-X 输入。分字段与合并字段同时存在时必须一致；别名冲突会报错。

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

`examples/smplx/gbtc/gbtc.pt` 是附带的一组 30 fps GVHMR 缓存，可直接替换上面示例中的 `.pt` 路径，运行方式见 [README](../README.md)。

`check-input` 只检查内容结构；实际模型文件、人体几何和支撑证据充分性在 `convert` 时检查。`--max-frames` 仅用于短输入接口测试，不能作为全片质量验收。SMPL-X 同样支持 `--mode fast`（默认）和 `--mode refine`。两种模式的 `--budget` 都限制包含人体准备在内的总时间。精修另支持 `--repair-budget` 和检查点续跑，见 [流程说明](PIPELINE.md#检查点与续跑)。

人体准备在 `--smplx-python` 指定的独立子进程中执行。准备后的 `input/human.npz` 是内部几何缓存，不是可重新当作 SMPL-X 参数提交的公共输入格式。

### 可选接地修正的含义

`--grounding contact` 利用静止概率大于 0.8、且接近较低脚底的观测，拟合一条平滑的公共垂直漂移曲线；对所有人体点减去同一高度偏移，不改局部旋转和肢体长度。没有支撑观测的区间只延续/插值漂移，**不会每帧把最低脚压到地面**。缺少置信度或有效支撑证据不足时直接提示失败，不静默切换算法。

这种修正假设平地；静止概率不等于真实承重。长时间腾空、静态抬腿、台阶和平台可能影响结果，应复核起跳、落地及长时间无支撑观测的区间。已可靠接地的动捕使用 `off`。

## 附带示例

| 路径 | 用途 |
|---|---|
| `examples/bvh/nobody/nobody.bvh` | 可直接转换的 mocap 骨架 BVH：2085 帧、90 fps、厘米、Y-up |
| `examples/bvh/nobody/nobody_50Hz-Retarget.npz` | 该输入的 50 Hz 参考输出，1158 帧 |
| `examples/smplx/gbtc/gbtc.pt` | 420 帧、30 fps 的 GVHMR 人体缓存；需另备 SMPL-X 模型和依赖 |
| `examples/smplx/gbtc/gbtc_50Hz-Retarget.npz` | 该输入的 50 Hz 参考输出，699 帧 |

参考 NPZ 是机器人动作（格式见下节），只能用于对照，不能当作人体输入，也不表示当前版本的验收结果。

## 输出格式

每条完成动作的输出结构为：

```text
outputs/motion/
├── selection.json
├── accepted.json
└── frax/
    ├── motion.pkl
    ├── motion.npz
    ├── quality.json
    ├── identity.json
    └── run.json
```

| 文件 | 内容 |
|---|---|
| `selection.json` | 该动作的结果状态、选中阶段及路径 |
| `accepted.json` | 训练准入列表；下游读取 `motions`，路径相对于输出根目录 |
| `frax/motion.pkl` / `motion.npz` | 完整 50 Hz 机器人动作，待审结果也保留 |
| `frax/quality.json` | 质量状态、失败标记、指标及检查范围 |
| `frax/identity.json` / `run.json` | 输入、代码、依赖、资产身份与执行状态、耗时、产物哈希 |
| `frax/motion.json` / `motion.diagnostics.npz` | 接触、目标和求解诊断 |
| `frax/pipeline.json` / `pipeline.md` | 根据实际记录生成的流程解释 |
| `frax/training.npz` | 仅质量通过时生成的训练导出 |

预览动作字段：

| 字段 | 形状或含义 |
|---|---|
| `root_pos` | `(T,3)`，米 |
| `root_rot` | `(T,4)`，XYZW 四元数 |
| `dof_pos` | `(T,29)`，弧度，顺序见 `joint_names` |
| `fps` | 50 |
| `joint_names`、`quat_order`、`model_sha256` | 关节顺序、四元数约定及模型身份 |

启用局部减速时另有 `motion.time_map.npz`；动作文件保留 `source_time_s`、`source_frame_float` 和 `source_fps`。机器人按 `i / 50` 播放，人体对照按源时间映射采样。

`training.npz` 使用 `storage_format=root_joint_lossless_v1`、`converter=dex_frax`：`joint_pos` 与预览关节数据逐值一致，`body_names` 仅含 pelvis，`body_pos_w` 为 `(T,1,3)`，`body_quat_w` 为 `(T,1,4)`，根四元数 **WXYZ**。下游按模型身份和关节顺序读取；全身刚体 FK、速度或其他训练特征需另行计算。

`run.json` 的 `complete` 表示执行完成，质量状态查看 `quality.json`。`accepted` 只覆盖所列运动学检查，不表示动力学或真机验证通过。
