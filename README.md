# Dex-Retarget

将人体动作转换为 **Dex 29 关节、50 Hz 的机器人运动学参考**，用于动作预览和训练数据准备。支持已适配的 BVH 骨架及 SMPL-X 人体参数（`.pt`、`.npz`、`.pkl`）。

当前版本为 **0.8.2**。版本与验证结果见 [CURRENT_VERSION.md](docs/CURRENT_VERSION.md)。

## 安装

使用 Python **3.11 或 3.12**。在项目根目录执行；以下以 Python 3.12 为例。

若交付目录中没有 `Dex-StanfordASL-frax/`，先获取 Frax：

```sh
git clone https://github.com/StanfordASL/frax.git Dex-StanfordASL-frax
git -C Dex-StanfordASL-frax checkout f5e87ab59a0b82ba43095e882c1b88c9b0dacf07
```

然后安装：

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e './Dex-StanfordASL-frax[ik]'
.venv/bin/python -m pip install -e '.[preview]'
.venv/bin/dex-frax --help
```

目录搬迁后重新创建虚拟环境，不复制 `.venv`。运行依赖见 [pyproject.toml](pyproject.toml)，验证环境的版本快照见 [requirements-lock.txt](requirements-lock.txt)。

以上命令适用于完整源码目录。单独安装 wheel 或源码分发包时，须另备机器人资产，并设置 `DEX_FRAX_ASSETS`。LAFAN/SOMA 另需参考几何表；SMPL-X 另需人体模型与依赖。准备方法见 [输入规范](docs/INPUTS.md)。

## 转换动作

先运行附带的 BVH 示例，输出目录必须尚不存在：

```sh
.venv/bin/dex-frax convert \
  examples/bvh/nobody/nobody.bvh \
  outputs/nobody --source mocap
```

| 模式 | 行为 |
|---|---|
| `--mode fast`（默认） | 完整 Frax 求解、导出和质量检查 |
| `--mode refine` | 自动选择腿脚初解路线，再修复问题窗口并整片复查 |

`--mode` 只有这两个取值；旧版的 `auto`、`v8`、`hybrid` 和 `--pipeline two-stage` 已移除，传入会被参数校验拒绝。

`refine` 检查原始 Frax 轨迹的下肢碰撞。碰撞少时保留原始腿脚，仅修对应窗口；碰撞多或窗口修复未通过检查时，使用整片腿脚求解。实际路线记录在 `motion.json` 的 `frax.body_priority.lower_routing`。

```sh
.venv/bin/dex-frax convert input.bvh outputs/refined \
  --source mocap --mode refine --repair-budget 240
```

`--repair-budget` 限制局部修复时间；预算结束时保留已采用的改进及待处理记录。单文件 `--budget` 限制包含准备、初解和检查在内的总时间，默认 86400 秒。精修续跑见 [流程说明](docs/PIPELINE.md#检查点与续跑)。

BVH 来源使用 `lafan`、`soma` 或 `mocap`；`mocap` 需要已适配的骨架，不能直接接收任意 BVH。BVH 默认厘米、`absolute` 位置通道，可通过 `--bvh-units` 和 `--bvh-translations` 指定。

`examples/` 提供人体输入、两版机器人动作和逐帧对照：

| 路径 | 内容 |
|---|---|
| `examples/bvh/nobody/nobody.bvh` | 2085 帧、90 fps 的 mocap 骨架 BVH（厘米、Y-up），配合 `--source mocap` |
| `examples/bvh/nobody/` | `nobody_v1-50Hz-Retarget.npz`、`nobody_v2-50Hz-Retarget.npz`，各 1158 帧；`nobody_逐帧对照.csv` |
| `examples/smplx/gbtc/gbtc.pt` | 420 帧、30 fps 的 GVHMR SMPL-X 参数缓存，需另备人体模型与依赖 |
| `examples/smplx/gbtc/` | `gbtc_v1-50Hz-Retarget.npz`、`gbtc_v2-50Hz-Retarget.npz`，各 699 帧；`gbtc_逐帧对照.csv` |

`v1` / `v2` 对应 `--foot-contact v1` / `v2`（默认 v2，见下文）。两版 NPZ 均为 50 Hz，与转换生成的 `motion.npz` 同格式：`root_pos`、`root_rot`（XYZW）、`dof_pos`、`joint_names`、`quat_order`、`model_sha256`、`robot`。逐帧 CSV 对比两版的关节最大差异和根平移差异，以及左右脚的支撑相位、目标与实际 roll、sole 三轴角、鞋底与外缘最低点。

![人体输入与 v1 / v2 参考动作同屏回放](docs/pt-v1-v2.gif)

同一段人体动作与两版 50 Hz 参考动作在一个查看器里同步回放：初始状态左为人体骨架，中为 `v1`，右为 `v2`。回放是运动学示意，不含动力学。

`preview` 只接受转换结果中的 `motion.pkl`，不能直接播放这些参考 NPZ；它们用于数值对照，不代表当前 `accepted` 结论。

SMPL-X 输入先按 [输入规范](docs/INPUTS.md#smpl-x) 配置 `SMPLX_PYTHON` 和 `SMPLX_MODEL_DIR`，再运行：

```sh
.venv/bin/dex-frax check-input examples/smplx/gbtc/gbtc.pt \
  --smplx-python "$SMPLX_PYTHON" --input-fps 30
.venv/bin/dex-frax convert examples/smplx/gbtc/gbtc.pt outputs/gbtc \
  --source smplx --body-models "$SMPLX_MODEL_DIR" \
  --smplx-python "$SMPLX_PYTHON" --input-fps 30 --grounding contact
```

`--input-fps` 应填真实源帧率，`.pt` 后缀本身不代表 30 Hz。示例中的 `.pt` 是人体输入，已有机器人 `.npz` / `.pkl` 不能作为 SMPL-X 参数输入。

### 足部接触规则

单文件转换和 BVH 批处理默认使用 v2.0。单端支撑要求另一端持续抬起，足姿修正以约 0.1 秒过渡。SMPL-X 鞋底四区域的几何用于调节支撑侧倾，BVH 使用保守的侧倾目标。

复算 v1.0 动作时加 `--foot-contact v1`。续跑时使用与原任务相同的足部接触规则。

## 查看与使用结果

输出根目录的 `selection.json` 记录结果选择，动作及检查报告位于 `frax/`；实际模式查看 `profile` 字段。

| 结果 | 含义 |
|---|---|
| `accepted` | 通过当前配置的运动学质量检查，生成 `training.npz` |
| `review` | 已生成完整动作，但仍有未通过项；查看 `frax/quality.json` 的 `flags` |
| `failed` / `input_error` / 中断或超时 | 转换未完成；查看执行记录和日志 |

完整结果包含 `motion.pkl`、`motion.npz`、`quality.json`、`identity.json` 和 `run.json`。预览动作使用米、弧度、根四元数 **XYZW**；训练导出的根四元数为 **WXYZ**，格式为 `root_joint_lossless_v1`，读取方式见 [输出格式](docs/INPUTS.md#输出格式)。

```sh
.venv/bin/dex-frax preview outputs/nobody/frax/motion.pkl \
  --output outputs/nobody/preview.mp4
.venv/bin/dex-frax explain outputs/nobody
```

`preview` 生成 MP4、GIF 和关键帧；`explain` 只读解释已有报告。播放成功或执行状态为 `complete`，均不代表质量通过。下游训练入口读取 `accepted.json` 的 `motions`，其中路径相对于输出根目录。

## 帧率与时间对齐

机器人输出固定 **50 Hz**，第 `i` 帧的播放时间是 `i / 50` 秒，帧号从 0 开始。

默认保持源动作时间尺度。仅 `refine --allow-local-slowdown` 允许将骨盆运动过快的片段放慢，**这会延长动作时长，并改变其与原视频或音乐的同步关系**。该选项不默认开启，续跑时须保持一致。

启用后，`source_time_s[i]` 表示输出第 `i` 帧对应的原人体时刻；它可以非均匀，不能当作机器人播放时钟。人体对照应按此映射插值，详见 [时间映射](docs/PIPELINE.md#帧率与时间映射)。

## 批量转换

批处理支持 BVH。按 [批处理说明](docs/BATCH.md) 生成带输入哈希的清单，然后运行：

```sh
.venv/bin/dex-frax batch manifest.json outputs/batch --budget 604800
```

默认 `fast`，使用可用 CPU 并按内存准入控制并发。批次总预算默认 1800 秒，磁盘默认预留 80 GiB；请按任务规模设置 `--budget`、`--workers` 和 `--disk-reserve-gib`。停止与续跑方法，以及以全部可用逻辑核为并发上限的 `refine` 参数，见批处理文档的[启动与预算](docs/BATCH.md#启动与预算)。

首次批处理会编译 MuJoCo 模型与 JAX kernel 并写入 `~/.cache/dex-frax`（可用 `DEX_FRAX_STATE_DIR` 改），后续批次和同一进程里的后续动作复用这些产物；换机器架构、改依赖、改权重或改 `--ik-block-frames` 都会重新编译。分层与失效规则见 [模型与编译缓存](docs/BATCH.md#模型与编译缓存)。

### 并发、内存预算与复用

`--workers 32` 只设置 **32 个执行槽上限**，不保证同时运行 32 条动作；`auto` 使用当前 CPU 配额及亲和性允许的逻辑核数。实际并发受主机可用内存、cgroup 余量、在途任务增长和编译预留共同限制。修改内存估算不会降低求解器的真实内存用量。

| 参数 | 默认值 | 实际作用 |
|---|---|---|
| `--worker-resident-gib` | `1` | Frax 每槽常驻预算基数，另加逐帧预算 |
| `--worker-peak-gib` | `3.75` | 冷启动基数；超过 resident 的部分作为单独的编译预留，不逐槽重复相加 |
| `--frame-memory-kib` | `32` | 每个 **50 Hz 输出帧** 的估算增量；`16` 对应 16,384 字节 |
| `--memory-target-percent` | `90` | dynamic 模式下整台主机的占用目标，包括系统和其他程序 |
| `--memory-reserve-gib` | 未指定 | dynamic 模式的额外最小保留；与占用目标推导的保留量取较大值；manual 默认保留 10 GiB |
| `--memory-limit-gib` | 未指定 | 收紧 Linux 批次 cgroup 的内存硬上限，不预先分配这些内存 |

未积累足够内存样本时，Frax 单槽预算为 `resident × GiB + 输出帧数 × frame_memory_kib × 1024`，编译另预留 `max(0, worker_peak − resident) × GiB`。例如 `--worker-peak-gib 2.1 --frame-memory-kib 16` 且未指定 resident 时，单槽基数仍是 **1 GiB**，编译预留为 **1.1 GiB**，不能按 `2.1 × 32` 推算。默认开启的内存学习会使用已完成动作的阶段峰值更新估算；运行时策略文件也可覆盖估算，详见 [资源设置](docs/BATCH.md#资源设置)。

dynamic 模式实际主机保留量为 `max(总内存 × (100 − target) / 100, 显式 reserve)`，紧急底线为该保留量的一半。例如总内存约 62.4 GiB、target=95、reserve=2 时，实际保留约 **3.12 GiB**，紧急底线约 **1.56 GiB**。旧版 `emergency_memory_floor` 表示主机可用内存低于底线后整批停止，并不等于内核 OOM；当前批处理按下述流程降并发、排队恢复。

**复用默认开启，无需额外开关。** 调度器保留热 worker、连续处理后续动作，复用进程内模型与 kernel；磁盘上的 MJB/JAX 编译缓存可跨进程复用编译产物。各 spawn 进程仍有独立的求解工作内存，缓存不等于跨进程共享整份 JAX 堆。内存学习仅改估算，不等于释放内存。提高预算会限制准入，降低预算可能放入过多任务，两者都不是实际节省内存的方法。

是否满载应查看 `batch.running.json` 的活动任务、阶段、`worker_pss_bytes`、`sampled_resource_peaks` 和等待原因，并结合 CPU 采样；不要只看执行槽数或累加各进程 RSS。不同模式（旧 `auto` 与当前 `refine`）、动作长度、修复阶段及版本的峰值不能直接互套。新运行使用新输出目录；不要为解决内存问题删除已有转换结果或缓存。

### 资源不足时自动排队

批处理默认持续排队到资源恢复或总预算耗尽，普通内存不足不再直接终止整批：

1. **先校准，再扩并发。** 开启内存学习时，同模式、参数及来源尚无足够样本，会依次跑完两条完整动作，包含审计、修复与导出，再按完整流程峰值增加并发。`fast` 与 `refine` 的学习记录隔离；不能用初解阶段的低占用替代完整峰值。
2. **内存紧张先停派发。** 优先保存已完成结果、回收空闲 worker，运行中的任务继续；资源恢复后自动派发队列中的动作。
3. **持续增长时只回退必要的单条任务。** 到达紧急底线或 cgroup 余量过低时，逐次释放一个在途 worker，将该动作标记为 `memory_retry` 放回队列，并提高预算、降低并发；其他任务及已完成结果保留。未完成尝试归档，该动作可能从头重算，不承诺任意帧断点续算。只有确有 cgroup OOM 计数且 worker 被 SIGKILL 时，才按内存问题自动重排；普通程序错误仍单独报告。

状态文件增加 `queued` 和 `resources.memory_queue`，等待原因区分 `memory_calibration_wait`、`memory_pressure_wait`、`priority_head_wait:memory_reserve`。默认 `--resource-wait 0` 表示等待受总 `--budget` 限制；manual 模式显式设置正值才限定空闲等待时间。`--no-memory-learning` 同时跳过冷启动校准，须自行给出覆盖审计与修复峰值的预算，但仍保留压力回收和重排。

总预算、用户 STOP/信号、磁盘保护、身份校验与真实系统硬限制仍有效；若操作系统连调度器也杀掉，需要核验后 `--resume`。新调度逻辑不会热替换已经运行的旧进程，也不改变求解参数或验收门槛。详细边界见 [批处理说明](docs/BATCH.md#内存不足时的队列与恢复)。

## 检查范围

质量检查覆盖配置中的限位、限速、自碰撞、地面、支撑、抖动及姿态指标。碰撞基于声明的凸几何与帧间有限采样；同侧脚—小腿在踝 pitch、roll 均处于模型限位内时按既定规则排除，跨腿和双脚检查保留。

`accepted` 的具体范围以 `quality.json` 为准。腿形、前臂轴向和根位置等部分相似度默认仅记录指标，未设硬门槛；本项目输出尚不构成动力学平衡、控制器跟踪或真机验证结果。

## 文档

- [输入、模型与输出格式](docs/INPUTS.md)
- [转换流程与验收标准](docs/PIPELINE.md)
- [批处理、资源与续跑](docs/BATCH.md)
- [修复与回归规则](docs/SAFETY_REPAIR_RULES.md)
- [版本与验证记录](docs/CURRENT_VERSION.md)
- [第三方来源与许可](THIRD_PARTY.md)
