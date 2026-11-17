# 批处理与资源管理

批处理仅接收 BVH；SMPL-X 使用单文件 `convert`，见 [输入规范](INPUTS.md)。

## 创建清单

在项目根目录执行，按实际数据修改 `root`、`source` 和文件匹配规则：

```sh
.venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path('examples/bvh/nobody').resolve()
source = 'mocap'  # LAFAN1 用 lafan，SOMA Uniform 用 soma
inputs = sorted(root.rglob('*.bvh'))
if not inputs:
    raise SystemExit('没有找到 BVH，请检查目录与匹配规则')
cases = [dict(id=f'motion_{i:06d}', input=str(p), source=source,
              input_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
         for i, p in enumerate(inputs)]
Path('manifest.json').write_text(
    json.dumps(dict(cases=cases), ensure_ascii=False, indent=2), encoding='utf-8')
PY
```

每条记录包含唯一 `id`、`input`、`source` 和真实 `input_sha256`。输入路径可为绝对路径或相对于清单的路径。`id` 以字母或数字开头，仅允许字母、数字、点、下划线和连字符。

调度器校验全部输入哈希，按文件字节去重，别名写入 `source_aliases.json`；相同内容声明不同来源会报错。一个批次使用相同的 BVH 单位与位置通道约定，不同约定应分批。

## 启动与预算

```sh
.venv/bin/dex-frax batch manifest.json outputs/batch --budget 604800
```

首次运行的输出目录必须不存在。默认 `--mode fast --workers auto`，按动作时长从短到长处理；`--mode refine` 才进行窗口修复。待审动作保留完整输出，通过项进入 `accepted.json`。

| 参数 | 默认或用途 |
|---|---|
| `--budget` | 批次总预算，默认 1800 秒；耗尽时终止在途任务 |
| `--frax-timeout` | 单条动作时限，单位秒；该条超时不终止其他动作 |
| `--repair-budget` | 每条精修的局部修复预算；结束后保存待处理记录 |
| `--foot-contact` | 足部接触规则：`v1` 或 `v2`，默认 `v2`；整批统一，不支持逐条混用 |
| `--workers` | 最大并发，默认 `auto`；实际并发还受内存准入限制 |
| `--completion-worker-gib` | 一条受保护任务的完整常驻预算基数，另加逐帧项；默认冷启动基数或更大的已测完整峰值 |
| `--disk-reserve-gib` | 默认预留 80 GiB；低于底线时停止批次 |
| `--resource-wait` | 默认 `0`，等待受总预算限制；manual 模式显式设置正值才限制无在途任务时的等待时长 |

`refine` 未指定单条时限时，受批次剩余总预算限制。需要保持原节奏时，不启用 `--allow-local-slowdown`；其含义见 [时间映射](PIPELINE.md#帧率与时间映射)。

以全部可用逻辑核为并发上限的精修批次（实际并发仍受内存准入限制）：

```sh
.venv/bin/dex-frax batch manifest.json outputs/batch \
  --mode refine --budget 1800 --workers auto \
  --frax-timeout 600 --repair-budget 240
```

- `--workers auto` 等于 CPU 配额及亲和性允许的逻辑核数；32 核时等价于 `--workers 32`，两者都不保证同时启动 32 个任务。
- `--frax-timeout` 建议显式给出：精修未指定时单条动作时限等于批次剩余预算，第一条难动作可能把 1800 秒用完；超时动作保留部分输出并记为待审，不中断其他动作。
- `--repair-budget` 只计局部修复那一段，不含初解和最终整片审计。

## 停止与续跑

```sh
touch outputs/batch/STOP
```

也可向调度器发送 SIGTERM。等待进程退出后，在原输出目录续跑：

```sh
rm -f outputs/batch/STOP
.venv/bin/dex-frax batch manifest.json outputs/batch --budget 604800 --resume
```

续跑须使用相同清单及匹配的转换、资源参数。程序核验输入、实现、运行资产、依赖和完成产物哈希，保留完成结果并归档未完成尝试。版本或转换规则改变时使用新输出目录。

批次 `--resume` 不会自动继续已完成但待审的条目。对此类动作，使用 `convert --mode refine --continue-from outputs/batch/<id>/frax`，提供相同原输入及参数并写入新目录，见 [检查点与续跑](PIPELINE.md#检查点与续跑)。

磁盘不足以 `resource_exhausted` 停止后，释放空间再手动续跑。正常停止会保存检查点；强杀或断电可能丢失最近约 30 秒状态。

## 资源设置

- **CPU：** 默认以上下文允许的逻辑 CPU 数为并发上限，每进程单线程并在支持的平台绑定单核；编译、I/O 和内存会影响实际利用率。
- **内存：** 默认 `--resource-mode dynamic --memory-target-percent 90`，按系统及其他程序合计占用控制准入，并计入完整动作峰值、编译峰值与在途增长。内存不足时在总预算内等待；紧急压力下回收空闲进程，必要时逐条重排在途动作，不直接终止整批。
- **Linux：** 多进程需要 `systemd-run --user` 建立 cgroup 内存硬限制并禁用交换，无法建立时拒绝启动；可用 `--memory-limit-gib` 收紧上限。macOS 使用准入和紧急停止。
- **固定资源：** `--resource-mode manual` 默认预留主机内存 10 GiB；`--resource-wait 0` 默认同样等待至资源恢复或总预算耗尽，显式正值才限制空闲等待时间。Linux 硬上限不再按启动瞬间的剩余内存冻结；主机保留量和继承的 cgroup 限制仍约束后续准入。

小批次可按实际空间设置 `--disk-reserve-gib 5`，大批次按预期输出量留足余量。状态目录默认为 `~/.cache/dex-frax`，可通过 `DEX_FRAX_STATE_DIR` 指定；同目录锁避免重复占用资源。

启动后生成 `resource_policy.json`，约每 2 秒检查更新。可调整 `workers`、`memory_learning`、`compile_extra_gib`、`stages.frax`、`completion_worker_gib` 与 `phase_bases_gib`；并发不能超过启动上限。`phase_bases_gib` 可分别设置 `loading`、`preprocessing`、`initial`、`local_repair`、`audit`、`export_check`、`export` 等阶段的基数，逐帧项仍按该模式的 `bytes_per_output_frame` 增加；缺少某阶段时使用模式基数。修改资源策略不改变求解参数或验收门槛。完整选项见 `dex-frax batch --help`。

### 预算计算与压力判断

内存参数的默认值与含义见 [README：并发、内存预算与复用](../README.md#并发内存预算与复用)。Frax 初始单槽预算为 `worker_resident_gib × GiB + 输出帧数 × frame_memory_kib × 1024`，编译增量为 `max(0, worker_peak_gib − worker_resident_gib) × GiB`，作为单独预留，而非每槽重复预留。默认 resident=1、peak=3.75、frame=32；`--worker-peak-gib 6.5 --frame-memory-kib 48 --memory-reserve-gib 8` 是手动配置，不是默认值，也不保证满并发。

内存学习默认启用，第一条完整结果即可抬高预算；降低估算至少需要两条完整样本。此前按初始估算并行启动，不再强制串行校准。学习身份包含完整转换契约，因此模式、足部规则、修复预算等配置不会混用旧样本。阶段预测使用对应阶段的完整实测峰值样本，根据长度拟合保守包络，而不是平均内存。改源码、依赖或执行身份可能使历史学习值失效。`resource_policy.json` 可覆盖初始预算，因此核对实际运行时要同时查看启动参数、策略文件和 `resources.memory_learning`。

### 一条完成通道与阶段许可

一条任务始终按整批各长度完整预算的最大值预留，其他任务按各自长度预测的常规完整峰值记账；该估算取配置各阶段或已测完整峰值的上界，避免只按加载时的低占用塞进过多进程。编译增量只单独预留一次。每条动作都按其 50 Hz 输出帧数计算增量，不能把同一个固定 GiB 数作为所有长度的总预算。

在加载模型、初解、审计、修复与导出等安全入口，worker 先发送预算请求，收到许可才执行下一段。已有任务扩容优先于新任务启动，受保护任务优先获得许可。编译申请在取得共享编译锁之前等待，不在持锁的 `ik_prepare`/`ik_compile` 通知处阻塞。当前阶段中的任意库内部分配仍不能逐次拦截，超出预测时保留压力回退兜底。

受保护任务完成后，从已有在途任务中交接完成通道。任务在安全点等待扩容时，归还上一阶段未使用的额度，只保留实测驻留内存与 128 MiB 余量；受保护的完整预算不会因此撤销。获准运行的任务仍保留常规完整预测，避免所有任务的正常审计都要重新争抢额度。热 worker 的实际驻留内存仍计费，只有重新采样确认释放后才能归还容量。如果热缓存挡住完成通道，优先回收该空闲进程。单条等待时间从计算超时中扣除，仍计入总 `--budget`。

`batch.running.json` 分别提供 `runnable`、`waiting_memory`、`queued`；单条提供已获预算、请求预算、完成预算和内存等待时间。`resources.stage_memory` 标明受保护动作与许可计数，每约 30 秒在标准输出打印运行/等待/排队/完成量。持有进程但等待许可的动作不应算作正在计算。

准入时，剩余空间取 `min(主机 MemAvailable − 实际保留量, cgroup 可用余量)`，再与在途任务预计增长、新任务增量及单独编译预留比较；实际保留量在 dynamic 模式下为 `max(主机总内存 × (100 − target) / 100, 显式 reserve)`。活跃任务的真实占用可能超过初始估算，调度器不会仅因停止新增任务就自动缩小这些任务的工作集。

旧版 `emergency_memory_floor` 只表示主机 `MemAvailable` 小于实际保留量的一半，不等于内核/cgroup OOM。当前低内存行为见下一节。PSS 用于减少共享页的重复计数；`worker_pss_bytes` 不包括整个作用域所有开销。`sampled_resource_peaks` 为采样峰值，不能当作连续内核计量。结合阶段和多次 CPU/内存采样判断稳定负载，不用启动瞬间或一条短动作的平均值承诺整批 32 路峰值。

### 内存不足时的队列与恢复

| 条件 | 自动处理 |
|---|---|
| 尚无两条同配置、同来源的完整内存样本 | 按初始估算并行填槽，同时为一条任务保留较大完成预算；首批本身就是正式转换 |
| 下一阶段额度不足 | `stage_memory_wait`，该 worker 保留进程原位等待，其他已获许可的任务继续运行 |
| 准入预算不足或主机可用内存低于保留量 | 停止派发，接收完成结果，优先回收空闲 worker；`memory_pressure_wait` 或 `priority_head_wait:memory_reserve` |
| 主机可用内存低于保留量一半，或 cgroup 余量低于硬上限的 1%（最低 32、最高 256 MiB） | 先回收空闲进程，再逐次终止必要的单条在途动作并放回队列；每次重新采样，不预先重复计入释放量 |
| worker 被 SIGKILL 且本批次 cgroup 的 OOM kill 计数增加 | 该条按 `cgroup_oom_retry` 重排；没有 OOM 证据的程序退出仍报告失败 |
| 连一条动作的估算都放不下 | 保持队列、报告候选所需预算，等待容量恢复或总预算结束，不立即反复启动同一条动作 |

重排记录为 `memory_retry`，部分输出与日志归档，未完成的 `training.npz` 不进入 accepted 清单；已通过结果继续保留并校验哈希。重试的是当前动作，可能从该动作开头重算，不能把批次重排当成任意帧断点续算。

发生回退后，调度器提高同来源的内存估算下限（按峰值增加 25% 及 128 MiB 余量，并保留逐帧项），并下调并发上限；这些信息保存在 `resources.memory_queue`，续跑仍保留。每完成两条完整动作，上限可回升一槽，实际派发始终还受内存准入约束。实时降低策略文件里的预算不会清掉这层已观察到的压力下限。

`--no-memory-learning` 关闭估算学习，使用配置的阶段和长度预算；阶段许可、完成通道、压力回收、单条重排仍生效。若估算被实际分配突破且所有 worker 都等住，会优先回退一条非受保护动作来恢复通道；单条任务独占资源仍放不下时，等待本身不能解决容量不足。总预算计入排队和重试时间，不会自动无限延长。磁盘不足、STOP/信号、身份校验失败仍按各自规则处理；若硬 OOM 杀掉调度器本身，进程内队列无法自救，须检查现场后使用 `--resume`。不要删除输出目录来恢复。

## 模型与编译缓存

首次运行的时间主要花在 MuJoCo 模型编译和 JAX kernel 编译上；批处理会把产物缓存下来，供后续批次和后续动作复用。全部位于状态目录，默认 `~/.cache/dex-frax`，可用 `DEX_FRAX_STATE_DIR` 指定。

| 缓存 | 位置 | 内容 |
|---|---|---|
| MuJoCo 二进制模型 | `models/<key>/model.mjb` | 由 `assets/dex_evt/mjcf/dex_29.xml` 编译，加载前逐数组与 XML 版比对；`key` 由资产哈希与 mujoco 版本决定 |
| JAX 持久编译缓存 | `jax_cache/` | 固定计算块的编译产物，跨进程复用 |
| 编译锁 | `compiler/` | worker 间串行编译，避免重复占用，进程退出自动释放 |

- MuJoCo 模型缓存只在批处理启用：调度器编译一次并校验，spawn worker 直接加载二进制模型，各自持有独立 MjData，不 fork 已初始化的 JAX。单文件 `convert` 仍按 XML 解析，但共用同一份 JAX 持久缓存。
- 进程内热缓存只作用于同一进程的后继动作：kinematic model 最多 4 份，编译 kernel 最多 `DEX_FRAX_KERNEL_CACHE_SIZE`（默认 2）份。批处理 worker 连续处理多条动作，通常从第二条起不再编译。
- 热 worker、MJB/JAX 缓存和内存学习默认启用，无需额外的“复用内存”参数。各 spawn 进程仍各自持有求解工作内存；磁盘缓存共享编译产物，不等于所有进程共享 JAX 堆。删除输出目录不会减少运行中任务的工作集，还会丢失结果与批次检查点。
- 无完整内存样本时按初始估算并行，不再串行完成两条校准。编译锁只允许一个 worker 编译；开始阶段的 CPU 利用率偏低可能来自模型加载、编译或阶段预算等待，应结合运行数和等待数判断。
- 计算块长度固定为 `--ik-block-frames`（默认 128）才能复用编译结果；改动该值等于换一套缓存。
- 缓存按内容身份匹配，不做跨机器迁移：换 CPU 架构、换依赖版本、改权重或改块长都会重新编译，属于预期行为。
- `models/<key>/receipt.json` 记录二进制哈希、参与比对的数组和生成条件；校验不通过会直接报错，不会静默重新编译。需要强制冷启动时，删除状态目录下对应的 `models` 或 `jax_cache` 子目录即可，两者互不影响。

## 状态与结果

| 文件 | 用途 |
|---|---|
| `batch.running.json` | 约每 2 秒更新阶段、并发、资源与等待原因 |
| `batch.json` | 约每 30 秒及停止时保存完整检查点 |
| `accepted.json` | 通过项的训练文件相对路径及 SHA-256；下游读取 `motions` |
| `<id>/selection.json` | 该动作的最终结果选择 |
| `<id>/frax/`、`<id>/frax.log` | 动作、报告和日志，模式见 `profile` |

调度 `complete` 与动作 `accepted` 分开记录。所有动作均完成时，`accepted` 和 `review` 都返回退出码 0。批次因资源、预算、中断或执行失败未完成时返回非零，具体原因见 `batch.json`。精修的候选与检查点保存在动作目录的 `repair.json` 和 `refine_state.json`。

## 对照回归

回归目录使用同样的 `manifest.json`，每次运行结果保存在 `results/<label>/`。

```sh
.venv/bin/python scripts/run_regression.py suite --label revised \
  --source-root src --assets assets/dex_evt --modes fast refine
.venv/bin/python scripts/compare_regression_runs.py suite \
  --baseline previous --candidate revised
```

对照工具也接受放在 `results/` 下的两个批次输出目录。逐帧比较前会核对输入、关节顺序、输出帧率、源时间映射和验收口径。碰撞统计读取全身检查报告。
