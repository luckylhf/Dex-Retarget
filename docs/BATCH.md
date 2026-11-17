# 批处理与资源管理

本文按准备、启动、查看进度、停止与续跑的顺序说明操作。批处理仅接收 BVH；SMPL-X 使用单文件 `convert`，见[输入规范](INPUTS.md)。以下命令均在项目根目录执行。文中的 worker 指处理单条动作的工作进程，kernel 指 JAX 编译后的求解计算程序。

## 运行前准备

先按 [README](../README.md#安装) 安装环境，并备齐输入来源需要的文件。LAFAN1 与 SOMA 的参考几何表须另外取得，不能仅克隆仓库就开始这两类转换。

Linux 默认批处理需要可用的 **systemd 用户会话**：程序用 `systemd-run --user` 建立本批次的内存限制、禁用本批次交换；无法建立时会报错退出。SSH 会话报用户总线连接错误时，可先在远端同一用户下检查：

```sh
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
export DBUS_SESSION_BUS_ADDRESS="unix:path=$XDG_RUNTIME_DIR/bus"
systemctl --user show-environment
```

这只能连接已经存在的用户会话；若仍失败，须先修复登录会话或用户服务。macOS 不使用 systemd，也没有 Linux cgroup 硬限制，依靠内存准入、等待及压力下的单条重排。

状态与缓存默认放在 `~/.cache/dex-frax`，可用 `DEX_FRAX_STATE_DIR` 指定。使用同一状态目录的批次互斥；已有批次运行时，第二个批次会报占用错误。该锁不管理其他用户、其他状态目录或单独启动的 `convert`，因此仍须考虑机器上的其他负载。

## 创建清单

按实际数据修改 `root`、`source` 和文件匹配规则：

```sh
.venv/bin/python - <<'PYMANIFEST'
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
PYMANIFEST
```

每条记录包含唯一 `id`、`input`、`source` 和真实 `input_sha256`。输入路径可为绝对路径或相对于清单的路径。`id` 以字母或数字开头，仅允许字母、数字、点、下划线和连字符。

调度器校验全部输入哈希，按文件字节去重，别名写入 `source_aliases.json`；相同内容声明不同来源会报错。一个批次使用相同的 BVH 单位与位置通道约定，不同约定应分批。

## 启动与预算

新批次的输出目录必须不存在。下面两种启动方式择一使用，目录名称可自行选择。

```sh
# fast：完整转换、导出与检查
.venv/bin/dex-frax batch manifest.json outputs/batch-fast --budget 86400

# refine：分层初解、问题窗口修复与整片复查
.venv/bin/dex-frax batch manifest.json outputs/batch-refine \
  --mode refine --repair-budget 1800 --budget 86400 --workers auto
```

默认模式是 `fast`，不会自动升级为 `refine`。两种模式都按源动作时长从短到长排队；这是派发顺序，并行任务的完成顺序可能不同。无须先完成单条试跑或手动启动另一条大内存任务。

程序的启动顺序是：

1. Linux 先建立批次 cgroup，再取得批次资源锁。
2. 校验清单及全部输入哈希，读取 BVH 长度，核验代码、模型和依赖身份。
3. 准备或校验 MuJoCo 二进制模型缓存；此时可能还没有动作进程。
4. 给系统留出内存，为一条任务预留较大的完整流程预算；其余按各自长度估算的完整峰值并行启动，直到 CPU 槽或内存预算用尽。
5. worker 在加载、初解、修复、审计和导出等阶段入口取得许可后继续执行。首次需要的 JAX 编译通过编译锁依次完成；可计算的任务并行推进，结束后复用 worker 处理队列。

前面的校验和模型准备也可能耗时，不能仅凭 CPU 未满判断没有启动。阶段许可和缓存都默认启用。

### 三种时间预算

| 参数 | 计时范围与耗尽行为 |
|---|---|
| `--budget` | 整批调度的总时间，默认 **1800 秒**；包含准备、排队、计算、内存等待和重试，耗尽时停止在途任务，未完成条目留待续跑 |
| `--frax-timeout` | 每条动作的 worker 时限；扣除阶段内存许可等待，其余加载、编译等待、求解、检查均计时。达到时限会终止该 worker，本条尝试记为 `timeout`，其他动作继续 |
| `--repair-budget` | 每条 `refine` 动作的局部修复预算；不含初解和最终整片审计。耗尽后结束局部修复并执行最终检查，仍可能得到完整的 `review` 或 `accepted` 结果 |

示例中的 86400 秒是整批 24 小时，1800 秒是每条局部修复 30 分钟，均为示例值，应按数据量设置。`refine` 未给 `--frax-timeout` 时，单条最多运行到本次批次剩余总预算耗尽；`fast` 默认单条时限为 `max(900, 300 + 0.5 × 输出帧数)` 秒，同样不能越过整批截止时间。

需要限制单条时长时可加 `--frax-timeout <秒数>`，应覆盖加载、初解、修复及整片检查。**超时不等于待审完成**：部分文件可能保留，但不能作为完整训练结果。不设置局部修复预算也不会延长整批预算。需要保持源节奏时，不启用 `--allow-local-slowdown`，含义见[时间映射](PIPELINE.md#帧率与时间映射)。

## 查看进度与结果

运行期间，终端约每 30 秒输出一次计数。也可在另一个终端读取状态；将下面的路径换成实际输出目录：

```sh
.venv/bin/python - <<'PYSTATUS'
import json
from pathlib import Path

state = json.loads(Path('outputs/batch-refine/batch.running.json').read_text())
for key in ('status', 'phase', 'elapsed_s', 'runnable', 'waiting_memory',
            'queued', 'counts', 'resource_wait_reason'):
    print(f'{key}: {state.get(key)}')
for item in state.get('active', []):
    telemetry = item.get('telemetry', {})
    print(item['case'], item['memory_state'], telemetry.get('stage'),
          telemetry.get('phase'), 'pid=', item['pid'])
PYSTATUS
```

| 字段 | 应如何理解 |
|---|---|
| `runnable` | 已派发且没有等待内存许可的任务数；其中仍可能有编译锁等待或 I/O，不等于正在占满 CPU 的进程数 |
| `waiting_memory` | 已经启动、保留进程并等待阶段内存额度的任务数 |
| `queued` | 尚未派发的动作数 |
| `counts` | 结果状态计数；`pending` 包括排队和在途任务，不能当作未启动数 |
| `resources.stage_memory` | 当前受保护动作、预留完成预算及许可计数；预算字段单位为字节 |
| `worker_pss_bytes` | worker 的采样内存合计；Linux 尽量用 PSS 分摊共享页，macOS 使用 RSS，不包含调度器等全部开销 |

初始阶段还没有完整动作结束时，`accepted=0` 是正常现象。查看 `<id>/frax.log` 中推进的帧数及阶段，结合状态更新时间判断进展。`waiting_compile` 表示编译锁等待；`stage_memory_wait` 表示内存许可等待。它们的处理方式见[内存不足时的队列与恢复](#内存不足时的队列与恢复)。

| 文件 | 用途 |
|---|---|
| `batch.running.json` | 调度循环期间约每 2 秒更新资源、阶段与等待原因；较长的前置准备可能使更新时间间隔变长 |
| `batch.json` | 约每 30 秒及正常停止时保存完整检查点，包含逐条尝试和错误 |
| `accepted.json` | 通过项的训练路径及 SHA-256；下游读取 `motions`，路径相对于批次输出根目录 |
| `<id>/selection.json` | 该动作的结果选择 |
| `<id>/frax/`、`<id>/frax.log` | 动作、质量报告与日志 |

批次结束后先看 `batch.json` 的批次状态，再看每条记录及 `<id>/frax/quality.json`：

- `accepted`：完整执行且通过当前运动学检查，训练路径进入 `accepted.json`。
- `review`：完整执行但有质量未通过项，保留动作和报告。
- `failed`、`input_error` 或中断状态：执行没有形成可采用的完整结果，查看 `attempts`、`run.json` 和日志。单条超时位于尝试的 `status=timeout`，动作最终可归为 `failed`。

所有动作都完整执行时，`accepted` 与 `review` 均允许退出码 0；退出码 0 不表示全部质量通过。批次中断、预算/磁盘不足或存在执行失败时返回非零。精修候选与检查点位于动作目录的 `repair.json` 和 `refine_state.json`。

## 停止与续跑

停止当前批次：

```sh
touch outputs/batch-refine/STOP
```

也可向调度器发送 SIGTERM。停止会终止在途 worker 并保存批次状态，**不会等待当前动作完整算完**。等待调度器与其 worker 全部退出、状态变为 `interrupted` 后，再移除 STOP 并续跑。

下面对应前面的 `refine` 示例；若原来是 `fast` 或使用了额外转换选项，应复用自己的原启动命令并加 `--resume`：

```sh
rm -f outputs/batch-refine/STOP
.venv/bin/dex-frax batch manifest.json outputs/batch-refine \
  --mode refine --repair-budget 1800 --budget 86400 --workers auto --resume
```

清单文件须保持字节一致。转换模式、单位、位置通道、足部规则、迭代数、局部减速开关、修复预算和计算块长度等转换设置须匹配；程序还核验代码、依赖、资产与完整产物的哈希。`--budget` 是本次续跑的新总预算，可重新设置；CPU/内存资源设置可以调整，但仍受启动上限、保留量与策略文件约束。版本或转换规则改变时使用新输出目录。

`--resume` 保留已经完成的 `accepted` 和 `review`；中断条目的未完成尝试归档后重新派发，可能从该动作开头重算。它不自动继续待审的局部修复，也不自动重试已记录为 `failed` 或 `input_error` 的条目。处理这些条目时：

- 完整待审动作需继续修复：使用原输入和匹配参数执行 `convert --mode refine --continue-from outputs/batch-refine/<id>/frax`，写入新目录，见[检查点与续跑](PIPELINE.md#检查点与续跑)。
- 输入错误或执行失败：先按日志修正原因，再用新输出目录单独转换，或另建失败条目的新清单；不要改写旧清单后对原批次强行续跑。

磁盘不足以 `resource_exhausted` 停止时，先释放空间再续跑。强杀或断电可能丢失最近约 30 秒的批次状态，须先确认残余进程已退出。保留已有结果和检查点，不需要删掉整个输出目录来恢复。

## 资源设置

这些设置控制 `batch` 的派发和等待；单文件 `convert` 不运行多任务资源调度。完整选项可查 `.venv/bin/dex-frax batch --help`。

内存参数使用二进制单位：1 GiB = 1024³ 字节，1 KiB = 1024 字节。预算是调度器的容量记账，不会预先申请同等大小的物理内存。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--workers` | `auto` | CPU 配额及亲和性允许的逻辑核数作为槽位上限；手动值不能超过它，实际并发还受内存限制 |
| `--resource-mode` | `dynamic` | 根据主机可用内存决定是否派发；`manual` 使用固定主机保留量，但仍检查内存并排队 |
| `--memory-target-percent` | `90` | dynamic 模式的主机占用目标，系统和其他程序的内存也在内 |
| `--memory-reserve-gib` | 未指定 | 额外的主机最小保留量；manual 未指定时保留 10 GiB |
| `--memory-limit-gib` | 未指定 | 收紧 Linux 批次 cgroup 硬上限；它不是预先分配的内存，也不是 macOS 的硬限制 |
| `--worker-resident-gib` | `1` | 普通任务的常驻估算基数，另加长度增量，不代表实际占用恒定为 1 GiB |
| `--worker-peak-gib` | `3.75` | 冷启动估算基数；超过 resident 的部分形成一份编译预留 |
| `--completion-worker-gib` | 未指定 | 受保护任务的完整流程估算基数；未指定时由 resident 与编译预留之和推导，再取阶段预测和完整实测估算的较大值 |
| `--frame-memory-kib` | `32` | 每个 50 Hz 输出帧的估算增量；不是每个原始 BVH 帧 |
| `--disk-reserve-gib` | `80` | 输出磁盘最小剩余空间；低于该值会停止批次，而非继续等待 |
| `--resource-wait` | `0` | 默认等待受总预算限制；仅 manual 模式的正值会限制无在途任务时的资源等待时间 |

批处理使用 CPU 求解，每个 worker 限制数值库线程，并在支持的平台绑定一个逻辑核；先分配各物理核，再使用其 SMT 线程。CPU 忙碌程度仅作为状态信息，不单独降低槽位上限，内存与编译等待仍可能使 CPU 无法满载。

### 按长度估算，而非每条固定内存

没有足够实测样本、也未覆盖策略文件时，普通任务估算为：

```text
普通完整峰值 = resident × GiB + 50 Hz 输出帧数 × frame_memory_kib × 1024
单独编译预留 = max(0, worker_peak − resident) × GiB
```

受保护任务按**整批各动作预测完整预算的最大值**预留，普通任务按自己的长度计算。例如默认参数下，10000 个输出帧的普通预算约为 1.31 GiB，默认受保护预算基数为 3.75 GiB，另加长度增量；还单独留一份 2.75 GiB 的编译预算。完成预算与编译预算是不同预留，不能遗漏其中之一，也不能把编译预算再乘上所有槽位。

若只改 `--worker-peak-gib 2.1 --frame-memory-kib 16`，resident 仍是默认 1 GiB，普通任务基数仍为 1 GiB，编译预留为 1.1 GiB，不能把 `2.1 × workers` 当作总预算。较小的估算会放入更多任务，**不会降低求解器的真实占用**。

输出帧数按 BVH 长度换算到 50 Hz：`floor((源帧数 − 1) × Frame Time × 50) + 1`。这是准入时的长度预测；启用局部减速时，最终输出可能更长，实测增长仍须重新计入。内存学习按来源及匹配的转换身份积累完整动作执行期间的采样峰值：首条完整样本即可抬高估算，降低预测至少需要两条完整样本。不同模式、长度或版本不能直接套用同一个峰值。

### 系统保留量与硬上限

默认 dynamic 模式的实际主机保留量为：

```text
max(主机总内存 × (100 − memory_target_percent) / 100, 显式 memory_reserve_gib)
```

例如主机总内存约 62.4 GiB、target=95、reserve=2 时，实际保留约 3.12 GiB，并非 2 GiB。主机紧急底线为实际保留量的一半；cgroup 还有自己的紧急余量检查。

准入空间取主机可用内存减保留量、以及 cgroup 余量中的较小值。已占用内存已经反映在这些余量中，调度器另外记账的是在途任务尚可能增长的部分、新任务增量和编译预留，避免把整份已占用内存再加一次。

Linux 自动硬上限不高于主机总内存减保留量；显式指定 `--memory-limit-gib` 时，同时受该值、继承的 cgroup 上限和主机紧急保留量约束。它不随启动瞬间的空闲内存固定为一个小值；每次派发仍检查当时主机余量。占用目标是准入目标，不能阻止其他程序抢占内存，也不是每条 worker 的强制内存配额。

### 运行时调整策略

通常先使用命令行参数。需要不中断批次地调整并发或估算时，编辑运行中的资源策略文件，程序约每 2 秒检查一次：

- 默认文件为输出目录的 `resource_policy.json`；用 `--resource-policy /path/to/policy.json` 时，以指定文件为准。
- 文件不存在时按命令行参数生成；文件已存在时保留并读取它，**不会因为重启或续跑就被新的 CLI 估算参数覆盖**。
- 可调整 `workers`、`memory_learning`、`compile_extra_gib`、`completion_worker_gib`、`phase_bases_gib` 和 `stages` 中的估算。并发不可超过本次启动时的 CPU 槽上限；降低上限会停止补充新任务，已有任务继续。
- `phase_bases_gib` 是各阶段的常驻估算基数，长度项仍取 `stages.frax.bytes_per_output_frame`；未采用足够实测样本时，普通完整估算覆盖配置的最大阶段基数。缺少阶段覆盖时使用模式基数。资源策略不修改求解参数与质量门槛。
- 非法修改会被拒绝，保留上一份有效策略；查看输出目录的 `resource_policy_events.jsonl`。建议先修改临时副本，完成后原子替换，避免保存到一半被读取。

`--no-memory-learning` 决定新策略的初始 `memory_learning=false`；已有或显式策略文件中的值优先。它仅关闭峰值学习，不关闭阶段许可与压力重排。实际生效配置查看 `batch.running.json` 的 `resources.resource_policy` 和 `resources.memory_learning`。资源回退形成的估算下限，以及本次已提高的受保护完成预算，不会因调低策略值立即撤销。

### 内存不足时的队列与恢复

在途任务的阶段扩容优先于新任务，受保护任务优先获得许可；它结束后，从在途任务中选择实际占用较高者交接保护。安全点等待时，未用完的预算可归还，当前实际内存仍保留；这只是调整记账，不会释放仍被数组引用的物理内存。等待内存许可时不持有编译锁。

| 情况或等待原因 | 自动处理与用户动作 |
|---|---|
| `priority_head_wait:memory_reserve` | 队首动作的预计预算暂时放不下，等待在途动作释放容量；不需要重启 |
| `stage_memory_wait` | 已启动任务等待下一阶段额度；保留进程和中间结果，其他已获许可任务继续 |
| `memory_pressure_wait` | 主机余量不足，暂停新派发，必要时回收空闲 worker |
| 主机可用内存低于实际保留量一半，或 cgroup 余量低于硬上限的 1%（最低 32、最高 256 MiB） | 先回收空闲进程，再逐次终止必要的单条在途动作放回队列，优先选择非受保护任务，每次重新采样 |
| worker 被 SIGKILL，且本批次 cgroup 的 OOM kill 计数增加 | 该条按 `cgroup_oom_retry` 重排；没有 OOM 证据的退出按执行失败报告 |
| 所有任务因低估峰值而等住 | 必要时回退一条非受保护动作，给其他任务腾出容量 |
| 连一条完整预算也放不下 | 保持队列并报告所需预算，等待外部内存释放或总预算结束；若单条实际需求就超过机器容量，单纯排队无法解决 |

重排尝试记为 `memory_retry`，部分输出与日志归档，该动作可能从头重算；已通过结果保留。调度器提高同来源的估算下限并降低并发上限，每完成两条完整动作可回升一槽，实际派发仍受准入约束。

阶段申请不能拦截 NumPy/JAX 内部任意分配，预留的最大值也只是当前预测，不能保证绝对零 OOM。如果操作系统杀掉了调度器本身，就无法自动恢复队列；需确认进程状态后手动 `--resume`。旧日志中的 `emergency_memory_floor` 表示主机可用内存低于保留量一半，不等于已经发生内核 OOM。

## 模型与编译缓存

**缓存默认启用，无须额外的“复用内存”开关。** 应区分下面几类：

| 内容 | 复用范围 | 是否共用动作工作内存 |
|---|---|---|
| MuJoCo 二进制模型 `models/<key>/model.mjb` | 批处理在状态目录保存一次，后续 worker 加载同一文件；每个进程仍有自己的模型对象与 MjData | 否 |
| JAX 持久编译缓存 `jax_cache/` | 同一状态目录中的兼容进程和后续批次复用磁盘编译产物，单文件转换也可使用 | 否；每个进程仍需加载可执行对象和分配自己的求解数组 |
| worker 内存中的模型与已编译 kernel | 同一个常驻 worker 连续处理后续动作时复用；不跨进程合并整个 JAX 堆 | 仅在该 worker 内复用相关对象，不共享不同动作的求解结果 |
| 内存峰值学习 `memory_profiles/` | 相同转换身份的后续任务据此改进预算估算 | 否；这是统计数据，不是动作缓存或内存池 |

操作系统可能共享库或文件页，Linux 使用 PSS 尽量避免重复计数；这不改变独立 worker 各自需要工作内存的事实。一个 4 GiB 工作集的任务启动 32 份，不能按“一份模型”预期总共只用 4 GiB。

首次批处理先准备 MuJoCo 模型，再在 worker 需要时加载或编译 JAX kernel。编译锁同一时刻只让一个 worker 进入受控编译段，其他已经具备执行条件的任务继续计算。即使已有磁盘缓存，各进程也需要加载和初始化时间，不能据此保证启动即满载。

空闲 worker 通常保留以便处理下一条；若实际内存压力或受保护任务的预算需要容量，调度器会回收空闲进程。缓存占用仍计入实际内存，只有重新采样确认释放后才归还容量。

缓存按资产、编译内容及依赖等身份匹配；换模型、CPU 架构、依赖或 `--ik-block-frames`（默认 128）可能需要新编译，后继动作也不保证永远不再编译。MuJoCo 二进制在首次生成时与 XML 模型逐数组校验，后续按记录的哈希验证；不匹配时报错，不静默采用坏缓存。

排查损坏缓存时，先等相关批次退出，再处理对应缓存目录；不要在运行中删除状态目录、锁或模型缓存。为内存不足删除输出文件不会缩小运行中进程的工作集，却会破坏结果和续跑检查点。

## 对照回归

回归目录使用同样的 `manifest.json`，每次运行结果保存在 `results/<label>/`。

```sh
.venv/bin/python scripts/run_regression.py suite --label revised \
  --source-root src --assets assets/dex_evt --modes fast refine
.venv/bin/python scripts/compare_regression_runs.py suite \
  --baseline previous --candidate revised
```

对照工具也接受放在 `results/` 下的两个批次输出目录。逐帧比较前会核对输入、关节顺序、输出帧率、源时间映射和验收口径。碰撞统计读取全身检查报告。
