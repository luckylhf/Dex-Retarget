# 批处理与资源管理

批处理仅接收 BVH；SMPL-X 使用单文件 `convert`，见 [输入规范](INPUTS.md)。

## 创建清单

在项目根目录执行，按实际数据修改 `root`、`source` 和文件匹配规则：

```sh
.venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path('examples/inputs/bvh/nobody').resolve()
source = 'mocap'  # LAFAN1 用 lafan，SOMA Uniform 用 soma
inputs = sorted(root.rglob('*without_calibration.bvh'))
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
| `--workers` | 最大并发，默认 `auto`；实际并发还受内存准入限制 |
| `--disk-reserve-gib` | 默认预留 80 GiB；低于底线时停止批次 |

`refine` 未指定单条时限时，受批次剩余总预算限制。需要保持原节奏时，不启用 `--allow-local-slowdown`；其含义见 [时间映射](PIPELINE.md#帧率与时间映射)。

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
- **内存：** 默认 `--resource-mode dynamic --memory-target-percent 90`，按系统及其他程序合计占用控制准入，并计入编译峰值与在途增长。无可用内存时在总预算内等待，紧急压力下停止批次。
- **Linux：** 多进程需要 `systemd-run --user` 建立 cgroup 内存硬限制并禁用交换，无法建立时拒绝启动；可用 `--memory-limit-gib` 收紧上限。macOS 使用准入和紧急停止。
- **固定资源：** `--resource-mode manual` 默认预留主机内存 10 GiB；无任务在途且持续无法准入时，按 `--resource-wait` 结束等待。

小批次可按实际空间设置 `--disk-reserve-gib 5`，大批次按预期输出量留足余量。状态目录默认为 `~/.cache/dex-frax`，可通过 `DEX_FRAX_STATE_DIR` 指定；同目录锁避免重复占用资源。

启动后生成 `resource_policy.json`，约每 2 秒检查更新。可调整 `workers`、`memory_learning`、`compile_extra_gib` 和 `stages.frax` 内存估算；并发不能超过启动上限。修改资源策略不改变求解参数或验收门槛。完整选项见 `dex-frax batch --help`。

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
