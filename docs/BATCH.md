# 批处理与资源管理

批处理只接收 BVH。单条 SMPL-X 动作使用 `convert`，见 [输入规范](INPUTS.md)。

## 生成输入清单

在项目根目录执行以下脚本，修改 `root` 和 `source`：

```sh
.venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path('examples/inputs/bvh/nobody').resolve()
source = 'mocap'  # LAFAN1 用 lafan；SOMA Uniform 用 soma
inputs = sorted(root.rglob('*without_calibration.bvh'))  # 自有数据可改为 *.bvh
if not inputs:
    raise SystemExit('没有找到 BVH，请检查目录与文件匹配规则')
cases = [dict(id=f'motion_{i:06d}', input=str(p), source=source,
              input_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
         for i, p in enumerate(inputs)]
Path('manifest.json').write_text(json.dumps(dict(cases=cases), ensure_ascii=False, indent=2))
PY
```

清单结构：

```json
{
  "cases": [
    {
      "id": "walk_001",
      "input": "/path/to/walk.bvh",
      "source": "lafan",
      "input_sha256": "该文件的真实 SHA-256"
    }
  ]
}
```

`input` 可为绝对路径或相对于清单的路径。`id` 必须唯一，以字母或数字开头，仅含字母、数字、点、下划线和连字符。批处理校验全部输入哈希，按完整 BVH 字节去重，重复项写入 `source_aliases.json`；相同内容指定不同来源会被拒绝。

## 启动、停止和续跑

```sh
.venv/bin/dex-frax batch manifest.json outputs/batch --budget 604800
```

输出目录必须是新目录。默认 `--mode auto --workers auto`：所有动作（包括不少于 60 秒的长动作）先跑完整 Frax；整批 Frax 结束后，仅未通过项从人体输入独立跑完整 V8，包含重新初解和全部 V8 后处理，不复用 Frax 轨迹。Frax 短动作优先，V8 长动作优先；时长按 BVH 的 `Frames × Frame Time` 计算。内存不够派发高优先级动作时等待资源；阶段切换前等待剩余 Frax 完成，此时可能暂时不能占满 CPU。

默认保留 80 GiB 磁盘空间。小批次试跑时，若空余空间不足，可显式加 `--disk-reserve-gib 5`；大批次应按预计输出大小留足空间。磁盘空余低于预留量时，dynamic 和 manual 都会终止本批次工作进程并以 `resource_exhausted` 退出；释放空间后，使用同一清单、原输出目录及资源参数加 `--resume` 手动续跑。

`--mode frax` 或 `--mode v8` 可将整批固定为一种完整模式；`hybrid` 已移除。通过即停止升级；输入错误停止该动作，其他未通过项保留各阶段结果。V8 后仍未通过的动作保留待审，安全与质量门槛不变。

总预算默认 **1800 秒**，`--budget` 和 `--frax-timeout / --v8-timeout` 均为秒。

- **总预算耗尽：**终止整个批次的在途工作进程，以 `budget_exhausted` 结束批次。
- **单条阶段超时：**仅终止对应工作进程并记录 `timeout`；其他动作在剩余总预算内继续处理，自动模式可按分级规则尝试该动作的下一阶段。

未完成的结果不会进入验收清单。

运行中停止：

```sh
touch outputs/batch/STOP
```

也可向调度器发送 SIGTERM。等待进程退出后，移除 STOP 并续跑：

```sh
rm -f outputs/batch/STOP
.venv/bin/dex-frax batch manifest.json outputs/batch --budget 604800 --resume
```

续跑校验清单、输入、实现、资产、依赖、转换参数和已完成产物哈希；保留完成阶段，将未完成阶段归档后重试。资产身份包括目录内的 README 等全部文件，运行中及续跑前均应保持原字节。修改求解、审核或导出实现后应使用新目录。不要在运行中覆盖代码或资产文件。此次两阶段流程变更需要新输出目录，不能用调度兼容开关续跑旧三阶段批次；旧产物保留原身份。

仅更改 `managed_batch.py` / `resources.py` 的维护场景可显式设置 `DEX_FRAX_ALLOW_SCHEDULER_RESUME=1`，允许调度兼容续跑。其他身份仍严格核验，差异写入 `scheduler_resume.*.json`，原产物身份保留。

## 资源设置

- CPU：默认以上下文 affinity/cgroup 配额允许的全部逻辑 CPU 为上限，完成即补位；用 `--workers N` 限制并发。每进程单线程、单核绑定，并使用较低调度优先级。实际利用率受编译、I/O 和内存准入影响。
- 内存：默认 `--resource-mode dynamic --memory-target-percent 90`，目标为系统与程序合计约 90%；计入在途增长和可能的编译峰值。默认启用完成任务的内存峰值学习，内存不够时等待，实际压力下回收空闲进程，优先复用存活的热进程。
- 磁盘：默认保留 **80 GiB**，可用 `--disk-reserve-gib` 调整。低于预留量时立即停止本批次，释放空间后手动 `--resume`。主机或 cgroup 紧急内存不足时也停止本批次任务。
- Linux：多进程要求 `systemd-run --user` 建立 cgroup 内存硬限制并禁用交换；无法建立时拒绝启动。`--memory-limit-gib` 可显式收紧硬上限。macOS 使用内存准入和紧急停止，无此 cgroup 硬限制。
- 固定资源对照：`--resource-mode manual` 默认预留主机内存 10 GiB，无任务在途且内存准入持续不足时按 `--resource-wait` 退出；dynamic 在总预算内等待内存准入恢复。磁盘底线与紧急内存保护均直接停止批次。

默认状态目录为 `~/.cache/dex-frax`，可用 `DEX_FRAX_STATE_DIR` 指定。批次锁防止同一状态目录重复占用资源；共享 JAX 缓存应仅允许本用户写入。完整参数见 `dex-frax batch --help`。

启动时生成 `resource_policy.json`，也可用 `--resource-policy` 指定已有文件。每 2 秒检查修改，可调整 `workers`、`memory_learning`、`compile_extra_gib` 和 `stages.frax`、`stages.v8` 的内存估算；`workers` 不得超过启动上限。建议在生成的完整文件上修改并原子替换，非法配置保留最近有效值。热加载不改变求解参数、质量门槛或 cgroup 硬限制。

## 状态和结果

| 文件 | 用途 |
|---|---|
| `batch.running.json` | 约每 2 秒更新的阶段、并发、资源及等待原因 |
| `batch.json` | 每 30 秒及停止时保存的完整检查点、尝试与耗时 |
| `accepted.json` | 通过的训练文件、相对路径及 SHA-256 |
| `<id>/selection.json` | 该动作的最终状态、选中模式及路径 |
| `<id>/<mode>/` 和 `<id>/<mode>.log` | 自动模式各阶段产物和日志；固定模式产物直接在 `<id>/` |

下游读取 `accepted.json` 的 `motions`，路径相对于批次输出目录。`complete` 表示调度完成，逐动作还需区分 `accepted`、`review`、`failed`、`input_error`、`budget_exhausted`、`interrupted`、`resource_exhausted` 和 `identity_changed`。资源或执行失败不能当作动作质量结论。

强杀或断电可能丢失最后 30 秒检查点。使用 STOP 或 SIGTERM 并等待退出，可保留可续跑状态。
