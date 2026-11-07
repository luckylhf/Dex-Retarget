# Dex-Retarget

把人体动作转换为 **Dex 29 关节、50 Hz 的机器人运动学参考**，用于动作预览和后续训练数据准备。支持 LAFAN1、BONES-SEED SOMA Uniform、已适配骨架的 BVH，以及 SMPL-X 人体参数（`.pt` / `.npz` / `.pkl`）。

默认自动尝试求解、接触修正和避碰，再做统一质量检查。通过后才生成训练导出；结果仍需下游动力学跟踪与真机验证。

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

单文件默认依次尝试 `frax → hybrid → v8`，通过即停止。`frax` 生成快速候选，`hybrid` 加入完整后处理，`v8` 独立重新求解。可用 `--mode frax`、`--mode hybrid` 或 `--mode v8` 固定一种模式。

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

自动模式读取输出根目录的 `selection.json` 和 `accepted.json`；固定模式读取 `quality.json`。

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

批处理默认使用全部可用 CPU，受内存和磁盘准入约束；内存目标为系统与程序合计约 90%。小于 60 秒的动作先走 `frax → hybrid`，未通过项和长动作进入 V8。默认总预算只有 1800 秒，长批次需显式调整。SMPL-X 使用单文件入口。

默认磁盘保留量为 **80 GiB**。小批次试跑且空余空间不足时，可加 `--disk-reserve-gib 5`；按实际输出规模设置余量。低于预留量会停止批次，释放空间后手动 `--resume`。

**下游只读取 `accepted.json` 的 `motions` 列表**，其中路径相对于批次输出目录。停止、续跑、资源限制与实时状态见 [批处理说明](docs/BATCH.md)。

## 进一步说明

- [输入规范与示例](docs/INPUTS.md)
- [批处理、资源与续跑](docs/BATCH.md)
- [第三方来源和许可](THIRD_PARTY.md)

本项目使用 Stanford ASL 的 frax IK 和源自 Dex-SuperMimic 的 V8 转换组件。代码、机器人资产、人体参考表及示例数据各自遵循其许可。
