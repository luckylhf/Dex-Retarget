# 第三方来源与许可

代码、机器人资产、人体模型和示例数据分别遵循各自许可。

## Frax

使用 [Stanford ASL / frax](https://github.com/StanfordASL/frax) 0.2.0，安装说明指定参考提交 `f5e87ab59a0b82ba43095e882c1b88c9b0dacf07`。本地依赖目录为 `Dex-StanfordASL-frax/`，不随本项目 Git 仓库提交；交付目录未包含时按 README 获取。

MIT 许可及版权声明见 [FRAX_LICENSE](docs/FRAX_LICENSE)。每次转换记录实际导入的 Frax Python 源码哈希。

## 转换与检查组件

`src/dex_frax/v8` 及部分姿态、速度检查模块源自 Dex-SuperMimic 的 V8 转换组件。继承版本为 `V8_cpu_coupled_arms_20261105`，来源记录见 [provenance.json](src/dex_frax/v8/provenance.json)，继承许可保留于 [LICENSE](LICENSE)。本项目对输入适配、求解和修复作了修改，当前代码与原始快照不同。

GMR 和 NVIDIA 的声明保留于 `src/dex_frax/v8/reference/`。

## 机器人资产

`assets/dex_evt/` 包含独立的 `LICENSE`、来源清单、URDF、MJCF 和网格。机器人资产的许可独立于 Frax 和转换代码。Python wheel 与源码分发包不包含该资产目录，安装时须另备并设置 `DEX_FRAX_ASSETS`。

## 人体参考几何与模型

`lafan1_surfaces.json` 和 `soma_surfaces.json` 是源人体参考几何表，不随 Git 或 wheel 分发。来源及适用条款见 [SOURCE_GEOMETRY.md](src/dex_frax/v8/reference/SOURCE_GEOMETRY.md)。使用者须按对应来源条款获取，软件许可不覆盖这些表或原始数据集的再分发。

SMPL-X 人体模型参数不随项目分发，须从 [SMPL-X 官方网站](https://smpl-x.is.tue.mpg.de/) 按其许可获取。

## 示例数据

`examples/` 中的人体输入保留原始文件字节；`*_50Hz-Retarget.npz` 是本项目对该输入生成的机器人参考输出，只用于对照。示例数据不代表当前版本的转换或验收结论；数据使用与再分发遵循原始来源许可。
