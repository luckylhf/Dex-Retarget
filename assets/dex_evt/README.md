# Dex 29 关节机器人资产

本目录随 Git 分发，包含匹配的 `model.json`、`mjcf/dex_29.xml`、`meshes/`、`meshes_visual/`、`meshes_convex/`、`urdf/` 及原始 `LICENSE`。使用前保留原始许可；资产使用 OpenAtom Open Hardware License 1.0，不能用本仓库代码的 MIT 许可替代。

来源：[Dex-Video2Robot](https://github.com/luckylhf/Dex-Video2Robot) 的 `Dex-GMR/assets/dex`，或既有已核验的相同资产目录。

- 模型 XML SHA-256：`17573c55d1c9d9d43b2e44a5e3ebc237027bfcfb6f5b5d928aa0205792548194`。
- `model.json` 的 `model_sha256` / `urdf_sha256` 用于转换、训练、回放和缓存身份校验；模型不匹配时应重新转换、训练和验证，不要只改 JSON 哈希绕过检查。
- 转换、训练和回放默认使用本目录；可用 `DEX_ASSETS=/absolute/path/to/assets` 指向外部副本。

如需刷新为外部已核验资产，先清空本目录中除 `README.md` 外的内容，再运行：

```bash
python scripts/import_assets.py --source /path/to/Dex-Video2Robot/Dex-GMR/assets/dex
```

脚本核对 12341 XML 指纹并编译检查，然后复制 `model.json`、`mjcf/`、`meshes/`、`meshes_visual/`、`meshes_convex/`、`urdf/`、`LICENSE`；目标目录已有资产时脚本会拒绝覆盖。
