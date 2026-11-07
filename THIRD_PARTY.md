# Sources and licenses

- Dex-Retarget uses [`frax`](https://github.com/StanfordASL/frax) 0.2.0 from
  Stanford ASL, locally installed from the nested `Dex-StanfordASL-frax/`
  checkout (base commit `f5e87ab59a0b82ba43095e882c1b88c9b0dacf07`).
  This local upstream checkout is excluded by `.gitignore` and must be fetched
  separately when installing a fresh copy of Dex-Retarget.
  MIT notice, copyright 2026 Daniel Morton: [docs/FRAX_LICENSE](docs/FRAX_LICENSE).
  Each conversion hashes the imported frax Python sources.
- `src/dex_frax/v8` and the pose/velocity audit modules derive from the
  local `Dex-SuperMimic` V8 conversion component. The inherited project license
  is preserved in `LICENSE`. The inherited release is
  `V8_cpu_coupled_arms_20261105`; source provenance is recorded in
  [provenance.json](src/dex_frax/v8/provenance.json).
  This project modifies the conversion adapters, arm optimizer and collision
  recovery. The copied V8 sources are not identical to the original snapshot.
- `assets/dex_evt` is copied without geometry changes, including its own
  `LICENSE`, manifest, original URDF, derived URDF/MJCF and meshes. Robot assets
  are distinct from the frax and conversion-code licenses.
- GMR and NVIDIA notices remain under `src/dex_frax/v8/reference`.
- `lafan1_surfaces.json` and `soma_surfaces.json` are copied for this local
  authorized workflow from the V8 reference directory. They derive from human
  reference geometry and retain the source-specific terms described in
  [SOURCE_GEOMETRY.md](src/dex_frax/v8/reference/SOURCE_GEOMETRY.md).
  They are excluded from Git; do not interpret
  the software license as permission to redistribute these tables or datasets.
- Example inputs under `examples/inputs/` are byte-for-byte copies of the
  original motion files. The BVH directory's `manifest.json` records source
  paths, sizes and SHA-256; the SMPL-X directory's `说明.txt` lists file contents
  and SHA-256. Original inputs are not changed. Generated
  motions, reports and previews stay under ignored `outputs/`.

The wheel includes conversion code but does not bundle the 220 MiB robot asset
directory. For wheel installation set `DEX_FRAX_ASSETS` to the preserved asset
directory. Human reference tables must be supplied under their own terms.
