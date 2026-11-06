# Source reference geometry

lafan1_surfaces.json is derived locally from Ubisoft's official fbx/model_skeleton.fbx (SHA256 3d9349b5ec686248be6ab73ea3b918f486016a0d22e4b4b80d5418c8f3a608aa). Bone-local surfaces use dominant skin weights, with extra fingers folded into the original BVH Hand. Shoe landmarks and the sole normal are fixed reference-surface measurements, not fit per motion.

soma_surfaces.json is derived locally from NVIDIA soma_base_skel_minimal.usd (SHA256 4074b8d830f53af246b4d93d1871af38c8825e6b6b6bbaf034d4612673c89266), using its geometry bind and skeleton bind transforms. Official SOMA foot landmark configuration remains separate and unchanged.

These are rigid per-bone convex-envelope approximations of the reference skins. They do not contain force labels or exact deforming skin. Original dataset/model terms and the existing NVIDIA notices remain applicable. The original FBX/USD and extraction tools are retained in the research workspace; runtime uses the frozen numeric JSONs only.
