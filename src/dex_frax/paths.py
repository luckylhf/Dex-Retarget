import os
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(os.environ.get("DEX_FRAX_ASSETS", ROOT / "assets/dex_evt")).expanduser().resolve()


def is_runtime_asset(relative):
    path = Path(relative)
    return path.as_posix() not in {"README.md", "LICENSE"} and not any(
        part == ".DS_Store" or part.startswith("._") for part in path.parts
    )


def asset_fingerprints(root=None):
    root = ASSETS if root is None else Path(root)
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and is_runtime_asset(path.relative_to(root))
    }
