import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(os.environ.get("DEX_FRAX_ASSETS", ROOT / "assets/dex_evt")).expanduser().resolve()
