from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from AllIn_run_phospho_tail_coupling import run


if __name__ == "__main__":
    run()
