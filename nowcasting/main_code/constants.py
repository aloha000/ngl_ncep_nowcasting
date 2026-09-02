from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TSL_ROOT = ROOT / "Time-Series-Library"
if str(TSL_ROOT) not in sys.path:
    sys.path.insert(0, str(TSL_ROOT))

NCEP_VARS = ["p", "slp", "t2m", "r2m", "u10", "v10"]
NGL_VARS = ["ztd", "zwd"]
