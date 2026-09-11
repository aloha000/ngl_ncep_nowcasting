from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TSL_ROOT = ROOT / "Time-Series-Library"
if str(TSL_ROOT) not in sys.path:
    sys.path.insert(0, str(TSL_ROOT))

# ``q2m`` is specific humidity (kg kg-1), computed from the observed t2m,
# r2m and p fields using the IFS Cy48r1 mixed-phase saturation formulation.
NCEP_VARS = ["p", "slp", "t2m", "q2m", "u10", "v10"]
# Fields physically stored in the observation Zarr.  r2m remains a source
# field, but is not used directly as a model target.
NCEP_SOURCE_VARS = ["p", "slp", "t2m", "r2m", "u10", "v10"]
NGL_VARS = ["ztd"]  # GNSS input fields used by the model; ZWD is no longer used.

# ERA5 pressure-level background fields used by GNSS nowcasting. Surface
# fields (t2m/u10/v10/msl/tp) and levels below 700 hPa are intentionally out.
ERA5_BACKGROUND_CHANNELS = [
    f"{variable}{level}"
    for variable in ("z", "t", "u", "v", "r")
    for level in (50, 100, 150, 200, 250, 300, 400, 500, 600, 700)
]
