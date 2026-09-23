"""Shared grid / time helpers for the da_ngl assimilation Zarr builds.

Three stores are produced on one unified grid (0.25 deg, lat 36.50..56.25 N,
lon -5.25..24.50 E -> 80 x 120):

* ``ngl_europe_0p25_5min.zarr``   5-minute GNSS ZTD, one station per grid cell
* ``fuxi_europe_0p25.zarr``       6-hourly FuXi forecasts, (init, lead) x 69 ch
* ``label_europe_0p25.zarr``      6-hourly ERA5 (69 ch) + IMERG ``tp`` = 70 ch

All time coordinates are stored CF-style (int64 + ``units`` attribute) using a
common epoch of 2022-01-01 00:00 UTC.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DA_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = DA_ROOT / "dataset"

GRID_MAP = DATASET_DIR / "ngl_europe_0p25_80x120_station_grid_map.parquet"

NGL_ZARR = DATASET_DIR / "ngl_europe_0p25_5min.zarr"
FUXI_ZARR = DATASET_DIR / "fuxi_europe_0p25.zarr"
LABEL_ZARR = DATASET_DIR / "label_europe_0p25.zarr"

# Unified target grid: exact 0.25-deg multiples (method "jia").
LAT_MIN, LAT_MAX = 36.50, 56.25
LON_MIN, LON_MAX = -5.25, 24.50
RES = 0.25
N_LAT, N_LON = 80, 120

# Union coverage of the three stores (train + val + test, half-open).
# The test split stops at 2025-10-01, where the IMERG precipitation label ends.
TIME_START = pd.Timestamp("2022-01-01T00:00:00")
TIME_END = pd.Timestamp("2025-10-01T00:00:00")
EPOCH = "2022-01-01 00:00:00"

SPLITS = {
    "train": ("2022-01-01T00:00:00", "2024-05-01T00:00:00"),
    "val": ("2024-05-01T00:00:00", "2025-01-01T00:00:00"),
    "test": ("2025-01-01T00:00:00", "2025-10-01T00:00:00"),
}

# ---------------------------------------------------------------------------
# Source data roots
# ---------------------------------------------------------------------------
NGL_RAW_ROOT = Path(
    "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/ngl_ztd_all_downloaded/"
    "data/top10_2022_2026/Western_and_central_Europe"
)
FUXI_SOURCES = [
    Path("/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/huangyuanqing/data/Fuxi_pred_2017_2024"),
    Path("/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/huangyuanqing/data/FuXi_Pred_2024_2025"),
    Path("/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/huangyuanqing/data/FuXi_Pred_2025_other"),
]
ERA5_ZARR = Path("/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/huangyuanqing/data/ERA5_2017_2025")
IMERG_ZARR = Path(
    "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/database/fuxi-obs/imerg/zarr_25_720_more"
)
MEAN_STD_DIR = Path(
    "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/database/fuxi-obs/obs-grid_qc/mean_std"
)

# ERA5/FuXi share this 70-channel order; channel 69 is ``tp`` and is dropped
# (the precipitation label comes from IMERG instead).
SOURCE_CHANNELS = [
    "z50", "z100", "z150", "z200", "z250", "z300", "z400", "z500", "z600", "z700",
    "z850", "z925", "z1000",
    "t50", "t100", "t150", "t200", "t250", "t300", "t400", "t500", "t600", "t700",
    "t850", "t925", "t1000",
    "u50", "u100", "u150", "u200", "u250", "u300", "u400", "u500", "u600", "u700",
    "u850", "u925", "u1000",
    "v50", "v100", "v150", "v200", "v250", "v300", "v400", "v500", "v600", "v700",
    "v850", "v925", "v1000",
    "r50", "r100", "r150", "r200", "r250", "r300", "r400", "r500", "r600", "r700",
    "r850", "r925", "r1000",
    "t2m", "u10", "v10", "msl", "tp",
]
CHANNELS = SOURCE_CHANNELS[:69]          # model/label channels 0..68
TRAIN_LABEL_CHANNELS = CHANNELS + ["tp"]         # 70 channels: what the model predicts
# The store also carries ERA5's own ``tp`` (already log1p-standardised with the
# same mean/std).  It is *not* used for training -- only for evaluation/plots.
LABEL_CHANNELS = TRAIN_LABEL_CHANNELS + ["era5_tp"]   # 71 channels in the store


def target_grid() -> tuple[np.ndarray, np.ndarray]:
    """Return the unified 0.25-deg grid centres (80 lats, 120 lons)."""
    lat = np.round(np.arange(LAT_MIN, LAT_MAX + 1e-9, RES), 6)
    lon = np.round(np.arange(LON_MIN, LON_MAX + 1e-9, RES), 6)
    assert lat.size == N_LAT and lon.size == N_LON, (lat.size, lon.size)
    return lat, lon


class Region:
    """Nearest-neighbour mapping from a global 0.25-deg field to the target grid.

    The source latitude axis runs north -> south (descending); the target axis
    ascends, so the latitude block is flipped.  Longitude wraps through 0/360.
    """

    def __init__(self, lat_src: np.ndarray, lon_src: np.ndarray) -> None:
        tlat, tlon = target_grid()
        lat_src = np.asarray(lat_src, dtype=np.float64)
        lon_src = np.asarray(lon_src, dtype=np.float64)

        li = np.array([int(np.argmin(np.abs(lat_src - v))) for v in tlat])
        self.la0, self.la1 = int(li.min()), int(li.max())
        self.flip_lat = bool(li[0] > li[-1])
        self.lon_idx = np.array(
            [int(np.argmin(np.abs(lon_src - (v % 360.0)))) for v in tlon]
        )
        self.max_lat_err = float(np.max(np.abs(lat_src[li] - tlat)))
        lon_err = np.abs((lon_src[self.lon_idx] - tlon + 180.0) % 360.0 - 180.0)
        self.max_lon_err = float(np.max(lon_err))

    def extract(self, field: np.ndarray) -> np.ndarray:
        """(..., lat_src, lon_src) -> (..., 80, 120) in target order."""
        block = field[..., self.la0:self.la1 + 1, :]
        if self.flip_lat:
            block = block[..., ::-1, :]
        return np.ascontiguousarray(block[..., self.lon_idx])


def split_bounds(name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start, end = SPLITS[name]
    return pd.Timestamp(start), pd.Timestamp(end)


def load_grid_map() -> pd.DataFrame:
    return pd.read_parquet(GRID_MAP)


def era5_channel_stats() -> tuple[np.ndarray, np.ndarray]:
    """Per-channel ERA5/FuXi standardisation mean/std (70 values)."""
    mean = np.load(MEAN_STD_DIR / "mean_era5.npy").astype("float64").reshape(-1)
    std = np.load(MEAN_STD_DIR / "std_era5.npy").astype("float64").reshape(-1)
    return mean, std


def standardize_tp(tp_mm: np.ndarray, *, as_metres: bool) -> np.ndarray:
    """Reproduce the ERA5 ``tp`` pre-processing: clip(0) -> log1p -> z-score.

    ``as_metres`` multiplies by 1000 first, matching ERA5 (tp stored in m).
    """
    mean, std = era5_channel_stats()
    x = np.clip(tp_mm, 0.0, None)
    if as_metres:
        x = x * 1000.0
    x = np.log1p(x)
    return ((x - mean[69]) / std[69]).astype(np.float32)


NO_FILL_ARRAYS = (
    "time", "init", "step", "channel", "lat", "lon", "mask", "station",
    "ztd_train_mean", "ztd_train_std", "zwd_train_mean", "zwd_train_std",
)


def finalize_store(path) -> None:
    """Make a store clean for xarray.

    zarr's default ``fill_value`` (0 / False / '') collides with real
    coordinate values -- ``lon == 0.0``, ``time == 0``, ``mask == False`` --
    and xarray treats ``_FillValue`` as missing, silently turning them into
    NaN/NaT.  Clearing the fill value on coordinate/index arrays fixes that.
    Consolidated metadata is written so ``xr.open_zarr`` opens the store
    directly instead of falling back to the slow non-consolidated path.
    """
    import json

    import zarr

    path = Path(path)
    group = zarr.open_group(str(path), mode="r+")
    for name in NO_FILL_ARRAYS:
        meta_path = path / name / ".zarray"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("fill_value") is not None:
            meta["fill_value"] = None
            meta_path.write_text(json.dumps(meta, indent=4, sort_keys=True))
    zarr.consolidate_metadata(str(path), metadata_key=".zmetadata")


def decode_time_axis(path: Path, name: str) -> pd.DatetimeIndex:
    """Decode a CF-style int64 time axis (``units`` attribute) to timestamps."""
    import json

    units = json.loads((path / name / ".zattrs").read_text())["units"]
    value, _, ref = units.partition(" since ")
    ref = pd.Timestamp(ref.strip())
    import zarr

    raw = zarr.open(str(path), "r")[name][:]
    unit = value.strip().lower()
    if unit.startswith("hour"):
        return ref + pd.to_timedelta(raw, unit="h")
    if unit.startswith("day"):
        return ref + pd.to_timedelta(raw, unit="D")
    if unit.startswith("minute"):
        return ref + pd.to_timedelta(raw, unit="m")
    raise ValueError(f"unsupported time unit {value!r} in {path}/{name}")
