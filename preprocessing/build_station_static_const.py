#!/usr/bin/env python3
"""Sample global const.nc static fields at GNSS and NCEP station locations."""

from __future__ import annotations

import argparse
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONST = Path(
    "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/database/fuxi-obs/obs-grid_qc_new/other_info/const.nc"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--const-nc", type=Path, default=DEFAULT_CONST)
    parser.add_argument("--target-stations", type=Path, default=ROOT / "dataset" / "target_stations.parquet")
    parser.add_argument("--gnss-stations", type=Path, default=ROOT / "dataset" / "gnss_stations.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "dataset" / "station_static_const.npz")
    return parser.parse_args()


def lon_to_360(values):
    return np.asarray(values, dtype=np.float64) % 360.0


def nearest_indices(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if axis[0] <= axis[-1]:
        pos = np.searchsorted(axis, values)
        left = np.clip(pos - 1, 0, len(axis) - 1)
        right = np.clip(pos, 0, len(axis) - 1)
    else:
        rev = axis[::-1]
        pos = np.searchsorted(rev, values)
        left = len(axis) - 1 - np.clip(pos, 0, len(axis) - 1)
        right = len(axis) - 1 - np.clip(pos - 1, 0, len(axis) - 1)
    choose_right = np.abs(axis[right] - values) < np.abs(axis[left] - values)
    return np.where(choose_right, right, left).astype(np.int64)


def make_interpolator(lat: np.ndarray, lon: np.ndarray, field: np.ndarray) -> RegularGridInterpolator:
    if lat[0] > lat[-1]:
        lat_use = lat[::-1]
        field_use = field[::-1, :]
    else:
        lat_use = lat
        field_use = field
    return RegularGridInterpolator(
        (lat_use, lon), field_use, method="linear", bounds_error=False, fill_value=np.nan
    )


def sample(
    frame: pd.DataFrame,
    id_col: str,
    lat: np.ndarray,
    lon: np.ndarray,
    x: np.ndarray,
    onehot: np.ndarray,
):
    ids = frame[id_col].astype(str).to_numpy(dtype=object)
    st_lat = frame["lat"].to_numpy(dtype=np.float64)
    st_lon = lon_to_360(frame["lon"].to_numpy(dtype=np.float64))
    pts = np.stack([st_lat, st_lon], axis=1)
    li = nearest_indices(lat, st_lat)
    oi = nearest_indices(lon, st_lon)

    values = np.empty((len(frame), x.shape[0]), dtype=np.float32)
    for c in range(x.shape[0]):
        if onehot[c]:
            # tvh_*/tvl_* are categorical one-hot channels. Do not linearly
            # interpolate them, otherwise the category encoding becomes fractional.
            values[:, c] = x[c, li, oi]
        else:
            values[:, c] = make_interpolator(lat, lon, x[c])(pts).astype(np.float32)
    return ids, values, li, oi


def main():
    args = parse_args()
    with netCDF4.Dataset(args.const_nc) as ds:
        channels = np.asarray([str(c) for c in ds.variables["channel"][:]], dtype=object)
        lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
        x = np.asarray(ds.variables["x"][:], dtype=np.float32)

    onehot = np.asarray([name.startswith("tvh_") or name.startswith("tvl_") for name in channels], dtype=bool)
    target = pd.read_parquet(args.target_stations)
    gnss = pd.read_parquet(args.gnss_stations)
    target_ids, target_static, target_lat_idx, target_lon_idx = sample(target, "target_station_id", lat, lon, x, onehot)
    gnss_ids, gnss_static, gnss_lat_idx, gnss_lon_idx = sample(gnss, "gnss_station_id", lat, lon, x, onehot)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        channels=channels,
        onehot_mask=onehot,
        target_station_ids=target_ids,
        target_static=target_static,
        target_lat_idx=target_lat_idx,
        target_lon_idx=target_lon_idx,
        gnss_station_ids=gnss_ids,
        gnss_static=gnss_static,
        gnss_lat_idx=gnss_lat_idx,
        gnss_lon_idx=gnss_lon_idx,
        const_nc=str(args.const_nc),
        interpolation="linear for continuous channels; nearest for tvh_/tvl_ one-hot channels",
    )
    print(f"[save] {args.output}")
    print(f"target_static={target_static.shape} gnss_static={gnss_static.shape}")
    print("channels=" + ",".join(channels.tolist()))
    print("onehot=" + ",".join(channels[onehot].tolist()))
    print("interpolation=linear for continuous channels; nearest for tvh_/tvl_ one-hot channels")


if __name__ == "__main__":
    main()
