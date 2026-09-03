#!/usr/bin/env python3
"""Convert hourly ERA5 NetCDF files over the US domain to a restartable Zarr store.

Input files are expected as ``YYYYMMDDHH.nc`` with variables ``z`` shaped
(time=1, channel, lat, lon). The output stores the selected hourly range as one
standardized array ``z(time, channel, lat, lon)``. Standardization follows the
FuXi obs-grid convention: ``tp`` is clipped to non-negative values, converted to
millimetres, transformed by log1p, and then all channels are z-scored with
``mean_era5.npy`` and ``std_era5.npy``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ERA5_DIR = Path(
    "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/xuxiaoze/data_prep/era5_1h"
)
DEFAULT_STANDARD_DIR = Path(
    "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/database/fuxi-obs/obs-grid_qc/mean_std"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--era5-dir", type=Path, default=DEFAULT_ERA5_DIR)
    parser.add_argument("--standard-dir", type=Path, default=DEFAULT_STANDARD_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dataset" / "era5_hourly_us_2018_2024_standardized.zarr",
    )
    parser.add_argument("--start", default="2018-01-01T00:00Z", help="inclusive UTC start")
    parser.add_argument("--end", default="2024-12-31T23:00Z", help="inclusive UTC end")
    parser.add_argument("--lat-min", type=float, default=24.0)
    parser.add_argument("--lat-max", type=float, default=50.0)
    parser.add_argument(
        "--lon-min",
        type=float,
        default=235.0,
        help="minimum longitude on the ERA5 0..360 axis; 235 == -125",
    )
    parser.add_argument(
        "--lon-max",
        type=float,
        default=295.0,
        help="maximum longitude on the ERA5 0..360 axis; 295 == -65",
    )
    parser.add_argument("--chunk-hours", type=int, default=6)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true", help="replace an existing store")
    return parser.parse_args()


def parse_time_axis(start: str, end: str) -> np.ndarray:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None or end_ts.tzinfo is None:
        raise SystemExit("--start and --end must include a timezone, e.g. 2018-01-01T00:00Z")
    start_ts = start_ts.tz_convert("UTC")
    end_ts = end_ts.tz_convert("UTC")
    if end_ts < start_ts:
        raise SystemExit("--end must not precede --start")
    return pd.date_range(start_ts, end_ts, freq="h").tz_localize(None).to_numpy("datetime64[ns]")


def path_for_hour(era5_dir: Path, time64: np.datetime64) -> str | None:
    stamp = pd.Timestamp(time64).strftime("%Y%m%d%H")
    path = era5_dir / f"{stamp}.nc"
    return str(path) if path.exists() else None


def read_grid_metadata(sample_path: str, lat_min: float, lat_max: float, lon_min: float, lon_max: float):
    with netCDF4.Dataset(sample_path) as ds:
        channels = np.asarray([str(x) for x in ds.variables["channel"][:]], dtype=object)
        lat_all = np.asarray(ds.variables["lat"][:], dtype=np.float32)
        lon_all = np.asarray(ds.variables["lon"][:], dtype=np.float32)

    lat_idx = np.flatnonzero((lat_all >= lat_min) & (lat_all <= lat_max))
    lon_idx = np.flatnonzero((lon_all >= lon_min) & (lon_all <= lon_max))
    if lat_idx.size == 0 or lon_idx.size == 0:
        raise SystemExit("requested lat/lon box has no ERA5 grid cells")
    if not (np.all(np.diff(lat_idx) == 1) and np.all(np.diff(lon_idx) == 1)):
        raise SystemExit("lat/lon selection must be contiguous for efficient slicing")
    return channels, lat_all[lat_idx], lon_all[lon_idx], slice(lat_idx[0], lat_idx[-1] + 1), slice(lon_idx[0], lon_idx[-1] + 1)


def load_standard_arrays(standard_dir: Path, n_channel: int) -> tuple[np.ndarray, np.ndarray]:
    mean = np.load(standard_dir / "mean_era5.npy").astype(np.float32, copy=False)
    std = np.load(standard_dir / "std_era5.npy").astype(np.float32, copy=False)
    expected = (n_channel, 1, 1)
    if mean.shape != expected or std.shape != expected:
        raise SystemExit(f"mean/std shape mismatch: got {mean.shape}/{std.shape}, expected {expected}")
    if not np.isfinite(std).all() or np.any(std == 0):
        raise SystemExit("std_era5.npy contains non-finite or zero values")
    return mean, std


def create_store(path: Path, times, channels, lat, lon, files, chunk_hours, force: bool, attrs: dict):
    if path.exists() and force:
        shutil.rmtree(path)
    if path.exists():
        group = zarr.open_group(str(path), mode="a")
        expected_start = np.datetime_as_string(times[0], unit="s")
        expected_end = np.datetime_as_string(times[-1], unit="s")
        if (
            group.attrs.get("start_utc") != expected_start
            or group.attrs.get("end_utc") != expected_end
            or group["z"].shape != (len(times), len(channels), len(lat), len(lon))
        ):
            raise RuntimeError("existing store axes differ; rerun with --force")
        return group

    path.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(path), mode="w")
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    chunk = (min(max(1, chunk_hours), len(times)), len(channels), len(lat), len(lon))

    channel_width = max(len(str(c)) for c in channels)
    group.create_dataset("time", data=times, chunks=(len(times),), compressor=compressor)
    group.create_dataset("channel", data=np.asarray(channels, dtype=f"<U{channel_width}"), chunks=(len(channels),), compressor=compressor)
    group.create_dataset("lat", data=np.asarray(lat, dtype=np.float32), chunks=(len(lat),), compressor=compressor)
    group.create_dataset("lon", data=np.asarray(lon, dtype=np.float32), chunks=(len(lon),), compressor=compressor)
    group["time"].attrs.update({"_ARRAY_DIMENSIONS": ["time"], "standard_name": "time", "timezone": "UTC"})
    group["channel"].attrs.update({"_ARRAY_DIMENSIONS": ["channel"]})
    group["lat"].attrs.update({"_ARRAY_DIMENSIONS": ["lat"], "units": "degrees_north", "order": "north_to_south"})
    group["lon"].attrs.update({"_ARRAY_DIMENSIONS": ["lon"], "units": "degrees_east", "range": "0..360"})

    z = group.create_dataset(
        "z",
        shape=(len(times), len(channels), len(lat), len(lon)),
        chunks=chunk,
        dtype="f4",
        fill_value=np.nan,
        compressor=compressor,
    )
    z.attrs.update(
        {
            "_ARRAY_DIMENSIONS": ["time", "channel", "lat", "lon"],
            "description": "standardized ERA5 hourly fields over the US domain",
            "standardization": "tp=log1p(clip(tp,0)*1000); all channels=(value-mean_era5)/std_era5",
        }
    )

    complete = group.create_dataset("_hour_complete", shape=(len(times),), chunks=(len(times),), dtype="bool", fill_value=False)
    complete.attrs["description"] = "restart marker; removed after successful completion"
    complete[:] = np.asarray([path is None for path in files])

    group.attrs.update(
        {
            "title": "ERA5 hourly standardized fields over CONUS",
            "start_utc": np.datetime_as_string(times[0], unit="s"),
            "end_utc": np.datetime_as_string(times[-1], unit="s"),
            "source_files": "YYYYMMDDHH.nc",
            "complete": False,
            **attrs,
        }
    )
    return group


def process_block(task):
    paths, indices, lat_slice, lon_slice, mean, std, tp_index = task
    n_hours = len(paths)
    n_channel = mean.shape[0]
    # infer spatial shape from the first available file in this block
    n_lat = lat_slice.stop - lat_slice.start
    n_lon = lon_slice.stop - lon_slice.start
    block = np.full((n_hours, n_channel, n_lat, n_lon), np.nan, dtype=np.float32)
    ok_count = 0
    for j, path in enumerate(paths):
        if path is None:
            continue
        try:
            with netCDF4.Dataset(path) as ds:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*_FillValue not used.*")
                    arr = np.asarray(ds.variables["z"][0, :, lat_slice, lon_slice], dtype=np.float32)
        except Exception as exc:
            print(f"[warn] failed to read {path}: {exc}", flush=True)
            continue
        tp = np.clip(arr[tp_index], 0.0, None) * 1000.0
        arr[tp_index] = np.log1p(tp)
        block[j] = (arr - mean) / std
        ok_count += 1
    return indices, block, ok_count


def main():
    args = parse_args()
    times = parse_time_axis(args.start, args.end)
    files = [path_for_hour(args.era5_dir, t) for t in times]
    sample_path = next((p for p in files if p is not None), None)
    if sample_path is None:
        raise FileNotFoundError(f"no ERA5 files found in {args.era5_dir} for requested time range")

    channels, lat, lon, lat_slice, lon_slice = read_grid_metadata(
        sample_path, args.lat_min, args.lat_max, args.lon_min, args.lon_max
    )
    try:
        tp_index = list(channels).index("tp")
    except ValueError as exc:
        raise SystemExit("ERA5 channel list does not contain 'tp'") from exc
    mean, std = load_standard_arrays(args.standard_dir, len(channels))

    attrs = {
        "era5_dir": str(args.era5_dir),
        "standard_dir": str(args.standard_dir),
        "lat_min": float(args.lat_min),
        "lat_max": float(args.lat_max),
        "lon_min_0_360": float(args.lon_min),
        "lon_max_0_360": float(args.lon_max),
        "n_channels": int(len(channels)),
        "n_lat": int(len(lat)),
        "n_lon": int(len(lon)),
        "n_source_files": int(sum(path is not None for path in files)),
        "source_hours_missing": int(sum(path is None for path in files)),
    }
    group = create_store(args.output, times, channels, lat, lon, files, args.chunk_hours, args.force, attrs)
    if "_hour_complete" not in group:
        if group.attrs.get("complete"):
            print(f"already complete: {args.output}")
            return
        raise RuntimeError("incomplete store has no restart marker; rerun with --force")

    completed = group["_hour_complete"][:]
    pending = np.flatnonzero(~completed)
    tasks = []
    for begin in range(0, len(pending), args.chunk_hours):
        idx = pending[begin : begin + args.chunk_hours]
        tasks.append(([files[i] for i in idx], idx, lat_slice, lon_slice, mean, std, tp_index))

    print(
        f"[era5-zarr] output={args.output} hours={len(times)} pending={len(pending)} "
        f"grid=({len(lat)},{len(lon)}) channels={len(channels)} workers={args.workers}",
        flush=True,
    )
    written = 0
    if args.workers == 1:
        results = map(process_block, tasks)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=args.workers)
        results = pool.map(process_block, tasks, chunksize=1)
    try:
        for done, (indices, block, ok_count) in enumerate(results, 1):
            group["z"][indices, :, :, :] = block
            group["_hour_complete"][indices] = True
            written += ok_count
            if done % 20 == 0 or done == len(tasks):
                complete_now = int(group["_hour_complete"][:].sum())
                print(f"[era5-zarr] {complete_now}/{len(times)} hours marked complete; read {written} files this run", flush=True)
    finally:
        if pool is not None:
            pool.shutdown()

    if bool(group["_hour_complete"][:].all()):
        del group["_hour_complete"]
        group.attrs.update({"complete": True, "n_times": int(len(times)), "n_files_read_last_run": int(written)})
        zarr.consolidate_metadata(str(args.output))
    print(group.tree())


if __name__ == "__main__":
    main()
