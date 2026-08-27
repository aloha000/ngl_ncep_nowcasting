#!/usr/bin/env python3
"""Convert NCEP surface-observation pickles to one hourly, restartable Zarr store.

Station identity is the exact ``(latitude, longitude, height)`` triple reported
by the source, with longitude normalized to [-180, 180).  The station axis
follows ``dataset/target_stations.parquet`` (ids ``ncep_00001`` ...) so that
the store lines up with the sample index.  No interpolation is performed in
time or space: every UTC hour present in the source files becomes one time
step, and any (time, station) without a report is NaN.  If several source rows
share the same (lat, lon, h) key in one file, each variable drops one
maximum and one minimum and averages the remaining finite values.  Rows with
u10 == v10 == 0 are excluded from the wind averages (they look like missing
wind placeholders), and an all-NaN slp stays NaN.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc


SURF_RE = re.compile(r"surf_(\d{10})\.pkl$")
VARIABLES = ("p", "slp", "t2m", "r2m", "u10", "v10")
UNITS = {
    "p": "hPa",
    "slp": "hPa",
    "t2m": "K",
    "r2m": "%",
    "u10": "m/s",
    "v10": "m/s",
}
LONG_NAMES = {
    "p": "surface pressure",
    "slp": "sea level pressure",
    "t2m": "2 m temperature",
    "r2m": "2 m relative humidity",
    "u10": "10 m u-wind",
    "v10": "10 m v-wind",
}


def normalize_lon(values):
    return (np.asarray(values, dtype=np.float64) + 180.0) % 360.0 - 180.0


def trimmed_mean(values):
    """Drop one maximum and one minimum, then average the rest.

    With fewer than three finite values the available values are kept
    unchanged (one value is used directly, two are averaged).
    """
    n = len(values)
    if n == 0:
        return np.nan
    if n <= 2:
        return float(np.mean(values))
    values = np.sort(np.asarray(values, dtype=np.float64))
    return float(np.mean(values[1:-1]))


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surf-dir", type=Path, default=root / "surf_ncep")
    parser.add_argument(
        "--stations",
        type=Path,
        default=root / "dataset" / "target_stations.parquet",
        help="parquet with target_station_id, lat, lon, height_m columns",
    )
    parser.add_argument("--output", type=Path, default=root / "dataset" / "ncep_hourly.zarr")
    parser.add_argument("--start", default=None, help="inclusive UTC start (default: first file)")
    parser.add_argument("--end", default=None, help="inclusive UTC end (default: last file)")
    parser.add_argument("--chunk-hours", type=int, default=48)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true", help="replace an existing store")
    return parser.parse_args()


def surface_files_and_axis(surf_dir: Path, start=None, end=None):
    parsed = []
    for path in surf_dir.glob("surf_*.pkl"):
        match = SURF_RE.search(path.name)
        if match:
            parsed.append((pd.to_datetime(match.group(1), format="%Y%m%d%H", utc=True), path))
    if not parsed:
        raise FileNotFoundError(f"no surf_YYYYMMDDHH.pkl files in {surf_dir}")
    parsed.sort()
    first, last = parsed[0][0], parsed[-1][0]
    if start is None:
        start = first
    if end is None:
        end = last
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start.tzinfo is None or end.tzinfo is None:
        raise SystemExit("--start and --end must include a timezone")
    start, end = start.tz_convert("UTC"), end.tz_convert("UTC")
    if end < start:
        raise SystemExit("--end must not precede --start")
    times = pd.date_range(start, end, freq="h").tz_localize(None).to_numpy(dtype="datetime64[ns]")
    by_hour = {time.tz_localize(None).to_datetime64(): str(path) for time, path in parsed}
    files = [by_hour.get(time) for time in times]
    return files, times


def load_stations(path: Path):
    frame = pd.read_parquet(path)
    needed = ["target_station_id", "lat", "lon", "height_m"]
    missing = [name for name in needed if name not in frame.columns]
    if missing:
        raise SystemExit(
            f"station file {path} lacks columns {missing}; "
            "expected target_station_id, lat, lon, height_m"
        )
    frame = frame.dropna(subset=["lat", "lon", "height_m"]).reset_index(drop=True)
    station_ids = frame["target_station_id"].astype(str).tolist()
    keys = np.column_stack(
        [
            frame["lat"].to_numpy(dtype=np.float64),
            normalize_lon(frame["lon"].to_numpy(dtype=np.float64)),
            frame["height_m"].to_numpy(dtype=np.float64),
        ]
    )
    if len({tuple(row) for row in keys}) != len(keys):
        raise SystemExit(f"duplicate (lat, lon, height_m) keys in {path}")
    return station_ids, keys


def create_store(path: Path, station_ids, keys, times, files, chunk_hours, force: bool):
    if path.exists() and force:
        shutil.rmtree(path)
    if path.exists():
        group = zarr.open_group(str(path), mode="a")
        expected_start = np.datetime_as_string(times[0], unit="s")
        expected_end = np.datetime_as_string(times[-1], unit="s")
        if (
            group.attrs.get("start_utc") != expected_start
            or group.attrs.get("end_utc") != expected_end
            or group["station"].shape[0] != len(station_ids)
            or group["time"].shape[0] != len(times)
        ):
            raise RuntimeError("existing store axes differ; rerun with --force")
        if group["station"][:].astype(str).tolist() != station_ids:
            raise RuntimeError("existing station order differs; rerun with --force")
        return group

    path.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(path), mode="w")
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    n_time, n_station = len(times), len(station_ids)
    station_width = max(map(len, station_ids))
    chunk = (min(max(1, chunk_hours), n_time), n_station)

    group.create_dataset("time", data=times, chunks=(n_time,), compressor=compressor)
    group.create_dataset(
        "station",
        data=np.asarray(station_ids, dtype=f"<U{station_width}"),
        chunks=(n_station,),
        compressor=compressor,
    )
    group["time"].attrs.update(
        {"_ARRAY_DIMENSIONS": ["time"], "standard_name": "time", "timezone": "UTC"}
    )
    group["station"].attrs.update({"_ARRAY_DIMENSIONS": ["station"]})
    for name, values in zip(("lat", "lon", "height_m"), keys.T):
        group.create_dataset(name, data=values.astype(np.float64), chunks=(n_station,))
        group[name].attrs.update({"_ARRAY_DIMENSIONS": ["station"]})

    for variable in VARIABLES:
        array = group.create_dataset(
            variable,
            shape=(n_time, n_station),
            chunks=chunk,
            dtype="f4",
            fill_value=np.nan,
            compressor=compressor,
        )
        array.attrs.update(
            {
                "_ARRAY_DIMENSIONS": ["time", "station"],
                "units": UNITS[variable],
                "long_name": LONG_NAMES[variable],
            }
        )

    complete = group.create_dataset(
        "_hour_complete", shape=(n_time,), chunks=(n_time,), dtype="bool", fill_value=False
    )
    complete.attrs["description"] = "restart marker; removed after successful completion"
    # Hours without a source file have nothing to process.
    complete[:] = np.asarray([path is None for path in files])
    group.attrs.update(
        {
            "title": "NCEP surface observations hourly",
            "start_utc": np.datetime_as_string(times[0], unit="s"),
            "end_utc": np.datetime_as_string(times[-1], unit="s"),
            "time_sampling": "1 hour; exact UTC hours from source file names",
            "station_identity": "(latitude, longitude, height) exactly as reported; "
            "longitude normalized to [-180, 180)",
            "no_interpolation": "no temporal or spatial interpolation is applied",
            "missing_rule": "hours or stations without a report are NaN",
            "duplicate_rule": (
                "when several source rows share a (lat, lon, h) key in one file, "
                "each variable drops one maximum and one minimum and averages the "
                "remaining finite values; u10/v10 exclude rows with u10==0 and "
                "v10==0; slp stays NaN when all duplicates are NaN"
            ),
            "source_files": "surf_YYYYMMDDHH.pkl",
            "variables": {v: {"units": UNITS[v], "long_name": LONG_NAMES[v]} for v in VARIABLES},
            "complete": False,
        }
    )
    return group


def process_block(task):
    paths, hour_indices, keys = task
    n_hours, n_station = len(paths), keys.shape[0]
    n_vars = len(VARIABLES)
    block = np.full((n_hours, n_station, n_vars), np.nan, dtype=np.float32)
    counts = np.zeros(n_vars + 1, dtype=np.int64)
    for j, (path, hour) in enumerate(zip(paths, hour_indices)):
        if path is None:
            continue
        try:
            frame = pd.read_pickle(path)
        except Exception:
            continue
        lat = frame["lat"].to_numpy(dtype=np.float64)
        lon = normalize_lon(frame["lon"].to_numpy(dtype=np.float64))
        height = frame["h"].to_numpy(dtype=np.float64)
        finite = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(height)
        rows_by_key = {}
        for i in np.flatnonzero(finite):
            rows_by_key.setdefault((lat[i], lon[i], height[i]), []).append(int(i))
        columns = {variable: frame[variable].to_numpy(dtype=np.float64) for variable in VARIABLES}
        for s in range(n_station):
            rows = rows_by_key.get((keys[s, 0], keys[s, 1], keys[s, 2]))
            if not rows:
                continue
            for k, variable in enumerate(VARIABLES):
                candidates = []
                for i in rows:
                    value = columns[variable][i]
                    if not np.isfinite(value):
                        continue
                    if (
                        variable in ("u10", "v10")
                        and columns["u10"][i] == 0.0
                        and columns["v10"][i] == 0.0
                    ):
                        continue
                    candidates.append(value)
                aggregated = trimmed_mean(candidates)
                if np.isfinite(aggregated):
                    block[j, s, k] = aggregated
                    counts[k] += 1
            if np.any(np.isfinite(block[j, s])):
                counts[n_vars] += 1
    return hour_indices, block, counts


def main():
    args = parse_args()
    files, times = surface_files_and_axis(args.surf_dir, args.start, args.end)
    station_ids, keys = load_stations(args.stations)
    n_time, n_station = len(times), len(station_ids)
    group = create_store(args.output, station_ids, keys, times, files, args.chunk_hours, args.force)
    if "_hour_complete" not in group:
        if group.attrs.get("complete"):
            print(f"already complete: {args.output}")
            return
        raise RuntimeError("incomplete store has no restart marker; rerun with --force")

    completed = group["_hour_complete"][:]
    pending = np.flatnonzero(~completed)
    tasks = []
    for begin in range(0, len(pending), args.chunk_hours):
        indices = pending[begin : begin + args.chunk_hours]
        tasks.append(([files[i] for i in indices], indices, keys))
    per_variable = np.zeros(len(VARIABLES) + 1, dtype=np.int64)

    if args.workers == 1:
        results = map(process_block, tasks)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=args.workers)
        results = pool.map(process_block, tasks, chunksize=1)
    try:
        for done, (indices, block, counts) in enumerate(results, 1):
            for k, variable in enumerate(VARIABLES):
                group[variable][indices, :] = block[:, :, k]
            group["_hour_complete"][indices] = True
            per_variable += counts
            group.attrs["n_obs_records"] = int(per_variable[len(VARIABLES)])
            if done % 20 == 0 or done == len(tasks):
                print(
                    f"[ncep-zarr] {completed.sum() + done * args.chunk_hours}/{n_time} hours; "
                    f"{per_variable[len(VARIABLES)]} station-hour records",
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.shutdown()

    if bool(group["_hour_complete"][:].all()):
        del group["_hour_complete"]
        group.attrs.update(
            {
                "complete": True,
                "n_stations": n_station,
                "n_times": n_time,
                "n_source_files": sum(path is not None for path in files),
                "source_hours_missing": sum(path is None for path in files),
                "n_obs_records": int(per_variable[len(VARIABLES)]),
            }
        )
        zarr.consolidate_metadata(str(args.output))
    print(group.tree())


if __name__ == "__main__":
    main()
