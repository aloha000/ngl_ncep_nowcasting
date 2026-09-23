#!/usr/bin/env python3
"""Build the 5-minute NGL ZTD Zarr on the unified 0.25-deg 80x120 grid.

One GNSS station represents each grid cell (already chosen in the grid map);
cells without a station stay NaN (masked).  Two fields are stored at the native
5-minute sampling of the NGL archives, both in millimetres:

    TROTOT -> ``ztd``    total zenith delay
    TRWET  -> ``zwd``    wet zenith delay

They share the same time axis, grid, mask/station arrays and normalisation
convention (each field is standardised with its own train-split mean/std, stored
as ``{ztd,zwd}_train_mean`` / ``{ztd,zwd}_train_std``).  Pass ``--no-wet`` to
skip the wet delay.  Training-split statistics are written to the store
attributes.
"""

from __future__ import annotations

import argparse
import gzip
import io
import os
import shutil
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc

from common import (
    finalize_store,
    EPOCH,
    LAT_MIN,
    LON_MIN,
    N_LAT,
    N_LON,
    NGL_RAW_ROOT,
    NGL_ZARR,
    RES,
    SPLITS,
    TIME_END,
    TIME_START,
    load_grid_map,
)


def _parse_epoch(epoch: str) -> np.datetime64:
    year, day, seconds = (int(v) for v in epoch.split(":"))
    year += 2000 if year < 80 else 1900
    base = np.datetime64(f"{year:04d}-01-01T00:00:00", "s")
    return base + np.timedelta64(day - 1, "D") + np.timedelta64(seconds, "s")


def parse_station(task):
    """Return (station, TROTOT series, TRWET series, n_records) for one station.

    NGL 的 ``+TROP/SOLUTION`` 段列定义::

        *SITE ___EPOCH____ TROTOT _SIG TRWET TGETOT _SIG TGNTOT _SIG WVAPOR _SIG MTEMP

    用空白切分后 ``fields[2]`` = TROTOT（总延迟 mm）、``fields[4]`` = TRWET（湿延迟 mm）。
    """
    station, data_root, years, start_ns, end_ns, n_steps = task
    start = np.datetime64(start_ns, "s")
    end = np.datetime64(end_ns, "s")
    series = np.full(n_steps, np.nan, dtype=np.float32)
    wet = np.full(n_steps, np.nan, dtype=np.float32)
    n_records = 0
    root = Path(data_root)
    for year in years:
        archive = root / str(year) / f"{station}.{year}.trop.zip"
        if not archive.exists():
            continue
        try:
            with zipfile.ZipFile(archive) as outer:
                for name in outer.namelist():
                    try:
                        with gzip.GzipFile(fileobj=io.BytesIO(outer.read(name))) as inner:
                            in_solution = False
                            for raw in inner:
                                line = raw.decode("ascii", errors="replace").strip()
                                if line == "+TROP/SOLUTION":
                                    in_solution = True
                                    continue
                                if line == "-TROP/SOLUTION":
                                    break
                                if not in_solution or not line or line.startswith("*"):
                                    continue
                                fields = line.split()
                                if len(fields) < 5:
                                    continue
                                seconds = int(fields[1].rsplit(":", 1)[1])
                                if seconds % 300 != 0:
                                    continue
                                time = _parse_epoch(fields[1])
                                if time < start or time >= end:
                                    continue
                                step = int((time - start) / np.timedelta64(5, "m"))
                                if 0 <= step < n_steps:
                                    series[step] = float(fields[2])
                                    if len(fields) > 4:
                                        try:
                                            wet[step] = float(fields[4])
                                        except ValueError:
                                            pass
                                    n_records += 1
                    except (OSError, gzip.BadGzipFile, ValueError):
                        continue
        except (OSError, zipfile.BadZipFile):
            continue
    return station, series, wet, n_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=NGL_ZARR)
    parser.add_argument("--data-root", type=Path, default=NGL_RAW_ROOT)
    parser.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 1))
    parser.add_argument("--cache", type=Path, default=None,
                        help="optional .npz to cache per-station series")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--with-wet", dest="with_wet", action="store_true", default=True,
                        help="also store TRWET as ``zwd`` (default)")
    parser.add_argument("--no-wet", dest="with_wet", action="store_false",
                        help="only store TROTOT")
    args = parser.parse_args()

    grid = load_grid_map()
    grid["lat_idx"] = np.round((grid["lat"] - LAT_MIN) / RES).astype(int)
    grid["lon_idx"] = np.round((grid["lon"] - LON_MIN) / RES).astype(int)

    cells = grid[grid["n_stations"] > 0]
    station_cell = {
        str(row.station_id): (int(row.lat_idx), int(row.lon_idx))
        for row in cells.itertuples(index=False)
    }
    stations = sorted(station_cell)
    print(f"[grid] {len(cells)} cells with a station, {len(stations)} unique stations")

    times = pd.date_range(TIME_START, TIME_END, freq="5min", inclusive="left")
    n_time = len(times)
    years = sorted({t.year for t in (TIME_START, TIME_END - pd.Timedelta(seconds=1))})
    years = list(range(years[0], years[-1] + 1))
    print(f"[time] {times[0]} .. {times[-1]}  n={n_time}  years={years}")

    series_by_station: dict[str, np.ndarray] = {}
    wet_by_station: dict[str, np.ndarray] = {}
    if args.cache and args.cache.exists():
        cached = np.load(args.cache, allow_pickle=True)
        keys = list(cached.files)
        has_new = any(k.startswith("ztd__") for k in keys)
        has_wet = any(k.startswith("zwd__") for k in keys)
        if args.with_wet and not has_wet:
            print(f"[cache] {args.cache} 是旧格式（只有 TROTOT），"
                  f"为了取 TRWET 需要重新解析归档")
        else:
            n_cached = None
            for key in keys:
                values = cached[key]
                n_cached = values.size if n_cached is None else n_cached
                if values.size < n_time:
                    raise SystemExit(
                        f"cache {args.cache} holds {values.size} steps, need {n_time}")
                if not has_new:                      # 旧格式：纯站名 = TROTOT
                    series_by_station[key] = values[:n_time]
                elif key.startswith("ztd__"):
                    series_by_station[key[5:]] = values[:n_time]
                elif key.startswith("zwd__"):
                    wet_by_station[key[5:]] = values[:n_time]
            print(f"[cache] loaded {len(series_by_station)} ztd / {len(wet_by_station)} zwd "
                  f"series ({n_cached} steps -> {n_time}) from {args.cache}")
    if not series_by_station:
        tasks = [
            (s, str(args.data_root), years, TIME_START.to_datetime64(),
             TIME_END.to_datetime64(), n_time)
            for s in stations
        ]
        total_records = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for done, (station, series, wet, n_records) in enumerate(
                pool.map(parse_station, tasks, chunksize=1), 1
            ):
                series_by_station[station] = series
                wet_by_station[station] = wet
                total_records += n_records
                if done % 200 == 0 or done == len(tasks):
                    print(f"[ngl] parsed {done}/{len(tasks)} stations, "
                          f"{total_records} records", flush=True)
        if args.cache:
            payload = {f"ztd__{k}": v for k, v in series_by_station.items()}
            payload.update({f"zwd__{k}": v for k, v in wet_by_station.items()})
            np.savez_compressed(args.cache, **payload)
            print(f"[cache] wrote {args.cache} ({len(series_by_station)} ztd + "
                  f"{len(wet_by_station)} zwd series)")

    def _assemble(by_station):
        """每站序列 -> (time, lat, lon) 网格（站格外 NaN）。"""
        grid = np.full((n_time, N_LAT, N_LON), np.nan, dtype=np.float32)
        for station, series in by_station.items():
            i, j = station_cell[station]
            grid[:, i, j] = series
        return grid

    tr0, tr1 = pd.Timestamp(SPLITS["train"][0]), pd.Timestamp(SPLITS["train"][1])
    tr_mask = (times >= tr0) & (times < tr1)

    def _standardise(grid, name):
        """用 train 段有限值算均值/标准差并标准化（ztd / zwd 各自一套）。"""
        vals = grid[tr_mask]
        finite = np.isfinite(vals)
        mu, sd = float(vals[finite].mean()), float(vals[finite].std())
        print(f"[stats] {name} train {tr0}..{tr1}: mean={mu:.4f} std={sd:.4f} "
              f"(n={int(finite.sum())})")
        return ((grid - mu) / sd).astype(np.float32), mu, sd

    # ---- 总延迟 ----
    ztd = _assemble(series_by_station)
    covered = np.isfinite(ztd).sum(axis=0)
    print(f"[assemble] ztd finite fraction {np.isfinite(ztd).mean():.4f}; "
          f"cells with any data {int((covered > 0).sum())}")
    ztd, train_mean, train_std = _standardise(ztd, "ztd")

    # ---- 湿延迟（TRWET）----
    zwd = None
    if args.with_wet:
        zwd = _assemble(wet_by_station)
        print(f"[assemble] zwd finite fraction {np.isfinite(zwd).mean():.4f}; "
              f"cells with any data {int((np.isfinite(zwd).sum(axis=0) > 0).sum())}")
        zwd, w_mean, w_std = _standardise(zwd, "zwd")

    if args.output.exists():
        if not args.force:
            raise SystemExit(f"{args.output} exists; rerun with --force")
        shutil.rmtree(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    group = zarr.open_group(str(args.output), mode="w")
    group.create_dataset(
        "time",
        data=(times - pd.Timestamp(EPOCH)).total_seconds().to_numpy().astype("int64") // 60,
        chunks=(min(n_time, 4096),),
        compressor=compressor,
    )
    group.create_dataset("lat", data=grid["lat"].drop_duplicates().to_numpy(np.float64),
                         chunks=(N_LAT,), compressor=compressor)
    group.create_dataset("lon", data=grid["lon"].drop_duplicates().to_numpy(np.float64),
                         chunks=(N_LON,), compressor=compressor)
    station_arr = np.empty((N_LAT, N_LON), dtype=f"<U{max(map(len, stations), default=1)}")
    station_arr[:] = ""
    for station, (i, j) in station_cell.items():
        station_arr[i, j] = station
    group.create_dataset("station", data=station_arr, chunks=(N_LAT, N_LON),
                         compressor=compressor)
    group.create_dataset("mask", data=(station_arr == ""), chunks=(N_LAT, N_LON),
                         compressor=compressor)
    group.create_dataset("ztd", data=ztd, chunks=(288, N_LAT, N_LON), dtype="f4",
                         fill_value=np.nan, compressor=compressor)
    group.create_dataset("ztd_train_mean", data=np.array([train_mean], dtype="f4"),
                         chunks=(1,), compressor=compressor)
    group.create_dataset("ztd_train_std", data=np.array([train_std], dtype="f4"),
                         chunks=(1,), compressor=compressor)
    if zwd is not None:
        group.create_dataset("zwd", data=zwd, chunks=(288, N_LAT, N_LON), dtype="f4",
                             fill_value=np.nan, compressor=compressor)
        group.create_dataset("zwd_train_mean", data=np.array([w_mean], dtype="f4"),
                             chunks=(1,), compressor=compressor)
        group.create_dataset("zwd_train_std", data=np.array([w_std], dtype="f4"),
                             chunks=(1,), compressor=compressor)

    group["time"].attrs.update(
        {"_ARRAY_DIMENSIONS": ["time"], "standard_name": "time", "timezone": "UTC",
         "units": f"minutes since {EPOCH}"}
    )
    group["lat"].attrs.update({"_ARRAY_DIMENSIONS": ["lat"], "units": "degrees_north"})
    group["lon"].attrs.update({"_ARRAY_DIMENSIONS": ["lon"], "units": "degrees_east"})
    group["station"].attrs.update({"_ARRAY_DIMENSIONS": ["lat", "lon"]})
    group["mask"].attrs.update({"_ARRAY_DIMENSIONS": ["lat", "lon"],
                                "description": "True where no GNSS station is present"})
    group["ztd"].attrs.update({
        "_ARRAY_DIMENSIONS": ["time", "lat", "lon"], "_FillValue": np.nan,
        "long_name": "total zenith tropospheric delay (standardised)",
        "units": "1", "raw_units": "mm",
        "standardisation": "(ztd_mm - ztd_train_mean) / ztd_train_std",
    })
    group["ztd_train_mean"].attrs.update({"_ARRAY_DIMENSIONS": ["stat"], "units": "mm"})
    group["ztd_train_std"].attrs.update({"_ARRAY_DIMENSIONS": ["stat"], "units": "mm"})
    if zwd is not None:
        group["zwd"].attrs.update({
            "_ARRAY_DIMENSIONS": ["time", "lat", "lon"], "_FillValue": np.nan,
            "long_name": "wet zenith tropospheric delay (standardised)",
            "units": "1", "raw_units": "mm",
            "source_field": "TRWET (NGL wet zenith tropo delay)",
            "standardisation": "(zwd_mm - zwd_train_mean) / zwd_train_std",
        })
        group["zwd_train_mean"].attrs.update({"_ARRAY_DIMENSIONS": ["stat"], "units": "mm"})
        group["zwd_train_std"].attrs.update({"_ARRAY_DIMENSIONS": ["stat"], "units": "mm"})
    group.attrs.update(
        {
            "title": "NGL 5-minute zenith delays (TROTOT -> ztd, TRWET -> zwd) on the unified 0.25-deg Europe grid",
            "source": str(args.data_root),
            "source_fields": "TROTOT -> ztd (total), TRWET -> zwd (wet)",
            "grid": f"{N_LAT}x{N_LON} @ {RES} deg, lat {grid.lat.min()}..{grid.lat.max()}, "
                    f"lon {grid.lon.min()}..{grid.lon.max()}",
            "time_sampling": "5 minutes (exact UTC 5-minute records)",
            "start_utc": str(times[0]), "end_utc": str(times[-1]),
            "n_stations": len(stations),
            "masked_cells": int((station_arr == "").sum()),
            "ztd_train_mean": train_mean, "ztd_train_std": train_std,
            "zwd_stored": bool(zwd is not None),
            **({"zwd_train_mean": w_mean, "zwd_train_std": w_std} if zwd is not None else {}),
            "ztd_storage": "standardised (ztd_mm - mean) / std, raw units mm",
            "normalisation": f"mean/std over train {SPLITS['train'][0]}..{SPLITS['train'][1]}",
            "splits": {k: list(v) for k, v in SPLITS.items()},
        }
    )
    finalize_store(args.output)
    print(f"[output] wrote {args.output}"
          + (f"  (ztd μ/σ = {train_mean:.2f}/{train_std:.2f} mm, "
             f"zwd μ/σ = {w_mean:.2f}/{w_std:.2f} mm)" if zwd is not None
             else f"  (ztd μ/σ = {train_mean:.2f}/{train_std:.2f} mm)"))


if __name__ == "__main__":
    main()
