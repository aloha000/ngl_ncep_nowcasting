#!/usr/bin/env python3
"""Build the label Zarr: ERA5 (69 channels) + IMERG ``tp`` on the unified grid.

The 69 ERA5 channels are copied from the 6-hourly standardised ERA5 store;
``tp`` is taken from IMERG (the ERA5 ``tp`` channel is dropped) and written as
channel 69 so a single 70-channel ``label[time, channel, lat, lon]`` array
holds the complete target.
"""

from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc

from common import (
    finalize_store,
    CHANNELS,
    ERA5_ZARR,
    IMERG_ZARR,
    LABEL_CHANNELS,
    LABEL_ZARR,
    N_LAT,
    N_LON,
    SOURCE_CHANNELS,
    SPLITS,
    TRAIN_LABEL_CHANNELS,
    TIME_END,
    TIME_START,
    Region,
    decode_time_axis,
    era5_channel_stats,
    target_grid,
)

_CACHE: dict[str, tuple] = {}
_TP_MODE = "standardized"
_TP_STATS = None


def _open(path: str):
    cached = _CACHE.get(path)
    if cached is None:
        group = zarr.open(path, "r")
        cached = (group, Region(group["lat"][:], group["lon"][:]))
        _CACHE[path] = cached
    return cached


def _tp_transform(tp_mm: np.ndarray) -> np.ndarray:
    if _TP_MODE == "raw":
        return tp_mm.astype(np.float32)
    mean, std = _TP_STATS
    x = np.clip(tp_mm, 0.0, None)
    if _TP_MODE == "standardized_m":
        x = x * 1000.0
    x = np.log1p(x)
    return ((x - mean[69]) / std[69]).astype(np.float32)


def read_one(task):
    i, e_i, m_i, era5_path, imerg_path = task
    out = np.full((len(LABEL_CHANNELS), N_LAT, N_LON), np.nan, dtype=np.float32)
    i_tp = LABEL_CHANNELS.index("tp")            # IMERG precipitation (training target)
    i_era5_tp = LABEL_CHANNELS.index("era5_tp")  # ERA5 precipitation (evaluation only)
    if e_i >= 0:
        group, region = _open(era5_path)
        # ERA5 carries 70 channels: 0..68 state, 69 = tp (already log1p-standardised
        # with the same mean_era5/std_era5), so it can be copied straight in.
        era5 = region.extract(group["z"][e_i, : len(SOURCE_CHANNELS), :, :]).astype(np.float32)
        out[: len(CHANNELS)] = era5[: len(CHANNELS)]
        out[i_era5_tp] = era5[SOURCE_CHANNELS.index("tp")]
    if m_i >= 0:
        group, region = _open(imerg_path)
        names = [str(v) for v in group["channel"][:]]
        tp = region.extract(group["data"][m_i, names.index("tp"), :, :]).astype(np.float32)
        out[i_tp] = _tp_transform(tp)
    return i, out


def _init_worker(tp_mode: str) -> None:
    global _TP_MODE, _TP_STATS
    _TP_MODE = tp_mode
    _TP_STATS = era5_channel_stats()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=LABEL_ZARR)
    parser.add_argument("--era5", type=Path, default=ERA5_ZARR)
    parser.add_argument("--imerg", type=Path, default=IMERG_ZARR)
    parser.add_argument("--tp-transform", choices=("standardized", "standardized_m", "raw"),
                        default="standardized",
                        help="IMERG tp: log1p+z-score (mm), same but x1000, or raw mm")
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    times = pd.date_range(TIME_START, TIME_END, freq="6h", inclusive="left")
    n_time = len(times)
    print(f"[time] {times[0]} .. {times[-1]}  n={n_time}")

    era5_index = decode_time_axis(args.era5, "time")
    imerg_index = decode_time_axis(args.imerg, "time")
    e_map = pd.Series(np.arange(era5_index.size), index=era5_index)
    m_map = pd.Series(np.arange(imerg_index.size), index=imerg_index)
    e_idx = e_map.reindex(times).fillna(-1).to_numpy("int64")
    m_idx = m_map.reindex(times).fillna(-1).to_numpy("int64")
    print(f"[src] ERA5 missing {int((e_idx < 0).sum())}, IMERG missing {int((m_idx < 0).sum())}")

    tasks = [
        (i, int(e_idx[i]), int(m_idx[i]), str(args.era5), str(args.imerg))
        for i in range(n_time)
    ]
    label = np.full((n_time, len(LABEL_CHANNELS), N_LAT, N_LON), np.nan, dtype=np.float32)
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(args.tp_transform,)) as pool:
        futures = [pool.submit(read_one, t) for t in tasks]
        for fut in as_completed(futures):
            i, block = fut.result()
            label[i] = block
            done += 1
            if done % 500 == 0 or done == n_time:
                print(f"[label] {done}/{n_time} times read", flush=True)

    if args.output.exists():
        if not args.force:
            raise SystemExit(f"{args.output} exists; rerun with --force")
        shutil.rmtree(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    lat, lon = target_grid()
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    root = zarr.open_group(str(args.output), mode="w")
    root.create_dataset(
        "time",
        data=np.asarray((times - TIME_START).total_seconds() // 3600, dtype="int64"),
        chunks=(min(n_time, 4096),), compressor=compressor,
    )
    ch_width = max(len(c) for c in LABEL_CHANNELS)
    root.create_dataset("channel", data=np.array(LABEL_CHANNELS, dtype=f"<U{ch_width}"),
                        chunks=(len(LABEL_CHANNELS),), compressor=compressor)
    root.create_dataset("lat", data=lat, chunks=(N_LAT,), compressor=compressor)
    root.create_dataset("lon", data=lon, chunks=(N_LON,), compressor=compressor)
    root.create_dataset("label", data=label, chunks=(6, len(LABEL_CHANNELS), N_LAT, N_LON),
                        dtype="f4", fill_value=np.nan, compressor=compressor)

    root["time"].attrs.update({
        "_ARRAY_DIMENSIONS": ["time"], "standard_name": "time",
        "units": f"hours since {TIME_START:%Y-%m-%d %H:%M:%S}",
    })
    root["channel"].attrs.update({"_ARRAY_DIMENSIONS": ["channel"]})
    root["lat"].attrs.update({"_ARRAY_DIMENSIONS": ["lat"], "units": "degrees_north"})
    root["lon"].attrs.update({"_ARRAY_DIMENSIONS": ["lon"], "units": "degrees_east"})
    root["label"].attrs.update({
        "_ARRAY_DIMENSIONS": ["time", "channel", "lat", "lon"],
        "_FillValue": np.nan,
        "long_name": "ERA5 analysis (0..68) + IMERG tp (69) + ERA5 tp (70, evaluation only)",
    })
    root.attrs.update({
        "title": "Assimilation label on the unified 0.25-deg Europe grid",
        "era5_source": str(args.era5),
        "imerg_source": str(args.imerg),
        "channels": LABEL_CHANNELS,
        "era5_normalisation": "standardised with mean_era5.npy / std_era5.npy",
        "tp_transform": args.tp_transform + " (IMERG tp); era5_tp copied from ERA5 channel 69",
        "train_channels": TRAIN_LABEL_CHANNELS,
        "eval_only_channels": ["era5_tp"],
        "time_sampling": "6-hourly (00/06/12/18 UTC)",
        "splits": {k: list(v) for k, v in SPLITS.items()},
    })
    finalize_store(args.output)
    print(f"[output] wrote {args.output}")


if __name__ == "__main__":
    main()
