#!/usr/bin/env python3
"""Build the FuXi Zarr: 6-hourly forecasts on the unified 0.25-deg 80x120 grid.

The three FuXi stores are concatenated by initialisation time and cropped to
the Europe window.  Both ``init`` and ``step`` (lead time) are kept; only the
requested leads are stored (default: just the first one, 6 h), so the array is
``z[init, step, channel(69), lat, lon]`` with ``step == [6]``.  Values are
stored exactly as provided (already standardised with the shared ERA5 mean/std).
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
    CHANNELS,
    FUXI_SOURCES,
    FUXI_ZARR,
    N_LAT,
    N_LON,
    SPLITS,
    TIME_END,
    TIME_START,
    Region,
    decode_time_axis,
    finalize_store,
    target_grid,
)

_SRC_CACHE: dict[str, tuple] = {}
_OUT_CACHE: dict[str, zarr.hierarchy.Group] = {}
_LEAD_INDEX: list[int] = [0]
_N_LEAD = 1


def _source(path: str):
    cached = _SRC_CACHE.get(path)
    if cached is None:
        group = zarr.open(path, "r")
        cached = (group, Region(group["lat"][:], group["lon"][:]))
        _SRC_CACHE[path] = cached
    return cached


def _output(path: str):
    group = _OUT_CACHE.get(path)
    if group is None:
        group = zarr.open(path, "r+")
        _OUT_CACHE[path] = group
    return group


def _init_worker(lead_index, n_lead):
    global _LEAD_INDEX, _N_LEAD
    _LEAD_INDEX = list(lead_index)
    _N_LEAD = int(n_lead)


def build_one(task):
    out_i, src_path, src_i, out_path = task
    group, region = _source(src_path)
    z = group["z"]
    block = np.empty((_N_LEAD, len(CHANNELS), N_LAT, N_LON), dtype=np.float32)
    for k, j in enumerate(_LEAD_INDEX):
        block[k] = region.extract(z[src_i, j, : len(CHANNELS), :, :])
    _output(out_path)["z"][out_i] = block
    return out_i


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FUXI_ZARR)
    parser.add_argument("--sources", type=Path, nargs="+", default=FUXI_SOURCES)
    parser.add_argument("--leads", default="6",
                        help="comma-separated lead hours to keep (default: 6)")
    parser.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    lead_hours = [int(v) for v in args.leads.split(",")]
    src_steps = None
    for src in args.sources:
        group = zarr.open(str(src), "r")
        names = [str(v) for v in group["channel"][:]]
        if names[: len(CHANNELS)] != CHANNELS:
            raise SystemExit(f"{src}: channel order differs from the expected 69")
        steps = np.asarray(group["step"][:])
        if src_steps is None:
            src_steps = steps
        elif not np.array_equal(src_steps, steps):
            raise SystemExit(f"{src}: lead axis differs between sources")
    missing = [h for h in lead_hours if h not in src_steps]
    if missing:
        raise SystemExit(f"leads {missing} not in source lead axis {src_steps.tolist()}")
    lead_index = [int(np.where(src_steps == h)[0][0]) for h in lead_hours]
    n_lead = len(lead_hours)
    print(f"[lead] keeping {lead_hours} h (source lead indices {lead_index})")

    frames = []
    for src in args.sources:
        idx = decode_time_axis(src, "time")
        frames.append(pd.DataFrame({"init": idx, "src": str(src),
                                    "src_i": np.arange(idx.size)}))
    allinit = (
        pd.concat(frames, ignore_index=True)
        .sort_values("init")
        .drop_duplicates("init", keep="first")
        .reset_index(drop=True)
    )
    sel = allinit[(allinit["init"] >= TIME_START) & (allinit["init"] < TIME_END)]
    sel = sel.reset_index(drop=True)
    n_init = len(sel)
    print(f"[time] {sel['init'].iloc[0]} .. {sel['init'].iloc[-1]}  n_init={n_init}")
    for src in args.sources:
        k = int((sel["src"] == str(src)).sum())
        print(f"[src] {src.name}: {k} inits used")

    if args.output.exists():
        if not args.force:
            raise SystemExit(f"{args.output} exists; rerun with --force")
        shutil.rmtree(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    lat, lon = target_grid()
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    root = zarr.open_group(str(args.output), mode="w")
    root.create_dataset(
        "init",
        data=((sel["init"] - TIME_START).dt.total_seconds() // 3600).to_numpy("int64"),
        chunks=(min(n_init, 4096),), compressor=compressor,
    )
    root.create_dataset("step", data=np.array(lead_hours, dtype="int64"),
                        chunks=(n_lead,), compressor=compressor)
    root.create_dataset("channel", data=np.array(CHANNELS, dtype="<U5"),
                        chunks=(len(CHANNELS),), compressor=compressor)
    root.create_dataset("lat", data=lat, chunks=(N_LAT,), compressor=compressor)
    root.create_dataset("lon", data=lon, chunks=(N_LON,), compressor=compressor)
    root.create_dataset(
        "z",
        shape=(n_init, n_lead, len(CHANNELS), N_LAT, N_LON),
        chunks=(1, n_lead, len(CHANNELS), N_LAT, N_LON),
        dtype="f4", fill_value=np.nan, compressor=compressor,
    )

    root["init"].attrs.update({
        "_ARRAY_DIMENSIONS": ["init"], "standard_name": "forecast_reference_time",
        "units": f"hours since {TIME_START:%Y-%m-%d %H:%M:%S}",
    })
    root["step"].attrs.update({"_ARRAY_DIMENSIONS": ["step"], "units": "hours"})
    root["channel"].attrs.update({"_ARRAY_DIMENSIONS": ["channel"]})
    root["lat"].attrs.update({"_ARRAY_DIMENSIONS": ["lat"], "units": "degrees_north"})
    root["lon"].attrs.update({"_ARRAY_DIMENSIONS": ["lon"], "units": "degrees_east"})
    root["z"].attrs.update({
        "_ARRAY_DIMENSIONS": ["init", "step", "channel", "lat", "lon"],
        "long_name": "FuXi forecast fields (standardised)",
        "_FillValue": np.nan,
    })
    root.attrs.update({
        "title": "FuXi forecasts on the unified 0.25-deg Europe grid",
        "sources": [str(s) for s in args.sources],
        "channels": CHANNELS,
        "leads_hours": lead_hours,
        "time_sampling": f"6-hourly initialisations, lead {'/'.join(map(str, lead_hours))} h",
        "start_utc": str(sel["init"].iloc[0]), "end_utc": str(sel["init"].iloc[-1]),
        "normalisation": "standardised with mean_era5.npy / std_era5.npy",
        "splits": {k: list(v) for k, v in SPLITS.items()},
    })

    tasks = [
        (int(i), str(row.src), int(row.src_i), str(args.output))
        for i, row in enumerate(sel.itertuples(index=False))
    ]
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(lead_index, n_lead)) as pool:
        futures = [pool.submit(build_one, t) for t in tasks]
        for fut in as_completed(futures):
            fut.result()
            done += 1
            if done % 500 == 0 or done == n_init:
                print(f"[fuxi] {done}/{n_init} inits written", flush=True)
    finalize_store(args.output)
    print(f"[output] wrote {args.output}")


if __name__ == "__main__":
    main()
