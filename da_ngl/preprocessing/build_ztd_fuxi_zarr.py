#!/usr/bin/env python3
"""Build the FuXi-implied ZTD on the unified station grid (6-hourly).

For every background valid time (the label 6-hourly axis, 5476 steps) this
applies the ZTD observation operator to the FuXi forecast the assimilation uses
as background (``init = T - 6 h``, the only lead in the store) and writes

    zhd / zwd / ztd_fuxi   [mm]      NaN outside the station cells

on the unified 0.25 deg grid.  The innovation the model consumes is then
``obs(t) - ztd_fuxi(T)``, computed on the fly by the dataset, so this store stays
physical and the convention lives in one place.

Operator: ``ztd_profile_surface`` -- ZHD from the surface pressure (msl reduced
hypsometrically to the station height) plus ZWD integrated over the 13 ERA5
pressure levels with a surface node spliced in at ``p_s`` (t2m + the lowest
above-ground level's RH); levels at or below the surface pressure are collapsed
onto ``p_s``.  Validated against the NGL ZTD in ``check_ztd_operator.py``:
total RMSE 14.3 mm, anomaly RMSE 12.1 mm (93 % of the observed anomaly variance).

Heights are the GNSS station heights (``ngl_europe_stations.parquet``): the
station height gives 14.3 mm RMSE, the ETOPO2 grid value 80 mm, because ZHD is
0.0022768 m/hPa and a 100 m height error is already ~12 hPa.

Usage (from da_ngl/main_code):
    python preprocessing/build_ztd_fuxi_zarr.py --workers 16
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "main_code"))
sys.path.insert(0, str(HERE))

from common import CHANNELS, DATASET_DIR, era5_channel_stats, finalize_store  # noqa: E402
from ztd_operator import (G_0, LEV_HPA, R_D, elevation_from_etopo,  # noqa: E402
                          ztd_profile_surface)

_STORE_CACHE = {}
_BLOCK_OUT = {}


def _prepare(configs):
    """Per-process setup: open stores once, expose the station cells."""
    global _T_CH, _R_CH, _I_T2M, _I_MSL, _MEAN, _STD, _IY, _IX, _H, _CELLS, _H_GRID, _LAT_GRID
    cfg = importlib.import_module(configs)
    gf = zarr.open(cfg.fuxi_zarr, "r")
    dsit = Path(cfg.ngl_zarr).parent
    mp = pd.read_parquet(dsit / "ngl_europe_0p25_80x120_station_grid_map.parquet")
    st = pd.read_parquet(dsit / "ngl_europe_stations.parquet")
    cells = mp[~mp["mask"].astype(bool) & mp["station_id"].notna()].reset_index(drop=True)
    cells = cells.merge(st[["gnss_station_id", "height_m"]],
                        left_on="station_id", right_on="gnss_station_id", how="left")
    _CELLS = cells
    _IY = np.searchsorted(np.asarray(cfg.lat, float), cells["lat"].values)
    _IX = np.searchsorted(np.asarray(cfg.lon, float), cells["lon"].values)
    _H = cells["height_m"].values.astype(float)
    _T_CH = [CHANNELS.index(f"t{int(L)}") for L in LEV_HPA]
    _R_CH = [CHANNELS.index(f"r{int(L)}") for L in LEV_HPA]
    _I_T2M, _I_MSL = CHANNELS.index("t2m"), CHANNELS.index("msl")
    m, s = era5_channel_stats()
    _MEAN, _STD = m[:69], s[:69]
    hg = np.full((80, 120), np.nan)
    hg[_IY, _IX] = _H
    _H_GRID = hg
    _LAT_GRID = np.repeat(np.asarray(cfg.lat, float)[:, None], 120, axis=1)
    return gf, np.asarray(cfg.lat, float)[_IY], np.asarray(cfg.lon, float)[_IX]


def _block(bg_idx):
    """FuXi ZTD [mm] for one block of background indices, on the full grid.

    Everything is computed on the (H, W) grid with a height array that is NaN
    away from the station cells, so the operator's NaN propagation leaves those
    cells empty and the result can be written as one contiguous block (zarr's
    ``oindex`` is orthogonal, so point-wise (time, lat, lon) assignment would
    need the (n_t, n_cell, n_cell) outer product instead).
    """
    gf = _STORE_CACHE["gf"]
    chans = _T_CH + _R_CH + [_I_T2M, _I_MSL]
    raw = np.asarray(gf["z"].oindex[np.asarray(bg_idx), 0, chans])      # (b, 28, H, W)
    ph = _MEAN[chans][None, :, None, None] + _STD[chans][None, :, None, None] * raw
    h2 = _H_GRID[None]                                                  # (1, H, W)
    T_lev = ph[:, :13].transpose(0, 2, 3, 1)                            # (b, H, W, 13)
    R_lev = ph[:, 13:26].transpose(0, 2, 3, 1)
    t2m, msl = ph[:, 26], ph[:, 27]
    p_s = msl / 100.0 * np.exp(-G_0 * h2 / (R_D * t2m))
    lat2 = np.asarray(_LAT_GRID)[None]
    out = ztd_profile_surface(T_lev, R_lev, t2m, p_s, np.broadcast_to(h2, p_s.shape),
                              np.broadcast_to(lat2, p_s.shape))
    return out["ZHD_mm"], out["ZWD_mm"], out["ZTD_mm"], p_s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", default="configs")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--out", type=Path,
                    default=Path(DATASET_DIR) / "ztd_fuxi_europe_0p25_6h.zarr")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.out.exists():
        if not args.force:
            raise SystemExit(f"{args.out} exists -- pass --force to rebuild")
        shutil.rmtree(args.out)

    cfg = importlib.import_module(args.configs)
    gf, lat_c, lon_c = _prepare(args.configs)
    _STORE_CACHE["gf"] = gf

    # ---- time alignment: background valid at label time T has init = T - lead
    from common import decode_time_axis
    lab_t = decode_time_axis(Path(cfg.label_zarr), "time")
    init_t = decode_time_axis(Path(cfg.fuxi_zarr), "init")
    lead = pd.Timedelta(hours=int(cfg.fcst_step) * 6)
    pos = np.searchsorted(init_t.values, (lab_t - lead).values)
    pos = np.clip(pos, 0, init_t.size - 1)
    exact = init_t.values[pos] == (lab_t - lead).values
    n_valid = int(exact.sum())
    print(f"[build] label steps {lab_t.size}, fuxi init steps {init_t.size}, "
          f"lead {lead}, exact matches {n_valid}")
    assert n_valid >= lab_t.size - 1, "label/fuxi time axes do not line up"

    n_t, n_cell = lab_t.size, _IY.size
    shape = (n_t, 80, 120)
    z = zarr.open(str(args.out), "w")

    def arr(name, fill):
        a = z.create_dataset(name, shape=shape, chunks=(64, 80, 120), dtype="f4",
                             fill_value=fill,
                             compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE))
        a.attrs["_ARRAY_DIMENSIONS"] = ["time", "lat", "lon"]
        a.attrs["units"] = "mm"
        return a

    a_zhd, a_zwd, a_ztd = arr("zhd", np.nan), arr("zwd", np.nan), arr("ztd_fuxi", np.nan)
    z.create_dataset("lat", data=np.asarray(cfg.lat, "f4"), chunks=80)
    z.create_dataset("lon", data=np.asarray(cfg.lon, "f4"), chunks=120)
    z["lat"].attrs.update({"_ARRAY_DIMENSIONS": ["lat"], "units": "degrees_north"})
    z["lon"].attrs.update({"_ARRAY_DIMENSIONS": ["lon"], "units": "degrees_east"})
    ref = pd.Timestamp("1970-01-01")
    n_hours = np.asarray((lab_t - ref).total_seconds() // 3600, dtype="int64")
    z.create_dataset("time", data=n_hours, chunks=4096)
    z["time"].attrs.update({"units": "hours since 1970-01-01 00:00:00",
                            "standard_name": "time", "_ARRAY_DIMENSIONS": ["time"]})
    # station geometry, copied so the store is self-describing
    gn = zarr.open(cfg.ngl_zarr, "r")
    for name in ("mask", "station"):
        d = z.create_dataset(name, shape=(80, 120), dtype=gn[name].dtype, chunks=(80, 120))
        d[:] = gn[name][:]
        d.attrs["_ARRAY_DIMENSIONS"] = ["lat", "lon"]
    z["station"].attrs["station_id"] = list(_CELLS["station_id"].astype(str))
    z.create_dataset("height_m", data=_H.astype("f4"), chunks=(n_cell,), fill_value=np.nan)
    z["height_m"].attrs.update({"_ARRAY_DIMENSIONS": ["station"], "units": "m"})

    blocks = [(s, min(s + args.block, n_t)) for s in range(0, n_t, args.block)]
    print(f"[build] {n_t} times x {n_cell} stations, {len(blocks)} blocks, "
          f"{args.workers} workers")
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for s, e in blocks:
            k = np.arange(s, e)
            keep = exact[k]
            if not keep.any():
                continue
            futs[ex.submit(_block, pos[k][keep])] = (k[keep],)
        for fut in as_completed(futs):
            (k,) = futs[fut]
            zhd, zwd, ztd, _ = fut.result()
            a_zhd[k] = zhd.astype("f4")
            a_zwd[k] = zwd.astype("f4")
            a_ztd[k] = ztd.astype("f4")
            done += len(k)
            if done % (10 * args.block) < args.block or done >= n_valid:
                print(f"  [{done}/{n_valid}] time steps written", flush=True)

    z.attrs.update({
        "title": "FuXi-implied zenith total delay on the unified 0.25 deg grid",
        "operator": "ztd_profile_surface (13 ERA5 levels + surface node at p_s)",
        "background": "FuXi 6 h forecast, init = T - 6 h (label time axis)",
        "units": "mm", "fill_value": "NaN outside the station cells",
        "levels_hpa": list(map(int, LEV_HPA)),
        "heights": "GNSS station heights (ngl_europe_stations.parquet)",
        "surface_node": "t2m + RH of the lowest level above p_s, e=RH/100*es(T)",
        "zhd": "0.0022768 * p_s / (1 - 0.00266 cos2lat - 0.00028 h_km)",
        "zwd": "1e-6 * (R_d/g) * int[k2'/T + k3/T^2) e dp], k2'=16.52, k3=3.776e5",
        "validation": "vs NGL ZTD: total RMSE 14.3 mm, r=0.9959; anomaly RMSE 12.1 mm",
        "stations": n_cell,
    })
    finalize_store(args.out)
    print(f"[build] done -> {args.out}")


if __name__ == "__main__":
    main()
