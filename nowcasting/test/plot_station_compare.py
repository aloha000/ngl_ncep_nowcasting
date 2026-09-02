#!/usr/bin/env python3
"""Station-level comparison (no field interpolation): NCEP obs, test preds, and
ERA5 sampled at station locations, for selected UTC hours. 3x6 grid per time
(rows = obs, pred, ERA5; columns = 6 variables), with one shared horizontal
colorbar per variable below its column (1-99 pct of the union of the three datasets)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import netCDF4
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/linan/linan_dev/gnss"
RUN = f"{ROOT}/nowcasting/outputs/gnss_nowcast_hg_ll_s1915_off0_h6_dm128_el2_nh4_df512_sp_thf"
ERA5_DIR = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/xuxiaoze/data_prep/era5_1h"
VARS = ["p", "slp", "t2m", "r2m", "u10", "v10"]
TIMES = ["2024-03-14T00:00", "2024-05-20T12:00", "2024-06-01T12:00",
         "2024-07-04T18:00", "2024-08-10T06:00"]
G, R = 9.80665, 287.05


def era5_at_stations(fname: str, lat: np.ndarray, lon: np.ndarray,
                     h: np.ndarray) -> dict[str, np.ndarray]:
    """Sample ERA5 fields at station points; p derived (user formula), r2m proxy."""
    ds = netCDF4.Dataset(fname)
    ch = [str(c) for c in ds.variables["channel"][:]]
    latg = np.flip(np.asarray(ds.variables["lat"][:], np.float64))
    long = np.asarray(ds.variables["lon"][:], np.float64)
    z = ds.variables["z"]
    li = np.flatnonzero((latg >= 24.0) & (latg <= 50.0))
    oi = np.flatnonzero((long >= 235.0) & (long <= 295.0))
    lat_s = latg[li[0]:li[-1] + 1]
    lon_s = long[oi[0]:oi[-1] + 1]

    def raw(name):
        a = np.flip(np.asarray(z[0, ch.index(name), :, :], np.float64), axis=0)
        return a[li[0]:li[-1] + 1, oi[0]:oi[-1] + 1]

    interp = {}
    for name in ["msl", "t2m", "u10", "v10", "r1000", "r925"]:
        interp[name] = RegularGridInterpolator(
            (lat_s, lon_s), raw(name), bounds_error=False, fill_value=np.nan)

    pts = np.stack([lat, lon + 360.0], axis=-1)
    msl = interp["msl"](pts) / 100.0                      # hPa
    t2m = interp["t2m"](pts)                              # K
    r2m = np.where(np.isfinite(interp["r1000"](pts)), interp["r1000"](pts),
                   interp["r925"](pts))
    p_der = msl * np.exp(-G * h / (R * (t2m + 0.0065 * h / 2.0)))
    ds.close()
    return {"p": p_der, "slp": msl, "t2m": t2m, "r2m": r2m,
            "u10": interp["u10"](pts), "v10": interp["v10"](pts)}


d = np.load(f"{RUN}/test_predictions.npz")
preds = d["preds"][:, 0, :]
trues = d["trues"][:, 0, :]
m = np.load(f"{RUN}/test_sample_map.npz", allow_pickle=True)
station_ids = m["station_ids"]
times = pd.DatetimeIndex(pd.to_datetime(m["times"]))

st = pd.read_parquet(f"{ROOT}/dataset/target_stations.parquet")
geo = st.set_index("target_station_id")[["lat", "lon", "height_m"]].to_dict("index")

for tstr in TIMES:
    ts = pd.Timestamp(tstr, tz="UTC")
    idx = np.flatnonzero(times == ts)
    if len(idx) == 0:
        print(f"[warn] no samples at {tstr}")
        continue
    sid = station_ids[idx]
    lat = np.array([geo[s]["lat"] for s in sid])
    lon = np.array([geo[s]["lon"] for s in sid])
    h = np.array([geo[s]["height_m"] for s in sid])
    tr = trues[idx]
    pr = preds[idx]
    stamp = f"{tstr[:10].replace('-', '')}{tstr[11:13]}"
    era = era5_at_stations(f"{ERA5_DIR}/{stamp}.nc", lat, lon, h)
    era_arr = np.stack([era[v] for v in VARS], axis=1)
    print(f"{tstr}: {len(idx)} stations", flush=True)

    fig, axes = plt.subplots(
        4, 6, figsize=(48, 16),
        gridspec_kw={"height_ratios": [1, 1, 1, 0.055], "wspace": 0.12, "hspace": 0.16},
    )
    panels = ((tr, "obs (NCEP)"), (pr, "pred (test)"), (era_arr, "ERA5"))
    for j, var in enumerate(VARS):
        comb = np.concatenate([tr[:, j], pr[:, j], era[var]])
        lo, hi = np.nanpercentile(comb, [1, 99])
        pm = None
        for row, (data, label) in enumerate(panels):
            ax = axes[row, j]
            pm = ax.scatter(lon, lat, c=data[:, j], cmap="turbo", s=100,
                            vmin=lo, vmax=hi, linewidths=0.5)
            ax.set_aspect(1.0 / np.cos(np.deg2rad(35.0)))
            ax.tick_params(labelsize=8)
            if row == 0:
                ax.set_title(var, fontsize=14)
            if j == 0:
                ax.set_ylabel(label, fontsize=14)
        cax = axes[3, j]
        fig.colorbar(pm, cax=cax, orientation="horizontal")
        cax.tick_params(labelsize=8)
    note = ("p: ERA5 derived (msl+DEM+t2m, user formula) | r2m: ERA5 RH@1000hPa (925 fallback)")
    fig.suptitle(f"Station obs vs pred vs ERA5  {tstr} UTC  ({len(idx)} stations, no interpolation)\n{note}",
                 fontsize=12)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.055, top=0.90)
    out = f"{RUN}/station_cmp_{stamp}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[save] {out}", flush=True)
