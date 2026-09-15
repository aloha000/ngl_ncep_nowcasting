#!/usr/bin/env python3
"""Sanity check: the FuXi-implied ZTD vs the NGL ZTD observations.

For a spread of analysis times this computes

    ZTD_fuxi = ZHD(p_s) + ZWD(t, r profiles)

from the FuXi 6 h background (13 pressure levels + t2m + msl, all standardised
in the store) at every station cell, and compares it with the NGL ZTD that was
actually measured there.  Both the total delay and the station-anomaly part
(each station's own time mean removed) are reported, because the anomaly part is
what a DA increment can use.

Run from da_ngl/main_code:
    python preprocessing/check_ztd_operator.py --stride 8
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

HERE = Path(__file__).resolve().parent
MAIN = HERE.parent / "main_code"
sys.path.insert(0, str(MAIN))
sys.path.insert(0, str(HERE))

from common import CHANNELS, era5_channel_stats          # noqa: E402
from ztd_operator import (LEV_HPA, elevation_from_etopo,  # noqa: E402
                          ztd_profile, ztd_profile_surface, ztd_surface)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", default="configs")
    ap.add_argument("--stride", type=int, default=8, help="take every Nth sample")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--height", default="station",
                    choices=("station", "etopo", "both"))
    args = ap.parse_args()

    cfg = importlib.import_module(args.configs)
    from main.utils import AssimilationDataset

    dates = {"train": cfg.dates_train_range, "val": cfg.dates_val_range,
             "test": cfg.dates_test_range}[args.split]
    ds = AssimilationDataset(cfg, dates, n_label_chans=69)
    samples = ds.samples[::args.stride]
    n = len(samples)
    print(f"[check] split={args.split} samples={n} (stride {args.stride})")

    # ---- station geometry -------------------------------------------------
    dsit = Path(cfg.ngl_zarr).parent
    mp = pd.read_parquet(dsit / "ngl_europe_0p25_80x120_station_grid_map.parquet")
    st = pd.read_parquet(dsit / "ngl_europe_stations.parquet")
    # ``mask=True`` marks the 8222 cells *without* a station (see the builder),
    # so the station cells are the complement.
    cells = mp[~mp["mask"].astype(bool) & mp["station_id"].notna()].reset_index(drop=True)
    cells = cells.merge(st[["gnss_station_id", "height_m"]],
                        left_on="station_id", right_on="gnss_station_id", how="left")
    iy = np.searchsorted(np.asarray(cfg.lat, float), cells["lat"].values)
    ix = np.searchsorted(np.asarray(cfg.lon, float), cells["lon"].values)
    h_station = cells["height_m"].values.astype(float)
    h_etopo = np.asarray(elevation_from_etopo(cells["lat"].values, cells["lon"].values))
    print(f"[check] station cells {len(cells)}  heights: metadata "
          f"mean={np.nanmean(h_station):6.1f} m  ETOPO mean={h_etopo.mean():7.1f} m")

    # ---- store handles ----------------------------------------------------
    gf = zarr.open(cfg.fuxi_zarr, "r")
    gl = zarr.open(cfg.label_zarr, "r")
    gn = zarr.open(cfg.ngl_zarr, "r")
    t_ch = [CHANNELS.index(f"t{int(L)}") for L in LEV_HPA]
    r_ch = [CHANNELS.index(f"r{int(L)}") for L in LEV_HPA]
    i_t2m, i_msl = CHANNELS.index("t2m"), CHANNELS.index("msl")
    mean, std = era5_channel_stats()
    mean, std = mean[:69], std[:69]
    z_mean = float(gn["ztd_train_mean"][0])
    z_std = float(gn["ztd_train_std"][0])

    bg_idx = np.array([s[1] for s in samples])
    frames = np.array([s[2] + cfg.obs_frames - 1 for s in samples])   # window end = T

    def phys(chans):
        raw = np.asarray(gf["z"].oindex[bg_idx, ds.lead_index, chans])   # (n, c, 80, 120)
        return mean[chans][None, :, None, None] + std[chans][None, :, None, None] * raw

    T_lev = phys(t_ch)[:, :, iy, ix]                    # (n, 13, ncell)
    R_lev = phys(r_ch)[:, :, iy, ix]
    T2m = phys([i_t2m])[:, 0, iy, ix]                   # (n, ncell)
    msl = phys([i_msl])[:, 0, iy, ix]                   # Pa
    obs = np.asarray(gn["ztd"].oindex[frames])[:, iy, ix] * z_std + z_mean   # mm

    lat_c = np.asarray(cfg.lat, float)[iy]
    results = {}
    for hname, h in (("station", h_station),
                     ("etopo", h_etopo)) if args.height == "both" else ((args.height,
                     h_station if args.height == "station" else h_etopo),):
        p_s = msl / 100.0 * np.exp(-9.80665 * h[None, :] / (287.05 * T2m))
        # ztd_profile wants the level axis last
        prof = ztd_profile(T_lev.transpose(0, 2, 1), R_lev.transpose(0, 2, 1),
                           p_s, h[None, :], lat_c[None, :])
        # lowest level as a stand-in for 2 m humidity (the reference snippet)
        surf = ztd_surface(T2m, p_s, R_lev[:, -1], h[None, :], lat_c[None, :],
                           temp_unit="K")
        # + surface node at p_s (t2m, lowest-level RH), below-ground levels collapsed
        sfc = ztd_profile_surface(T_lev.transpose(0, 2, 1), R_lev.transpose(0, 2, 1),
                                  T2m, p_s, h[None, :], lat_c[None, :])
        results[hname] = dict(prof=prof, surf=surf, sfc=sfc, p_s=p_s)

    ok = np.isfinite(obs)
    print(f"\n[check] valid (station,time) pairs: {ok.sum()} of {obs.size} "
          f"({100.0*ok.sum()/obs.size:.1f}%)")

    def stats(x, y, m):
        x, y = x[m], y[m]
        d = x - y
        return (np.mean(x), np.mean(d), np.sqrt(np.mean(d**2)),
                np.corrcoef(x, y)[0, 1])

    print("\n[check] ZTD (mm):  obs(NGL) vs FuXi operator")
    print(f"  NGL observed      mean={np.nanmean(obs):8.1f}  std={np.nanstd(obs):7.1f}")
    for hname, res in results.items():
        for tag, key in (("profile", "prof"), ("+surface", "sfc"), ("surface", "surf")):
            z = res[key]["ZTD_mm"]
            m = ok & np.isfinite(z)
            mu, bias, rmse, r = stats(z, obs, m)
            print(f"  H(FuXi) {hname:7s} {tag:7s} mean={mu:8.1f}  "
                  f"bias={bias:+7.1f}  rmse={rmse:7.1f}  r={r:.4f}")
        print(f"           {hname:7s} ZHD mean={np.nanmean(res['prof']['ZHD_mm']):8.1f}  "
              f"ZWD(profile) mean={np.nanmean(res['prof']['ZWD_mm']):6.1f}  "
              f"ZWD(surface) mean={np.nanmean(res['surf']['ZWD_mm']):6.1f}  "
              f"p_s mean={np.nanmean(res['p_s']):7.1f} hPa")
        # anomaly part: remove each station's own time mean from both sides --
        # this is the component a DA increment can actually use.
        print(f"           {hname:7s} anomalies (per-station time mean removed):")
        for tag, key in (("profile", "prof"), ("+surface", "sfc"), ("surface", "surf")):
            zm = res[key]["ZTD_mm"] - np.nanmean(res[key]["ZTD_mm"], axis=0)
            om = obs - np.nanmean(obs, axis=0)
            m = ok & np.isfinite(zm)
            x, y = zm[m], om[m]
            d = x - y
            print(f"             {tag:8s} std(H)={x.std():5.1f}  std(obs)={y.std():5.1f}  "
                  f"bias={d.mean():+5.1f}  rmse={np.sqrt((d**2).mean()):5.1f}  "
                  f"r={np.corrcoef(x, y)[0,1]:.4f}  var_expl={1-(d**2).mean()/(y**2).mean():6.3f}")


if __name__ == "__main__":
    main()
