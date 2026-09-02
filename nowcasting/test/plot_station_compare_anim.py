#!/usr/bin/env python3
"""Animated station-level comparison for a continuous UTC time range.

Rows are obs (NCEP), pred (test), ERA5. Columns are variables. Each subplot is
updated for every hour in the requested period; each variable has one shared
horizontal colorbar below its column.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import netCDF4
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

ROOT = Path("/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/linan/linan_dev/gnss")
RUN = ROOT / "nowcasting/outputs/gnss_nowcast_hg_ll_s1915_off0_h6_dm128_el2_nh4_df512_sp_thf"
ERA5_DIR = Path("/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/xuxiaoze/data_prep/era5_1h")
VARS = ["p", "slp", "t2m", "r2m", "u10", "v10"]
G, R = 9.80665, 287.05


def era5_at_stations(fname: Path, lat: np.ndarray, lon: np.ndarray, h: np.ndarray) -> dict[str, np.ndarray]:
    """Sample ERA5 fields at station points; p uses the requested exponential formula."""
    with netCDF4.Dataset(fname) as ds:
        ch = [str(c) for c in ds.variables["channel"][:]]
        latg = np.flip(np.asarray(ds.variables["lat"][:], np.float64))
        long = np.asarray(ds.variables["lon"][:], np.float64)
        z = ds.variables["z"]
        li = np.flatnonzero((latg >= 24.0) & (latg <= 50.0))
        oi = np.flatnonzero((long >= 235.0) & (long <= 295.0))
        lat_s = latg[li[0]:li[-1] + 1]
        lon_s = long[oi[0]:oi[-1] + 1]

        def raw(name: str) -> np.ndarray:
            arr = np.flip(np.asarray(z[0, ch.index(name), :, :], np.float64), axis=0)
            return arr[li[0]:li[-1] + 1, oi[0]:oi[-1] + 1]

        interp = {}
        for name in ["msl", "t2m", "u10", "v10", "r1000", "r925"]:
            interp[name] = RegularGridInterpolator(
                (lat_s, lon_s), raw(name), bounds_error=False, fill_value=np.nan
            )

        pts = np.stack([lat, lon + 360.0], axis=-1)
        msl = interp["msl"](pts) / 100.0
        t2m = interp["t2m"](pts)
        r2m = np.where(np.isfinite(interp["r1000"](pts)), interp["r1000"](pts), interp["r925"](pts))
        p_der = msl * np.exp(-G * h / (R * (t2m + 0.0065 * h / 2.0)))
        return {
            "p": p_der,
            "slp": msl,
            "t2m": t2m,
            "r2m": r2m,
            "u10": interp["u10"](pts),
            "v10": interp["v10"](pts),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--era5-dir", type=Path, default=ERA5_DIR)
    parser.add_argument("--start", default="2024-05-19T00:00")
    parser.add_argument("--end", default="2024-05-20T23:00")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out or args.run / "station_cmp_anim_20240519_20240520.gif"

    d = np.load(args.run / "test_predictions.npz")
    preds = d["preds"][:, 0, :]
    trues = d["trues"][:, 0, :]

    sample_map = np.load(args.run / "test_sample_map.npz", allow_pickle=True)
    station_ids = sample_map["station_ids"].astype(str)
    times = pd.DatetimeIndex(pd.to_datetime(sample_map["times"]))
    if times.tz is None:
        times = times.tz_localize("UTC")
    else:
        times = times.tz_convert("UTC")

    target_st = pd.read_parquet(ROOT / "dataset/target_stations.parquet")
    geo = target_st.set_index("target_station_id")[["lat", "lon", "height_m"]].to_dict("index")

    frame_times = pd.date_range(args.start, args.end, freq="1h", tz="UTC")
    frames = []
    color_values = {v: [] for v in VARS}
    for ts in frame_times:
        idx = np.flatnonzero(times == ts)
        if len(idx) == 0:
            print(f"[warn] no samples at {ts.isoformat()}", flush=True)
            continue
        sid = station_ids[idx]
        lat = np.array([geo[s]["lat"] for s in sid])
        lon = np.array([geo[s]["lon"] for s in sid])
        h = np.array([geo[s]["height_m"] for s in sid])
        tr = trues[idx]
        pr = preds[idx]
        stamp = ts.strftime("%Y%m%d%H")
        era = era5_at_stations(args.era5_dir / f"{stamp}.nc", lat, lon, h)
        era_arr = np.stack([era[v] for v in VARS], axis=1)
        frames.append({"time": ts, "lat": lat, "lon": lon, "obs": tr, "pred": pr, "era5": era_arr})
        for j, v in enumerate(VARS):
            color_values[v].append(np.concatenate([tr[:, j], pr[:, j], era_arr[:, j]]))
        print(f"[load] {ts.strftime('%Y-%m-%dT%H:%M')}: {len(idx)} stations", flush=True)

    if not frames:
        raise SystemExit("no frames found in requested time range")

    clim = {}
    for v in VARS:
        vals = np.concatenate(color_values[v])
        clim[v] = np.nanpercentile(vals, [1, 99])

    fig, axes = plt.subplots(
        4,
        6,
        figsize=(36, 12),
        gridspec_kw={"height_ratios": [1, 1, 1, 0.055], "wspace": 0.12, "hspace": 0.16},
    )
    panel_keys = ("obs", "pred", "era5")
    panel_labels = ("obs (NCEP)", "pred (test)", "ERA5")
    scatters = {}

    first = frames[0]
    for j, var in enumerate(VARS):
        lo, hi = clim[var]
        last_pm = None
        for row, (key, label) in enumerate(zip(panel_keys, panel_labels)):
            ax = axes[row, j]
            last_pm = ax.scatter(
                first["lon"],
                first["lat"],
                c=first[key][:, j],
                cmap="turbo",
                s=90,
                vmin=lo,
                vmax=hi,
                linewidths=0.5,
            )
            scatters[(row, j)] = last_pm
            ax.set_xlim(-126, -66)
            ax.set_ylim(24, 50)
            ax.set_aspect(1.0 / np.cos(np.deg2rad(35.0)))
            ax.tick_params(labelsize=8)
            if row == 0:
                ax.set_title(var, fontsize=14)
            if j == 0:
                ax.set_ylabel(label, fontsize=14)
        cbar = fig.colorbar(last_pm, cax=axes[3, j], orientation="horizontal")
        cbar.ax.tick_params(labelsize=8)

    note = "p: ERA5 derived (msl+station height+t2m, exponential formula) | r2m: ERA5 RH@1000hPa (925 fallback)"
    title = fig.suptitle("", fontsize=12)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.055, top=0.90)

    def update(i: int):
        frame = frames[i]
        xy = np.column_stack([frame["lon"], frame["lat"]])
        for j in range(len(VARS)):
            for row, key in enumerate(panel_keys):
                sc = scatters[(row, j)]
                sc.set_offsets(xy)
                sc.set_array(frame[key][:, j])
        title.set_text(
            f"Station obs vs pred vs ERA5  {frame['time'].strftime('%Y-%m-%d %H:%M')} UTC"
            f"  ({len(frame['lat'])} stations, no interpolation)\n{note}"
        )
        return [*scatters.values(), title]

    anim = matplotlib.animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / args.fps, blit=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    plt.close(fig)
    print(f"[save] {out}", flush=True)


if __name__ == "__main__":
    main()
