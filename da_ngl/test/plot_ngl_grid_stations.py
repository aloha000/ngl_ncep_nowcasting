#!/usr/bin/env python3
"""Visualise the gridded NGL data: where the stations are on the 0.25 deg grid.

Six panels, all English (this machine has no CJK font):

  a) grid occupancy  -- the 80x120 lattice, station cells vs masked cells
  b) the 1870 raw NGL stations, coloured by station height
  c) how many raw stations collapsed into each 0.25 deg cell
  d) the gridded product itself: NGL ZTD at one analysis time, in mm
  e) observation availability per cell (fraction of 6-hourly samples present)
  f) height distribution of the 1378 representative (one-per-cell) stations

Usage (from da_ngl/test):
    python plot_ngl_grid_stations.py
    python plot_ngl_grid_stations.py --time 2025-07-01T00:00 --output ../plots/x.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
DA = HERE.parent
sys.path.insert(0, str(DA / "preprocessing"))
from common import decode_time_axis  # noqa: E402

GRID_MAP = "ngl_europe_0p25_80x120_station_grid_map.parquet"
STATIONS = "ngl_europe_stations.parquet"
ZARR = "ngl_europe_0p25_5min.zarr"


def load(dataset: Path):
    gm = pd.read_parquet(dataset / GRID_MAP)
    st = pd.read_parquet(dataset / STATIONS)
    g = zarr.open(str(dataset / ZARR), "r")
    lat = np.asarray(g["lat"][:], dtype="float64")
    lon = np.asarray(g["lon"][:], dtype="float64")
    mask = np.asarray(g["mask"][:])                     # True = no station
    ztd = g["ztd"]
    mean = float(np.asarray(g["ztd_train_mean"][:]).reshape(-1)[0])
    std = float(np.asarray(g["ztd_train_std"][:]).reshape(-1)[0])
    occupied = ~mask
    cells = gm[~gm["mask"].astype(bool)].reset_index(drop=True)
    rep = cells.merge(st[["gnss_station_id", "height_m"]],
                      left_on="station_id", right_on="gnss_station_id", how="left")
    return gm, st, g, lat, lon, mask, occupied, cells, rep, ztd, mean, std


def extent(lat, lon):
    dy, dx = np.abs(lat[1] - lat[0]) / 2, np.abs(lon[1] - lon[0]) / 2
    return [lon[0] - dx, lon[-1] + dx, lat[0] - dy, lat[-1] + dy]


def polarmap(ax):
    ax.set_aspect(1.0 / np.cos(np.radians(46.0)))
    ax.grid(alpha=0.15, ls=":")
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")


def availability(ztd, lat, lon, step=72):
    """Fraction of sampled frames with a finite ZTD, per grid cell."""
    n = ztd.shape[0]
    idx = np.arange(0, n, step)
    cnt = np.zeros((lat.size, lon.size), dtype="float64")
    for i in idx:
        cnt += np.isfinite(np.asarray(ztd[i], dtype="float32"))
    return cnt / idx.size, idx.size


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=DA / "dataset")
    ap.add_argument("--output", type=Path, default=DA / "plots" / "ngl_grid_stations_0p25.png")
    ap.add_argument("--time", default="2025-07-01T00:00",
                    help="analysis time for the ZTD snapshot (UTC)")
    ap.add_argument("--avail-step", type=int, default=72,
                    help="subsample the 5-min axis by this factor for panel (e) "
                         "(72 = 6-hourly)")
    args = ap.parse_args()

    (gm, st, g, lat, lon, mask, occupied, cells, rep, ztd, mean, std) = load(args.dataset)
    ext = extent(lat, lon)
    n_cell = int(occupied.sum())

    times = decode_time_axis(args.dataset / ZARR, "time")
    want = pd.Timestamp(args.time)
    ti = int(np.searchsorted(times.values, want.to_datetime64()))
    snap = np.asarray(ztd[ti], dtype="float32")
    snap_mm = np.where(np.isfinite(snap), snap * std + mean, np.nan)

    avail, n_sampled = availability(ztd, lat, lon, args.avail_step)

    fig, axes = plt.subplots(2, 3, figsize=(21, 12), dpi=120)

    # ---------------------------------------------------------------- (a)
    ax = axes[0, 0]
    ax.imshow(occupied.astype("float32"), origin="lower", extent=ext,
              cmap=ListedColormap(["#eef2f7", "#1f77b4"]), vmin=0, vmax=1,
              interpolation="nearest", aspect="auto")
    ax.plot(rep["lon"], rep["lat"], ".", ms=1.6, color="#08306b", alpha=0.8)
    polarmap(ax)
    ax.set_title(f"(a) 0.25$^\\circ$ grid occupancy: {n_cell} of "
                 f"{lat.size * lon.size} cells hold a station\n"
                 f"{lat.size}$\\times${lon.size} cells, "
                 f"lat {lat[0]:.2f}..{lat[-1]:.2f}, lon {lon[0]:.2f}..{lon[-1]:.2f}")

    # ---------------------------------------------------------------- (b)
    ax = axes[0, 1]
    sc = ax.scatter(st["lon"], st["lat"], c=st["height_m"], s=7,
                    cmap="viridis", norm=LogNorm(vmin=30, vmax=3200),
                    linewidths=0, alpha=0.9, rasterized=True)
    polarmap(ax)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.045)
    cb.set_label("station height (m, log scale)")
    ax.set_title(f"(b) {len(st)} NGL stations in the domain\n"
                 "all raw stations (before gridding)")

    # ---------------------------------------------------------------- (c)
    ax = axes[0, 2]
    counts = np.zeros((lat.size, lon.size))
    iy = np.searchsorted(lat, cells["lat"].values)
    ix = np.searchsorted(lon, cells["lon"].values)
    counts[iy, ix] = cells["n_stations"].values
    vals = np.unique(cells["n_stations"].values.astype(int))
    cmap = plt.get_cmap("turbo", len(vals))
    im = ax.imshow(np.ma.masked_where(counts == 0, counts), origin="lower",
                   extent=ext, cmap=cmap, vmin=vals.min() - 0.5,
                   vmax=vals.max() + 0.5, interpolation="nearest", aspect="auto")
    im.cmap.set_bad("#f2f2f2")
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.045,
                      ticks=vals[vals <= 6])
    cb.set_label("raw stations per cell")
    polarmap(ax)
    dist = ", ".join(f"{v}:{int((cells['n_stations'] == v).sum())}"
                     for v in vals if v > 1)
    ax.set_title("(c) raw stations collapsed per 0.25$^\\circ$ cell\n"
                 f"{len(cells)} cells <- {int(cells['n_stations'].sum())} stations"
                 f"  (multi-station cells  {dist})")

    # ---------------------------------------------------------------- (d)
    ax = axes[1, 0]
    m = np.ma.masked_invalid(snap_mm)
    cmap = plt.get_cmap("RdYlBu_r").copy()
    cmap.set_bad("#e9e9e9")
    im = ax.imshow(m, origin="lower", extent=ext, cmap=cmap,
                   vmin=np.nanpercentile(snap_mm, 1), vmax=np.nanpercentile(snap_mm, 99),
                   interpolation="nearest", aspect="auto")
    ax.plot(st["lon"], st["lat"], "o", ms=1.2, mfc="none", mec="0.25", mew=0.4)
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.045)
    cb.set_label("ZTD (mm)")
    polarmap(ax)
    valid = int(np.isfinite(snap).sum())
    ax.set_title(f"(d) gridded product: NGL ZTD at {pd.Timestamp(times[ti])} UTC\n"
                 f"valid cells {valid} / {n_cell} station cells "
                 f"(grey = no station, {mean:.1f} $\\pm$ {std:.1f} mm train stats)")

    # ---------------------------------------------------------------- (e)
    ax = axes[1, 1]
    # grey = no station cell (same convention as panels a/d); otherwise
    # dark = rarely reporting, bright = always reporting
    im = ax.imshow(np.ma.masked_where(~occupied, avail),
                   origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=1,
                   interpolation="nearest", aspect="auto")
    im.cmap.set_bad("#f2f2f2")
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.045)
    cb.set_label("fraction of samples with data")
    polarmap(ax)
    a = avail[occupied]
    ax.set_title("(e) observation availability per cell\n"
                 f"{n_sampled} samples ({args.avail_step * 5 // 60} h spacing), "
                 f"min {a.min():.2f} / median {np.median(a):.2f} / max {a.max():.2f}, "
                 f"full {(a > 0.999).sum()}/{n_cell}")

    # ---------------------------------------------------------------- (f)
    ax = axes[1, 2]
    bins = np.logspace(np.log10(30), np.log10(3300), 45)
    ax.hist(rep["height_m"].dropna(), bins=bins, color="#1f77b4", alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("station height (m, log scale)")
    ax.set_ylabel("grid cells")
    ax.grid(alpha=0.2, ls=":")
    h = rep["height_m"].dropna()
    ax.axvline(h.median(), color="crimson", ls="--", lw=1.2,
               label=f"median {h.median():.0f} m")
    ax.axvline(h.quantile(0.9), color="darkorange", ls=":", lw=1.2,
               label=f"p90 {h.quantile(0.9):.0f} m")
    ax.legend(fontsize=9)
    ax.set_title("(f) heights of the 1378 representative stations\n"
                 "(one station per cell, seed 2021; ZHD needs the true height)")

    fig.suptitle("NGL GNSS ZTD on the unified 0.25$^\\circ$ Europe grid "
                 "(da_ngl/dataset/ngl_europe_0p25_5min.zarr)",
                 fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=120, bbox_inches="tight")
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
