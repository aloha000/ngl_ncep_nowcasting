#!/usr/bin/env python3
"""Plot the lat/lon of ``ngl_europe_0p25_80x120_station_grid_map.parquet``.

The parquet holds all 80x120 = 9600 cells of the unified grid; 1378 of them
carry at least one NGL station (the rest are masked out).  This draws

  (a) every cell as a faint lattice, with the station cells coloured by how
      many raw NGL stations fall into that 0.25 deg cell, and
  (b) the distribution of that station count.

Usage (from da_ngl/test):
    python plot_grid_map_points.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DA = HERE.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid-map", type=Path,
                    default=DA / "dataset" / "ngl_europe_0p25_80x120_station_grid_map.parquet")
    ap.add_argument("--output", type=Path,
                    default=DA / "plots" / "ngl_grid_map_points_0p25.png")
    args = ap.parse_args()

    gm = pd.read_parquet(args.grid_map)
    occ = gm[~gm["mask"].astype(bool)].reset_index(drop=True)
    emp = gm[gm["mask"].astype(bool)]

    lat, lon = gm["lat"].values, gm["lon"].values
    print(f"[grid-map] rows {len(gm)} | lat {lat.min()} .. {lat.max()} "
          f"({len(np.unique(lat))} values, d={np.diff(np.unique(lat))[0]:.2f})")
    print(f"[grid-map] lon {lon.min()} .. {lon.max()} ({len(np.unique(lon))} values, "
          f"d={np.diff(np.unique(lon))[0]:.2f})")
    print(f"[grid-map] station cells {len(occ)} | masked cells {len(emp)} | "
          f"raw stations mapped {int(occ['n_stations'].sum())}")
    assert len(occ) + len(emp) == len(gm)
    assert not occ[["lat", "lon"]].duplicated().any()

    fig, axes = plt.subplots(1, 2, figsize=(17, 8), dpi=130,
                             gridspec_kw={"width_ratios": [3.2, 1.0]})

    ax = axes[0]
    ax.scatter(emp["lon"], emp["lat"], s=0.5, color="0.85", linewidths=0,
               rasterized=True, label=f"masked cells ({len(emp)})")
    sc = ax.scatter(occ["lon"], occ["lat"], c=occ["n_stations"], s=22,
                    cmap="turbo", vmin=1, vmax=occ["n_stations"].max(),
                    edgecolors="k", linewidths=0.25, zorder=3,
                    label=f"cells with a station ({len(occ)})")
    cb = fig.colorbar(sc, ax=ax, pad=0.015, fraction=0.04)
    cb.set_label("raw NGL stations in the cell")
    ax.set_aspect(1.0 / np.cos(np.radians(46.0)))
    ax.set_xlim(lon.min() - 0.5, lon.max() + 0.5)
    ax.set_ylim(lat.min() - 0.5, lat.max() + 0.5)
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")
    ax.grid(alpha=0.2, ls=":")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    ax.set_title("ngl_europe_0p25_80x120_station_grid_map.parquet\n"
                 f"{len(gm)} grid cells @ 0.25 deg, {len(occ)} station cells, "
                 f"lat {lat.min():.2f}..{lat.max():.2f}, lon {lon.min():.2f}..{lon.max():.2f}")

    ax = axes[1]
    vc = occ["n_stations"].astype(int).value_counts().sort_index()
    ax.bar(vc.index, vc.values, color="tab:blue", width=0.7)
    ax.set_yscale("log")
    for x, y in zip(vc.index, vc.values):
        ax.text(x, y * 1.15, str(int(y)), ha="center", fontsize=9)
    ax.set_xlabel("raw NGL stations per 0.25 deg cell")
    ax.set_ylabel("cells (log scale)")
    ax.set_xticks(sorted(vc.index))
    ax.grid(alpha=0.25, ls=":", axis="y")
    ax.set_ylim(0.6, vc.values.max() * 3)
    ax.set_title(f"(b) station count per cell\n{len(occ)} cells, "
                 f"{int(occ['n_stations'].sum())} stations total")

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=130, bbox_inches="tight")
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
