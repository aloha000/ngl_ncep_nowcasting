#!/usr/bin/env python3
"""Scatter plot of 0.05-deg grid points and NGL station distribution."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid-map",
        type=Path,
        default=here / "dataset" / "ngl_europe_0p05_station_grid_map.parquet",
    )
    parser.add_argument(
        "--station-info",
        type=Path,
        default=here / "dataset" / "ngl_europe_stations.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "plots" / "europe_grid_station_distribution.png",
    )
    args = parser.parse_args()

    grid = pd.read_parquet(args.grid_map)
    stations = pd.read_parquet(args.station_info)
    occupied = grid[grid["n_stations"] > 0]
    masked = grid[grid["n_stations"] == 0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 10), dpi=130)

    ax.scatter(
        masked["lon"], masked["lat"], s=0.25, color="0.82",
        linewidths=0, alpha=0.35, rasterized=True,
        label=f"masked grid points ({len(masked)})",
    )
    ax.scatter(
        occupied["lon"], occupied["lat"], s=5, color="tab:blue",
        linewidths=0, alpha=0.7, rasterized=True,
        label=f"grid cells with a station ({len(occupied)})",
    )
    ax.scatter(
        stations["lon"], stations["lat"], s=10, marker="*",
        color="tab:red", edgecolors="k", linewidths=0.15, zorder=5,
        label=f"NGL stations ({len(stations)})",
    )

    ax.set_xlim(-5.25, 24.25)
    ax.set_ylim(36.25, 56.50)
    ax.set_aspect(1.0 / np.cos(np.radians(46.0)))
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(
        "Western and central Europe: 0.05° grid vs NGL station distribution"
    )
    ax.grid(alpha=0.15, ls=":")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(args.output, dpi=130, bbox_inches="tight")
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
