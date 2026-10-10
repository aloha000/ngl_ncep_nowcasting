#!/usr/bin/env python3
"""Compare NGL observations with nearest-cell FuXi and ERA5 ZTD.

The project preprocessing has already performed the expensive spatial and
vertical matching and stores the strictly aligned samples in
``dataset/era5_37lev_ztd_matched_57232.csv``.  Each row is the same
(observation time, GNSS station), with:

* ``obs_ztd``: raw NGL GNSS observation (mm)
* ``fx_ztd``: FuXi 24-h ZTD at the station's unified nearest grid cell (mm)
* ``ztd37``: ERA5 37-level ZTD at its nearest grid cell, referenced to the
  GNSS station height (mm)

This script reports overall and per-station bias/MAE/RMSE/correlation and
creates a compact comparison figure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_INPUT = ROOT / "dataset" / "era5_37lev_ztd_matched_57232.csv"


def statistics(obs: np.ndarray, model: np.ndarray) -> dict:
    valid = np.isfinite(obs) & np.isfinite(model)
    obs, model = obs[valid], model[valid]
    if obs.size == 0:
        return {"n": 0, "bias_mm": None, "mae_mm": None,
                "rmse_mm": None, "correlation": None}
    error = model - obs
    corr = np.corrcoef(model, obs)[0, 1] if obs.size > 1 else np.nan
    return {
        "n": int(obs.size),
        "bias_mm": float(error.mean()),
        "mae_mm": float(np.abs(error).mean()),
        "rmse_mm": float(np.sqrt(np.mean(error ** 2))),
        "correlation": float(corr) if np.isfinite(corr) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--era5", choices=("37", "13"), default="37",
                        help="ERA5 pressure-level product to compare (default: 37)")
    parser.add_argument("--start", help="inclusive start time, e.g. 2025-01-01")
    parser.add_argument("--end", help="inclusive end time, e.g. 2025-09-30 23:59")
    parser.add_argument("--out-dir", type=Path, default=HERE / "ztd_comparison")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input does not exist: {args.input}")
    df = pd.read_csv(args.input)
    era_col = "ztd37" if args.era5 == "37" else "sd_ztd_op13"
    required = {"time", "st", "station_id", "dist_km", "obs_ztd", "fx_ztd", era_col}
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in {args.input}: {sorted(missing)}")

    df["time"] = pd.to_datetime(df["time"])
    if args.start:
        df = df[df.time >= pd.Timestamp(args.start)]
    if args.end:
        df = df[df.time <= pd.Timestamp(args.end)]
    if df.empty:
        raise SystemExit("No samples remain after time filtering")

    # Use explicit, readable names in the output pair table.
    pairs = df[["time", "st", "station_id", "dist_km",
                "obs_ztd", "fx_ztd", era_col]].copy()
    pairs = pairs.rename(columns={
        "dist_km": "station_to_grid_km",
        "obs_ztd": "obs_ztd_mm",
        "fx_ztd": "fuxi_ztd_mm",
        era_col: "era5_ztd_mm",
    })

    summary = {
        "era5_levels": int(args.era5),
        "rows": int(len(pairs)),
        "times": int(pairs.time.nunique()),
        "stations": int(pairs.station_id.nunique()),
        "nearest_grid_distance_km": {
            "mean": float(pairs.station_to_grid_km.mean()),
            "max": float(pairs.station_to_grid_km.max()),
        },
        "fuxi_vs_obs": statistics(pairs.obs_ztd_mm.to_numpy(float),
                                   pairs.fuxi_ztd_mm.to_numpy(float)),
        "era5_vs_obs": statistics(pairs.obs_ztd_mm.to_numpy(float),
                                   pairs.era5_ztd_mm.to_numpy(float)),
    }

    station_rows = []
    for (st, station_id), part in pairs.groupby(["st", "station_id"], sort=True):
        for source, column in (("fuxi", "fuxi_ztd_mm"), ("era5", "era5_ztd_mm")):
            station_rows.append({
                "st": int(st), "station_id": station_id, "source": source,
                "station_to_grid_km": float(part.station_to_grid_km.iloc[0]),
                **statistics(part.obs_ztd_mm.to_numpy(float),
                             part[column].to_numpy(float)),
            })
    station_metrics = pd.DataFrame(station_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.out_dir / "ztd_nearest_pairs.csv", index=False)
    station_metrics.to_csv(args.out_dir / "ztd_nearest_station_metrics.csv", index=False)
    with open(args.out_dir / "ztd_nearest_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.no_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        obs = pairs.obs_ztd_mm.to_numpy(float)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=140)
        colors = {"FuXi 24h": ("fuxi_ztd_mm", "tab:blue"),
                  f"ERA5 {args.era5}L": ("era5_ztd_mm", "tab:orange")}
        for label, (column, color) in colors.items():
            model = pairs[column].to_numpy(float)
            axes[0].scatter(obs, model, s=3, alpha=.12, color=color,
                            rasterized=True, label=label)
            axes[1].hist(model - obs, bins=100, histtype="step", lw=1.3,
                         color=color, label=label)
        lo = float(np.nanmin([obs, pairs.fuxi_ztd_mm, pairs.era5_ztd_mm]))
        hi = float(np.nanmax([obs, pairs.fuxi_ztd_mm, pairs.era5_ztd_mm]))
        axes[0].plot([lo, hi], [lo, hi], "k--", lw=.8)
        axes[0].set(xlabel="Observed NGL ZTD (mm)", ylabel="Model ZTD (mm)",
                    title="Nearest-grid ZTD comparison")
        axes[1].axvline(0, color="k", lw=.8)
        axes[1].set(xlabel="Model - observation (mm)", ylabel="Count",
                    title="Error distribution")
        for axis in axes:
            axis.legend()
            axis.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(args.out_dir / "ztd_nearest_comparison.png")
        plt.close(fig)

    print(f"Outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
