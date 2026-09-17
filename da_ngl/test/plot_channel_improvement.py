#!/usr/bin/env python3
"""Per-channel MAE change of one run (analysis vs background), all 70 channels.

NOTE: this container has no CJK font, so every label is in English.

``plot_results.py`` writes the numbers into the run folder as
``metrics.csv`` (all 9600 grid cells) and ``metrics_station.csv`` (the 1378
station cells).  This turns them into one figure per run:

    channel_improvement.png
        row 1 : improvement [%]  = 100 * (bg - analysis) / bg      (symlog)
        row 2 : absolute change  = analysis - bg  [standardised MAE]

Left column = all grid cells, right column = station cells only.  The bar order
is the background error on the full grid, so the channels that carry the most
absolute error stay on the left; bars are coloured by variable family.

Usage (from da_ngl/main_code):
    python ../test/plot_channel_improvement.py <run_dir> [--out <png>]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

FAMILIES = (("z", "tab:blue"), ("t", "tab:red"), ("u", "tab:green"),
            ("v", "tab:orange"), ("r", "tab:purple"))
SURFACE = {"t2m": "tab:brown", "u10": "tab:brown", "v10": "tab:brown", "msl": "tab:brown"}
TP = {"era5_tp", "imerg_tp", "tp"}


def colour(channel: str) -> str:
    if channel in TP:
        return "tab:cyan"
    if channel in SURFACE:
        return SURFACE[channel]
    for pre, col in FAMILIES:
        if channel.startswith(pre) and channel[len(pre):].isdigit():
            return col
    return "0.5"


def family(channel: str) -> str:
    if channel in TP:
        return "tp"
    if channel in SURFACE:
        return "surface"
    for pre, _ in FAMILIES:
        if channel.startswith(pre) and channel[len(pre):].isdigit():
            return pre
    return "other"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path, help="a {model_id}_{exp_tag} results folder")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--top-annotate", type=int, default=8,
                    help="annotate this many largest |improvement| values per panel")
    args = ap.parse_args()

    full = pd.read_csv(args.run_dir / "metrics.csv").set_index("channel")
    stat = pd.read_csv(args.run_dir / "metrics_station.csv").set_index("channel")
    order = full["mae_bg_std"].sort_values(ascending=False).index
    cols = list(order)
    x = np.arange(len(cols))
    colours = [colour(c) for c in cols]

    views = (
        ("all 9600 grid cells", full.loc[cols],
         "mae_bg_std", "mae_analysis_std", "improve_pct"),
        ("1378 station cells only", stat.loc[cols], "mae_bg_std_station",
         "mae_analysis_std_station", "improve_pct_station"),
    )

    fig, axes = plt.subplots(2, 2, figsize=(19, 9.5), dpi=130)
    for col, (name, tbl, cbg, can, cimp) in enumerate(views):
        imp = tbl[cimp].to_numpy(dtype=float)
        delta = (tbl[can] - tbl[cbg]).to_numpy(dtype=float)

        # ---- row 1: relative improvement (symlog, msl is far off scale) -----
        ax = axes[0, col]
        ax.bar(x, imp, color=colours, width=.75)
        ax.axhline(0, c="k", lw=.8)
        ax.set_yscale("symlog", linthresh=1.0, linscale=.6)
        # msl can be -100 % while every other channel is inside a few %, so scale
        # to the actual data instead of wasting the panel on empty space
        ax.set_ylim(min(-3.0, float(np.nanmin(imp)) * 1.4),
                    max(3.0, float(np.nanmax(imp)) * 1.4))
        for i in np.argsort(-np.abs(imp))[:args.top_annotate]:
            ax.annotate(f"{imp[i]:+.1f}", (x[i], imp[i]), fontsize=6.5,
                        ha="center", va="bottom" if imp[i] > 0 else "top",
                        rotation=90, color="k")
        bg_mean = tbl[cbg].mean()
        mv = "mean_abs_analysis_minus_bg" + ("_station" if col else "")
        ax.set_title(f"{name}: per-channel improvement [%], symlog\n"
                     f"69ch mean {100*np.nansum(delta[:69])/69/bg_mean:+.3f}%   "
                     f"mean |analysis - bg| {np.nanmean(tbl[mv]):.4f}",
                     fontsize=10)
        ax.set_ylabel("improvement [%]  (positive = analysis better)")

        # ---- row 2: absolute change in MAE (what the accounting sums) ------
        ax = axes[1, col]
        ax.bar(x, delta, color=colours, width=.75)
        ax.axhline(0, c="k", lw=.8)
        for i in np.argsort(-np.abs(delta))[:args.top_annotate]:
            ax.annotate(f"{delta[i]:+.3f}", (x[i], delta[i]), fontsize=6.5,
                        ha="center", va="bottom" if delta[i] > 0 else "top",
                        rotation=90, color="k")
        ax.set_title(f"absolute change, analysis MAE - background MAE (standardised)\n"
                     f"69ch sum {np.nansum(delta[:69]):+.5f}  "
                     f"(msl {delta[cols.index('msl')]:+.5f}, "
                     f"other 68ch {np.nansum(delta[:69]) - delta[cols.index('msl')]:+.5f})",
                     fontsize=10)
        ax.set_ylabel("delta MAE  (positive = worse)")

        for r in (0, 1):
            axes[r, col].set_xticks(x)
            axes[r, col].set_xticklabels(cols, rotation=90, fontsize=6.5)
            for tick, c in zip(axes[r, col].get_xticklabels(), colours):
                tick.set_color(c)
            axes[r, col].grid(alpha=.3, axis="y")

    handles = [Line2D([], [], marker="s", ls="", color=col, label=lbl)
               for lbl, col in (("z* geopotential", "tab:blue"),
                                ("t* temperature", "tab:red"),
                                ("u* wind", "tab:green"), ("v* wind", "tab:orange"),
                                ("r* rel. humidity", "tab:purple"),
                                ("surface t2m/u10/v10/msl", "tab:brown"),
                                ("tp precipitation", "tab:cyan"))]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=9, frameon=False)
    fig.suptitle(f"{args.run_dir.name}\n"
                 f"per-channel MAE change (test split, latitude-weighted, "
                 f"channels sorted by background error)", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    out = args.out or args.run_dir / "channel_improvement.png"
    fig.savefig(out)
    print(f"[done] {out}")

    # also drop the tidy per-channel table next to the figure
    tidy = pd.DataFrame({
        "channel": cols,
        "family": [family(c) for c in cols],
        "mae_bg_full": full.loc[cols, "mae_bg_std"].to_numpy(),
        "mae_analysis_full": full.loc[cols, "mae_analysis_std"].to_numpy(),
        "improve_pct_full": full.loc[cols, "improve_pct"].to_numpy(),
        "delta_full": (full.loc[cols, "mae_analysis_std"]
                       - full.loc[cols, "mae_bg_std"]).to_numpy(),
        "mae_bg_station": stat.loc[cols, "mae_bg_std_station"].to_numpy(),
        "mae_analysis_station": stat.loc[cols, "mae_analysis_std_station"].to_numpy(),
        "improve_pct_station": stat.loc[cols, "improve_pct_station"].to_numpy(),
        "delta_station": (stat.loc[cols, "mae_analysis_std_station"]
                          - stat.loc[cols, "mae_bg_std_station"]).to_numpy(),
    })
    csv = (args.out.with_suffix(".csv") if args.out
           else args.run_dir / "channel_improvement.csv")
    tidy.to_csv(csv, index=False)
    print(f"[done] {csv}")


if __name__ == "__main__":
    main()
