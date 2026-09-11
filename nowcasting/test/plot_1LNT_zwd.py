#!/usr/bin/env python3
"""Plot 5-minute NGL Z*D for station 1LNT over a requested UTC interval."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr


ROOT = Path("/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/linan/linan_dev/gnss")
STORE = ROOT / "dataset" / "ngl_5min.zarr"
STATION = "1LNT"
START = pd.Timestamp("2024-05-01T00:00:00")
END = pd.Timestamp("2024-05-02T23:55:00")
OUT = ROOT / "nowcasting" / "test" / "1LNT_2024_0501_0502_ztd-zwd.png"


def main() -> None:
    group = zarr.open_group(str(STORE), mode="r")
    stations = np.asarray(group["station"][:]).astype(str).tolist()
    times = pd.DatetimeIndex(group["time"][:])
    t0 = times[0]
    i0 = int((START - t0).total_seconds() // 300)
    i1 = int((END - t0).total_seconds() // 300)
    station_idx = stations.index(STATION)
    values = np.asarray(group["ztd"][i0 : i1 + 1, station_idx]-group["zwd"][i0 : i1 + 1, station_idx], dtype=float)
    times = times[i0 : i1 + 1]

    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=160)
    ax.plot(times, values, color="#0072b2", lw=1.4, marker=".", ms=3,
            label=f"{STATION} ZTD-ZWD")
    ax.fill_between(times, values, min(values), color="#0072b2", alpha=0.12,
                    linewidth=0)
    ax.set_title(
        f"NGL station {STATION} zenith wet delay (ZTD-ZWD)\n"
        f"{START:%Y-%m-%d %H:%M} to {END:%Y-%m-%d %H:%M} UTC",
        loc="left",
        fontsize=13,
    )
    ax.set_xlabel("UTC time")
    ax.set_ylabel("ZTD-ZWD (mm)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 6)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.tick_params(axis="x", rotation=0)
    ax.set_xlim(START, END)
    ax.legend(loc="upper right")

    n_points = len(times)
    n_missing = int(np.isnan(values).sum())
    ax.text(
        0.01,
        0.01,
        f"{n_points} samples | {n_missing} missing | "
        f"min={np.nanmin(values):.1f} mm | max={np.nanmax(values):.1f} mm",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85},
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    print(f"[save] {OUT}")


if __name__ == "__main__":
    main()
