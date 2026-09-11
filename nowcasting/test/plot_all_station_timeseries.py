from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def plot_all_station_timeseries(out_dir: Path) -> dict:
    """Plot full test-period truth/pred curves for every station in predictions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(out_dir / "test_predictions.npz", allow_pickle=True)
    preds = np.asarray(z["preds"])[:, 0, :]
    trues = np.asarray(z["trues"])[:, 0, :]
    names = np.asarray(z["variables"]).astype(str).tolist()
    station_ids = np.asarray(z["station_ids"]).astype(str)
    time_utc = np.asarray(z["time_utc"])

    available = np.unique(station_ids)
    if not len(available):
        return {}

    summary = {}
    for station_id in available:
        idx = np.flatnonzero(station_ids == station_id)
        idx = idx[np.argsort(time_utc[idx])]

        fig, axes = plt.subplots(3, 2, figsize=(18, 11), sharex=True)
        for j, (ax, name) in enumerate(zip(axes.ravel(), names)):
            ax.plot(time_utc[idx], trues[idx, j], label="truth", lw=1.0)
            ax.plot(time_utc[idx], preds[idx, j], label="pred", lw=0.9, alpha=0.85)
            ax.set_title(name)
            ax.grid(alpha=0.3)
            if j == 0:
                ax.legend(fontsize=8)

        fig.suptitle(
            f"Test set full time series: {station_id} "
            f"(n={len(idx)}, {str(time_utc[idx].min())[:10]} .. {str(time_utc[idx].max())[:10]})"
        )
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out_dir / f"test_timeseries_{station_id}.png", dpi=150)
        plt.close(fig)

        summary[str(station_id)] = {
            "n_samples": int(len(idx)),
            "first_utc": str(time_utc[idx].min()),
            "last_utc": str(time_utc[idx].max()),
        }

    (out_dir / "test_all_station_timeseries.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/linan/linan_dev/gnss/"
            "nowcasting/outputs/ztd_only_min100-max131_valid_60-80-131_s14_off0_h2_"
            "dm128_el2_nh4_df256_spconcat_thf_const_dt"
        ),
    )
    args = p.parse_args()
    summary = plot_all_station_timeseries(args.out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
