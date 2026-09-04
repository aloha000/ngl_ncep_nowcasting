from __future__ import annotations

from pathlib import Path

import numpy as np

def plot_loss_curve(train_losses: list[float], val_losses: list[float], out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(train_losses) + 1))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, "o-", label="train")
    ax.plot(epochs, val_losses, "s-", label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss (normalized)")
    ax.set_title("Train / validation loss per epoch")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close(fig)


def plot_random_station_timeseries(preds, trues, names: list[str], station_ids, time_utc, out_dir: Path, seed: int, n_stations: int = 5):
    """Save full test-period truth/prediction plots for reproducibly random stations."""
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    preds = np.asarray(preds)[:, 0, :]
    trues = np.asarray(trues)[:, 0, :]
    station_ids = np.asarray(station_ids, dtype=str)
    time_utc = np.asarray(time_utc)
    available = np.unique(station_ids)
    if not len(available):
        return
    selected = np.random.default_rng(seed).choice(available, size=min(n_stations, len(available)), replace=False)
    (out_dir / "test_random_station_timeseries.json").write_text(
        json.dumps({"seed": int(seed), "station_ids": selected.tolist()}, indent=2), encoding="utf-8"
    )
    for station_id in selected:
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
        fig.suptitle(f"Test set full time series: {station_id}")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out_dir / f"test_timeseries_{station_id}.png", dpi=150)
        plt.close(fig)


def plot_test_analysis(preds, trues, names: list[str], out_dir: Path):
    """Scatter, error histogram and time-series snippet for the test predictions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    preds = np.asarray(preds)[:, 0, :]  # (N, V)
    trues = np.asarray(trues)[:, 0, :]
    n_vars = len(names)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols

    # 1) scatter pred vs truth with 1:1 line and R2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 9))
    for j, (ax, name) in enumerate(zip(np.ravel(axes), names)):
        t, pr = trues[:, j], preds[:, j]
        # A scatter plot hides the concentration of this multi-million-sample
        # test set. Hexagonal bins retain the truth/prediction relationship
        # while colour shows the local sample count on a log scale.
        density = ax.hexbin(
            t, pr, gridsize=90, mincnt=1, bins="log", cmap="viridis",
            linewidths=0, rasterized=True,
        )
        colorbar = fig.colorbar(density, ax=ax, pad=0.02)
        colorbar.set_label("sample count (log scale)", fontsize=8)
        colorbar.ax.tick_params(labelsize=7)
        lo = float(min(t.min(), pr.min()))
        hi = float(max(t.max(), pr.max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ss_res = float(((t - pr) ** 2).sum())
        ss_tot = float(((t - t.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        ax.set_title(f"{name}  R2={r2:.3f}")
        ax.set_xlabel("truth")
        ax.set_ylabel("pred")
    for ax in np.ravel(axes)[n_vars:]:
        ax.set_visible(False)
    fig.suptitle("Test set: prediction vs truth (best checkpoint)")
    fig.tight_layout()
    fig.savefig(out_dir / "test_scatter.png", dpi=150)
    plt.close(fig)

    # 2) error histograms
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 9))
    for j, (ax, name) in enumerate(zip(np.ravel(axes), names)):
        err = trues[:, j] - preds[:, j]
        ax.hist(err, bins=100, alpha=0.7)
        ax.axvline(0, color="r", lw=1)
        ax.set_title(f"{name}  err std={err.std():.3f}")
        ax.set_xlabel("error (truth - pred)")
    for ax in np.ravel(axes)[n_vars:]:
        ax.set_visible(False)
    fig.suptitle("Test set: prediction error distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "test_error_hist.png", dpi=150)
    plt.close(fig)

    # 3) time-series snippet (first samples in flattened station order)
    m = min(300, len(trues))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8))
    for j, (ax, name) in enumerate(zip(np.ravel(axes), names)):
        ax.plot(trues[:m, j], label="truth", lw=1)
        ax.plot(preds[:m, j], label="pred", lw=1, alpha=0.8)
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    for ax in np.ravel(axes)[n_vars:]:
        ax.set_visible(False)
    fig.suptitle(f"Test set: first {m} samples per variable (best checkpoint)")
    fig.tight_layout()
    fig.savefig(out_dir / "test_timeseries.png", dpi=150)
    plt.close(fig)
