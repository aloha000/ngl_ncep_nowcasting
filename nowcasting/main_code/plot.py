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
        ax.scatter(t, pr, s=2, alpha=0.3, rasterized=True)
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
