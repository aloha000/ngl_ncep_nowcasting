#!/usr/bin/env python3
"""Evaluate a trained checkpoint on a split and plot the results.

Outputs (into ``<work_dir>/plots``):
  loss_curve.png          train/val loss + learning rate
  channel_metrics.png     per-channel MAE, analysis vs FuXi background
  maps_<channel>.png      ERA5 truth / background / analysis / analysis-background
  timeseries.png          domain-mean time series over the split
  tp_maps.png             IMERG tp: truth vs analysis
  metrics.csv/.json       the numbers behind the plots

The background baseline is always evaluated on the *same* channels as the
analysis, so the comparison is apples to apples (the training log compared a
69-channel background loss with a 70-channel model loss).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "preprocessing"))   # da_ngl codes (era5 stats)

from main.model import AssimilationNetv6                     # noqa: E402
from main.utils import AssimilationDataset                   # noqa: E402
from train_FSDP import process_bg, process_obs               # noqa: E402
from common import (CHANNELS, LABEL_CHANNELS, TRAIN_LABEL_CHANNELS,  # noqa: E402
                    era5_channel_stats)


def latest_checkpoint(work_dir: Path) -> Path:
    ckpts = sorted((work_dir / "model").glob("iteration_*.pth"),
                   key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        raise SystemExit(f"no checkpoints under {work_dir}/model")
    return ckpts[-1]


def load_model(cfg, ckpt: Path, device):
    model = AssimilationNetv6(bg_chans=cfg.model_bg_chans,
                              obs_chans=cfg.model_obs_chans,
                              obs_frames=cfg.model_obs_frames,
                              out_chans=cfg.model_out_chans,
                              embed_dim=cfg.model_embed_dim,
                              depth=cfg.model_depth)
    try:
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(ckpt, map_location="cpu")
    state = payload["model"]
    # drop any FSDP/DDP prefixes just in case
    clean = {}
    for k, v in state.items():
        for pre in ("_fsdp_wrapped_module.", "module."):
            while k.startswith(pre):
                k = k[len(pre):]
        clean[k] = v
    missing, unexpected = model.load_state_dict(clean, strict=False)
    if missing or unexpected:
        print(f"[warn] missing={list(missing)[:3]} unexpected={list(unexpected)[:3]}")
    model.to(device).eval()
    return model, payload.get("iteration")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", default="configs")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--split", choices=("val", "test", "train"), default="test")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import importlib
    cfg = importlib.import_module(args.configs)
    work_dir = Path(cfg.work_dir)
    out_dir = args.out or (work_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = args.checkpoint or latest_checkpoint(work_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] checkpoint {ckpt}")
    model, iteration = load_model(cfg, ckpt, device)
    print(f"[eval] iteration {iteration} | split {args.split} | device {device}")

    dates = {"train": cfg.dates_train_range, "val": cfg.dates_val_range,
             "test": cfg.dates_test_range}[args.split]
    # read every channel of the store (incl. the eval-only ``era5_tp``); the
    # model itself only ever consumes the first ``n_ch`` channels
    dataset = AssimilationDataset(cfg, dates, n_label_chans=len(LABEL_CHANNELS))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device == "cuda"))

    lat = np.asarray(cfg.lat, dtype=np.float64)
    wlat = np.cos(np.deg2rad(lat))
    wlat = (wlat / wlat.mean()).astype(np.float64)
    # (1, lat, 1): broadcasts against (B, C, H, W)
    w3d = torch.as_tensor(wlat, dtype=torch.float32, device=device).view(1, -1, 1)

    mean_era5, std_era5 = era5_channel_stats()
    mean_era5, std_era5 = mean_era5[:70], std_era5[:70]
    n_ch = len(TRAIN_LABEL_CHANNELS)    # 70 = model I/O (ERA5 69 + IMERG tp)
    n_bg = len(CHANNELS)                # 69 (the background has no tp)

    acc = {
        "model_abs": np.zeros(n_ch), "model_sq": np.zeros(n_ch), "model_bias": np.zeros(n_ch),
        "bg_abs": np.zeros(n_bg), "bg_sq": np.zeros(n_bg), "bg_bias": np.zeros(n_bg),
        "clim_abs": np.zeros(n_ch), "delta_abs": np.zeros(n_bg),
        "wsum": np.zeros(n_ch), "wsum_bg": np.zeros(n_bg),
    }
    series_model, series_bg, series_true = [], [], []
    series_etp, series_etp_err = [], []
    etp_stats = {"vs_imerg": 0.0, "model_vs_etp": 0.0, "n": 0.0}
    series_an_err, series_bg_err = [], []
    keep_maps, keep_tp = [], []

    label_time = dataset.label_time
    # label_time is the whole store axis (2022-01-01 onward); a split only
    # uses a slice of it, so index by the sampled lb_i -- otherwise the
    # series/map labels silently show the stores first n timestamps.
    sample_times = label_time[[s[0] for s in dataset.samples]]
    with torch.no_grad():
        for bi, (batch_fcst, batch_obs, batch_era5) in enumerate(loader):
            bg = process_bg(batch_fcst, device)
            obs = process_obs(batch_obs, cfg, device)
            full = batch_era5.to(device).float()               # (B, 71, H, W)
            era5_tp = full[:, -1]                              # evaluation-only channel
            truth = full[:, :n_ch].unsqueeze(1)                # model I/O channels
            out = model(bg, obs).float()

            # drop the singleton time dim: work in (B, C, H, W)
            B = truth.shape[0]
            truth = truth[:, 0]
            out = out[:, 0]
            bg69 = bg[:, 0, :69]

            valid = torch.isfinite(truth)
            m = valid.float()
            m69 = m[:, :69]
            w = w3d
            tgt = torch.nan_to_num(truth)
            tgt69 = torch.nan_to_num(truth[:, :69])

            bias_m = out - tgt
            bias_b = bg69 - tgt69

            acc["model_abs"] += (bias_m.abs() * w * m).sum(dim=(0, 2, 3)).cpu().numpy()
            acc["model_sq"] += (bias_m ** 2 * w * m).sum(dim=(0, 2, 3)).cpu().numpy()
            acc["model_bias"] += (bias_m * w * m).sum(dim=(0, 2, 3)).cpu().numpy()
            acc["wsum"] += (w * m).sum(dim=(0, 2, 3)).cpu().numpy()

            # climatology baseline: predict the standardised mean (= 0)
            acc["clim_abs"] += (tgt.abs() * w * m).sum(dim=(0, 2, 3)).cpu().numpy()
            # how far the analysis moves away from the background
            acc["delta_abs"] += ((out[:, :n_bg] - bg69).abs() * w * m69).sum(dim=(0, 2, 3)).cpu().numpy()

            acc["bg_abs"] += (bias_b.abs() * w * m69).sum(dim=(0, 2, 3)).cpu().numpy()
            acc["bg_sq"] += (bias_b ** 2 * w * m69).sum(dim=(0, 2, 3)).cpu().numpy()
            acc["bg_bias"] += (bias_b * w * m69).sum(dim=(0, 2, 3)).cpu().numpy()
            acc["wsum_bg"] += (w * m69).sum(dim=(0, 2, 3)).cpu().numpy()

            # latitude-weighted domain means, per sample
            den_m = (w * m).sum(dim=(2, 3)).clamp_min(1e-6)      # (B, 70)
            m_etp = torch.isfinite(era5_tp).float()
            den_etp = (w[:, 0] * m_etp).sum(dim=(1, 2)).clamp_min(1e-6)
            series_etp.append(((torch.nan_to_num(era5_tp) * w[:, 0] * m_etp).sum(dim=(1, 2))
                               / den_etp).cpu().numpy())
            # domain-mean |ERA5 tp - IMERG tp| (product disagreement)
            a_etp = (era5_tp - tgt[:, 69]).abs() * w[:, 0] * m_etp
            series_etp_err.append((a_etp.sum(dim=(1, 2)) / den_etp).cpu().numpy())
            both = (torch.isfinite(era5_tp) & torch.isfinite(truth[:, 69])).float()
            etp_stats["vs_imerg"] += float(((era5_tp - torch.nan_to_num(truth[:, 69])).abs()
                                             * both).sum())
            etp_stats["model_vs_etp"] += float(((out[:, 69] - torch.nan_to_num(era5_tp)).abs()
                                                 * both).sum())
            etp_stats["n"] += float(both.sum())
            den_b = (w * m69).sum(dim=(2, 3)).clamp_min(1e-6)    # (B, 69)
            series_model.append(((out * w * m).sum(dim=(2, 3)) / den_m).cpu().numpy())
            series_true.append(((tgt * w * m).sum(dim=(2, 3)) / den_m).cpu().numpy())
            series_bg.append(((bg69 * w * m69).sum(dim=(2, 3)) / den_b).cpu().numpy())
            # per-step domain-mean absolute error (this is where the difference shows)
            series_an_err.append(((bias_m[:, :n_bg].abs() * w * m69).sum(dim=(2, 3)) / den_b).cpu().numpy())
            series_bg_err.append(((bias_b.abs() * w * m69).sum(dim=(2, 3)) / den_b).cpu().numpy())

            idx = np.arange(bi * args.batch_size, bi * args.batch_size + B)
            if len(keep_maps) < 3:
                for k in range(B):
                    if len(keep_maps) < 3:
                        keep_maps.append((idx[k], bg69[k].cpu().numpy(), out[k].cpu().numpy(),
                                          truth[k].cpu().numpy()))
            if len(keep_tp) < 2:
                for k in range(B):
                    if len(keep_tp) < 2:
                        keep_tp.append((idx[k], out[k, 69].cpu().numpy(),
                                        truth[k, 69].cpu().numpy(), era5_tp[k].cpu().numpy()))
            if bi % 50 == 0:
                print(f"  batch {bi}/{len(loader)}", flush=True)

    def pad_bg(v):
        out = np.full(n_ch, np.nan)
        out[:n_bg] = v
        return out

    mae_model = acc["model_abs"] / np.maximum(acc["wsum"], 1e-9)
    rmse_model = np.sqrt(acc["model_sq"] / np.maximum(acc["wsum"], 1e-9))
    bias_model = acc["model_bias"] / np.maximum(acc["wsum"], 1e-9)
    mae_clim = acc["clim_abs"] / np.maximum(acc["wsum"], 1e-9)
    delta = acc["delta_abs"] / np.maximum(acc["wsum_bg"], 1e-9)
    mae_bg = acc["bg_abs"] / np.maximum(acc["wsum_bg"], 1e-9)
    rmse_bg = np.sqrt(acc["bg_sq"] / np.maximum(acc["wsum_bg"], 1e-9))
    bias_bg = acc["bg_bias"] / np.maximum(acc["wsum_bg"], 1e-9)

    # physical units: the stores are standardised with mean_era5 / std_era5
    mae_model_phys = mae_model * std_era5
    mae_bg_phys = pad_bg(mae_bg * std_era5[:n_bg])
    # tp is log1p-standardised: report the difference in the log space (mm is non-linear)
    df = pd.DataFrame({
        "channel": TRAIN_LABEL_CHANNELS,
        "mae_analysis_std": mae_model,
        "mae_bg_std": pad_bg(mae_bg),
        "rmse_analysis_std": rmse_model,
        "rmse_bg_std": pad_bg(rmse_bg),
        "bias_analysis_std": bias_model,
        "bias_bg_std": pad_bg(bias_bg),
        "std_era5": std_era5,
        "mae_climatology_std": mae_clim,
        "mean_abs_analysis_minus_bg": pad_bg(delta),
        "mae_analysis_phys": mae_model_phys,
        "mae_bg_phys": mae_bg_phys,
        "improve_pct": 100.0 * (mae_bg.mean() - mae_model[:n_bg].mean()) / mae_bg.mean(),
    })
    df.to_csv(out_dir / "metrics.csv", index=False)

    summary = {
        "checkpoint": str(ckpt), "iteration": iteration, "split": args.split,
        "n_samples": len(dataset),
        "mae_analysis_std_69ch": float(mae_model[:n_bg].mean()),
        "mae_bg_std_69ch": float(mae_bg.mean()),
        "improve_pct_69ch": float(100 * (mae_bg.mean() - mae_model[:n_bg].mean()) / mae_bg.mean()),
        "mae_analysis_std_70ch": float(mae_model.mean()),
        "mae_tp_std": float(mae_model[69]),
        "mae_era5tp_vs_imerg_std": etp_stats["vs_imerg"] / max(etp_stats["n"], 1.0),
        "mae_modeltp_vs_era5tp_std": etp_stats["model_vs_etp"] / max(etp_stats["n"], 1.0),
        "n_channels_better": int((mae_model[:n_bg] < mae_bg).sum()),
        "mae_climatology_std_69ch": float(mae_clim[:n_bg].mean()),
        "mean_abs_analysis_minus_bg_69ch": float(delta.mean()),
        "mae_tp_climatology_std": float(mae_clim[69]),
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # ---------------------------------------------------------------- plots
    tr = np.load(work_dir / "train_loss.npy")
    va = np.load(work_dir / "val_loss.npy")
    lr = np.load(work_dir / "lr.npy")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), dpi=130)
    ep = np.arange(1, len(tr) + 1)
    ax[0].plot(ep, tr, "-o", ms=3, label="train")
    ax[0].plot(ep, va, "-s", ms=3, label="val")
    bg_tr = 0.0938
    ax[0].axhline(bg_tr, ls="--", c="0.4", lw=1, label=f"background (logged, 69ch)={bg_tr:.4f}")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("MAE (standardised)")
    ax[0].set_title("loss"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].plot(np.arange(1, len(lr) + 1), lr)
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel("lr"); ax[1].set_yscale("log")
    ax[1].set_title("learning rate"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out_dir / "loss_curve.png"); plt.close(fig)

    # per-channel MAE
    order = np.argsort(-mae_bg)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5), dpi=130)
    x = np.arange(69)
    ax[0].bar(x - .2, mae_bg[order], .4, label="background (FuXi)")
    ax[0].bar(x + .2, mae_model[:n_bg][order], .4, label="analysis (model)")
    ax[0].set_xticks(x); ax[0].set_xticklabels(np.array(CHANNELS)[order], rotation=90, fontsize=6)
    ax[0].set_ylabel("MAE (standardised)"); ax[0].legend(fontsize=8)
    ax[0].set_title("per-channel MAE, sorted by background error"); ax[0].grid(alpha=.3, axis="y")
    rel = 100 * (mae_bg - mae_model[:n_bg]) / mae_bg
    ax[1].bar(x, rel[order], .6,
              color=np.where(rel[order] > 0, "tab:green", "tab:red"))
    ax[1].axhline(0, c="k", lw=.8)
    ax[1].set_xticks(x); ax[1].set_xticklabels(np.array(CHANNELS)[order], rotation=90, fontsize=6)
    ax[1].set_ylabel("MAE reduction vs background (%)")
    ax[1].set_title("improvement (positive = analysis better)"); ax[1].grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(out_dir / "channel_metrics.png"); plt.close(fig)

    # maps
    map_chans = [("z500", 7), ("t850", 23), ("r700", 61), ("t2m", 65), ("msl", 68)]
    vmax_all = float(np.nanpercentile(np.abs(np.concatenate(
        [np.ravel(m[2]) for m in keep_maps] + [np.ravel(m[3]) for m in keep_maps])), 99))
    for cname, ci in map_chans:
        fig, axes = plt.subplots(1, 4, figsize=(20, 3.6), dpi=130)
        idx, bgk, outk, truthk = keep_maps[0]
        t = sample_times[idx]
        panels = [(truthk[ci], f"ERA5 truth {cname} {t}", "RdBu_r", None),
                  (bgk[ci], "FuXi background", "RdBu_r", None),
                  (outk[ci], "analysis (model)", "RdBu_r", None),
                  (outk[ci] - bgk[ci], "analysis - background", "coolwarm", None)]
        for a, (data, title, cmap, _) in zip(axes, panels):
            lim = np.nanpercentile(np.abs(data), 99) or 1.0
            im = a.imshow(data, origin="lower", cmap=cmap, vmin=-lim, vmax=lim,
                          extent=[cfg.lon.min(), cfg.lon.max(), cfg.lat.min(), cfg.lat.max()],
                          aspect="auto")
            a.set_title(title, fontsize=9); plt.colorbar(im, ax=a, fraction=.046)
        fig.suptitle(f"{cname} (standardised units)", y=1.02)
        fig.tight_layout(); fig.savefig(out_dir / f"maps_{cname}.png", bbox_inches="tight"); plt.close(fig)

    # ------------------------------------------------------------- time series
    # NOTE: the raw domain-mean curves of FuXi and ERA5 overlap almost exactly
    # (the domain-mean error is ~0.013 while the signal is ~0.66), so the
    # informative panels here are the per-step domain-mean |error| curves.
    sm = np.concatenate(series_model, 0)
    st = np.concatenate(series_true, 0)
    sb = np.concatenate(series_bg, 0)
    ea = np.concatenate(series_an_err, 0)
    eb = np.concatenate(series_bg_err, 0)
    n = sm.shape[0]
    times = sample_times[:n]
    pick = [("z500", 7), ("t850", 23), ("r700", 61), ("t2m", 65), ("u10", 66), ("msl", 68)]

    fig, axes = plt.subplots(len(pick), 2, figsize=(15, 2.1 * len(pick)), dpi=130, sharex=True)
    for row, (cname, ci) in enumerate(pick):
        a0 = axes[row, 0]
        a0.plot(times, st[:, ci], "k-", lw=1.1, label="ERA5 truth")
        a0.plot(times, sb[:, ci], "--", c="tab:orange", lw=1, label="FuXi background")
        a0.plot(times, sm[:, ci], "-", c="tab:blue", lw=1, label="analysis")
        gap = np.abs(sb[:, ci] - st[:, ci]).mean()
        a0.set_ylabel(f"{cname}\ndomain mean", fontsize=8)
        a0.grid(alpha=.3)
        if row == 0:
            a0.legend(fontsize=7, ncol=3, loc="lower left")
            a0.set_title("domain mean  (FuXi/ERA5 differ by only ~%.3f here)" % gap, fontsize=9)

        a1 = axes[row, 1]
        a1.plot(times, eb[:, ci], "--", c="tab:orange", lw=1,
                label=f"|background - truth|  mean {eb[:, ci].mean():.4f}")
        a1.plot(times, ea[:, ci], "-", c="tab:blue", lw=1,
                label=f"|analysis - truth|  mean {ea[:, ci].mean():.4f}")
        a1.set_ylabel("domain-mean |err|", fontsize=8)
        a1.grid(alpha=.3)
        a1.legend(fontsize=7, loc="upper left")
        if row == 0:
            a1.set_title("per-step domain-mean absolute error", fontsize=9)

    axes[-1, 0].set_xlabel("time"); axes[-1, 1].set_xlabel("time")
    fig.suptitle(f"test-set time series ({args.split})", y=1.0)
    fig.tight_layout(); fig.savefig(out_dir / "timeseries.png", bbox_inches="tight"); plt.close(fig)

    # tp separately (no background available)
    fig, ax = plt.subplots(2, 1, figsize=(13, 5), dpi=130, sharex=True)
    setp = np.concatenate(series_etp, 0)
    ax[0].plot(times, st[:, 69], "k-", lw=1.1, label="IMERG tp (training target)")
    ax[0].plot(times, setp, "-", c="tab:green", lw=1, label="ERA5 tp (reference)")
    ax[0].plot(times, sm[:, 69], "-", c="tab:blue", lw=1, label="analysis tp")
    ax[0].set_ylabel("tp domain mean"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].plot(times, np.abs(sm[:, 69] - st[:, 69]), "-", c="tab:blue", lw=1,
               label=f"|analysis - IMERG| mean {np.abs(sm[:, 69] - st[:, 69]).mean():.4f}")
    ax[1].plot(times, np.abs(sm[:, 69] - setp), "-", c="tab:green", lw=1,
               label=f"|analysis - ERA5tp| mean {np.abs(sm[:, 69] - setp).mean():.4f}")
    ax[1].plot(times, np.abs(setp - st[:, 69]), "--", c="0.45", lw=1,
               label=f"|ERA5tp - IMERG| mean {np.abs(setp - st[:, 69]).mean():.4f}")
    ax[1].axhline(np.abs(st[:, 69]).mean(), ls="--", c="0.4", lw=1,
                  label=f"climatology mean {np.abs(st[:, 69]).mean():.4f}")
    ax[1].set_ylabel("tp domain-mean |err|"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    ax[1].set_xlabel("time")
    fig.suptitle("tp (standardised log1p space)", y=1.0)
    fig.tight_layout(); fig.savefig(out_dir / "timeseries_tp.png", bbox_inches="tight"); plt.close(fig)

    # tp maps
    if keep_tp:
        idx, tp_out, tp_true, tp_era5 = keep_tp[0]
        fig, axes = plt.subplots(1, 4, figsize=(21, 3.6), dpi=130)
        t = sample_times[idx]
        for a, (d, title) in zip(axes, [(tp_true, f"IMERG tp truth {t}"),
                                        (tp_out, "analysis tp"),
                                        (tp_era5, "ERA5 tp (reference)"),
                                        (tp_out - tp_era5, "analysis - ERA5 tp")]):
            lim = np.nanpercentile(np.abs(d), 99) or 1.0
            cmap = "coolwarm" if "diff" in title or "- truth" in title else "viridis"
            im = a.imshow(d, origin="lower", cmap=cmap,
                          vmin=(-lim if cmap == "coolwarm" else None),
                          vmax=(lim if cmap == "coolwarm" else np.nanpercentile(d, 99)),
                          extent=[cfg.lon.min(), cfg.lon.max(), cfg.lat.min(), cfg.lat.max()],
                          aspect="auto")
            a.set_title(title, fontsize=9); plt.colorbar(im, ax=a, fraction=.046)
        fig.suptitle("tp (standardised log1p space)", y=1.02)
        fig.tight_layout(); fig.savefig(out_dir / "tp_maps.png", bbox_inches="tight"); plt.close(fig)

    print(f"[done] figures -> {out_dir}")


if __name__ == "__main__":
    main()
