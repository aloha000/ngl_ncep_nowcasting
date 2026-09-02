from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from .constants import NCEP_VARS
from .dataset import NowcastData, make_loader
from .model import Model
from .plot import plot_loss_curve, plot_test_analysis

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, cfg, criterion, device) -> float:
    model.eval()
    losses = []
    for bx, by, bxm, bym, bxg, bxt in loader:
        bx = bx.float().to(device)
        by = by.float().to(device)
        if cfg.n_time_features > 0:
            bxm = bxm.float().to(device)
        else:
            bxm = None
        bxg = bxg.float().to(device)
        bxt = bxt.float().to(device)
        dec_inp = torch.zeros(bx.shape[0], cfg.pred_len, cfg.c_out, device=device)
        if cfg.n_time_features > 0:
            y_mark = torch.zeros(bx.shape[0], cfg.pred_len, cfg.n_time_features, device=device)
        else:
            y_mark = None
        out = model(bx, bxm, dec_inp, y_mark, x_geo=bxg, x_tgt=bxt)
        loss = criterion(out[:, -cfg.pred_len:, :], by[:, -cfg.pred_len:, :])
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def train(args: argparse.Namespace, data: NowcastData, device, cfg) -> tuple[nn.Module, Path]:
    train_ds, train_loader = make_loader(data, "train", args, shuffle=True)
    val_ds, val_loader = make_loader(data, "val", args, shuffle=False)
    print(f"[train] batches/epoch={len(train_loader)}  [val] batches={len(val_loader)}", flush=True)

    model = Model(cfg).float().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    setting = f"{args.model_id}_s{data.n_stations}_off{args.station_offset}_h{args.hour_stride}" \
              f"_dm{args.d_model}_el{args.e_layers}_nh{args.n_heads}_df{args.d_ff}"
    if args.spatial_enc:
        setting += "_sp"
    if args.target_h_feat:
        setting += "_thf"
    out_dir = args.out_root / setting
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                       f, sort_keys=False)

    data.save_scalers(out_dir)

    best_val = float("inf")
    patience = 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_losses = []
        t0 = time.time()
        for step, (bx, by, bxm, bym, bxg, bxt) in enumerate(train_loader, 1):
            bx = bx.float().to(device)
            by = by.float().to(device)
            if cfg.n_time_features > 0:
                bxm = bxm.float().to(device)
            else:
                bxm = None
            bxg = bxg.float().to(device)
            bxt = bxt.float().to(device)
            dec_inp = torch.zeros(bx.shape[0], cfg.pred_len, cfg.c_out, device=device)
            if cfg.n_time_features > 0:
                y_mark = torch.zeros(bx.shape[0], cfg.pred_len, cfg.n_time_features, device=device)
            else:
                y_mark = None
            optimizer.zero_grad()
            out = model(bx, bxm, dec_inp, y_mark, x_geo=bxg, x_tgt=bxt)
            loss = criterion(out[:, -cfg.pred_len:, :], by[:, -cfg.pred_len:, :])
            loss.backward()
            optimizer.step()
            ep_losses.append(loss.item())
            if step % 200 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {loss.item():.6f}", flush=True)

        train_loss = float(np.mean(ep_losses))
        val_loss = evaluate(model, val_loader, cfg, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"[epoch {epoch}] train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
              f"({time.time() - t0:.1f}s)", flush=True)
        with open(out_dir / "loss_curve.json", "w", encoding="utf-8") as f:
            json.dump({"train": train_losses, "val": val_losses}, f, indent=2)
        plot_loss_curve(train_losses, val_losses, out_dir)

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch},
                       out_dir / "checkpoint.pth")
            print(f"  saved best checkpoint (val_loss={val_loss:.6f})")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"  early stopping after {epoch} epochs")
                break

    ckpt = torch.load(out_dir / "checkpoint.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"loaded best checkpoint from epoch {ckpt['epoch']} (val_loss={best_val:.6f})")
    return model, out_dir


@torch.no_grad()
def test(args: argparse.Namespace, data: NowcastData, model: nn.Module, cfg, out_dir: Path, device):
    test_ds, test_loader = make_loader(data, "test", args, shuffle=False)
    print(f"[test] samples={len(test_ds)} batches={len(test_loader)}")

    preds_norm, trues_norm = [], []
    model.eval()
    for bx, by, bxm, bym, bxg, bxt in test_loader:
        bx = bx.float().to(device)
        if cfg.n_time_features > 0:
            bxm = bxm.float().to(device)
        else:
            bxm = None
        bxg = bxg.float().to(device)
        bxt = bxt.float().to(device)
        dec_inp = torch.zeros(bx.shape[0], cfg.pred_len, cfg.c_out, device=device)
        if cfg.n_time_features > 0:
            y_mark = torch.zeros(bx.shape[0], cfg.pred_len, cfg.n_time_features, device=device)
        else:
            y_mark = None
        out = model(bx, bxm, dec_inp, y_mark, x_geo=bxg, x_tgt=bxt)
        preds_norm.append(out[:, -cfg.pred_len:, :].detach().cpu().numpy())
        trues_norm.append(by[:, -cfg.pred_len:, :].numpy())
    preds_norm = np.concatenate(preds_norm, axis=0)
    trues_norm = np.concatenate(trues_norm, axis=0)

    preds = preds_norm * data.ys + data.ym
    trues = trues_norm * data.ys + data.ym

    metrics = {"variables": NCEP_VARS, "per_variable": []}
    for j, name in enumerate(NCEP_VARS):
        err = trues[:, :, j] - preds[:, :, j]
        mae = float(np.mean(np.abs(err)))
        mse = float(np.mean(err ** 2))
        metrics["per_variable"].append({"variable": name, "mae": mae, "mse": mse, "rmse": float(np.sqrt(mse))})
    err = trues - preds
    metrics["overall"] = {
        "mae": float(np.mean(np.abs(err))),
        "mse": float(np.mean(err ** 2)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
    }
    metrics["n_samples"] = int(trues.shape[0])

    np.savez(out_dir / "test_predictions.npz", preds=preds, trues=trues,
             preds_norm=preds_norm, trues_norm=trues_norm, variables=np.asarray(NCEP_VARS))
    plot_test_analysis(preds, trues, NCEP_VARS, out_dir)
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
