#!/usr/bin/env python3
"""Train/test an iTransformer nowcasting model on GNSS -> NCEP surface data.

Task
----
Use hourly NGL ZTD/ZWD of the nearest GNSS stations over T-6..T to nowcast
the NCEP surface variables (p, slp, t2m, r2m, u10, v10) at time T.

Temporal splits (UTC; [start, end) half-open ranges):
    train: 2018-01-01T00:00 <= T <  2023-11-01T00:00
    val:   2023-11-01T00:00 <= T <  2024-03-01T00:00
    test:  2024-03-01T00:00 <= T <= last available hour

Input tensor per sample: 7 x (2*max_neighbors + max_neighbors) channels, i.e.
ztd/zwd per neighbor (padded with zeros when missing/invalid) plus one 0/1
validity-mask channel per neighbor. Target tensor: the 6 NCEP variables at T
(z-scored with train statistics when --target-scale is on).

Usage
-----
    python nowcasting/train_iTransformer_nowcast.py            # all params in nowcasting/config.yaml
    python nowcasting/train_iTransformer_nowcast.py --config my_config.yaml
    python nowcasting/train_iTransformer_nowcast.py --set stations=128 --set epochs=20
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader, Dataset

import yaml

ROOT = Path(__file__).resolve().parents[1]
TSL_ROOT = ROOT / "Time-Series-Library"
sys.path.insert(0, str(TSL_ROOT))

import zarr  # noqa: E402
from models.iTransformer import Model  # noqa: E402
from utils.timefeatures import time_features  # noqa: E402

NCEP_VARS = ["p", "slp", "t2m", "r2m", "u10", "v10"]
NGL_VARS = ["ztd", "zwd"]
WINDOW_HOURS = 7          # T-6 .. T
NGL_OFFSET = 21           # ngl index = ncep index + 21: NGL zarr starts 2017-12-31T00:00, NCEP starts 2017-12-31T21:00
TIME_FEATURES = 4         # freq='h' with embed='timeF'


def parse_utc(value: str):
    if not value:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


DEFAULTS: dict = {
    # data
    "ngl_zarr": "dataset/ngl_hourly.zarr",
    "ncep_zarr": "dataset/ncep_hourly.zarr",
    "neighbors_parquet": "dataset/target_gnss_neighbors.parquet",
    # temporal splits
    "train_start": "2018-01-01T00:00",
    "train_end": "2023-11-01T00:00",
    "val_start": "2023-11-01T00:00",
    "val_end": "2024-03-01T00:00",
    "test_start": "2024-03-01T00:00",
    "test_end": "",
    # station/time sampling
    "stations": 64,
    "station_offset": 0,
    "max_neighbors": 5,
    "min_valid_neighbors": 3,
    "hour_stride": 6,
    "val_stride": 1,
    "test_stride": 1,
    "load_full_arrays": False,
    "load_full_ncep": True,
    # model
    "seq_len": WINDOW_HOURS,
    "pred_len": 1,
    "d_model": 128,
    "n_heads": 4,
    "e_layers": 2,
    "d_ff": 512,
    "dropout": 0.1,
    "activation": "gelu",
    # training
    "epochs": 10,
    "batch_size": 256,
    "learning_rate": 1e-3,
    "patience": 3,
    "num_workers": 0,
    "seed": 2021,
    "target_scale": True,
    # run
    "device": "auto",
    "model_id": "gnss_nowcast",
    "out_root": "nowcasting/outputs",
}

PATH_KEYS = ("ngl_zarr", "ncep_zarr", "neighbors_parquet", "out_root")


def _flatten(mapping: dict) -> dict:
    """Flatten nested YAML sections to leaf key names (e.g. data.ngl_zarr -> ngl_zarr)."""
    out = {}
    for key, value in mapping.items():
        if isinstance(value, dict):
            out.update(_flatten(value))
        else:
            out[key] = value
    return out


def coerce_value(text: str):
    if text in ("true", "True"):
        return True
    if text in ("false", "False"):
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def load_config(config_path: Path, overrides: list[str]) -> argparse.Namespace:
    if not config_path.exists():
        raise SystemExit(f"config file not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    flat = _flatten(raw)

    unknown = sorted(set(flat) - set(DEFAULTS))
    if unknown:
        raise SystemExit(f"unknown config key(s): {', '.join(unknown)}")

    for spec in overrides:
        if "=" not in spec:
            raise SystemExit(f"--set expects KEY=VALUE, got: {spec!r}")
        key, value = spec.split("=", 1)
        if key not in DEFAULTS:
            raise SystemExit(f"unknown config key in --set: {key!r}")
        flat[key] = coerce_value(value)

    cfg = dict(DEFAULTS)
    cfg.update(flat)
    args = argparse.Namespace(**cfg)
    for key in PATH_KEYS:
        path = Path(getattr(args, key))
        setattr(args, key, path if path.is_absolute() else ROOT / path)
    return args


def build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="All run parameters live in the YAML config (default: nowcasting/config.yaml). "
               "Override any key with --set KEY=VALUE (repeatable).",
    )
    p.add_argument("--config", type=Path, default=ROOT / "nowcasting/config.yaml",
                   help="path to the YAML config file")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config key, e.g. --set stations=128 --set epochs=20")
    return p


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit(f"device {name} requested but CUDA is not available")
        return torch.device(name)
    return torch.device("cpu")


class NowcastData:
    """Shared metadata, sample indices and cached per-station arrays."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.ngl = zarr.open(args.ngl_zarr, mode="r")
        self.ncep = zarr.open(args.ncep_zarr, mode="r")
        self.ngl_full = None
        self.ncep_full = None
        if args.load_full_arrays:
            print("[data] loading full NGL ZTD/ZWD arrays into RAM (slow on HDD) ...", flush=True)
            self.ngl_full = {
                "ztd": np.asarray(self.ngl["ztd"][:]),
                "zwd": np.asarray(self.ngl["zwd"][:]),
            }
        if args.load_full_ncep:
            print("[data] loading full NCEP arrays row-major ...", flush=True)
            self.ncep_full = {}
            for v in NCEP_VARS:
                arr = self.ncep[v]
                n_t, n_s = arr.shape
                out = np.empty((n_t, n_s), dtype=np.float32)
                for t0 in range(0, n_t, arr.chunks[0]):    # 把“按列读很慢”的 NCEP 数据，用“按行带读”的方式一次性搬到内存里
                    t1 = min(t0 + arr.chunks[0], n_t)
                    out[t0:t1] = arr[t0:t1, :]
                self.ncep_full[v] = out
                print(f"  loaded {v} {out.shape}", flush=True)
        self.ngl_time = pd.DatetimeIndex(pd.to_datetime(np.asarray(self.ngl["time"][:])))
        self.ncep_time = pd.DatetimeIndex(pd.to_datetime(np.asarray(self.ncep["time"][:])))
        # Zarr stores naive UTC strings; make them tz-aware so they compare
        # with the tz-aware split boundaries.
        if self.ngl_time.tz is None:
            self.ngl_time = self.ngl_time.tz_localize("UTC")
        if self.ncep_time.tz is None:
            self.ncep_time = self.ncep_time.tz_localize("UTC")
        self.ngl_col = {str(s): i for i, s in enumerate(np.asarray(self.ngl["station"][:]))}
        self.ncep_col = {str(s): i for i, s in enumerate(np.asarray(self.ncep["station"][:]))}

        # per-target ordered neighbor list (rank 1..max_neighbors)
        nb = pd.read_parquet(args.neighbors_parquet)
        nb = nb[nb["rank"] <= args.max_neighbors].sort_values(["target_station_id", "rank"])   # 从邻近站表里筛选出前 N 个邻近站，并按（目标站, 排名）排好序
        neighbor_ids = nb.groupby("target_station_id")["gnss_station_id"].apply(list).to_dict()

        candidates = sorted(
            sid for sid, ids in neighbor_ids.items()
            if len(ids) >= args.min_valid_neighbors
        )
        # stations <= 0 (or None) means "use all usable stations" from the offset onward.
        if args.stations is None or args.stations <= 0:
            selected = candidates[args.station_offset:]
        else:
            selected = candidates[args.station_offset: args.station_offset + args.stations]

        self.station_ids: list[str] = []
        self.station_ngl: list[tuple[np.ndarray, np.ndarray]] = []  # (ztd (T_ngl,k), zwd)
        self.station_ncep_col: list[int] = []
        self.target: list[np.ndarray] = []  # (T_ncep, n_vars) float32

        for si, sid in enumerate(selected):
            if si % 10 == 0:
                print(f"[data] loading station {si}/{len(selected)} ({sid})", flush=True)
            ncep_i = self.ncep_col.get(sid)
            if ncep_i is None:
                continue
            cols = [self.ngl_col[g] for g in neighbor_ids[sid][: args.max_neighbors]]
            if any(c is None for c in cols) or not cols:
                continue
            if self.ngl_full is not None:
                ztd = self.ngl_full["ztd"][:, cols].astype(np.float32, copy=False)
                zwd = self.ngl_full["zwd"][:, cols].astype(np.float32, copy=False)
            else:
                ztd = np.stack([self.ngl["ztd"][:, c] for c in cols], axis=1).astype(np.float32)
                zwd = np.stack([self.ngl["zwd"][:, c] for c in cols], axis=1).astype(np.float32)
            if self.ncep_full is not None:
                tgt = np.stack([self.ncep_full[v][:, ncep_i] for v in NCEP_VARS], axis=1).astype(np.float32)
            else:
                tgt = np.stack([self.ncep[v][:, ncep_i] for v in NCEP_VARS], axis=1).astype(np.float32)
            self.station_ids.append(sid)
            self.station_ngl.append((ztd, zwd))
            self.station_ncep_col.append(ncep_i)
            self.target.append(tgt)

        self.n_stations = len(self.station_ids)
        if not self.station_ids:
            raise SystemExit("no usable target stations (check --station-offset/--stations)")

        self.split_ranges = {
            "train": (parse_utc(args.train_start), parse_utc(args.train_end)),
            "val": (parse_utc(args.val_start), parse_utc(args.val_end)),
            "test": (parse_utc(args.test_start), parse_utc(args.test_end)),
        }
        self.split_idx: dict[str, np.ndarray] = {}
        for flag, (start, end) in self.split_ranges.items():
            mask = self.ncep_time >= start
            if end is not None:
                mask &= self.ncep_time < end
            self.split_idx[flag] = np.flatnonzero(mask)

        self.samples: dict[str, list[np.ndarray]] = {f: [] for f in self.split_ranges}
        self._build_sample_index()
        self.ym, self.ys = self._train_scaler()
        for flag in self.samples:
            print(f"[{flag}] stations={len(self.samples[flag])} samples={sum(len(s) for s in self.samples[flag])}", flush=True)

    def _build_sample_index(self): 
        args = self.args
        n_ncep = len(self.ncep_time)
        n_ngl = len(self.ngl_time)
        win_slide = WINDOW_HOURS
        for i, (ztd, zwd) in enumerate(self.station_ngl):
            good = np.isfinite(ztd) & np.isfinite(zwd)  # (T_ngl, k)
            win_good = sliding_window_view(good, win_slide, axis=0).all(axis=2)  # (T_ngl-6, k)
            count_valid = win_good.sum(axis=1).astype(np.int16)
            # ncep index t needs the ngl window [t+NGL_OFFSET-6, t+NGL_OFFSET] (T-6..T).
            # sliding_window_view index k covers ngl [k, k+6], so k = t + NGL_OFFSET - 6.
            win_start = NGL_OFFSET - (WINDOW_HOURS - 1)  # 21 - 6 = 15
            n_valid = np.zeros(n_ncep, dtype=np.int16)
            n_len = min(n_ncep, len(count_valid) - win_start)
            n_valid[:n_len] = count_valid[win_start:win_start + n_len]
            # Target validity: all 6 NCEP variables at T must be finite (real observations).
            tgt_ok = np.isfinite(self.target[i]).all(axis=1)  # (T_ncep,)
            for flag in self.samples:
                idx = self.split_idx[flag]
                keep = idx[(n_valid[idx] >= args.min_valid_neighbors) & tgt_ok[idx]]
                stride = {"train": args.hour_stride, "val": args.val_stride, "test": args.test_stride}[flag]
                self.samples[flag].append(keep[::stride])

    def _train_scaler(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.args.target_scale:
            return np.zeros(len(NCEP_VARS), dtype=np.float32), np.ones(len(NCEP_VARS), dtype=np.float32)
        idx = self.split_idx["train"]
        sums = np.zeros(len(NCEP_VARS), dtype=np.float64)
        sqs = np.zeros(len(NCEP_VARS), dtype=np.float64)
        cnt = np.zeros(len(NCEP_VARS), dtype=np.float64)
        for tgt in self.target:
            y = tgt[idx]
            for v in range(len(NCEP_VARS)):
                x = y[:, v]
                x = x[np.isfinite(x)]
                if x.size:
                    sums[v] += x.sum()
                    sqs[v] += (x * x).sum()
                    cnt[v] += x.size
        if (cnt == 0).any():
            raise SystemExit("train split has no finite NCEP targets for some variables")
        mean = sums / cnt
        var = sqs / cnt - mean * mean
        std = np.sqrt(np.clip(var, 1e-12, None))
        return mean.astype(np.float32), std.astype(np.float32)

    def make_sample(self, station_pos: int, t: int):
        ztd, zwd = self.station_ngl[station_pos]
        k = ztd.shape[1]
        max_k = self.args.max_neighbors
        n_chan = 2 * max_k + max_k
        g0 = t - (WINDOW_HOURS - 1) + NGL_OFFSET   # first ngl index of the window
        g1 = g0 + WINDOW_HOURS                     # exclusive
        wz = np.isfinite(ztd[g0:g1]) & np.isfinite(zwd[g0:g1])  # (7, k)
        valid = wz.all(axis=0)
        x = np.zeros((WINDOW_HOURS, n_chan), dtype=np.float32)
        for j in range(k):
            if valid[j]:
                x[:, 2 * j] = ztd[g0:g1, j]
                x[:, 2 * j + 1] = zwd[g0:g1, j]
            x[:, 2 * max_k + j] = 1.0 if valid[j] else 0.0
        y = self.target[station_pos][t].astype(np.float32)
        y = np.nan_to_num((y - self.ym) / self.ys, nan=0.0).reshape(1, len(NCEP_VARS))
        mark = time_features(self.ngl_time[g0:g1], freq="h").T.astype(np.float32)  # (7, 4)
        y_mark = np.zeros((1, TIME_FEATURES), dtype=np.float32)
        return x, y, mark, y_mark


class GNSSNowcastDataset(Dataset):
    def __init__(self, data: NowcastData, flag: str):
        self.data = data
        self.flag = flag
        self.samples = data.samples[flag]
        self.offsets = np.concatenate([[0], np.cumsum([len(s) for s in self.samples])])
        self.n = int(self.offsets[-1])

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int):
        i = int(np.searchsorted(self.offsets, index + 1, side="left") - 1)
        t = int(self.samples[i][index - int(self.offsets[i])])
        return self.data.make_sample(i, t)


def make_model_config(args: argparse.Namespace) -> argparse.Namespace:
    cfg = argparse.Namespace()
    cfg.task_name = "long_term_forecast"
    cfg.seq_len = args.seq_len
    cfg.pred_len = args.pred_len
    cfg.enc_in = 2 * args.max_neighbors + args.max_neighbors
    cfg.c_out = len(NCEP_VARS)
    cfg.d_model = args.d_model
    cfg.n_heads = args.n_heads
    cfg.e_layers = args.e_layers
    cfg.d_ff = args.d_ff
    cfg.dropout = args.dropout
    cfg.embed = "timeF"
    cfg.freq = "h"
    cfg.activation = args.activation
    cfg.factor = 1
    cfg.separate_output = True
    return cfg


def make_loader(data: NowcastData, flag: str, args: argparse.Namespace, shuffle: bool):
    ds = GNSSNowcastDataset(data, flag)
    if len(ds) == 0:
        raise SystemExit(
            f"[{flag}] no valid samples for the current station selection "
            f"(stations={args.stations}, offset={args.station_offset}, "
            f"min_valid_neighbors={args.min_valid_neighbors}). "
            "Try --set station_offset=... or increase --set stations=..."
        )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                    num_workers=args.num_workers, pin_memory=True, drop_last=False)
    return ds, dl


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, cfg, criterion, device) -> float:
    model.eval()
    losses = []
    for bx, by, bxm, bym in loader:
        bx = bx.float().to(device)
        by = by.float().to(device)
        bxm = bxm.float().to(device)
        dec_inp = torch.zeros(bx.shape[0], cfg.pred_len, cfg.c_out, device=device)
        y_mark = torch.zeros(bx.shape[0], cfg.pred_len, TIME_FEATURES, device=device)
        out = model(bx, bxm, dec_inp, y_mark)
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
    out_dir = args.out_root / setting
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                       f, sort_keys=False)

    best_val = float("inf")
    patience = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_losses = []
        t0 = time.time()
        for step, (bx, by, bxm, bym) in enumerate(train_loader, 1):
            bx = bx.float().to(device)
            by = by.float().to(device)
            bxm = bxm.float().to(device)
            dec_inp = torch.zeros(bx.shape[0], cfg.pred_len, cfg.c_out, device=device)
            y_mark = torch.zeros(bx.shape[0], cfg.pred_len, TIME_FEATURES, device=device)
            optimizer.zero_grad()
            out = model(bx, bxm, dec_inp, y_mark)
            loss = criterion(out[:, -cfg.pred_len:, :], by[:, -cfg.pred_len:, :])
            loss.backward()
            optimizer.step()
            ep_losses.append(loss.item())
            if step % 200 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {loss.item():.6f}", flush=True)

        train_loss = float(np.mean(ep_losses))
        val_loss = evaluate(model, val_loader, cfg, criterion, device)
        print(f"[epoch {epoch}] train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
              f"({time.time() - t0:.1f}s)", flush=True)

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
    for bx, by, bxm, bym in test_loader:
        bx = bx.float().to(device)
        bxm = bxm.float().to(device)
        dec_inp = torch.zeros(bx.shape[0], cfg.pred_len, cfg.c_out, device=device)
        y_mark = torch.zeros(bx.shape[0], cfg.pred_len, TIME_FEATURES, device=device)
        out = model(bx, bxm, dec_inp, y_mark)
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
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def main():
    cli = build_cli_parser().parse_args()
    args = load_config(cli.config, cli.set)
    print("effective config:")
    print(json.dumps(vars(args), indent=2, default=str))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"device={device}  zarr ngl={args.ngl_zarr}  ncep={args.ncep_zarr}")

    t0 = time.time()
    data = NowcastData(args)
    print(f"data index built in {time.time() - t0:.1f}s")

    cfg = make_model_config(args)
    model, out_dir = train(args, data, device, cfg)
    test(args, data, model, cfg, out_dir, device)
    print(f"outputs saved under {out_dir}")


if __name__ == "__main__":
    main()
