#!/usr/bin/env python3
"""Train/test an iTransformer nowcasting model on GNSS -> NCEP surface data.

Task
----
Use 5-minute NGL ZTD/ZWD of the nearest GNSS stations over T-2h..T to nowcast
the NCEP surface variables (p, slp, t2m, r2m, u10, v10) at time T.

Temporal splits (UTC; [start, end) half-open ranges):
    train: 2018-01-01T00:00 <= T <  2023-11-01T00:00
    val:   2023-11-01T00:00 <= T <  2024-03-01T00:00
    test:  2024-03-01T00:00 <= T <= last available hour

Input tensor per sample: (window_hours*60/ngl_step_minutes + 1) x (2*max_neighbors + max_neighbors)
channels, i.e.
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
from torch.utils.data import DataLoader, Dataset

import yaml

ROOT = Path(__file__).resolve().parents[1]
TSL_ROOT = ROOT / "Time-Series-Library"
sys.path.insert(0, str(TSL_ROOT))

import zarr  # noqa: E402
from models.iTransformer import Model  # noqa: E402
from utils.timefeatures import time_features, time_features_from_frequency_str  # noqa: E402

NCEP_VARS = ["p", "slp", "t2m", "r2m", "u10", "v10"]
NGL_VARS = ["ztd", "zwd"]


def parse_utc(value: str):
    if not value:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


# Run parameters come exclusively from the YAML config (nowcasting/config.yaml);
# the script holds no value defaults. Missing/unknown keys are reported at startup.
REQUIRED_KEYS = {
    # data
    "ngl_zarr", "ncep_zarr", "neighbors_parquet", "target_stations_parquet",
    "gnss_stations_parquet", "ngl_step_minutes",
    # temporal splits
    "train_start", "train_end", "val_start", "val_end", "test_start", "test_end",
    # station/time sampling
    "stations", "station_offset", "max_neighbors", "min_valid_neighbors",
    "hour_stride", "val_stride", "test_stride", "load_full_arrays", "load_full_ncep",
    # model
    "window_hours", "pred_len", "time_encoding", "time_freq", "spatial_enc", "n_geo", "spatial_mlp_hidden",
    "target_h_feat",
    "d_model", "n_heads", "e_layers", "d_ff", "dropout", "activation",
    # training
    "epochs", "batch_size", "learning_rate", "patience", "num_workers", "seed",
    "target_scale",
    # run
    "device", "model_id", "out_root",
}

PATH_KEYS = ("ngl_zarr", "ncep_zarr", "neighbors_parquet",
             "target_stations_parquet", "gnss_stations_parquet", "out_root")


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

    unknown = sorted(set(flat) - REQUIRED_KEYS)
    if unknown:
        raise SystemExit(f"unknown config key(s): {', '.join(unknown)}")

    for spec in overrides:
        if "=" not in spec:
            raise SystemExit(f"--set expects KEY=VALUE, got: {spec!r}")
        key, value = spec.split("=", 1)
        if key not in REQUIRED_KEYS:
            raise SystemExit(f"unknown config key in --set: {key!r}")
        flat[key] = coerce_value(value)

    missing = sorted(REQUIRED_KEYS - set(flat))
    if missing:
        raise SystemExit(
            "missing config key(s); add them to the YAML config: " + ", ".join(missing)
        )

    args = argparse.Namespace(**flat)
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


_WGS84_A = 6378137.0
_WGS84_E2 = 0.0066943799901413165  # WGS84 (f = 1/298.257223563)


def _ecef(lat_deg: float, lon_deg: float, h_m: float) -> tuple[float, float, float]:
    """Geodetic (WGS84) -> geocentric ECEF."""
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    n = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * np.sin(lat) ** 2)
    x = (n + h_m) * np.cos(lat) * np.cos(lon)
    y = (n + h_m) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - _WGS84_E2) + h_m) * np.sin(lat)
    return x, y, z


def _enu(t_info: tuple[float, float, float], g_info: tuple[float, float, float]) -> tuple[float, float, float]:
    """Neighbor position in the target's local ENU frame (target = origin).

    Returns (dE_km, dN_km, dU_m); dU uses the station metadata heights directly
    (consistent with target_gnss_neighbors.height_difference_m).
    """
    tx, ty, tz = _ecef(*t_info)
    gx, gy, gz = _ecef(*g_info)
    dx, dy, dz = gx - tx, gy - ty, gz - tz
    lat, lon = np.deg2rad(t_info[0]), np.deg2rad(t_info[1])
    slat, clat = np.sin(lat), np.cos(lat)
    slon, clon = np.sin(lon), np.cos(lon)
    e = -slon * dx + clon * dy
    n = -slat * clon * dx - slat * slon * dy + clat * dz
    du = g_info[2] - t_info[2]
    return float(e / 1000.0), float(n / 1000.0), float(du)


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
        # Precompute time marks once for the whole NGL time axis (optional).
        if args.time_encoding == "sincos":
            # Four calendar cycles, each encoded as a (sin, cos) pair:
            # hour-of-day (24h), day-of-week (7d), day-of-month (31d), day-of-year (365d).
            hours = np.asarray(self.ngl_time.hour, dtype=np.float32)
            dows = np.asarray(self.ngl_time.dayofweek, dtype=np.float32)
            doms = np.asarray(self.ngl_time.day - 1, dtype=np.float32)
            doys = np.asarray(self.ngl_time.dayofyear - 1, dtype=np.float32)
            cols = []
            for values, period in ((hours, 24.0), (dows, 7.0), (doms, 31.0), (doys, 365.0)):
                ang = 2.0 * np.pi * values / period
                cols.extend([np.sin(ang), np.cos(ang)])
            self.time_mark = np.stack(cols, axis=1).astype(np.float32)  # (T_ngl, 8)
        elif args.time_encoding == "hour_sincos":
            hours = np.asarray(self.ngl_time.hour, dtype=np.float32)
            ang = 2.0 * np.pi * hours / 24.0
            self.time_mark = np.stack([np.sin(ang), np.cos(ang)], axis=1).astype(np.float32)  # (T_ngl, 2)
        elif args.time_encoding == "linear":
            self.time_mark = time_features(self.ngl_time, freq=args.time_freq).T.astype(np.float32)  # (T_ngl, n_tf)
        else:
            self.time_mark = None

        # NGL resolution / window geometry (works for hourly or 5-minute stores).
        self.ngl_step_minutes = int(args.ngl_step_minutes)
        self.ngl_steps_per_hour = 60 // self.ngl_step_minutes
        self.window_steps = int(args.window_hours * 60 // self.ngl_step_minutes) + 1
        self.ngl_offset = int(
            (self.ncep_time[0] - self.ngl_time[0]).total_seconds() // 60 // self.ngl_step_minutes
        )
        print(f"[data] ngl_step={self.ngl_step_minutes}min window={self.window_steps} steps "
              f"(T-{args.window_hours}h..T) ngl_offset={self.ngl_offset} steps", flush=True)
        self.ngl_col = {str(s): i for i, s in enumerate(np.asarray(self.ngl["station"][:]))}
        self.ncep_col = {str(s): i for i, s in enumerate(np.asarray(self.ncep["station"][:]))}

        # per-target ordered neighbor list (rank 1..max_neighbors)
        nb = pd.read_parquet(args.neighbors_parquet)
        nb = nb[nb["rank"] <= args.max_neighbors].sort_values(["target_station_id", "rank"])   # 从邻近站表里筛选出前 N 个邻近站，并按（目标站, 排名）排好序
        neighbor_ids = nb.groupby("target_station_id")["gnss_station_id"].apply(list).to_dict()

        # Station geometry for the positional encoding: per (target, neighbor)
        # relative ENU position (target as origin) plus absolute heights.
        target_st = pd.read_parquet(args.target_stations_parquet)
        gnss_st = pd.read_parquet(args.gnss_stations_parquet)
        t_geo = {str(s): (float(r.lat), float(r.lon), float(r.height_m))
                 for s, r in target_st.set_index("target_station_id").iterrows()}
        g_geo = {str(s): (float(r.lat), float(r.lon), float(r.height_m))
                 for s, r in gnss_st.set_index("gnss_station_id").iterrows()}

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
        self.station_geo_raw: list[np.ndarray] = []  # (max_k, n_geo) pre-normalization
        self.target_h_raw: list[float] = []  # target station height (m) per selected station

        for si, sid in enumerate(selected):
            if si % 10 == 0:
                print(f"[data] loading station {si}/{len(selected)} ({sid})", flush=True)
            ncep_i = self.ncep_col.get(sid)
            if ncep_i is None:
                continue
            cols = [self.ngl_col[g] for g in neighbor_ids[sid][: args.max_neighbors]]
            if any(c is None for c in cols) or not cols:
                continue
            row = np.zeros((args.max_neighbors, args.n_geo), dtype=np.float32)
            t_info = t_geo.get(sid)
            for j, gid in enumerate(neighbor_ids[sid][: args.max_neighbors]):
                g_info = g_geo.get(gid)
                if t_info is None or g_info is None:
                    continue
                de, dn, du = _enu(t_info, g_info)
                row[j] = (de, dn, du, g_info[2])
            self.station_geo_raw.append(row)
            self.target_h_raw.append(t_info[2] if t_info is not None else np.nan)
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

        # Normalize geometry over real (target, neighbor) pairs; padding stays zero.
        if args.spatial_enc:
            stacked = np.stack(self.station_geo_raw, axis=0)          # (S, max_k, n_geo)
            valid_geo = np.abs(stacked).sum(axis=2) > 0               # nonzero rows = real pairs
            flat = stacked[valid_geo]
            self.geo_mean = flat.mean(axis=0).astype(np.float32)
            self.geo_std = (flat.std(axis=0) + 1e-6).astype(np.float32)
            self.station_geo_arr = ((stacked - self.geo_mean) / self.geo_std * valid_geo[..., None]).astype(np.float32)
        else:
            self.geo_mean = np.zeros(args.n_geo, dtype=np.float32)
            self.geo_std = np.ones(args.n_geo, dtype=np.float32)
            self.station_geo_arr = np.zeros((self.n_stations, args.max_neighbors, args.n_geo), dtype=np.float32)
        print(f"[data] station_geo={self.station_geo_arr.shape} "
              f"mean={self.geo_mean.tolist()} std={self.geo_std.tolist()}", flush=True)

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
        self._target_h_scaler()
        for flag in self.samples:
            print(f"[{flag}] stations={len(self.samples[flag])} samples={sum(len(s) for s in self.samples[flag])}", flush=True)

    def _build_sample_index(self): 
        args = self.args
        n_ncep = len(self.ncep_time)
        n_ngl = len(self.ngl_time)
        win_slide = self.window_steps
        # ncep index t needs ngl window [t*sph+off-(win-1), t*sph+off],
        # i.e. sliding-window start k = t*sph+off-(win-1).
        win_start = (
            np.arange(n_ncep, dtype=np.int64) * self.ngl_steps_per_hour
            + self.ngl_offset
            - (win_slide - 1)
        )
        for i, (ztd, zwd) in enumerate(self.station_ngl):
            good = np.isfinite(ztd) & np.isfinite(zwd)  # (T_ngl, k)
            # Window fully-finite count via cumulative sums (equivalent to
            # sliding_window_view(...).all(), but much faster at 5-minute scale).
            cum = np.zeros((good.shape[0] + 1, good.shape[1]), dtype=np.int64)
            np.cumsum(good, axis=0, out=cum[1:])
            n_valid = np.zeros(n_ncep, dtype=np.int16)
            ok = (win_start >= 0) & (win_start + win_slide <= good.shape[0])
            ws = win_start[ok]
            full = (cum[ws + win_slide] - cum[ws]) == win_slide  # (n_ok, k)
            n_valid[ok] = full.sum(axis=1)
            # Target validity: all 6 NCEP variables at T must be finite (real observations).
            tgt_ok = np.isfinite(self.target[i]).all(axis=1)  # (T_ncep,)
            for flag in self.samples:
                idx = self.split_idx[flag]
                keep = idx[(n_valid[idx] >= args.min_valid_neighbors) & tgt_ok[idx]]
                stride = {"train": args.hour_stride, "val": args.val_stride, "test": args.test_stride}[flag]
                self.samples[flag].append(keep[::stride])
            if (i + 1) % 100 == 0 or i + 1 == len(self.station_ngl):
                print(f"[data] sample-index {i + 1}/{len(self.station_ngl)}", flush=True)

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

    def _target_h_scaler(self) -> None:
        """Z-score target-station heights with train-station statistics.

        The target height is constant across the input window, so it cannot
        pass through the per-variate instance normalization inside the model;
        it is fed as a per-sample scalar to the separate_output linear layer.
        Statistics come from stations that actually contribute train samples
        (fall back to all selected stations if none do).
        """
        raw = np.asarray(self.target_h_raw, dtype=np.float64)  # (S,)
        finite = np.isfinite(raw)
        train_ok = np.array([len(s) > 0 for s in self.samples["train"]])
        mask = finite & train_ok
        if not mask.any():
            mask = finite
        if mask.any():
            mean = float(raw[mask].mean())
            std = float(raw[mask].std())
        else:
            mean, std = 0.0, 1.0
        arr = np.where(finite, (raw - mean) / (std + 1e-6), 0.0)
        self.target_h_mean = mean
        self.target_h_std = std
        self.target_h_arr = arr.astype(np.float32).reshape(-1, 1)  # (S, 1)
        print(f"[data] target_h_mean={mean:.2f} m target_h_std={std:.2f} m "
              f"(train stations={int(mask.sum())}/{len(raw)})", flush=True)

    def save_scalers(self, out_dir: Path) -> None:
        """Save normalization statistics needed for inference on new points.

        geo_mean/geo_std (per-neighbor spatial encoding), target_h_mean/std
        (target-station height feature) and ym/ys (target z-scoring) are all
        computed from the training split; grid inference must reuse exactly
        these values.
        """
        np.savez(
            out_dir / "scalers.npz",
            geo_mean=self.geo_mean,
            geo_std=self.geo_std,
            target_h_mean=np.float64(self.target_h_mean),
            target_h_std=np.float64(self.target_h_std),
            ym=self.ym,
            ys=self.ys,
            station_ids=np.asarray(self.station_ids, dtype=object),
            target_h_raw=np.asarray(self.target_h_raw, dtype=np.float64),
        )
        print(f"[data] scalers saved to {out_dir / 'scalers.npz'}", flush=True)

    def make_sample(self, station_pos: int, t: int):
        ztd, zwd = self.station_ngl[station_pos]
        k = ztd.shape[1]
        max_k = self.args.max_neighbors
        n_chan = 2 * max_k + max_k
        g0 = t * self.ngl_steps_per_hour + self.ngl_offset - (self.window_steps - 1)
        g1 = g0 + self.window_steps                # exclusive
        wz = np.isfinite(ztd[g0:g1]) & np.isfinite(zwd[g0:g1])  # (7, k)
        valid = wz.all(axis=0)
        x = np.zeros((self.window_steps, n_chan), dtype=np.float32)
        for j in range(k):
            if valid[j]:
                x[:, 2 * j] = ztd[g0:g1, j]
                x[:, 2 * j + 1] = zwd[g0:g1, j]
            x[:, 2 * max_k + j] = 1.0 if valid[j] else 0.0
        y = self.target[station_pos][t].astype(np.float32)
        y = np.nan_to_num((y - self.ym) / self.ys, nan=0.0).reshape(1, len(NCEP_VARS))
        if self.time_mark is not None:
            mark = self.time_mark[g0:g1]  # (win, n_tf) precomputed
            y_mark = np.zeros((1, self.args.n_time_features), dtype=np.float32)
        else:
            mark = np.zeros((self.window_steps, 0), dtype=np.float32)
            y_mark = np.zeros((1, 0), dtype=np.float32)
        # Positional encoding: per-neighbor ENU + heights, gated by per-sample validity.
        v = np.zeros(self.args.max_neighbors, dtype=np.float32)
        v[:k] = valid.astype(np.float32)
        x_geo = self.station_geo_arr[station_pos] * v[:, None]  # (max_k, n_geo) float32
        x_tgt = self.target_h_arr[station_pos]  # (1,) z-scored target height
        return x, y, mark, y_mark, x_geo, x_tgt


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
    cfg.freq = args.time_freq
    cfg.n_time_features = int(args.n_time_features)
    cfg.activation = args.activation
    cfg.factor = 1
    cfg.separate_output = True
    cfg.spatial_enc = bool(args.spatial_enc)
    cfg.n_geo = int(args.n_geo)
    cfg.spatial_mlp_hidden = int(args.spatial_mlp_hidden)
    cfg.target_h_feat = bool(args.target_h_feat)
    cfg.max_neighbors = int(args.max_neighbors)
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


def main():
    cli = build_cli_parser().parse_args()
    args = load_config(cli.config, cli.set)
    args.seq_len = int(args.window_hours * 60 // args.ngl_step_minutes) + 1
    if args.time_encoding == "none":
        args.n_time_features = 0
    elif args.time_encoding == "linear":
        args.n_time_features = len(time_features_from_frequency_str(args.time_freq))
    elif args.time_encoding == "hour_sincos":
        args.n_time_features = 2
    elif args.time_encoding == "sincos":
        args.n_time_features = 8  # hour/day-of-week/day-of-month/day-of-year, each sin+cos
    else:
        raise SystemExit(f"unknown time_encoding: {args.time_encoding!r} (choose none|linear|hour_sincos|sincos)")
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
