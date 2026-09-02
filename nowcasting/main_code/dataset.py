from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

import zarr

from .config import parse_utc
from .constants import NCEP_VARS
from .geometry import _enu
from utils.timefeatures import time_features


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
        # n_geo=4: [dE_km, dN_km, dU_m, ngl_h_m].
        # n_geo=8: append [target_lat, target_lon, gnss_lat, gnss_lon].
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
                if args.n_geo == 4:
                    row[j] = (de, dn, du, g_info[2])
                elif args.n_geo == 8:
                    row[j] = (de, dn, du, g_info[2], t_info[0], t_info[1], g_info[0], g_info[1])
                else:
                    raise SystemExit(
                        f"unsupported n_geo={args.n_geo}; use 4 or 8 "
                        "([dE,dN,dU,ngl_h] plus optional target/GNSS lat/lon)"
                    )
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
        self.target_feat_mean = np.asarray([mean], dtype=np.float64)
        self.target_feat_std = np.asarray([std], dtype=np.float64)
        self.target_feat_raw = raw.reshape(-1, 1).astype(np.float64)
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
            target_feat_mean=self.target_feat_mean,
            target_feat_std=self.target_feat_std,
            ym=self.ym,
            ys=self.ys,
            station_ids=np.asarray(self.station_ids, dtype=object),
            target_h_raw=np.asarray(self.target_h_raw, dtype=np.float64),
            target_feat_raw=self.target_feat_raw,
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
        x_tgt = self.target_h_arr[station_pos]  # z-scored target-level output-head features
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
