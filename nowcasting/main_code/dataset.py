from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, Sampler

import zarr

from .config import parse_utc
from .constants import ERA5_BACKGROUND_CHANNELS, NCEP_SOURCE_VARS, NCEP_VARS, NGL_VARS
from .geometry import _enu
from .humidity import specific_humidity_from_t_rh_p
from utils.timefeatures import time_features


class NowcastData:
    """Shared metadata, sample indices and cached per-station arrays."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.ngl = zarr.open(args.ngl_zarr, mode="r")
        self.ncep = zarr.open(args.ncep_zarr, mode="r")
        missing_ncep = [name for name in NCEP_SOURCE_VARS if name not in self.ncep]
        if missing_ncep:
            raise SystemExit(f"NCEP store is missing required source fields: {missing_ncep}")
        self.use_era5 = bool(args.use_era5)
        self.era5 = zarr.open(args.era5_zarr, mode="r") if self.use_era5 else None
        self.ngl_full = None
        self.ncep_full = None
        if args.load_full_arrays:
            print("[data] loading full NGL input arrays into RAM (slow on HDD) ...", flush=True)
            self.ngl_full = {
                name: np.asarray(self.ngl[name][:])
                for name in NGL_VARS
            }
        if args.load_full_ncep:
            print("[data] loading full NCEP arrays row-major ...", flush=True)
            self.ncep_full = {}
            for v in NCEP_SOURCE_VARS:
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
        if self.use_era5:
            self.era5_time = pd.DatetimeIndex(pd.to_datetime(np.asarray(self.era5["time"][:])))
            if self.era5_time.tz is None:
                self.era5_time = self.era5_time.tz_localize("UTC")
            # Samples are indexed on the NCEP hourly axis. Keep the matching ERA5 row explicitly.
            self.era5_ncep_idx = self.era5_time.get_indexer(self.ncep_time)
            self.era5_available = self.era5_ncep_idx >= 0
            if not self.era5_available.all():
                missing = int((~self.era5_available).sum())
                print(f"[data] ERA5 missing {missing} NCEP timestamps; those samples are excluded", flush=True)
            era5_channels = np.asarray(self.era5["channel"][:]).astype(str)
            era5_lookup = {name: i for i, name in enumerate(era5_channels)}
            missing_channels = [name for name in ERA5_BACKGROUND_CHANNELS if name not in era5_lookup]
            if missing_channels:
                raise SystemExit(f"ERA5 background channels missing: {missing_channels}")
            self.era5_channel_idx = np.asarray([era5_lookup[name] for name in ERA5_BACKGROUND_CHANNELS], dtype=np.int64)
            self.era5_dim = len(ERA5_BACKGROUND_CHANNELS)
            self.era5_lat = np.asarray(self.era5["lat"][:], dtype=np.float32)
            self.era5_lon = np.asarray(self.era5["lon"][:], dtype=np.float32)
            self.era5_z = self.era5["z"]
            self.era5_time_chunk = int(self.era5_z.chunks[0])
            self.era5_cache: dict[int, np.ndarray] = {}
            self.era5_cache_order: list[int] = []
            self.era5_cache_hours = int(args.era5_cache_hours)
            self.era5_cache_chunks = max(1, self.era5_cache_hours // self.era5_time_chunk)
            print(f"[data] ERA5 background channels={self.era5_dim} "
                  f"chunk_hours={self.era5_time_chunk} cache_chunks={self.era5_cache_chunks}", flush=True)
        else:
            self.era5_available = np.ones(len(self.ncep_time), dtype=bool)
            self.era5_dim = 0
            print("[data] ERA5 disabled; using GNSS-only inputs and ordinary random batches", flush=True)
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

        filter_max_neighbors = int(getattr(args, "filter_max_neighbors", args.max_neighbors))
        filter_min_valid_neighbors = int(getattr(args, "filter_min_valid_neighbors", args.min_valid_neighbors))
        args.filter_max_neighbors = filter_max_neighbors
        args.filter_min_valid_neighbors = filter_min_valid_neighbors

        # per-target ordered neighbor list (rank 1..filter_max_neighbors)
        nb = pd.read_parquet(args.neighbors_parquet)
        nb = nb[nb["rank"] <= filter_max_neighbors].sort_values(["target_station_id", "rank"])
        neighbor_ids = nb.groupby("target_station_id")["gnss_station_id"].apply(list).to_dict()

        # Station geometry/static features for the positional encoding: per (target, neighbor)
        # relative ENU position (target as origin), absolute heights and const.nc features.
        # n_geo=4: [dE_km, dN_km, dU_m, ngl_h_m].
        # n_geo=8: append [target_lat, target_lon, gnss_lat, gnss_lon].
        target_st = pd.read_parquet(args.target_stations_parquet)
        gnss_st = pd.read_parquet(args.gnss_stations_parquet)
        t_geo = {str(s): (float(r.lat), float(r.lon), float(r.height_m))
                 for s, r in target_st.set_index("target_station_id").iterrows()}
        g_geo = {str(s): (float(r.lat), float(r.lon), float(r.height_m))
                 for s, r in gnss_st.set_index("gnss_station_id").iterrows()}
        static = np.load(args.station_static_npz, allow_pickle=True)
        self.static_channels = np.asarray(static["channels"], dtype=object).astype(str)
        self.static_onehot_mask = np.asarray(static["onehot_mask"], dtype=bool)
        g_static = {str(s): static["gnss_static"][i].astype(np.float32)
                    for i, s in enumerate(static["gnss_station_ids"])}
        t_static = {str(s): static["target_static"][i].astype(np.float32)
                    for i, s in enumerate(static["target_station_ids"])}
        self.static_dim = int(len(self.static_channels))
        self.base_n_geo = int(args.n_geo)
        args.encoder_static_dim = self.static_dim
        args.decoder_static_dim = self.static_dim
        args.n_geo_total = self.base_n_geo + self.static_dim
        print(f"[data] station static features={self.static_dim} "
              f"onehot={self.static_channels[self.static_onehot_mask].tolist()}", flush=True)

        candidates = sorted(
            sid for sid, ids in neighbor_ids.items()
            if len(ids) >= filter_min_valid_neighbors
        )
        # stations <= 0 (or None) means "use all usable stations" from the offset onward.
        if args.stations is None or args.stations <= 0:
            selected = candidates[args.station_offset:]
        else:
            selected = candidates[args.station_offset: args.station_offset + args.stations]

        self.station_ids: list[str] = []
        self.station_ngl: list[np.ndarray] = []  # ztd (T_ngl, max_neighbors)
        self.station_ncep_col: list[int] = []
        self.target: list[np.ndarray] = []  # (T_ncep, n_vars) float32
        self.station_geo_raw: list[np.ndarray] = []  # (max_k, n_geo + static_dim) pre-normalization
        self.target_h_raw: list[float] = []  # target station height (m) per selected station
        self.target_static_raw: list[np.ndarray] = []  # const.nc static features per selected target
        self.station_era5_grid: list[np.ndarray] = []  # bilinear ERA5 weights per GNSS neighbor
        self.target_era5_grid: list[np.ndarray] = []   # bilinear ERA5 weights per target

        for si, sid in enumerate(selected):
            if si % 10 == 0:
                print(f"[data] loading station {si}/{len(selected)} ({sid})", flush=True)
            ncep_i = self.ncep_col.get(sid)
            if ncep_i is None:
                continue
            cols = [self.ngl_col[g] for g in neighbor_ids[sid][: filter_max_neighbors]]
            if any(c is None for c in cols) or not cols:
                continue
            row = np.zeros((args.max_neighbors, args.n_geo_total), dtype=np.float32)
            t_info = t_geo.get(sid)
            for j, gid in enumerate(neighbor_ids[sid][: args.max_neighbors]):
                g_info = g_geo.get(gid)
                if t_info is None or g_info is None:
                    continue
                de, dn, du = _enu(t_info, g_info)
                if args.n_geo == 4:
                    geo_values = (de, dn, du, g_info[2])
                elif args.n_geo == 8:
                    geo_values = (de, dn, du, g_info[2], t_info[0], t_info[1], g_info[0], g_info[1])
                else:
                    raise SystemExit(
                        f"unsupported n_geo={args.n_geo}; use 4 or 8 "
                        "([dE,dN,dU,ngl_h] plus optional target/GNSS lat/lon)"
                    )
                row[j, :args.n_geo] = geo_values
                row[j, args.n_geo:] = g_static.get(gid, np.zeros(self.static_dim, dtype=np.float32))
            self.station_geo_raw.append(row)
            if self.use_era5:
                # Store bilinear ERA5 weights once; dynamic sampling below only reads the target hour.
                neighbor_grid = np.zeros((args.max_neighbors, 4, 3), dtype=np.float32)
                for j, gid in enumerate(neighbor_ids[sid][: args.max_neighbors]):
                    g_info = g_geo.get(gid)
                    if g_info is not None:
                        neighbor_grid[j] = self._era5_bilinear_grid(g_info[0], g_info[1])
                self.station_era5_grid.append(neighbor_grid)
                self.target_era5_grid.append(self._era5_bilinear_grid(t_info[0], t_info[1]) if t_info is not None else np.zeros((4, 3), dtype=np.float32))
            self.target_h_raw.append(t_info[2] if t_info is not None else np.nan)
            self.target_static_raw.append(t_static.get(sid, np.zeros(self.static_dim, dtype=np.float32)))
            if self.ngl_full is not None:
                ztd = self.ngl_full["ztd"][:, cols].astype(np.float32, copy=False)
            else:
                ztd = np.stack([self.ngl["ztd"][:, c] for c in cols], axis=1).astype(np.float32)
            if self.ncep_full is not None:
                source = [self.ncep_full[v][:, ncep_i] for v in NCEP_SOURCE_VARS]
            else:
                source = [self.ncep[v][:, ncep_i] for v in NCEP_SOURCE_VARS]
            p, slp, t2m, r2m, u10, v10 = source
            # Observation pressure is stored in hPa; the IFS formulation uses Pa.
            q2m = specific_humidity_from_t_rh_p(t2m, r2m, p * 100.0)
            tgt = np.stack([p, slp, t2m, q2m, u10, v10], axis=1).astype(np.float32)
            self.station_ids.append(sid)
            self.station_ngl.append(ztd)
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
            # const.nc tvh_*/tvl_* channels are one-hot and must stay 0/1.
            static_onehot = np.zeros(stacked.shape[2], dtype=bool)
            static_onehot[self.base_n_geo:] = self.static_onehot_mask
            self.geo_mean[static_onehot] = 0.0
            self.geo_std[static_onehot] = 1.0
            self.station_geo_arr = ((stacked - self.geo_mean) / self.geo_std * valid_geo[..., None]).astype(np.float32)
        else:
            self.geo_mean = np.zeros(args.n_geo_total, dtype=np.float32)
            self.geo_std = np.ones(args.n_geo_total, dtype=np.float32)
            self.station_geo_arr = np.zeros((self.n_stations, args.max_neighbors, args.n_geo_total), dtype=np.float32)
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
        self.station_ngl = [ztd[:, : args.max_neighbors] for ztd in self.station_ngl]
        if getattr(args, "zero_ztd", False):
            # Build sample indices with the real ZTD first so the ablation uses
            # exactly the same samples as the non-zeroed run, then replace the
            # input values with zeros.
            self.station_ngl = [np.zeros_like(ztd) for ztd in self.station_ngl]
        self.input_mean, self.input_std, self.input_count = self._train_input_scaler()
        self.ym, self.ys = self._train_scaler()
        self._target_h_scaler()
        for flag in self.samples:
            print(f"[{flag}] stations={len(self.samples[flag])} samples={sum(len(s) for s in self.samples[flag])}", flush=True)

    def _era5_bilinear_grid(self, lat: float, lon: float) -> np.ndarray:
        """Return four ERA5 grid corners as (lat_index, lon_index, weight)."""
        lon360 = lon % 360.0
        lat_asc = self.era5_lat[::-1]
        iy1_asc = int(np.clip(np.searchsorted(lat_asc, lat), 1, len(lat_asc) - 1))
        ix1 = int(np.clip(np.searchsorted(self.era5_lon, lon360), 1, len(self.era5_lon) - 1))
        iy0_asc = iy1_asc - 1
        ix0 = ix1 - 1
        lat0, lat1 = lat_asc[iy0_asc], lat_asc[iy1_asc]
        lon0, lon1 = self.era5_lon[ix0], self.era5_lon[ix1]
        fy = float(np.clip((lat - lat0) / (lat1 - lat0), 0.0, 1.0))
        fx = float(np.clip((lon360 - lon0) / (lon1 - lon0), 0.0, 1.0))
        # ERA5 latitude is stored north-to-south, hence reverse the ascending indices.
        iy0 = len(self.era5_lat) - 1 - iy0_asc
        iy1 = len(self.era5_lat) - 1 - iy1_asc
        return np.asarray([
            (iy0, ix0, (1.0 - fy) * (1.0 - fx)),
            (iy0, ix1, (1.0 - fy) * fx),
            (iy1, ix0, fy * (1.0 - fx)),
            (iy1, ix1, fy * fx),
        ], dtype=np.float32)

    def _era5_at_grid(self, ncep_t: int, grid: np.ndarray) -> np.ndarray:
        """Sample one NCEP target hour at station-specific bilinear corners.

        grid has shape (stations, 4, 3); output is (stations, era5_dim).
        """
        era5_t = int(self.era5_ncep_idx[ncep_t])
        chunk_id = era5_t // self.era5_time_chunk
        fields = self.era5_cache.get(chunk_id)
        if fields is None:
            t0 = chunk_id * self.era5_time_chunk
            t1 = min(t0 + self.era5_time_chunk, self.era5_z.shape[0])
            fields = np.asarray(self.era5_z[t0:t1, self.era5_channel_idx, :, :], dtype=np.float32)
            if self.era5_cache_hours > 0:
                self.era5_cache[chunk_id] = fields
                self.era5_cache_order.append(chunk_id)
                while len(self.era5_cache_order) > self.era5_cache_chunks:
                    self.era5_cache.pop(self.era5_cache_order.pop(0), None)
        fields_t = fields[era5_t - chunk_id * self.era5_time_chunk]
        iy = grid[:, :, 0].astype(np.intp)
        ix = grid[:, :, 1].astype(np.intp)
        weight = grid[:, :, 2].astype(np.float32)
        return (fields_t[:, iy, ix] * weight[None, :, :]).sum(axis=2).T.astype(np.float32)

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
        filter_min = int(getattr(args, "filter_min_valid_neighbors", args.min_valid_neighbors))
        for i, ztd in enumerate(self.station_ngl):
            good = np.isfinite(ztd)  # (T_ngl, filter_max_neighbors)
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
                keep = idx[(n_valid[idx] >= filter_min) & tgt_ok[idx] & self.era5_available[idx]]
                stride = {"train": args.hour_stride, "val": args.val_stride, "test": args.test_stride}[flag]
                self.samples[flag].append(keep[::stride])
            if (i + 1) % 100 == 0 or i + 1 == len(self.station_ngl):
                print(f"[data] sample-index {i + 1}/{len(self.station_ngl)}", flush=True)

    def _train_input_scaler(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute per-token ZTD statistics from all valid training windows.

        The statistics match the actual model inputs: a neighbour contributes
        only when its ZTD is finite throughout an input window.
        They are then reused unchanged by train, validation and test samples.
        """
        n_chan = len(NGL_VARS) * self.args.max_neighbors
        sums = np.zeros(n_chan, dtype=np.float64)
        sqs = np.zeros(n_chan, dtype=np.float64)
        count = np.zeros(n_chan, dtype=np.int64)
        offsets = np.arange(self.window_steps, dtype=np.int64)
        for station_pos, times in enumerate(self.samples["train"]):
            if not len(times):
                continue
            ztd = self.station_ngl[station_pos]
            starts = (times * self.ngl_steps_per_hour + self.ngl_offset
                      - (self.window_steps - 1))
            rows = starts[:, None] + offsets[None, :]
            for neighbor in range(ztd.shape[1]):
                values = ztd[rows, neighbor]
                valid = np.isfinite(values).all(axis=1)
                if not valid.any():
                    continue
                channel = len(NGL_VARS) * neighbor
                values = values[valid].astype(np.float64, copy=False).ravel()
                sums[channel] += values.sum()
                sqs[channel] += np.square(values).sum()
                count[channel] += values.size
        missing = count == 0
        if missing.any():
            # Padded/unavailable neighbour ranks are always zero in make_sample.
            sums[missing] = 0.0
            count[missing] = 1
        mean = sums / count
        var = sqs / count - mean * mean
        std = np.sqrt(np.clip(var, 1e-12, None))
        std[missing] = 1.0
        print(f"[data] input global scaler count={count.tolist()} "
              f"mean={mean.tolist()} std={std.tolist()}", flush=True)
        return mean.astype(np.float32), std.astype(np.float32), count

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
        target_static = np.stack(self.target_static_raw, axis=0).astype(np.float64)
        cont = ~self.static_onehot_mask
        static_mean = np.zeros(self.static_dim, dtype=np.float64)
        static_std = np.ones(self.static_dim, dtype=np.float64)
        if cont.any():
            static_mean[cont] = target_static[:, cont].mean(axis=0)
            static_std[cont] = target_static[:, cont].std(axis=0) + 1e-6
        target_static_norm = (target_static - static_mean) / static_std
        # const.nc tvh_*/tvl_* channels are one-hot and must stay 0/1.
        target_static_norm[:, self.static_onehot_mask] = target_static[:, self.static_onehot_mask]
        self.target_feat_mean = np.concatenate([[mean], static_mean]).astype(np.float64)
        self.target_feat_std = np.concatenate([[std], static_std]).astype(np.float64)
        self.target_feat_raw = np.concatenate([raw.reshape(-1, 1), target_static], axis=1).astype(np.float64)
        self.target_h_arr = np.concatenate([
            arr.astype(np.float32).reshape(-1, 1),
            target_static_norm.astype(np.float32),
        ], axis=1)
        self.args.target_feat_dim = int(self.target_h_arr.shape[1])
        print(f"[data] target_h_mean={mean:.2f} m target_h_std={std:.2f} m "
              f"(train stations={int(mask.sum())}/{len(raw)})", flush=True)

    def save_scalers(self, out_dir: Path) -> None:
        """Save normalization statistics needed for inference on new points.

        input_mean/input_std (per ZTD token), geo_mean/geo_std
        (per-neighbor spatial encoding), target_h_mean/std (target-station
        height feature) and ym/ys (target z-scoring) are all computed from
        the training split. Validation, test, and inference must reuse them.
        """
        np.savez(
            out_dir / "scalers.npz",
            input_mean=self.input_mean,
            input_std=self.input_std,
            input_count=self.input_count,
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
            static_channels=self.static_channels,
            static_onehot_mask=self.static_onehot_mask,
        )
        print(f"[data] scalers saved to {out_dir / 'scalers.npz'}", flush=True)

    def make_sample(self, station_pos: int, t: int):
        ztd = self.station_ngl[station_pos]
        k = ztd.shape[1]
        max_k = self.args.max_neighbors
        n_chan = len(NGL_VARS) * max_k
        g0 = t * self.ngl_steps_per_hour + self.ngl_offset - (self.window_steps - 1)
        g1 = g0 + self.window_steps                # exclusive
        valid_window = np.isfinite(ztd[g0:g1])  # (window, k)
        valid = valid_window.all(axis=0)
        # Dynamic GNSS channels remain the only input variates. Static/ERA5
        # values are attached to each corresponding token embedding in Model.
        x = np.zeros((self.window_steps, n_chan), dtype=np.float32)
        for j in range(k):
            if valid[j]:
                x[:, j] = (ztd[g0:g1, j] - self.input_mean[j]) / self.input_std[j]
        y = self.target[station_pos][t].astype(np.float32)
        y = np.nan_to_num((y - self.ym) / self.ys, nan=0.0).reshape(1, len(NCEP_VARS))
        if self.time_mark is not None:
            mark = self.time_mark[g0:g1]  # (win, n_tf) precomputed
            y_mark = self.time_mark[g1 - 1:g1]  # target time T, used by the decoder head
        else:
            mark = np.zeros((self.window_steps, 0), dtype=np.float32)
            y_mark = np.zeros((1, 0), dtype=np.float32)
        # Per-neighbour encoder features are gated by input validity. They are
        # concatenated onto each neighbour's ZTD embedding in Model.
        v = np.zeros(self.args.max_neighbors, dtype=np.float32)
        v[:k] = valid.astype(np.float32)
        x_geo = self.station_geo_arr[station_pos] * v[:, None]  # (max_k, n_geo_total) float32
        # ERA5 is already standardized in its source Zarr and is sampled only
        # at target hour T.  Repeating an hourly field over the 5-minute window
        # would be removed by the model's per-token instance normalization.
        # Instead, GNSS fields condition their corresponding encoder tokens and
        # target fields are concatenated with target static/time output features.
        if self.use_era5:
            x_era5_enc = self._era5_at_grid(t, self.station_era5_grid[station_pos]) * v[:, None]
            x_era5_tgt = self._era5_at_grid(t, self.target_era5_grid[station_pos][None, :, :])[0]
        else:
            x_era5_enc = np.zeros((self.args.max_neighbors, 0), dtype=np.float32)
            x_era5_tgt = np.zeros(0, dtype=np.float32)
        x_tgt = self.target_h_arr[station_pos]  # z-scored target-level output-head features
        return x, y, mark, y_mark, x_geo, x_tgt, x_era5_enc, x_era5_tgt


class Era5ChunkBatchSampler(Sampler[list[int]]):
    """Batch samples by physical ERA5 time chunk to maximize cache reuse.

    The training order is randomized at the six-hour chunk level.  Samples
    inside a chunk remain contiguous, so a single worker decompresses each
    ERA5 Zarr slab once instead of once per random sample.
    """

    def __init__(self, dataset: "GNSSNowcastDataset", batch_size: int, shuffle: bool):
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        data = dataset.data
        chunk_ids = np.empty(len(dataset), dtype=np.int32)
        for station_pos, times in enumerate(dataset.samples):
            start, stop = dataset.offsets[station_pos], dataset.offsets[station_pos + 1]
            chunk_ids[start:stop] = data.era5_ncep_idx[times] // data.era5_time_chunk
        order = np.argsort(chunk_ids, kind="stable")
        boundaries = np.flatnonzero(np.diff(chunk_ids[order])) + 1
        self.groups = np.split(order, boundaries)

    def __iter__(self):
        group_order = np.random.permutation(len(self.groups)) if self.shuffle else np.arange(len(self.groups))
        for group_idx in group_order:
            indices = self.groups[group_idx]
            if self.shuffle:
                indices = np.random.permutation(indices)
            for start in range(0, len(indices), self.batch_size):
                yield indices[start:start + self.batch_size].tolist()

    def __len__(self) -> int:
        return sum((len(group) + self.batch_size - 1) // self.batch_size for group in self.groups)


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
        # Preserve the target station and NCEP time index for test-time maps;
        # train/validation loops receive and discard these two metadata fields.
        return (*self.data.make_sample(i, t), self.data.station_ids[i], t)

def make_loader(data: NowcastData, flag: str, args: argparse.Namespace, shuffle: bool):
    ds = GNSSNowcastDataset(data, flag)
    if len(ds) == 0:
        raise SystemExit(
            f"[{flag}] no valid samples for the current station selection "
            f"(stations={args.stations}, offset={args.station_offset}, "
            f"min_valid_neighbors={args.min_valid_neighbors}). "
            "Try --set station_offset=... or increase --set stations=..."
        )
    # Chunked batches make the ERA5 six-hour slab cache effective.  A single
    # worker is intentional: multiple workers would each decompress the same
    # slab and reintroduce the HDD bottleneck.
    if data.use_era5:
        sampler = Era5ChunkBatchSampler(ds, args.batch_size, shuffle)
        dl = DataLoader(ds, batch_sampler=sampler, num_workers=args.num_workers,
                        pin_memory=True)
    else:
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                        num_workers=args.num_workers, pin_memory=True, drop_last=False)
    return ds, dl
