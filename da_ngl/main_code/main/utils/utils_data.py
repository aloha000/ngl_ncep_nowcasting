"""Zarr-backed data layer for the GNSS/FuXi -> ERA5 assimilation task.

Replaces the original per-file readers (``read_era5`` / ``read_fcst`` /
``read_obs``) with the three stores built in ``da_ngl/dataset``:

* background : ``fuxi_europe_0p25.zarr``   ``z[init, step, channel(69), lat, lon]``
* observation: ``ngl_europe_0p25_5min.zarr`` ``ztd[time(5min), lat, lon]``
* label      : ``label_europe_0p25.zarr``  ``label[time(6h), channel(70), lat, lon]``

One sample = one 6-hourly analysis time ``T``:

* background = FuXi initialised at ``T - bg_lead_hours`` (lead 6 h by default)
* observation = the ``obs_frames`` 5-minute ZTD frames ending at ``T``
* label = ERA5 (69 channels) + IMERG tp at ``T``
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import DataLoader, Dataset, SequentialSampler, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler

__all__ = ["AssimilationDataset", "build_dataloader", "decode_axis"]


def decode_axis(store, name):
    """Decode a CF-style int64 axis (``units`` attribute) into timestamps."""
    utils = json.loads((Path(store) / name / ".zattrs").read_text())["units"]
    value, _, ref = utils.partition(" since ")
    ref = pd.Timestamp(ref.strip())
    unit = value.strip().lower()
    raw = zarr.open(str(store), "r")[name][:]
    if unit.startswith("minute"):
        delta = pd.to_timedelta(raw, unit="m")
    elif unit.startswith("hour"):
        delta = pd.to_timedelta(raw, unit="h")
    elif unit.startswith("day"):
        delta = pd.to_timedelta(raw, unit="D")
    else:
        raise ValueError(f"unsupported time units {utils!r}")
    return pd.DatetimeIndex(ref + delta)


def _assert_regular(index, name):
    if len(index) < 2:
        raise ValueError(f"{name} time axis has fewer than 2 entries")
    step = index[1] - index[0]
    if not np.all(np.diff(index.values) == np.timedelta64(step)):
        # a gap is tolerable, but then arithmetic indexing would be wrong
        raise ValueError(f"{name} time axis is not regular; arithmetic indexing is unsafe")
    return step


class AssimilationDataset(Dataset):
    def __init__(self, cfg, dates_range, n_label_chans=None):
        # NOTE: only plain values are kept -- the config *module* is not
        # picklable and DataLoader workers are started with 'forkserver'.
        # ``n_label_chans`` selects how many channels of the label store are
        # returned: the store holds 71 (ERA5 69 + IMERG tp + ERA5 tp) but the
        # model only ever sees the first 70.
        self.n_label_chans = int(
            n_label_chans if n_label_chans is not None
            else getattr(cfg, "label_n_chans", 70))
        self.bg_path = str(cfg.fuxi_zarr)
        self.obs_path = str(cfg.ngl_zarr)
        self.label_path = str(cfg.label_zarr)

        self.bg_time = decode_axis(cfg.fuxi_zarr, "init")
        self.obs_time = decode_axis(cfg.ngl_zarr, "time")
        self.label_time = decode_axis(cfg.label_zarr, "time")
        self.bg_step = _assert_regular(self.bg_time, "fuxi.init")
        self.obs_step = _assert_regular(self.obs_time, "ngl.time")
        self.label_step = _assert_regular(self.label_time, "label.time")

        self.obs_frames = int(cfg.obs_frames)
        self.obs_end_offset = pd.Timedelta(minutes=int(getattr(cfg, "obs_end_offset_minutes", 0)))

        # Background (FuXi) convention, mirroring the reference ``read_bg``:
        #     init = T - fcst_step * 6h        (forecast reference time)
        #     step = fcst_step * 6h            (lead time; stored in the zarr)
        self.fcst_step = int(getattr(cfg, "fcst_step", 1))
        self.lead_hours = self.fcst_step * 6
        self.bg_lead = pd.Timedelta(hours=self.lead_hours)
        steps = np.asarray(zarr.open(self.bg_path, "r")["step"][:])
        if self.lead_hours not in steps:
            raise ValueError(
                f"{self.bg_path} has no lead of {self.lead_hours} h "
                f"(step axis = {steps.tolist()}); rebuild it or change fcst_step")
        self.lead_index = int(np.where(steps == self.lead_hours)[0][0])
        print(f"[Dataset] bg: init = T - {self.lead_hours}h, step = {self.lead_hours}h "
              f"(fcst_step={self.fcst_step}, zarr step axis {steps.tolist()})")

        start = pd.to_datetime(str(dates_range[0]), format="%Y%m%d%H")
        end = pd.to_datetime(str(dates_range[1]), format="%Y%m%d%H")
        times = pd.date_range(start, end, freq=self.label_step, inclusive="left")

        bg0, bg1 = self.bg_time[0], self.bg_time[-1]
        lb0, lb1 = self.label_time[0], self.label_time[-1]
        ob0, ob1 = self.obs_time[0], self.obs_time[-1]

        self.samples = []
        n_missing_bg = n_missing_label = n_missing_obs = 0
        for T in times:
            init = T - self.bg_lead
            if init < bg0 or init > bg1:
                n_missing_bg += 1
                continue
            if T < lb0 or T > lb1:
                n_missing_label += 1
                continue
            obs_end = T + self.obs_end_offset
            obs_start = obs_end - (self.obs_frames - 1) * self.obs_step
            if obs_start < ob0 or obs_end > ob1:
                n_missing_obs += 1
                continue
            bg_i = int((init - bg0) / self.bg_step)
            lb_i = int((T - lb0) / self.label_step)
            obs_i = int((obs_start - ob0) / self.obs_step)
            self.samples.append((lb_i, bg_i, obs_i))

        self._handles = None
        self._handles_pid = None
        print(f"[Dataset] label channels used = {self.n_label_chans}")
        print(f"[Dataset] {times[0]} ~ {times[-1]}  wanted={len(times)}  usable={len(self.samples)}"
              f"  (skip bg={n_missing_bg}, label={n_missing_label}, obs={n_missing_obs})")

    # -- stores are opened lazily so every DataLoader worker has its own handle
    def _store(self):
        pid = os.getpid()
        if self._handles is None or self._handles_pid != pid:
            self._handles = (
                zarr.open(self.bg_path, "r"),
                zarr.open(self.obs_path, "r"),
                zarr.open(self.label_path, "r"),
            )
            self._handles_pid = pid
        return self._handles

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        lb_i, bg_i, obs_i = self.samples[idx]
        bg_store, obs_store, label_store = self._store()

        bg = np.asarray(bg_store["z"][bg_i, self.lead_index],
                        dtype=np.float32)[None]                         # (1, 69, H, W)
        obs = np.asarray(obs_store["ztd"][obs_i:obs_i + self.obs_frames],
                         dtype=np.float32)[:, None]                     # (F, 1, H, W)
        label = np.asarray(label_store["label"][lb_i, : self.n_label_chans],
                           dtype=np.float32)                                # (N, H, W)

        return torch.from_numpy(bg), torch.from_numpy(obs), torch.from_numpy(label)


def build_dataloader(cfg, dates_range, batch_size, num_workers, world_size, rank,
                     shuffle=True, persistent_workers=True, prefetch_factor=3,
                     multiprocessing_context="forkserver", pin_memory=False,
                     n_label_chans=None):
    dataset = AssimilationDataset(cfg, dates_range, n_label_chans=n_label_chans)
    if len(dataset) == 0:
        raise RuntimeError("dataset is empty; check the date range and the obs window")
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                     shuffle=shuffle, drop_last=False)
    elif shuffle:
        sampler = SubsetRandomSampler(list(range(len(dataset))))
    else:
        sampler = SequentialSampler(dataset)
    kwargs = dict(dataset=dataset, sampler=sampler, batch_size=batch_size,
                  num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0:
        kwargs.update(prefetch_factor=prefetch_factor,
                      persistent_workers=persistent_workers,
                      multiprocessing_context=multiprocessing_context)
    return DataLoader(**kwargs)
