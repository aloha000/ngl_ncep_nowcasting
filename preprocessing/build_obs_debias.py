#!/usr/bin/env python3
"""Per-station static bias of the ZTD innovation (HANDOFF 11.8 item 2).

    b_s = mean over the *train* split of ( obs_mm(T, s) - H(bg)_mm(T, s) )

``H(bg)`` is the FuXi-implied ZTD the run uses
(``dataset/ztd_fuxi_europe_0p25_{lead}h.zarr``), ``obs`` is the NGL ZTD at the
analysis time ``T``, and the samples are enumerated by ``AssimilationDataset``
so the time alignment is exactly the training one.  Only the train split is
used -- the bias is a fitted quantity and must not see val/test.

Why: the absolute ZTD is ~85 % station-static variance, so ``obs - H(bg)``
carries a large per-station constant (height error, local meteorology, operator
bias).  Without removing it the network has to learn 1378 offsets before any
time-varying signal becomes visible.

Writes ``dataset/obs_debias_lead{lead}h.npz``:

    bias_mm     (1378,)  float32   mean(obs - H(bg)) per station
    bias_grid   (80,120) float32   same, 0 outside the station cells
    bias_std_mm (1378,)  float32   temporal std of (obs - H(bg))
    station_id  (1378,)  string    cell order of the grid map (checked on load)
    plus lead_hours / n_samples / split bounds

Usage (from da_ngl/main_code):
    python ../preprocessing/build_obs_debias.py --set fcst_step=4
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import zarr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "main_code"))
sys.path.insert(0, str(HERE))

from common import decode_time_axis  # noqa: E402
from main.utils import AssimilationDataset, station_geometry  # noqa: E402


def apply_overrides(configs, items):
    cfg = importlib.import_module(configs)
    for item in items:
        key, _, value = item.partition("=")
        for cast in (int, float):
            try:
                value = cast(value)
                break
            except ValueError:
                pass
        setattr(cfg, key.strip(), value)
    return cfg


def read_frames(arr, idx, block=512):
    """``arr[idx]`` for a 1-D index array, in blocks (keeps memory bounded)."""
    out = np.empty((idx.size,) + tuple(arr.shape[1:]), dtype=np.float32)
    for s in range(0, idx.size, block):
        part = idx[s:s + block]
        out[s:s + block] = np.asarray(arr.oindex[part], dtype=np.float32)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", default="configs")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: dataset/obs_debias_lead{lead}h.npz")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    cfg = apply_overrides(args.configs, args.set)
    lead = int(cfg.fcst_step) * 6
    out = args.out or Path(cfg.ngl_zarr).parent / f"obs_debias_lead{lead}h.npz"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists -- pass --force to rebuild")

    # the dataset is used purely to enumerate (label, background, obs) triples;
    # the de-bias map itself is what we are about to build, so keep it off
    cfg.obs_debias = False
    ds = AssimilationDataset(cfg, cfg.dates_train_range)
    samples = np.asarray(ds.samples, dtype=np.int64)          # (n, 3)
    lb_idx = samples[:, 0]
    obs_end = samples[:, 2] + int(cfg.obs_frames) - 1         # frame at T
    print(f"[debias] lead {lead} h, train samples {samples.shape[0]}, "
          f"label axis {ds.label_time[0]} .. {ds.label_time[-1]}")

    iy, ix, height, station_id = station_geometry(cfg)
    obs_store = zarr.open(str(cfg.ngl_zarr), "r")
    res_store = zarr.open(str(cfg.ztd_fuxi_zarr), "r")
    z_mean = float(np.asarray(obs_store["ztd_train_mean"][:]).reshape(-1)[0])
    z_std = float(np.asarray(obs_store["ztd_train_std"][:]).reshape(-1)[0])

    print(f"[debias] reading {obs_end.size} NGL frames at the analysis times ...")
    obs_std = read_frames(obs_store["ztd"], obs_end)[:, iy, ix]        # (n, n_cell)
    obs_mm = obs_std * np.float32(z_std) + np.float32(z_mean)
    bg_mm = np.asarray(res_store["ztd_fuxi"].oindex[lb_idx], dtype=np.float32)[:, iy, ix]

    diff = obs_mm - bg_mm                                              # (n, n_cell) mm
    ok = np.isfinite(diff)
    n_ok = ok.sum(axis=0)
    if (n_ok == 0).any():
        print(f"WARNING: {(n_ok == 0).sum()} station cells have no valid train sample")
    bias = np.where(n_ok > 0, np.nan_to_num(diff).sum(axis=0) /
                    np.maximum(n_ok, 1), 0.0).astype(np.float32)
    with np.errstate(invalid="ignore"):
        bias_std = np.nanstd(np.where(ok, diff, np.nan), axis=0).astype(np.float32)

    grid = np.zeros(tuple(int(v) for v in obs_store["mask"].shape), dtype=np.float32)
    grid[iy, ix] = bias

    resid = np.where(ok, diff - bias[None, :], np.nan)
    print(f"[debias] bias   : mean {bias.mean():+.2f} mm, |b| mean "
          f"{np.abs(bias).mean():.2f} mm, std {bias.std():.2f} mm, "
          f"range [{bias.min():.1f}, {bias.max():.1f}]")
    print(f"[debias] innovation std before {np.nanstd(np.where(ok, diff, np.nan)):.2f} mm "
          f"-> after {np.nanstd(resid):.2f} mm "
          f"({100 * (1 - np.nanstd(resid) / np.nanstd(np.where(ok, diff, np.nan))):.1f} % "
          f"of the variance is station-static)")
    r = np.corrcoef(bias, height)[0, 1] if bias.std() > 0 else float("nan")
    print(f"[debias] corr(bias, station height) = {r:+.3f}")

    np.savez_compressed(
        out, bias_mm=bias, bias_grid=grid, bias_std_mm=bias_std,
        station_id=station_id.astype("U16"), n_samples=n_ok.astype("int32"),
        lead_hours=np.int32(lead), fcst_step=np.int32(cfg.fcst_step),
        ztd_fuxi_zarr=str(cfg.ztd_fuxi_zarr),
        fuxi_zarr=str(cfg.fuxi_zarr),
        split=np.asarray([str(cfg.dates_train_range[0]), str(cfg.dates_train_range[1])]),
        obs_frames=np.int32(cfg.obs_frames))
    print(f"[debias] wrote {out}")


if __name__ == "__main__":
    main()
