#!/usr/bin/env python3
"""测试集上 FuXi（背景）与 ERA5（label）逐通道 MAE，只统计"实际用到的格点"。

配对方式与训练完全一致（``main/utils/utils_data.py``）：

    init = T - fcst_step*6h     （fcst_step = 4 -> lead 24 h）
    step = fcst_step*6h         （store 里那一个 lead）

统计的空间范围可选：

* ``station``   1378 个有 GNSS 站的格点（``station_cell_mask``）
* ``halo3``     上面这些格点膨胀 3 格（``loss_domain='station_halo'``,
                ``loss_halo_cells=3``），即当前配置真正拿来算 label loss 的 5778 格
* ``region``    整个 80x120（默认不算，只是为了对照）

两个 store 都用同一套 ``mean_era5/std_era5`` 标准化，所以先反标准化成物理量再算 MAE。
FuXi 的第 70 个通道是它自己的降水预报，和第 69 通道（IMERG 降水）不是同一个量，
单独列出来与 label 的第 70 通道 ``era5_tp``（ERA5 自己的降水）比较。

用法（**gnss** 环境，在 da_ngl/main_code 下）
-------------------------------------------
    python ../test/mae_fuxi_vs_era5.py --out /tmp/mae_fuxi_vs_era5_test.csv
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DA_ROOT = HERE.parent
sys.path.insert(0, str(DA_ROOT / "main_code"))
sys.path.insert(0, str(DA_ROOT / "preprocessing"))


def build_mask(cfg, mode: str) -> tuple[np.ndarray, str]:
    from main.utils import loss_region_mask, station_cell_mask

    if mode == "station":
        return station_cell_mask(cfg), "1378 个 GNSS 站格"
    if mode == "halo3":
        # 强制用当前配置的 halo 定义，即使 cfg.loss_domain 被改过
        old = getattr(cfg, "loss_domain", None)
        cfg.loss_domain = "station_halo"
        m = loss_region_mask(cfg)
        if old is not None:
            cfg.loss_domain = old
        return m, f"station_halo({int(getattr(cfg, 'loss_halo_cells', 3))}) 格"
    if mode == "region":
        m = np.zeros((int(np.asarray(cfg.lat).size), int(np.asarray(cfg.lon).size)), bool)
        m[:] = True
        return m, "80x120 全域"
    raise ValueError(mode)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default="configs")
    ap.add_argument("--out", type=Path, default=Path("/tmp/mae_fuxi_vs_era5_test.csv"))
    ap.add_argument("--masks", default="station,halo3",
                    help="逗号分隔：station,halo3,region")
    args = ap.parse_args()

    import importlib
    cfg = importlib.import_module(args.configs)
    from common import decode_time_axis, era5_channel_stats

    modes = [m.strip() for m in args.masks.split(",") if m.strip()]
    masks = {}
    for mode in modes:
        m, desc = build_mask(cfg, mode)
        masks[mode] = m
        print(f"[mask] {mode:>8}: {int(m.sum()):>5d} 格  ({desc})")

    lead_h = int(cfg.fcst_step) * 6
    fx = zarr.open(str(cfg.fuxi_zarr), "r")
    lab = zarr.open(str(cfg.label_zarr), "r")["label"]
    names = [str(c) for c in np.asarray(zarr.open(str(cfg.label_zarr), "r")["channel"][:])]
    n_ch = 70                                   # 69 状态 + tp
    bg_ch = int(fx["z"].shape[2])

    steps = np.asarray(fx["step"][:], dtype=float) if "step" in fx else None
    if steps is None:
        raise SystemExit("FuXi store 缺 step 轴")
    lead_idx = int(np.where(steps == lead_h)[0][0])

    bg_t = decode_time_axis(Path(cfg.fuxi_zarr), "init")
    lb_t = decode_time_axis(Path(cfg.label_zarr), "time")
    t0 = pd.to_datetime(str(cfg.dates_test_range[0]), format='%Y%m%d%H')
    t1 = pd.to_datetime(str(cfg.dates_test_range[1]), format='%Y%m%d%H')
    lb_idx = np.flatnonzero((lb_t.values >= t0.to_datetime64()) &
                            (lb_t.values < t1.to_datetime64()))
    print(f"[split] test {t0} ~ {t1}: {lb_idx.size} 个时刻；lead {lead_h} h")
    print(f"[store] FuXi 通道 {bg_ch}（取前 {n_ch}），label 通道 {len(names)}")

    mean, std = era5_channel_stats()
    mean, std = mean[:n_ch].astype("f8"), std[:n_ch].astype("f8")

    acc = {m: {"s": np.zeros(n_ch), "n": np.zeros(n_ch),
               "s2": np.zeros(n_ch), "sa": np.zeros(n_ch), "sb": np.zeros(n_ch)}
           for m in modes}
    used, skipped = 0, 0
    for k, lb_i in enumerate(lb_idx):
        T = pd.Timestamp(lb_t.values[lb_i])
        init = (T - pd.Timedelta(hours=lead_h)).to_datetime64()
        bg_i = int(np.searchsorted(bg_t.values, init))
        if bg_i >= bg_t.values.size or bg_t.values[bg_i] != init:
            skipped += 1
            continue
        a = np.asarray(fx["z"][bg_i, lead_idx, :n_ch], dtype="f8") * std[:, None, None] \
            + mean[:, None, None]
        b = np.asarray(lab[lb_i, :n_ch], dtype="f8") * std[:, None, None] \
            + mean[:, None, None]
        d = a - b
        ok = np.isfinite(d)
        for m, msk in masks.items():
            mm = np.broadcast_to(msk[None], d.shape) & ok
            acc[m]["s"] += np.where(mm, np.abs(d), 0.0).sum(axis=(1, 2))
            acc[m]["n"] += mm.sum(axis=(1, 2))
            acc[m]["s2"] += np.where(mm, d ** 2, 0.0).sum(axis=(1, 2))
            acc[m]["sa"] += np.where(mm, a, 0.0).sum(axis=(1, 2))
            acc[m]["sb"] += np.where(mm, b, 0.0).sum(axis=(1, 2))
        used += 1
        if (k + 1) % 200 == 0:
            print(f"  {k+1}/{lb_idx.size}", flush=True)

    print(f"[done] 用了 {used} 个时刻（缺背景 {skipped} 个）")

    rows = {}
    for m in modes:
        A = acc[m]
        n = np.maximum(A["n"], 1)
        rows[m] = pd.DataFrame({
            f"mae_{m}": A["s"] / n,
            f"rmse_{m}": np.sqrt(A["s2"] / n),
            f"bias_{m}": (A["sa"] - A["sb"]) / n,
            f"fuxi_mean_{m}": A["sa"] / n,
            f"era5_mean_{m}": A["sb"] / n,
        })
    out = pd.DataFrame({"channel": names[:n_ch]})
    for m in modes:
        out = pd.concat([out, rows[m]], axis=1)
    out.to_csv(args.out, index=False)

    print("\n" + "=" * 96)
    print(f"测试集 FuXi(lead {lead_h}h) vs ERA5 逐通道 MAE（物理量，n = {used} 个时刻）")
    print("=" * 96)
    hdr = f'{"通道":>7}' + "".join(f'{("MAE_" + m):>14}' for m in modes) \
        + f'{"bias_" + modes[0]:>14}{"FuXi均值":>12}{"ERA5均值":>12}'
    print(hdr)
    for j, name in enumerate(names[:n_ch]):
        line = f"{name:>7}"
        for m in modes:
            line += f'{rows[m][f"mae_{m}"].iloc[j]:>14.4g}'
        line += f'{rows[modes[0]][f"bias_{modes[0]}"].iloc[j]:>+14.4g}'
        line += f'{rows[modes[0]][f"fuxi_mean_{modes[0]}"].iloc[j]:>12.6g}'
        line += f'{rows[modes[0]][f"era5_mean_{modes[0]}"].iloc[j]:>12.6g}'
        print(line)
    print("-" * 96)
    for m in modes:
        col = rows[m]["mae_" + m]
        all_m = float(col.iloc[:69].mean())
        with_tp = float(col.mean())
        print(f"  {m:>8}: 状态通道(0..68) 平均 MAE = {all_m:.5g}")
        print(f"  {'':>8}  含 tp(69) 平均 MAE = {with_tp:.5g}")
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
