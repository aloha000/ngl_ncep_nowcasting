#!/usr/bin/env python3
"""逐通道背景误差标准差 sigma_b,c（增量惩罚的分母）。

用途
----
``IncrementPenalty``（见 ``main/model/build_optimizer.py``）用

    J_B = mu * mean_{c in 状态通道, 站点格} |x_a - x_b|_c / sigma_b,c

来给"改动分析场"标价。分母 sigma_b,c 是**背景误差的标准差**（比较不同通道
"动多少算多"）。因为 store 已经用 mean_era5/std_era5 标准化，
直接取气候态 std 会让每个通道都等于 1、失去区分度；必须是**背景误差**的 std。

定义
----
    sigma_b,c = std_{T ∈ train, cells}( FuXi(T) - ERA5(T) )      在标准化空间

只在 train split 上统计（这是要被拟合的量，不能看 val/test）。同时输出两个版本：

* ``all_cells``  —— 全域 9600 格
* ``station``    —— 只用 1378 个站点格（增量惩罚实际作用的区域）

输出 ``dataset/bg_err_std.npz``：
    sigma_all (70,), sigma_station (70,), mean_all, mean_station, channel (70,), meta…

用法（在 da_ngl/main_code 下）
----
    python ../preprocessing/build_bg_err_std.py --stride 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

HERE = Path(__file__).resolve().parent
DA_ROOT = HERE.parent
sys.path.insert(0, str(DA_ROOT / "main_code"))
sys.path.insert(0, str(HERE))

from common import DATASET_DIR, SPLITS, decode_time_axis  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default="configs")
    ap.add_argument("--stride", type=int, default=5, help="train 每 N 个时刻取 1 个")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import importlib
    cfg = importlib.import_module(args.configs)
    out = args.out or (DATASET_DIR / "bg_err_std.npz")
    lead = pd.Timedelta(hours=int(cfg.fcst_step) * 6)

    lab = zarr.open(str(cfg.label_zarr), "r")["label"]
    fx = zarr.open(str(cfg.fuxi_zarr), "r")
    lb_t = decode_time_axis(Path(cfg.label_zarr), "time")
    init_t = decode_time_axis(Path(cfg.fuxi_zarr), "init")
    init_vals = init_t.values
    n_bg = int(fx["z"].shape[2])
    n_ch = min(int(lab.shape[1]), n_bg)          # 只比较两边都有的通道

    t0, t1 = pd.Timestamp(SPLITS["train"][0]), pd.Timestamp(SPLITS["train"][1])
    idx = np.flatnonzero((lb_t >= t0) & (lb_t < t1))[::args.stride]
    print(f"[cfg] label={cfg.label_zarr}\n[cfg] fuxi ={cfg.fuxi_zarr} (lead {int(cfg.fcst_step)*6} h)")
    print(f"[train] {t0} ~ {t1}：{idx.size} 个时刻（stride {args.stride}），比较 {n_ch} 个通道")

    gm = pd.read_parquet(Path(cfg.ngl_zarr).parent /
                         "ngl_europe_0p25_80x120_station_grid_map.parquet")
    st_mask = (~gm["mask"].astype(bool)).values.reshape(80, 120)

    # Welford：累积和与平方和（单精度数组，用 float64 累加）
    s1 = np.zeros(n_ch); s2 = np.zeros(n_ch); n_ok = np.zeros(n_ch)
    s1_st = np.zeros(n_ch); s2_st = np.zeros(n_ch); n_st = np.zeros(n_ch)
    used = 0
    for k, i in enumerate(idx):
        bg_i = int(np.searchsorted(init_vals, (lb_t[i] - lead).to_datetime64()))
        if bg_i >= init_vals.size or abs(init_vals[bg_i] -
                                        (lb_t[i] - lead).to_datetime64()) > np.timedelta64(0, "s"):
            continue
        e = np.asarray(lab[i, :n_ch], dtype=np.float32)
        f = np.asarray(fx["z"][bg_i, 0, :n_ch], dtype=np.float32)
        d = (f - e).astype(np.float64)
        ok = np.isfinite(d)
        s1 += np.nansum(np.where(ok, d, np.nan), axis=(1, 2))
        s2 += np.nansum(np.where(ok, d ** 2, np.nan), axis=(1, 2))
        n_ok += ok.sum(axis=(1, 2))
        ok_st = ok & st_mask[None, :, :]
        s1_st += np.nansum(np.where(ok_st, d, np.nan), axis=(1, 2))
        s2_st += np.nansum(np.where(ok_st, d ** 2, np.nan), axis=(1, 2))
        n_st += ok_st.sum(axis=(1, 2))
        used += 1
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{idx.size} 时刻", flush=True)

    def finish(s1, s2, n):
        mu = np.where(n > 0, s1 / np.maximum(n, 1), np.nan)
        var = np.where(n > 1, s2 / np.maximum(n, 1) - mu ** 2, np.nan)
        return mu, np.sqrt(np.maximum(var, 0))

    mu_all, sd_all = finish(s1, s2, n_ok)
    mu_st, sd_st = finish(s1_st, s2_st, n_st)
    ch = [str(c) for c in np.asarray(zarr.open(str(cfg.label_zarr), "r")["channel"][:])[:n_ch]]
    np.savez_compressed(out, sigma_all=sd_all.astype("f4"), sigma_station=sd_st.astype("f4"),
                        mean_all=mu_all.astype("f4"), mean_station=mu_st.astype("f4"),
                        channel=np.asarray(ch), n_sample=np.int32(used),
                        lead_hours=np.int32(int(cfg.fcst_step) * 6),
                        fuxi_zarr=str(cfg.fuxi_zarr), label_zarr=str(cfg.label_zarr),
                        split=np.asarray(SPLITS["train"]))
    print(f"\n[out] {out}（用了 {used} 个时刻）")
    print(f'{"通道":>8}{"sigma_all":>11}{"sigma_station":>15}{"mean_all":>10}{"mean_station":>14}')
    for name in ("z500", "z850", "t850", "t2m", "r700", "r850", "r1000", "u10", "v10", "msl", "tp"):
        if name in ch:
            j = ch.index(name)
            print(f'{name:>8}{sd_all[j]:>11.4f}{sd_st[j]:>15.4f}{mu_all[j]:>10.4f}{mu_st[j]:>14.4f}')
    order = np.argsort(sd_st)
    print("\n站点格上前 10 个'最便宜'（sigma 最小）与后 10 个'最贵'的通道：")
    print("  便宜:", ", ".join(f"{ch[j]}({sd_st[j]:.3f})" for j in order[:10]))
    print("  贵  :", ", ".join(f"{ch[j]}({sd_st[j]:.3f})" for j in order[-10:]))


if __name__ == "__main__":
    main()
