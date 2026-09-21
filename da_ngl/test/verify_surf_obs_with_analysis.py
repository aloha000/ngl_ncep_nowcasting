#!/usr/bin/env python3
"""地表观测检验三件东西：ERA5 标签 / FuXi 24h 背景 / CNN 分析场。

和 ``verify_surf_obs.py`` 同一套口径（域内站点、最近邻落格、轻量合理性过滤），
额外把某个 run 的 **分析场** 加进来，并且按需求用"**先算每个时次的 MAE，再对时次取平均**"
作为主指标（同时也给出 pooled 口径做对照）。

三个被检验的对象：
    era5    label store 在 T 时刻的物理量
    fuxi    fuxi store 在 init = T − fcst_step×6h 的物理量（24 小时预报）
    ana     模型(checkpoint)在同样输入下输出的分析场

变量：t2m [K]、msl vs slp [hPa]、u10 / v10 [m/s]（都是 0.25° 网格上的最近邻格点值）。

注意 checkpoint 是在 val 期选的；如果 --start/--end 落在 test 期，那就是干净的泛化检验。

用法（在 da_ngl/main_code 下）
----
    CUDA_VISIBLE_DEVICES=0 python ../test/verify_surf_obs_with_analysis.py \
        --model_id lead24h_halo3_ztd_same_meanstd_stationarea \
        --exp_tag  lead24h_obs6h_era5tp_w1-0_halo3_bgtp_both_obsstd_oc0.2_occ_debias \
        --obs-dir /cpfs01/.../prod_output/qc/surf_ncep/2025 \
        --start 2025-01-01 --end 2025-10-01
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
DA_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DA_ROOT / "main_code"))
sys.path.insert(0, str(DA_ROOT / "preprocessing"))

# 观测列 -> (store 通道名, 模型侧再除的单位因子)
VARS = (("t2m", "t2m", 1.0), ("slp", "msl", 100.0),
        ("u10", "u10", 1.0), ("v10", "v10", 1.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default="configs")
    ap.add_argument("--model_id", default="lead24h_halo3_ztd_same_meanstd_stationarea")
    ap.add_argument("--exp_tag",
                    default="lead24h_obs6h_era5tp_w1-0_halo3_bgtp_both_obsstd_oc0.2_occ_debias")
    ap.add_argument("--arch_tag", default=None)
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--obs-dir", required=True,
                    help="含 surf_YYYYMMDDHH.pkl 的目录（例如 .../surf_ncep/2025）")
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2025-10-01", help="半开区间上界")
    ap.add_argument("--hours", default="0,6,12,18")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-filter", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="逐时次指标存 csv")
    args = ap.parse_args()

    import importlib
    cfg = importlib.import_module(args.configs)
    cfg.model_id = args.model_id
    cfg.exp_tag = args.exp_tag
    if args.arch_tag:
        cfg.arch_tag = args.arch_tag

    import zarr
    from common import CHANNELS, era5_channel_stats
    from main.utils import AssimilationDataset, checkpoint_file, exp_tag
    from plot_results import load_model
    from train_FSDP import process_bg, process_obs

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = Path(checkpoint_file(cfg))
    if not ckpt.exists():
        raise SystemExit(f"找不到 checkpoint：{ckpt}")
    model, iteration = load_model(cfg, ckpt, device)
    print(f"[run] {cfg.model_id}_{exp_tag(cfg)}\n[ckpt] {ckpt} (iteration {iteration})")
    print(f"[model] device={device} obs_chans={cfg.model_obs_chans} bg_chans={cfg.model_bg_chans}")

    mean, std = era5_channel_stats()
    ci = {name: CHANNELS.index(name) for _, name, _ in VARS}
    lat_ax, lon_ax = np.asarray(cfg.lat, float), np.asarray(cfg.lon, float)
    res = float(lat_ax[1] - lat_ax[0])

    lab = zarr.open(str(cfg.label_zarr), "r")["label"]
    gf = zarr.open(str(cfg.fuxi_zarr), "r")

    # ---- 数据集：只留窗口内、指定时次的样本 --------------------------------
    dates = {"train": cfg.dates_train_range, "val": cfg.dates_val_range,
             "test": cfg.dates_test_range}[args.split]
    ds = AssimilationDataset(cfg, dates, n_label_chans=cfg.label_n_chans)
    lb_idx = np.array([s[0] for s in ds.samples])
    times_all = ds.label_time[lb_idx]
    hours = {int(v) for v in str(args.hours).split(",")}
    keep = np.array([(t >= pd.Timestamp(args.start)) and (t < pd.Timestamp(args.end))
                     and (t.hour in hours) for t in times_all])
    sub_idx = np.flatnonzero(keep)
    times_keep = times_all[sub_idx]
    print(f"[data] {args.split} split {len(ds)} 样本 -> 窗口内 {sub_idx.size} 个时次")
    loader = DataLoader(Subset(ds, sub_idx.tolist()), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers, pin_memory=False)

    acc = {v: {k: np.zeros(2) for k in ("era5", "fuxi", "ana")} for v, _, _ in VARS}
    per_time = []
    k = 0
    t0 = _time.time()
    with torch.no_grad():
        for batch_fcst, batch_obs, batch_label in loader:
            n = batch_fcst.shape[0]
            bg = process_bg(batch_fcst, device)
            obs = process_obs(batch_obs, cfg, device)
            out = model(bg, obs).float()
            blk = {"era5": batch_label.numpy(), "fuxi": batch_fcst[:, 0].numpy(),
                   "ana": out[:, 0].cpu().numpy()}
            for b in range(n):
                t = times_keep[k + b]
                path = Path(args.obs_dir) / f"surf_{t:%Y%m%d%H}.pkl"
                if not path.exists():
                    continue
                with open(path, "rb") as fh:
                    d = pickle.load(fh)
                lon_s = (d["lon"].values % 360.0 + 180.0) % 360.0 - 180.0
                lat_s = d["lat"].values.astype(float)
                inside = ((lat_s >= lat_ax[0] - res / 2) & (lat_s <= lat_ax[-1] + res / 2)
                          & (lon_s >= lon_ax[0] - res / 2) & (lon_s <= lon_ax[-1] + res / 2))
                if not inside.any():
                    continue
                iy = np.clip(np.round((lat_s - lat_ax[0]) / res).astype(int), 0, lat_ax.size - 1)[inside]
                ix = np.clip(np.round((lon_s - lon_ax[0]) / res).astype(int), 0, lon_ax.size - 1)[inside]
                sub = d[inside]
                for ocol, cname, scale in VARS:
                    o = sub[ocol].values.astype(float)
                    m = np.isfinite(o)
                    if not args.no_filter:
                        if ocol == "t2m":
                            m &= (o > 200) & (o < 340)
                        elif ocol == "slp":
                            m &= (o > 850) & (o < 1100)
                        else:
                            m &= np.abs(o) < 60
                    if not m.any():
                        continue
                    vals = {}
                    for src in ("era5", "fuxi", "ana"):
                        f = blk[src][b][ci[cname]][iy, ix] * np.float32(std[ci[cname]]) \
                            + np.float32(mean[ci[cname]])
                        if scale != 1.0:
                            f = f / scale
                        vals[src] = f[m]
                    om = o[m]
                    row = {"time": t, "var": ocol, "n": int(m.sum())}
                    for src in vals:
                        mae = float(np.abs(vals[src] - om).mean())
                        acc[ocol][src] += [mae * m.sum(), m.sum()]
                        row[f"mae_{src}"] = mae
                    per_time.append(row)
            k += n
            if (k // args.batch_size) % 40 == 0:
                print(f"  {k}/{sub_idx.size} 时次，{_time.time()-t0:.0f}s", flush=True)

    pt = pd.DataFrame(per_time)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pt.to_csv(args.out, index=False)
    print(f"\n[run] 完成：{pt['time'].nunique()} 个时次有观测，逐时次记录 {len(pt)} 行，"
          f"耗时 {_time.time()-t0:.0f}s")

    print("\n=== 主指标：每个时次先算 MAE，再对时次取平均 ===")
    print(f'{"变量":>6}{"时次数":>7}{"每时次平均站数":>14}{"ERA5":>10}{"FuXi24h":>10}{"分析场":>10}'
          f'{"分析 vs 背景":>13}')
    for v, _, _ in VARS:
        g = pt[pt["var"] == v]
        if g.empty:
            continue
        e, f, a = g["mae_era5"].mean(), g["mae_fuxi"].mean(), g["mae_ana"].mean()
        print(f'{v:>6}{len(g):>7d}{g["n"].mean():>14.0f}{e:>10.3f}{f:>10.3f}{a:>10.3f}'
              f'{100*(f-a)/f:>12.2f}%')

    print("\n=== 对照：pooled（所有记录一起算）===")
    print(f'{"变量":>6}{"ERA5":>10}{"FuXi24h":>10}{"分析场":>10}{"分析 vs 背景":>13}')
    for v, _, _ in VARS:
        s = acc[v]
        e = s["era5"][0] / max(s["era5"][1], 1)
        f = s["fuxi"][0] / max(s["fuxi"][1], 1)
        a = s["ana"][0] / max(s["ana"][1], 1)
        print(f'{v:>6}{e:>10.3f}{f:>10.3f}{a:>10.3f}{100*(f-a)/f:>12.2f}%')

    print("\n=== 分析场相对背景的逐时次改善（%）===")
    for v, _, _ in VARS:
        g = pt[pt["var"] == v]
        imp = 100 * (g["mae_fuxi"] - g["mae_ana"]) / g["mae_fuxi"]
        print(f'{v:>6}  平均 {imp.mean():+.2f}%  中位 {imp.median():+.2f}%  '
              f'改善时次占比 {100*(imp > 0).mean():.1f}%')


if __name__ == "__main__":
    main()
