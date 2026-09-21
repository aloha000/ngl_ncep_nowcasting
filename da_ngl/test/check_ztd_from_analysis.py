#!/usr/bin/env python3
"""某个训练结果的分析场隐含 ZTD，与 NGL GNSS 实测 ZTD 比较。

做的事
------
加载一个 run 的 val-best checkpoint，在指定时间窗内前向：

    H(x_bg)  = ZTD 算子(FuXi 背景)          ← 背景场隐含的 ZTD
    H(x_a)   = ZTD 算子(模型分析场)          ← 网络同化后的 ZTD
    obs      = NGL GNSS 实测 ZTD [mm]        ← 真值

用与训练完全相同的可微算子 ``main/model/ztd_torch.StationZTD``（= 廓线版
``ztd_profile_surface``：ZHD 由地面气压算，ZWD 对 13 层 + 地面节点积分），
只在 1378 个站格上求值。

同时给出三个口径：
  ``bg``        H(x_bg) vs obs
  ``ana``       H(x_a)  vs obs
  ``ana+b_s``   H(x_a) + b_s vs obs     ← 训练时一致性损失的目标是 obs − b_s，
                                          所以加回逐站静态偏差后才是偏"原始观测"的
                                          口径，和地面观测算 ZTD 那张表可比

另外打印 ``|H(x_a) − (obs − b_s)|`` 的 MAE —— 这是训练日志里的
`[obs-consistency: |H(x)-obs|=… mm]`，用来交叉验证本脚本的口径没搞错。

注意：默认窗口（2024-05~2024-08）属于 val split，模型见过，做 ZTD 技能对比没问题，
但别当成泛化性能。要泛化口径请用 --split test（test 期没有地面观测，但 GNSS ZTD 有）。

用法（在 da_ngl/main_code 下）
----
    python ../test/check_ztd_from_analysis.py \
        --model_id lead24h_halo3_ztd_same_meanstd_stationarea \
        --exp_tag  lead24h_obs6h_era5tp_w1-0_halo3_bgtp_both_obsstd_oc0.2_occ_debias
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
DA_ROOT = HERE.parent
sys.path.insert(0, str(HERE))                    # 让脚本能在任何目录下运行
sys.path.insert(0, str(DA_ROOT / "main_code"))
sys.path.insert(0, str(DA_ROOT / "preprocessing"))

DEFAULT_RUN = ("lead24h_halo3_ztd_same_meanstd_stationarea_"
               "lead24h_obs6h_era5tp_w1-0_halo3_bgtp_both_obsstd_oc0.2_occ_debias")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default="configs")
    ap.add_argument("--model_id", default="lead24h_halo3_ztd_same_meanstd_stationarea")
    ap.add_argument("--exp_tag",
                    default="lead24h_obs6h_era5tp_w1-0_halo3_bgtp_both_obsstd_oc0.2_occ_debias")
    ap.add_argument("--arch_tag", default=None)
    ap.add_argument("--split", default="val", choices=("train", "val", "test"))
    ap.add_argument("--start", default="2024-05-01")
    ap.add_argument("--end", default="2024-08-16", help="半开区间上界")
    ap.add_argument("--hours", default="0,6,12,18")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import importlib
    cfg = importlib.import_module(args.configs)
    cfg.model_id = args.model_id
    cfg.exp_tag = args.exp_tag
    if args.arch_tag:
        cfg.arch_tag = args.arch_tag

    from main.model import StationZTD
    from main.utils import (AssimilationDataset, checkpoint_file, exp_tag,
                            load_obs_debias, station_geometry)
    from plot_results import load_model
    from train_FSDP import process_bg, process_obs, obs_at_valid_time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # obs_at_valid_time 要用 NGL 的 μ/σ（训练时由 build_obs_loss 顺带加载，这里手动读）
    import zarr
    _gn = zarr.open(str(cfg.ngl_zarr), "r")
    cfg.ztd_train_mean = float(np.asarray(_gn["ztd_train_mean"][:]).reshape(-1)[0])
    cfg.ztd_train_std = float(np.asarray(_gn["ztd_train_std"][:]).reshape(-1)[0])
    print(f"[ngl] μ={cfg.ztd_train_mean:.2f} mm  σ={cfg.ztd_train_std:.2f} mm")
    ckpt = Path(checkpoint_file(cfg))
    if not ckpt.exists():
        raise SystemExit(f"找不到 checkpoint：{ckpt}")
    print(f"[run] {cfg.model_id}_{exp_tag(cfg)}\n[ckpt] {ckpt}")
    model, iteration = load_model(cfg, ckpt, device)
    print(f"[model] iteration {iteration} | device {device} | "
          f"obs_chans={cfg.model_obs_chans} bg_chans={cfg.model_bg_chans}")

    # ---- 站点与静态偏差 ----------------------------------------------------
    iy, ix, height, station_id = station_geometry(cfg)
    op = StationZTD(cfg).to(device)
    bias = None
    if bool(getattr(cfg, "obs_debias", False)):
        bias, _ = load_obs_debias(cfg)
        print(f"[debias] b_s: mean {bias.mean():+.2f} mm, |b| mean "
              f"{np.abs(bias).mean():.2f} mm")

    # ---- 数据集：按时间窗与时次筛样本 --------------------------------------
    dates = {"train": cfg.dates_train_range, "val": cfg.dates_val_range,
             "test": cfg.dates_test_range}[args.split]
    ds = AssimilationDataset(cfg, dates, n_label_chans=cfg.label_n_chans)
    hours = {int(v) for v in str(args.hours).split(",")}
    lt = ds.label_time[[s[0] for s in ds.samples]]
    keep = [(t >= pd.Timestamp(args.start)) and (t < pd.Timestamp(args.end)) and (t.hour in hours)
            for t in lt]
    idx = np.flatnonzero(keep)
    print(f"[data] {args.split} split 共 {len(ds)} 样本，窗口内 {idx.size} 个 "
          f"({args.start} ~ {args.end}, 时次 {sorted(hours)})")
    loader = DataLoader(Subset(ds, idx.tolist()), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)

    # ---- 前向 + 算子 --------------------------------------------------------
    recs = []
    with torch.no_grad():
        for k, (batch_fcst, batch_obs, _) in enumerate(loader):
            bg = process_bg(batch_fcst, device)
            obs = process_obs(batch_obs, cfg, device)
            out = model(bg, obs)
            z_bg = op(bg).float().cpu().numpy()          # (B, n_cell) mm
            z_an = op(out).float().cpu().numpy()
            obs_mm = obs_at_valid_time(batch_obs, cfg, device)   # (B,H,W) mm，站格外 NaN
            o = obs_mm[:, iy, ix].float().cpu().numpy()
            times = lt[idx[k * args.batch_size: k * args.batch_size + z_an.shape[0]]]
            for b in range(z_an.shape[0]):
                recs.append(pd.DataFrame({
                    "time": times[b], "station": station_id,
                    "ztd_bg": z_bg[b], "ztd_ana": z_an[b], "ztd_gnss": o[b]}))
            if k % 20 == 0:
                print(f"  batch {k}/{len(loader)}", flush=True)
    df = pd.concat(recs, ignore_index=True)
    df = df[np.isfinite(df["ztd_gnss"].values)]
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
    print(f"[run] 有效记录 {len(df)}，站点 {df['station'].nunique()}")

    def line(tag, a, b):
        d = a - b
        r = np.corrcoef(a, b)[0, 1] if d.size > 2 else np.nan
        print(f"{tag:>10}{d.size:>9d}{a.mean():>11.2f}{b.mean():>11.2f}"
              f"{d.mean():>9.2f}{np.sqrt((d**2).mean()):>9.2f}{np.abs(d).mean():>9.2f}{r:>8.4f}")

    hdr = (f'{"口径":>10}{"n":>9}{"算子均值":>11}{"GNSS均值":>11}'
           f'{"Bias":>9}{"RMSE":>9}{"MAE":>9}{"r":>8}')
    print(f"\n=== ZTD（mm）：算子 − NGL GNSS 实测 ===")
    print(hdr)
    o = df["ztd_gnss"].values
    line("bg", df["ztd_bg"].values, o)
    line("ana", df["ztd_ana"].values, o)
    if bias is not None:
        line("ana+b_s", df["ztd_ana"].values + bias, o)
    if bias is not None:
        tgt = o - bias
        d = df["ztd_ana"].values - tgt
        print(f"\n[check] |H(x_a) − (obs − b_s)|  MAE = {np.abs(d).mean():.3f} mm，"
              f"RMSE = {np.sqrt((d**2).mean()):.3f} mm（应与训练日志的 "
              f"[obs-consistency] 同量级）")

    print(f"\n=== 距平（各站减去自身时间均值）===")
    print(f'{"口径":>10}{"n":>9}{"算子距平std":>13}{"GNSS距平std":>13}'
          f'{"Bias":>9}{"RMSE":>9}{"r":>8}')
    g = df.groupby("station")
    oa = (df["ztd_gnss"] - g["ztd_gnss"].transform("mean")).values
    for tag, col in (("bg", "ztd_bg"), ("ana", "ztd_ana")):
        a = (df[col] - g[col].transform("mean")).values
        d = a - oa
        print(f"{tag:>10}{d.size:>9d}{a.std():>13.2f}{oa.std():>13.2f}"
              f"{d.mean():>9.2f}{np.sqrt((d**2).mean()):>9.2f}{np.corrcoef(a, oa)[0,1]:>8.4f}")
    if args.out:
        print(f"\n逐条记录 -> {args.out}")


if __name__ == "__main__":
    main()
