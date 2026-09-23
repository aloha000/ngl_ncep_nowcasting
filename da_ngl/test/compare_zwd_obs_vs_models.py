#!/usr/bin/env python3
"""观测 ZWD（NGL TRWET）vs FuXi 背景 ZWD vs ERA5 分析 ZWD。

三方来源
--------
* 观测：``ngl_europe_0p25_5min.zarr`` 的 ``zwd``（= TRWET，方法 E 之前新增的数组），
  用 ``zwd_train_mean/std`` 还原成 mm
* FuXi：``ztd_fuxi_europe_0p25_24h_zdz.zarr`` 的 ``zwd``（同一算子对 FuXi 背景的湿项，
  已经是 mm，站格外 NaN）
* ERA5：**现算** —— 用 ``ztd_profile_zdz`` 对 label store 的 ERA5 廓线（t/r/t2m/msl/z）
  在同一批站格上求湿项（ERA5 没有预先算好的 ZTD 库）

三者共用：同一批站格（``station_geometry``）、同一个地面气压折算
（msl + 站点高度）、同一个"最低仍在地面之上的层"的 RH 作为地面湿度。

用法（在 da_ngl/main_code 下）
----
    python ../test/compare_zwd_obs_vs_models.py                       # test 期每 10 个时刻
    python ../test/compare_zwd_obs_vs_models.py --split all --stride 50
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default="configs")
    ap.add_argument("--split", default="test", choices=("train", "val", "test", "all"))
    ap.add_argument("--stride", type=int, default=10, help="每 N 个 6 小时时刻取 1 个")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import importlib
    cfg = importlib.import_module(args.configs)
    from common import CHANNELS, SPLITS, decode_time_axis
    from main.utils import station_geometry
    from ztd_operator import G_0, LEV_HPA, R_D, ztd_profile_zdz

    iy, ix, h_st, station_id = station_geometry(cfg)
    T_CH = [CHANNELS.index(f"t{int(L)}") for L in LEV_HPA]
    R_CH = [CHANNELS.index(f"r{int(L)}") for L in LEV_HPA]
    Z_CH = [CHANNELS.index(f"z{int(L)}") for L in LEV_HPA]
    I_T2M, I_MSL = CHANNELS.index("t2m"), CHANNELS.index("msl")
    OP = T_CH + R_CH + [I_T2M, I_MSL] + Z_CH

    ng = zarr.open(str(cfg.ngl_zarr), "r")
    w_mu = float(ng["zwd_train_mean"][0]); w_sd = float(ng["zwd_train_std"][0])
    obs_zwd = ng["zwd"]
    obs_t = decode_time_axis(Path(cfg.ngl_zarr), "time")
    t0_obs = obs_t[0]
    fx = zarr.open(str(getattr(cfg, "ztd_fuxi_zarr")), "r")
    lab = zarr.open(str(cfg.label_zarr), "r")["label"]
    lab_t = decode_time_axis(Path(cfg.label_zarr), "time")

    from common import era5_channel_stats
    mean, std = era5_channel_stats()

    if args.split == "all":
        rng = [SPLITS["train"][0], SPLITS["test"][1]]
    else:
        rng = SPLITS[args.split]
    a, b = pd.Timestamp(rng[0]), pd.Timestamp(rng[1])
    idx = np.flatnonzero((lab_t >= a) & (lab_t < b))[::args.stride]
    print(f"[cfg] 观测 zwd μ/σ = {w_mu:.2f}/{w_sd:.2f} mm；FuXi 库 {Path(cfg.ztd_fuxi_zarr).name}")
    print(f"[win] {a} ~ {b}：{idx.size} 个时刻（stride {args.stride}），{iy.size} 个站")

    rows = []
    for n, i in enumerate(idx):
        T = lab_t[i]
        fidx = int((T - t0_obs).total_seconds() // 300)
        o = np.asarray(obs_zwd[fidx, iy, ix], "float32") * w_sd + w_mu
        f = np.asarray(fx["zwd"][i], "float32")[iy, ix]
        # ERA5：现算
        blk = np.asarray(lab[i, :70], "float32")
        ph = blk[OP] * std[OP][:, None, None] + mean[OP][:, None, None]
        ph = ph[:, iy, ix]
        T_lev, R_lev = ph[:13].T, ph[13:26].T
        t2m, msl = ph[26], ph[27]
        H_lev = (ph[28:41] / G_0).T
        p_s = msl / 100.0 * np.exp(-G_0 * h_st / (R_D * t2m))
        e = ztd_profile_zdz(T_lev, R_lev, t2m, p_s, h_st, H_lev)
        rows.append(pd.DataFrame({
            "time": T, "st": np.arange(iy.size), "obs": o, "fuxi": f,
            "era5": e["ZWD_mm"], "p_s": p_s}))
        if (n + 1) % 25 == 0:
            print(f"  {n+1}/{idx.size}", flush=True)
    df = pd.concat(rows, ignore_index=True)
    if args.out:
        df.to_csv(args.out, index=False)

    print(f"\n=== ZWD（mm）统计：n={len(df)} ===")
    print(f'{"":>10}{"均值":>10}{"std":>9}{"最小":>9}{"最大":>9}')
    for tag, c in (("观测", "obs"), ("FuXi", "fuxi"), ("ERA5", "era5")):
        v = df[c].values
        print(f'{tag:>10}{np.nanmean(v):>10.2f}{np.nanstd(v):>9.2f}'
              f'{np.nanmin(v):>9.2f}{np.nanmax(v):>9.2f}')

    print(f'\n{"":>10}{"n":>9}{"Bias":>9}{"RMSE":>9}{"MAE":>9}{"r":>9}{"距平RMSE":>10}{"距平r":>8}')
    g = df.groupby("st")
    for tag, c in (("FuXi", "fuxi"), ("ERA5", "era5")):
        d = df[c].values - df["obs"].values
        m = np.isfinite(d)
        an = df[c].values - g[c].transform("mean").values
        ao = df["obs"].values - g["obs"].transform("mean").values
        ma = m & np.isfinite(an) & np.isfinite(ao)
        print(f'{tag:>10}{m.sum():>9d}{np.nanmean(d):>9.2f}'
              f'{np.sqrt(np.nanmean(d**2)):>9.2f}{np.nanmean(np.abs(d)):>9.2f}'
              f'{np.corrcoef(df[c].values[m], df["obs"].values[m])[0,1]:>9.4f}'
              f'{np.sqrt(np.nanmean((an[ma]-ao[ma])**2)):>10.2f}'
              f'{np.corrcoef(an[ma], ao[ma])[0,1]:>8.4f}')
    print("\n（距平 = 各站减去自身时间均值，反映时间变化能力）")
    if args.out:
        print(f"逐条记录 -> {args.out}")


if __name__ == "__main__":
    main()
