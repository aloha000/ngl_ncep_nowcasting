#!/usr/bin/env python3
"""ZHD / ZWD / ZTD 的三方对比：观测(NGL) vs FuXi 24h vs ERA5。

数据来源
--------
* 观测：``ngl_..._5min.zarr`` 的 ``ztd``（TROTOT）与 ``zwd``（TRWET）。
  **ZHD 不是独立观测**，而是由 NGL 处理链自己的静力模型反推：
  ``ZHD_obs = ZTD_obs - TRWET_obs``（所以"ZHD 误差"里含两套静力模型的差异）
* FuXi：``ztd_fuxi_..._zdz.zarr`` 的 ``zhd / zwd / ztd_fuxi``（方法 E，mm）
* ERA5：**现算** —— 用 ``ztd_profile_zdz`` 对 label store 的 ERA5 廓线求三项

三者共用同一批站格与同一套地面节点构造（p_s 由 msl + 站点高度折算，
地面湿度取"最低仍在地面之上的层"的 RH）。

用法（在 da_ngl/main_code 下）
----
    python ../test/compare_ztd_parts_three_way.py                      # test 期每 10 个时刻
    python ../test/compare_ztd_parts_three_way.py --split all --stride 50
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
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import importlib
    cfg = importlib.import_module(args.configs)
    from common import CHANNELS, SPLITS, decode_time_axis, era5_channel_stats
    from main.utils import station_geometry
    from ztd_operator import G_0, LEV_HPA, R_D, ztd_profile_zdz

    iy, ix, h_st, station_id = station_geometry(cfg)
    T_CH = [CHANNELS.index(f"t{int(L)}") for L in LEV_HPA]
    R_CH = [CHANNELS.index(f"r{int(L)}") for L in LEV_HPA]
    Z_CH = [CHANNELS.index(f"z{int(L)}") for L in LEV_HPA]
    OP = T_CH + R_CH + [CHANNELS.index("t2m"), CHANNELS.index("msl")] + Z_CH
    mean, std = era5_channel_stats()

    ng = zarr.open(str(cfg.ngl_zarr), "r")
    z_mu, z_sd = float(ng["ztd_train_mean"][0]), float(ng["ztd_train_std"][0])
    w_mu, w_sd = float(ng["zwd_train_mean"][0]), float(ng["zwd_train_std"][0])
    obs_t = decode_time_axis(Path(cfg.ngl_zarr), "time")
    fz = zarr.open(str(cfg.ztd_fuxi_zarr), "r")
    lab = zarr.open(str(cfg.label_zarr), "r")["label"]
    lab_t = decode_time_axis(Path(cfg.label_zarr), "time")

    rng = [SPLITS["train"][0], SPLITS["test"][1]] if args.split == "all" else SPLITS[args.split]
    a, b = pd.Timestamp(rng[0]), pd.Timestamp(rng[1])
    idx = np.flatnonzero((lab_t >= a) & (lab_t < b))[::args.stride]
    print(f"[win] {a.date()} ~ {b.date()}：{idx.size} 个时刻，{iy.size} 个站")
    print(f"[src] 观测 {Path(cfg.ngl_zarr).name} | FuXi {Path(cfg.ztd_fuxi_zarr).name} | "
          f"ERA5 由 {Path(cfg.label_zarr).name} 现算")

    rows = []
    for n, i in enumerate(idx):
        T = lab_t[i]
        fidx = int((T - obs_t[0]).total_seconds() // 300)
        z_o = np.asarray(ng["ztd"][fidx, iy, ix], "float32") * z_sd + z_mu
        w_o = np.asarray(ng["zwd"][fidx, iy, ix], "float32") * w_sd + w_mu
        blk = np.asarray(lab[i, :70], "float32")
        ph = blk[OP] * std[OP][:, None, None] + mean[OP][:, None, None]
        ph = ph[:, iy, ix]
        T_lev, R_lev = ph[:13].T, ph[13:26].T
        t2m, msl = ph[26], ph[27]
        H_lev = (ph[28:41] / G_0).T
        p_s = msl / 100.0 * np.exp(-G_0 * h_st / (R_D * t2m))
        e5 = ztd_profile_zdz(T_lev, R_lev, t2m, p_s, h_st, H_lev)
        rows.append(pd.DataFrame({
            "time": T, "st": np.arange(iy.size),
            "obs_ztd": z_o, "obs_zwd": w_o, "obs_zhd": z_o - w_o,
            "fx_ztd": np.asarray(fz["ztd_fuxi"][i], "float32")[iy, ix],
            "fx_zwd": np.asarray(fz["zwd"][i], "float32")[iy, ix],
            "fx_zhd": np.asarray(fz["zhd"][i], "float32")[iy, ix],
            "e5_ztd": e5["ZTD_mm"], "e5_zwd": e5["ZWD_mm"], "e5_zhd": e5["ZHD_mm"]}))
        if (n + 1) % 40 == 0:
            print(f"  {n+1}/{idx.size}", flush=True)
    df = pd.concat(rows, ignore_index=True)
    if args.out:
        df.to_csv(args.out, index=False)

    print(f"\n=== 均值 / 标准差（mm，n={len(df)}）===")
    print(f'{"":>12}{"ZHD":>20}{"ZWD":>20}{"ZTD":>20}')
    print(f'{"":>12}' + ''.join(f'{"mean":>10}{"std":>10}' for _ in range(3)))
    for tag, pre in (("观测(NGL)", "obs"), ("FuXi 24h", "fx"), ("ERA5", "e5")):
        print(f'{tag:>12}' + ''.join(
            f'{np.nanmean(df[f"{pre}_{c}"].values):>10.2f}{np.nanstd(df[f"{pre}_{c}"].values):>10.2f}'
            for c in ("zhd", "zwd", "ztd")))

    g = df.groupby("st")
    print(f'\n=== 相对观测的误差 ===')
    print(f'{"源":>10}{"项":>6}{"n":>9}{"Bias":>9}{"RMSE":>9}{"MAE":>9}{"r":>9}{"距平RMSE":>10}{"距平r":>8}')
    for tag, pre in (("FuXi", "fx"), ("ERA5", "e5")):
        for c, name in (("zhd", "ZHD"), ("zwd", "ZWD"), ("ztd", "ZTD")):
            o = df[f"obs_{c}"].values; v = df[f"{pre}_{c}"].values
            d = v - o; m = np.isfinite(d)
            an = v - g[f"{pre}_{c}"].transform("mean").values
            ao = o - g[f"obs_{c}"].transform("mean").values
            ma = m & np.isfinite(an) & np.isfinite(ao)
            r = np.corrcoef(v[m], o[m])[0, 1] if m.sum() > 2 else np.nan
            print(f'{tag:>10}{name:>6}{m.sum():>9d}{np.nanmean(d):>9.2f}'
                  f'{np.sqrt(np.nanmean(d**2)):>9.2f}{np.nanmean(np.abs(d)):>9.2f}{r:>9.4f}'
                  f'{np.sqrt(np.nanmean((an[ma]-ao[ma])**2)):>10.2f}'
                  f'{np.corrcoef(an[ma], ao[ma])[0,1]:>8.4f}')
    print("\n注：观测的 ZHD = ZTD - TRWET 是反推量，含 NGL 自己静力模型的定义差异。")
    if args.out:
        print(f"逐条记录 -> {args.out}")


if __name__ == "__main__":
    main()
