#!/usr/bin/env python3
"""新旧两套 ZHD/ZWD 实现，分别对 NGL 的 ZHD/ZWD 打分。

两套实现
--------
* **旧**（方法 A，``ztd_profile_surface``）：ZHD 用 Saastamoinen 闭式
  ``0.0022768·p_s/(1−0.00266cos2φ−0.00028h_km)``；ZWD 用
  ``1e-6·(R_d/g)·∫[k2'·e/p + k3·e/(T·p)]dp``（k2'=16.52, k3=3.776e5），在气压坐标上梯形积分。
  产物：``ztd_fuxi_europe_0p25_24h.zarr``
* **新**（方法 E，``ztd_profile_zdz``）：ZHD/ZWD 都在几何高度上梯形积分，层高取自 z 通道，
  湿项用 Bevis 常数（k2'=22.98, k3=375463），干项另加 p_top 以上的解析修正。
  产物：``ztd_fuxi_europe_0p25_24h_zdz.zarr``

两个参考基准
------------
* ``NGL ZWD`` = TRWET（直接产品）；``NGL ZHD`` = ZTD − TRWET（反推，含 VMF1/NWM 静力先验）
* ``NCEP ZHD`` = 实测站压 + Saastamoinen（**独立测量**，见 check_zhd_three_ways.py）

用法（在 da_ngl/main_code 下）
----
    python ../test/compare_zhd_zwd_old_vs_new.py --stride 10
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
    ap.add_argument("--old-ztd-store", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import importlib
    cfg = importlib.import_module(args.configs)
    from common import CHANNELS, SPLITS, decode_time_axis, era5_channel_stats
    from main.utils import station_geometry
    from ztd_operator import G_0, LEV_HPA, R_D, ztd_profile_surface, ztd_profile_zdz

    old_path = Path(args.old_ztd_store) if args.old_ztd_store else \
        (Path(cfg.ztd_fuxi_zarr).parent / "ztd_fuxi_europe_0p25_24h.zarr")
    new_path = Path(cfg.ztd_fuxi_zarr)
    print(f"[旧] {old_path.name}\n[新] {new_path.name}")

    iy, ix, h_st, _ = station_geometry(cfg)
    lat_st = np.asarray(cfg.lat, float)[iy]
    T_CH = [CHANNELS.index(f"t{int(L)}") for L in LEV_HPA]
    R_CH = [CHANNELS.index(f"r{int(L)}") for L in LEV_HPA]
    Z_CH = [CHANNELS.index(f"z{int(L)}") for L in LEV_HPA]
    OP = T_CH + R_CH + [CHANNELS.index("t2m"), CHANNELS.index("msl")] + Z_CH
    mean, std = era5_channel_stats()

    ng = zarr.open(str(cfg.ngl_zarr), "r")
    z_mu, z_sd = float(ng["ztd_train_mean"][0]), float(ng["ztd_train_std"][0])
    w_mu, w_sd = float(ng["zwd_train_mean"][0]), float(ng["zwd_train_std"][0])
    obs_t = decode_time_axis(Path(cfg.ngl_zarr), "time")
    fz_old = zarr.open(str(old_path), "r")
    fz_new = zarr.open(str(new_path), "r")
    lab = zarr.open(str(cfg.label_zarr), "r")["label"]
    lab_t = decode_time_axis(Path(cfg.label_zarr), "time")

    rng = [SPLITS["train"][0], SPLITS["test"][1]] if args.split == "all" else SPLITS[args.split]
    a, b = pd.Timestamp(rng[0]), pd.Timestamp(rng[1])
    idx = np.flatnonzero((lab_t >= a) & (lab_t < b))[::args.stride]
    print(f"[win] {a.date()} ~ {b.date()}：{idx.size} 个时刻 × {iy.size} 站")

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
        o_old = ztd_profile_surface(T_lev, R_lev, t2m, p_s, h_st, lat_st)
        o_new = ztd_profile_zdz(T_lev, R_lev, t2m, p_s, h_st, H_lev)
        rows.append(pd.DataFrame({
            "st": np.arange(iy.size),
            "obs_ztd": z_o, "obs_zwd": w_o, "obs_zhd": z_o - w_o,
            "fx_old_zhd": np.asarray(fz_old["zhd"][i], "float32")[iy, ix],
            "fx_old_zwd": np.asarray(fz_old["zwd"][i], "float32")[iy, ix],
            "fx_old_ztd": np.asarray(fz_old["ztd_fuxi"][i], "float32")[iy, ix],
            "fx_new_zhd": np.asarray(fz_new["zhd"][i], "float32")[iy, ix],
            "fx_new_zwd": np.asarray(fz_new["zwd"][i], "float32")[iy, ix],
            "fx_new_ztd": np.asarray(fz_new["ztd_fuxi"][i], "float32")[iy, ix],
            "e5_old_zhd": o_old["ZHD_mm"], "e5_old_zwd": o_old["ZWD_mm"],
            "e5_old_ztd": o_old["ZTD_mm"],
            "e5_new_zhd": o_new["ZHD_mm"], "e5_new_zwd": o_new["ZWD_mm"],
            "e5_new_ztd": o_new["ZTD_mm"]}))
        if (n + 1) % 30 == 0:
            print(f"  {n+1}/{idx.size}", flush=True)
    df = pd.concat(rows, ignore_index=True)
    if args.out:
        df.to_csv(args.out, index=False)

    print(f'\n{"实现":>10}{"源":>8}{"项":>5}{"n":>9}{"Bias":>9}{"RMSE":>9}{"MAE":>9}{"r":>9}')
    for tag, pre in (("旧(A)", "old"), ("新(E)", "new")):
        for src in ("fx", "e5"):
            for c, name in (("zhd", "ZHD"), ("zwd", "ZWD"), ("ztd", "ZTD")):
                o = df[f"obs_{c}"].values
                v = df[f"{src}_{pre}_{c}"].values
                m = np.isfinite(o) & np.isfinite(v)
                d = v[m] - o[m]
                print(f'{tag:>10}{"FuXi" if src=="fx" else "ERA5":>8}{name:>5}{m.sum():>9d}'
                      f'{d.mean():>9.2f}{np.sqrt((d**2).mean()):>9.2f}'
                      f'{np.abs(d).mean():>9.2f}{np.corrcoef(v[m],o[m])[0,1]:>9.4f}')
    # 相互差
    print(f'\n=== 新旧之差（新 − 旧，mm）===')
    for src in ("fx", "e5"):
        for c, name in (("zhd", "ZHD"), ("zwd", "ZWD"), ("ztd", "ZTD")):
            d = df[f"{src}_new_{c}"].values - df[f"{src}_old_{c}"].values
            m = np.isfinite(d)
            print(f'  {"FuXi" if src=="fx" else "ERA5":>5} {name}: {np.nanmean(d):+7.2f} '
                  f'(std {np.nanstd(d):5.2f})')
    if args.out:
        print(f"逐条记录 -> {args.out}")


if __name__ == "__main__":
    main()
