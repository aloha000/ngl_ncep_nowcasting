#!/usr/bin/env python3
"""两种 ZTD 算子实现的对比：气压坐标积分（项目用） vs 几何高度积分（用户提供）。

方法 A（项目，``ztd_operator.ztd_profile_surface``）
    ZHD = 0.0022768·p_s/(1-0.00266cos2φ-0.00028h_km)          ← 静力项解析积分
    ZWD = 1e-6·(R_d/g)·∫[k2'·e/p + k3·e/(T·p)]dp              ← 在气压坐标上梯形积分
    常数: k2'=16.52, k3=3.776e5, R_d=287.05, g=9.80665；Tetens 6.1078/17.269/237.3

方法 B（外部提供，本文件内 ``calc_zhd_zwd`` 逐字复制）
    ZHD = 1e-6·∫ k1·(p-e)/T dz                                 ← 在几何高度上积分
    ZWD = 1e-6·∫ [(k2-ε·k1)·e/T + k3·e/T²] dz                 ← 同上（可选 Z_w）
    常数: k1=77.6890, k2=71.2952, k3=375463.0, ε=0.62198；
          Tetens 6.112/17.67/243.5

两者在数学上只差一个变量替换 dz = −(R_d·T)/(g·p)·dp；差别来自
（a）积分坐标与积分格式、（b）常数与饱和水汽压公式、（c）干项是解析式还是数值积分、
（d）ZHD 用的是总压 p 还是干压 (p−e)。

本脚本用 FuXi 24h 背景与 ERA5 标签各自的温度/湿度廓线算出 ZTD，
再与 NGL GNSS 实测 ZTD 比较，看两种实现差多少、谁更接近观测。
几何高度直接取 store 自带的位势通道（``z*`` [m²/s²] ÷ 9.80665 = m），
不需要额外做静力重建；地面气压两种方法共用同一个（由 msl + 站点高度折算），
这样差别只来自公式本身。

用法（在 da_ngl/main_code 下）
----
    python ../test/compare_zwd_methods.py                 # 默认 test 期抽样 100 个时次
    python ../test/compare_zwd_methods.py --stride 4 --start 2025-01-01 --end 2025-10-01
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DA_ROOT = HERE.parent
sys.path.insert(0, str(DA_ROOT / "main_code"))
sys.path.insert(0, str(DA_ROOT / "preprocessing"))

G_0, R_D = 9.80665, 287.05


# --------------------------------------------------------------------------
# 方法 B：外部提供的实现，逐字复制（只加了类型说明与返回值）
# --------------------------------------------------------------------------
def _sat_vapor_pressure_B(T_k):
    Tc = T_k - 273.15
    return 6.112 * np.exp(17.67 * Tc / (Tc + 243.5))


def calc_zhd_zwd(p_hpa, T_k, rh_percent, h_m, H_s=None, zw_exact=False,
                 k1=77.6890, k2=71.2952, k3=375463.0, eps=0.62198):
    """外部版本：按几何高度积分。返回 (ZHD, ZWD, ZTD)，单位 m。"""
    p = np.asarray(p_hpa, dtype=np.float64)
    T = np.asarray(T_k, dtype=np.float64)
    rh = np.asarray(rh_percent, dtype=np.float64)
    h = np.asarray(h_m, dtype=np.float64)

    order = np.argsort(h)
    p, T, rh, h = p[order], T[order], rh[order], h[order]

    e = rh / 100.0 * _sat_vapor_pressure_B(T)

    if H_s is not None:
        H_s = float(H_s)
        if h[0] < H_s < h[-1]:
            p_s = np.interp(H_s, h, p)
            T_s = np.interp(H_s, h, T)
            e_s = np.interp(H_s, h, e)
            p = np.concatenate([[p_s], p[h > H_s]])
            T = np.concatenate([[T_s], T[h > H_s]])
            e = np.concatenate([[e_s], e[h > H_s]])
            h = np.concatenate([[H_s], h[h > H_s]])

    dh = np.diff(h)
    p_l, T_l, e_l = p[:-1], T[:-1], e[:-1]

    N_h = k1 * (p_l - e_l) / T_l
    if zw_exact:
        Tc = T_l - 273.15
        Zw_inv = 1.0 + 1650.0 * (e_l / T_l ** 3) * (
            1.0 - 0.01317 * Tc + 1.75e-4 * Tc ** 2 + 1.44e-6 * Tc ** 3)
    else:
        Zw_inv = 1.0
    N_w = ((k2 - eps * k1) * (e_l / T_l) + k3 * (e_l / T_l ** 2)) * Zw_inv

    ZHD = 1e-6 * np.sum(N_h * dh)
    ZWD = 1e-6 * np.sum(N_w * dh)
    return ZHD, ZWD, ZHD + ZWD


# p_top(50 hPa) 以上的干空气柱：用解析式补足，1e-6*k1*(R_d/g)*p_top
TOP_CORR_MM = 1e-6 * 77.6890 * (R_D / G_0) * 50.0 * 1000.0


def _dz_trapz_e(p_hpa, T_k, e_hpa, h_m,
                k1=77.6890, k2=71.2952, k3=375463.0, eps=0.62198):
    """同 B 的公式，梯形积分；水汽压直接传入。返回 (ZHD, ZWD) in m。"""
    p = np.asarray(p_hpa, float); T = np.asarray(T_k, float)
    e = np.asarray(e_hpa, float); h = np.asarray(h_m, float)
    o = np.argsort(h); p, T, e, h = p[o], T[o], e[o], h[o]
    N_h = k1 * (p - e) / T
    N_w = (k2 - eps * k1) * (e / T) + k3 * (e / T ** 2)
    dh = np.diff(h)
    return (1e-6 * np.sum(0.5 * (N_h[1:] + N_h[:-1]) * dh),
            1e-6 * np.sum(0.5 * (N_w[1:] + N_w[:-1]) * dh))


def _dz_trapz(p_hpa, T_k, rh_percent, h_m,
              k1=77.6890, k2=71.2952, k3=375463.0, eps=0.62198):
    """方法 C：与 B 完全相同的公式与常数，只把"下层值×层厚"换成梯形法。"""
    p = np.asarray(p_hpa, float); T = np.asarray(T_k, float)
    rh = np.asarray(rh_percent, float); h = np.asarray(h_m, float)
    o = np.argsort(h); p, T, rh, h = p[o], T[o], rh[o], h[o]
    e = rh / 100.0 * _sat_vapor_pressure_B(T)
    N_h = k1 * (p - e) / T
    N_w = (k2 - eps * k1) * (e / T) + k3 * (e / T ** 2)
    dh = np.diff(h)
    return (1e-6 * np.sum(0.5 * (N_h[1:] + N_h[:-1]) * dh),
            1e-6 * np.sum(0.5 * (N_w[1:] + N_w[:-1]) * dh))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default="configs")
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2025-10-01")
    ap.add_argument("--stride", type=int, default=10, help="每 N 个 6 小时时次取 1 个")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    args = ap.parse_args()

    import importlib
    import zarr
    cfg = importlib.import_module(args.configs)
    from common import CHANNELS, era5_channel_stats
    from main.utils.utils_data import decode_axis
    from ztd_operator import LEV_HPA, ztd_profile_surface

    mean, std = era5_channel_stats()
    T_CH = [CHANNELS.index(f"t{int(L)}") for L in LEV_HPA]
    R_CH = [CHANNELS.index(f"r{int(L)}") for L in LEV_HPA]
    Z_CH = [CHANNELS.index(f"z{int(L)}") for L in LEV_HPA]
    I_T2M, I_MSL = CHANNELS.index("t2m"), CHANNELS.index("msl")
    OP = T_CH + R_CH + [I_T2M, I_MSL] + Z_CH

    # 站点几何
    gm = pd.read_parquet(Path(cfg.ngl_zarr).parent /
                         "ngl_europe_0p25_80x120_station_grid_map.parquet")
    st = pd.read_parquet(Path(cfg.ngl_zarr).parent / "ngl_europe_stations.parquet")
    cells = gm[~gm["mask"].astype(bool) & gm["station_id"].notna()].reset_index(drop=True)
    # 只取高度；位置沿用格点映射里的格心坐标（与 utils.station_geometry 一致）
    cells = cells.merge(st[["gnss_station_id", "height_m"]],
                        left_on="station_id", right_on="gnss_station_id", how="left")
    lat_ax = np.asarray(cfg.lat, float)
    iy = np.searchsorted(lat_ax, cells["lat"].values)
    ix = np.searchsorted(np.asarray(cfg.lon, float), cells["lon"].values)
    h_st = cells["height_m"].values.astype(float)
    lat_st = cells["lat"].values.astype(float)
    n_cell = h_st.size

    ng = zarr.open(str(cfg.ngl_zarr), "r")
    ztd_obs = ng["ztd"]
    z_mu, z_sd = float(ng["ztd_train_mean"][0]), float(ng["ztd_train_std"][0])
    obs_t = decode_axis(cfg.ngl_zarr, "time")
    lab = zarr.open(str(cfg.label_zarr), "r")["label"]
    fx = zarr.open(str(cfg.fuxi_zarr), "r")
    lab_t = decode_axis(cfg.label_zarr, "time")
    init_t = decode_axis(cfg.fuxi_zarr, "init")
    lead = pd.Timedelta(hours=int(cfg.fcst_step) * 6)
    lb_pos = {t: i for i, t in enumerate(lab_t)}
    init_vals = init_t.values

    dates = {"train": cfg.dates_train_range, "val": cfg.dates_val_range,
             "test": cfg.dates_test_range}[args.split]
    times = [t for t in pd.date_range(pd.to_datetime(str(dates[0]), format="%Y%m%d%H"),
                                      pd.to_datetime(str(dates[1]), format="%Y%m%d%H"),
                                      freq="6h", inclusive="left")
             if (t >= pd.Timestamp(args.start)) and (t < pd.Timestamp(args.end))]
    times = times[::args.stride]
    print(f"[cfg] fcst_step={cfg.fcst_step} (lead {int(cfg.fcst_step)*6} h)；"
          f"抽样 {len(times)} 个时次，{n_cell} 个站")

    recs = []
    for t in times:
        lb_i = lb_pos.get(t)
        bg_i = int(np.searchsorted(init_vals, (t - lead).to_datetime64()))
        if lb_i is None or abs(init_vals[bg_i] - (t - lead).to_datetime64()) > np.timedelta64(0, "s"):
            continue
        fidx = int((t - obs_t[0]).total_seconds() // 300)
        obs = (np.asarray(ztd_obs[fidx, iy, ix], "float32") * z_sd + z_mu).astype(float)

        blks = {"era5": np.asarray(lab[lb_i], "float32"),
                "fuxi": np.asarray(fx["z"][bg_i, 0, :70], "float32")}
        for src, blk in blks.items():
            ph = blk[OP] * std[OP][:, None, None] + mean[OP][:, None, None]
            ph = ph[:, iy, ix]                                  # (41, n_cell)
            T_lev = ph[:13].T                                   # (n_cell, 13)
            R_lev = ph[13:26].T
            t2m = ph[26]
            msl = ph[27]
            H_lev = (ph[28:41] / G_0).T                          # 位势 -> 几何高度 [m]
            p_s = msl / 100.0 * np.exp(-G_0 * h_st / (R_D * t2m))

            # --- 方法 A：项目实现 ---
            a = ztd_profile_surface(T_lev, R_lev, t2m, p_s, h_st, lat_st)
            # --- 方法 B：外部实现（几何高度积分） ---
            zhdB = np.empty(n_cell); zwdB = np.empty(n_cell)
            zhdC = np.empty(n_cell); zwdC = np.empty(n_cell)
            zhdE = np.empty(n_cell); zwdE = np.empty(n_cell)
            for j in range(n_cell):
                up = LEV_HPA < p_s[j]                             # 地面之上的层
                rh_low = R_lev[j][up][-1] if up.any() else R_lev[j][-1]
                pp = np.concatenate([[p_s[j]], LEV_HPA[up]])
                TT = np.concatenate([[t2m[j]], T_lev[j][up]])
                RR = np.concatenate([[rh_low], R_lev[j][up]])
                HH = np.concatenate([[h_st[j]], H_lev[j][up]])
                zhdB[j], zwdB[j], _ = calc_zhd_zwd(pp, TT, RR, HH, H_s=h_st[j])
                zhdC[j], zwdC[j] = _dz_trapz(pp, TT, RR, HH)
                # --- 方法 E：地表节点改用 log-p 插值（T、e 都用夹住 p_s 的两层内插）---
                e_lev = RR[1:] / 100.0 * _sat_vapor_pressure_B(TT[1:])
                k2i = int(np.searchsorted(LEV_HPA, p_s[j]))
                lo = max(min(k2i - 1, LEV_HPA.size - 1), 0)
                hi = min(lo + 1, LEV_HPA.size - 1)
                l0, l1 = np.log(LEV_HPA[lo]), np.log(LEV_HPA[hi])
                w = (np.log(p_s[j]) - l0) / (l1 - l0) if l1 > l0 else 0.0
                T_s = T_lev[j][lo] + w * (T_lev[j][hi] - T_lev[j][lo])
                e_s_lv = (R_lev[j] / 100.0) * _sat_vapor_pressure_B(T_lev[j])
                e_s = e_s_lv[lo] + w * (e_s_lv[hi] - e_s_lv[lo])
                zhdE[j], zwdE[j] = _dz_trapz_e(
                    np.concatenate([[p_s[j]], pp[1:]]),
                    np.concatenate([[T_s], TT[1:]]),
                    np.concatenate([[e_s], e_lev]),
                    np.concatenate([[h_st[j]], HH[1:]]))
            recs.append(pd.DataFrame({
                "time": t, "src": src, "st": np.arange(n_cell), "obs": obs,
                "zhdA": a["ZHD_mm"], "zwdA": a["ZWD_mm"],
                "zhdB": zhdB * 1000.0, "zwdB": zwdB * 1000.0,
                "zhdC": zhdC * 1000.0, "zwdC": zwdC * 1000.0,
                "zhdD": zhdC * 1000.0 + TOP_CORR_MM, "zwdD": zwdC * 1000.0,
                "zhdE": zhdE * 1000.0 + TOP_CORR_MM, "zwdE": zwdE * 1000.0}))
    df = pd.concat(recs, ignore_index=True)
    df["ztdA"] = df.zhdA + df.zwdA
    df["ztdB"] = df.zhdB + df.zwdB
    df["ztdC"] = df.zhdC + df.zwdC
    df["ztdD"] = df.zhdD + df.zwdD
    df["ztdE"] = df.zhdE + df.zwdE

    def line(tag, a, b):
        m = np.isfinite(a) & np.isfinite(b)
        a, b = a[m], b[m]
        d = a - b
        print(f"{tag:>26}{d.size:>10d}{a.mean():>11.2f}{b.mean():>11.2f}"
              f"{d.mean():>9.2f}{np.sqrt((d**2).mean()):>9.2f}{np.abs(d).mean():>9.2f}"
              f"{np.corrcoef(a,b)[0,1]:>8.4f}")

    hdr = (f'{"口径":>26}{"n":>10}{"X均值":>11}{"obs均值":>11}'
           f'{"Bias":>9}{"RMSE":>9}{"MAE":>9}{"r":>8}')
    print("\n=== ZTD 对 NGL GNSS 实测（mm）===")
    print(hdr)
    for src in ("era5", "fuxi"):
        g = df[df.src == src]
        line(f"{src}: 方法A(ZHD解析+ZWD在p)", g.ztdA.values, g.obs.values)
        line(f"{src}: 方法B(在z积分,矩形)", g.ztdB.values, g.obs.values)
        line(f"{src}: 方法C(在z积分,梯形)", g.ztdC.values, g.obs.values)
        line(f"{src}: 方法D(C+顶层干柱)", g.ztdD.values, g.obs.values)
        line(f"{src}: 方法E(D+地表插值)", g.ztdE.values, g.obs.values)
    print("\n=== 距平（各站减去自身时间均值）===")
    print(hdr)
    for src in ("era5", "fuxi"):
        g = df[df.src == src]
        gg = g.groupby("st")
        o = (g.obs - gg.obs.transform("mean")).values
        line(f"{src}: 方法A", (g.ztdA - gg.ztdA.transform("mean")).values, o)
        line(f"{src}: 方法B", (g.ztdB - gg.ztdB.transform("mean")).values, o)
        line(f"{src}: 方法C", (g.ztdC - gg.ztdC.transform("mean")).values, o)
        line(f"{src}: 方法D", (g.ztdD - gg.ztdD.transform("mean")).values, o)
        line(f"{src}: 方法E", (g.ztdE - gg.ztdE.transform("mean")).values, o)

    print("\n=== 两方法的差别（B − A，mm）===")
    for src in ("era5", "fuxi"):
        g = df[df.src == src]
        for lab, ca, cb in (("ZHD", "zhdA", "zhdB"), ("ZWD", "zwdA", "zwdB"), ("ZTD", "ztdA", "ztdB"),
                            ("ZHD", "zhdA", "zhdC"), ("ZWD", "zwdA", "zwdC"), ("ZTD", "ztdA", "ztdC")):
            d = g[cb] - g[ca]
            print(f"  {src:>5} {lab}: 均值 {g[ca].mean():9.2f} -> {g[cb].mean():9.2f}"
                  f"  (差 {d.mean():+7.2f} ± {d.std():5.2f}, |差| 中位 {d.abs().median():5.2f})")
    print("\n=== 分量均值（mm）===")
    for src in ("era5", "fuxi"):
        g = df[df.src == src]
        print(f"  {src:>5}: ZHD  A {g.zhdA.mean():7.1f} B {g.zhdB.mean():7.1f} C {g.zhdC.mean():7.1f} "
              f"D {g.zhdD.mean():7.1f} E {g.zhdE.mean():7.1f}")
        print(f"  {'':>5}  ZWD  A {g.zwdA.mean():7.1f} B {g.zwdB.mean():7.1f} C {g.zwdC.mean():7.1f} "
              f"D {g.zwdD.mean():7.1f} E {g.zwdE.mean():7.1f}   obs ZTD {g.obs.mean():8.1f}   "
              f"顶层修正 {TOP_CORR_MM:.1f} mm")


if __name__ == "__main__":
    main()
