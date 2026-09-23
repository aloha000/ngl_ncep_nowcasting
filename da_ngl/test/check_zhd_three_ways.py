#!/usr/bin/env python3
"""判定 ZHD 偏差的来源：观测气压 / 我们的算子 / NGL 反推，三者互比。

三个 ZHD 估计
-------------
1. ``ZHD_surf`` —— **独立观测**：用地面站实测气压 ``p``（surf_ncep），按测高公式折到
   GNSS 站高后套 Saastamoinen：
       p_gnss = p_obs · exp( −g·(h_gnss − h_obs) / (R_d·T_obs) )
       ZHD    = 0.0022768 · p_gnss / (1 − 0.00266·cos2φ − 0.00028·h_km)
   只用地面站自己的 p/T/高度，不碰任何模式的廓线。
2. ``ZHD_op`` —— 我们的方法 E 算子（FuXi 库里已存好；ERA5 现算）
3. ``ZHD_gnss`` —— NGL 反推：``ZTD_obs − TRWET_obs``（含 VMF1/NWM 的静力项）

配对规则与 ``check_ztd_from_surf_obs.py`` 一致：每个 GNSS 站取最近的 NCEP 地面站，
球面距离 > ``--max-km``（默认 25 km）则丢弃。

判读
----
* ``ZHD_surf ≈ ZHD_op`` 且 ``ZHD_gnss`` 偏离 → **NGL 的 VMF1 静力项在偏**，我们的算子没问题
* ``ZHD_surf ≈ ZHD_gnss`` 且 ``ZHD_op`` 偏离 → **是我们的算子在偏**，需要回头查积分

用法（在 da_ngl/main_code 下）
----
    python ../test/check_zhd_three_ways.py \
        --obs-dir /cpfs01/.../gdas_process/prod_output/qc/surf_ncep/2025 \
        --start 2025-01-01 --end 2025-10-01 --stride 10
"""

from __future__ import annotations

import argparse
import pickle
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

R_EARTH = 6371.0088      # km
G_0, R_D, ZHD_COEF = 9.80665, 287.05, 0.0022768


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default="configs")
    ap.add_argument("--obs-dir", required=True)
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2025-10-01")
    ap.add_argument("--stride", type=int, default=10, help="每 N 个 6 小时时刻取 1 个")
    ap.add_argument("--max-km", type=float, default=25.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import importlib
    cfg = importlib.import_module(args.configs)
    from common import CHANNELS, decode_time_axis, era5_channel_stats
    from main.utils import station_geometry
    from ztd_operator import LEV_HPA, ztd_profile_zdz

    iy_g, ix_g, h_g, station_id = station_geometry(cfg)
    lat_ax = np.asarray(cfg.lat, float)
    LON_ax = np.asarray(cfg.lon, float)
    lat_g = lat_ax[iy_g]
    # GNSS 站的真实经纬度（格心坐标用于落格，这里用真实坐标做距离配对）
    st = pd.read_parquet(Path(cfg.ngl_zarr).parent / "ngl_europe_stations.parquet")
    st["gnss_station_id"] = st["gnss_station_id"].astype(str)
    meta = st.set_index("gnss_station_id").reindex(station_id)
    lat_true = meta["lat"].values.astype(float)
    lon_true = meta["lon"].values.astype(float)

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

    times = [t for t in pd.date_range(args.start, args.end, freq="6h", inclusive="left")][::args.stride]
    print(f"[win] {times[0].date()} ~ {times[-1].date()}：{len(times)} 个时刻，{iy_g.size} 个 GNSS 站")

    def xyz(lat, lon):
        la, lo = np.deg2rad(lat), np.deg2rad(lon)
        return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)

    recs, n_nopair = [], 0
    for k, T in enumerate(times):
        path = Path(args.obs_dir) / f"surf_{T:%Y%m%d%H}.pkl"
        if not path.exists():
            continue
        i = int(np.flatnonzero(lab_t == T)[0])
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        lon_n = (d["lon"].values % 360.0 + 180.0) % 360.0 - 180.0
        lat_n = d["lat"].values.astype(float)
        p_n = d["p"].values.astype(float)
        slp_n = d["slp"].values.astype(float)
        h_n = d["h"].values.astype(float) / G_0
        t_n = d["t2m"].values.astype(float)
        need = ~np.isfinite(p_n) & np.isfinite(slp_n) & np.isfinite(h_n) & np.isfinite(t_n)
        p_n[need] = slp_n[need] * np.exp(-G_0 * h_n[need] / (R_D * t_n[need]))
        ok_n = np.isfinite(p_n) & np.isfinite(t_n) & np.isfinite(h_n)
        if not ok_n.any():
            continue
        idx_n = np.flatnonzero(ok_n)
        Xn, Xg = xyz(lat_n[ok_n], lon_n[ok_n]), xyz(lat_true, lon_true)
        best = np.full(lat_true.size, np.inf); bj = np.full(lat_true.size, -1, int)
        for s in range(0, lat_true.size, 256):
            e = min(s + 256, lat_true.size)
            c = np.linalg.norm(Xg[s:e, None, :] - Xn[None, :, :], axis=-1)
            j = c.argmin(axis=1)
            dkm = 2 * R_EARTH * np.arcsin(np.clip(c[np.arange(e - s), j] / 2, 0, 1))
            best[s:e], bj[s:e] = dkm, idx_n[j]
        keep = best <= args.max_km
        if not keep.any():
            n_nopair += 1
            continue
        j = bj[keep]
        hh, ll = h_g[keep], lat_g[keep]
        p_fold = p_n[j] * np.exp(-G_0 * (hh - h_n[j]) / (R_D * t_n[j]))
        # ZHD_COEF 配 hPa 得到米，×1000 转毫米
        zhd_surf = 1000.0 * ZHD_COEF * p_fold / (1.0 - 0.00266 * np.cos(2 * np.deg2rad(ll))
                                                 - 0.00028 * hh / 1000.0)
        fidx = int((T - obs_t[0]).total_seconds() // 300)
        zhd_gnss = (np.asarray(ng["ztd"][fidx, iy_g, ix_g], "float32")[keep] * z_sd + z_mu
                    - (np.asarray(ng["zwd"][fidx, iy_g, ix_g], "float32")[keep] * w_sd + w_mu))
        zhd_fx = np.asarray(fz["zhd"][i], "float32")[iy_g, ix_g][keep]
        blk = np.asarray(lab[i, :70], "float32")
        ph = blk[OP] * std[OP][:, None, None] + mean[OP][:, None, None]
        ph = ph[:, iy_g, ix_g]
        T_lev, R_lev = ph[:13].T, ph[13:26].T
        t2m, msl = ph[26], ph[27]
        H_lev = (ph[28:41] / G_0).T
        p_s = msl / 100.0 * np.exp(-G_0 * h_g / (R_D * t2m))
        zhd_e5 = ztd_profile_zdz(T_lev, R_lev, t2m, p_s, h_g, H_lev)["ZHD_mm"][keep]
        denom = (1.0 - 0.00266 * np.cos(2 * np.deg2rad(ll)) - 0.00028 * hh / 1000.0)
        recs.append(pd.DataFrame({
            "time": T, "st": np.arange(iy_g.size)[keep], "dist_km": best[keep],
            "h_gnss": hh, "h_ncep": h_n[j], "dh": h_n[j] - hh,
            "p_obs": p_n[j], "p_fold": p_fold, "p_s_era5": p_s[keep],
            "zhd_surf": zhd_surf, "zhd_op_fuxi": zhd_fx, "zhd_op_era5": zhd_e5,
            "zhd_saas_era5": 1000.0 * ZHD_COEF * p_s[keep] / denom,
            "zhd_gnss": zhd_gnss}))
        if (k + 1) % 40 == 0:
            print(f"  {k+1}/{len(times)}", flush=True)
    df = pd.concat(recs, ignore_index=True)
    if args.out:
        df.to_csv(args.out, index=False)

    print(f"\n配对记录 {len(df)}（{df['time'].nunique()} 个时刻，"
          f"{df['st'].nunique()} 个 GNSS 站；无配对时刻 {n_nopair}）")
    print(f"距离 km：均值 {df.dist_km.mean():.2f}  中位 {df.dist_km.median():.2f}  "
          f"最大 {df.dist_km.max():.2f}")
    print(f"高度差(NCEP−GNSS) m：均值 {df.dh.mean():+.1f}  中位 {df.dh.median():+.1f}  "
          f"|差|>100m 占 {100*(df.dh.abs()>100).mean():.1f}%  "
          f"（折算到气压 ≈ {abs(df.dh.mean())*0.012:.2f} hPa 的系统项，已用测高公式扣掉）")
    print(f"\n=== 三个 ZHD 估计的均值（mm）===")
    for tag, c in (("ZHD_saas(ERA5 p_s，同 NCEP 方法)", "zhd_saas_era5"),
                   ("ZHD_surf（观测气压，同 NCEP 方法）", "zhd_surf"),
                   ("ZHD_op（FuXi，我们的算子）", "zhd_op_fuxi"),
                   ("ZHD_op（ERA5，我们的算子）", "zhd_op_era5"),
                   ("ZHD_gnss（NGL 反推：ZTD−TRWET）", "zhd_gnss")):
        v = df[c].values
        print(f"  {tag:<34} {np.nanmean(v):9.2f}  (std {np.nanstd(v):5.2f})")

    print(f'\n{"对比":>34}{"n":>9}{"Bias":>9}{"RMSE":>9}{"MAE":>9}{"r":>9}')
    pairs = (("同方法: ZHD_saas(ERA5) − ZHD_surf", "zhd_saas_era5", "zhd_surf"),
             ("同方法: ZHD_saas(ERA5) − ZHD_gnss", "zhd_saas_era5", "zhd_gnss"),
             ("方法差: ZHD_op(ERA5) − ZHD_saas(ERA5)", "zhd_op_era5", "zhd_saas_era5"),
             ("ZHD_op(FuXi) − ZHD_surf", "zhd_op_fuxi", "zhd_surf"),
             ("ZHD_op(ERA5) − ZHD_surf", "zhd_op_era5", "zhd_surf"),
             ("ZHD_gnss − ZHD_surf", "zhd_gnss", "zhd_surf"),
             ("ZHD_gnss − ZHD_op(ERA5)", "zhd_gnss", "zhd_op_era5"))
    for tag, ca, cb in pairs:
        a, b = df[ca].values, df[cb].values
        m = np.isfinite(a) & np.isfinite(b)
        d = a[m] - b[m]
        print(f"{tag:>34}{m.sum():>9d}{d.mean():>9.2f}{np.sqrt((d**2).mean()):>9.2f}"
              f"{np.abs(d).mean():>9.2f}{np.corrcoef(a[m], b[m])[0,1]:>9.4f}")
    print("\n判读：若 ZHD_op ≈ ZHD_surf 而 ZHD_gnss 偏离 → 是 NGL 的 VMF1 静力项在偏；"
          "\n      若 ZHD_gnss ≈ ZHD_surf 而 ZHD_op 偏离 → 是我们的算子在偏。")
    if args.out:
        print(f"逐条记录 -> {args.out}")


if __name__ == "__main__":
    main()
