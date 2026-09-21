#!/usr/bin/env python3
"""用 NCEP 地面观测算 ZTD，与 NGL GNSS 实测 ZTD 逐站比较。

思路
----
``ztd_operator.ztd_surface`` 是经典的地面形式（Saastamoinen + Tetens），只需要
**地面气压 p、2 米气温 T、2 米相对湿度 RH、测站高度 h、纬度**：

    ZHD = 0.0022768 · p / (1 − 0.00266·cos2φ − 0.00028·h_km)
    ZWD = 0.002277 · (1255/T + 0.05) · e,      e = RH/100 · es(T)
    ZTD = ZHD + ZWD

而 ``obs-pkl_qc/convi/surf_ncep`` 恰好提供这四样（p / t2m / r2m / h）。于是可以
完全绕开模式场，用独立观测算出一个 ZTD，再和 NGL 的 GNSS ZTD 比 —— 这检验的是
**算子 + 地面观测**这一条链路，和 ``check_ztd_operator.py``（用 FuXi 廓线算）互补。

配对规则（按需求）
------------
* 对每个 NGL GNSS 站，取**最近的 NCEP 地面站**；两者球面距离 **> --max-km（默认 25 km）
  就丢弃该点**，不参与统计；
* 时间上取整点：NCEP 是逐小时文件，NGL 是 5 分钟，取同一时刻的 NGL 帧；
* 只用同时有 p（或 slp 可折算出 p）、t2m、r2m 的 NCEP 站。

两个口径都会给出：
  ``raw``      —— 用 NCEP 站自己的 p 和 h（两个站高度不同，ZHD 差异会直接进误差）
  ``zcorr``    —— 把 NCEP 的 p 按测高公式折到 GNSS 站高度再算 ZHD（只改静力项），
                  用来把"高度不匹配"从误差里剥离出来

注意：地表点观测 vs GNSS 柱积分本身含代表性误差，所以别把这里的 RMSE 当成算子精度上限；
与 ``check_ztd_operator.py`` 的 FuXi 版本（14.3 mm）对比时要记住输入完全不同。

用法（在 da_ngl/main_code 下）
----
    python ../test/check_ztd_from_surf_obs.py
    python ../test/check_ztd_from_surf_obs.py --start 2024-05-01 --end 2024-08-16 --hours 0,6,12,18
    python ../test/check_ztd_from_surf_obs.py --max-km 50 --out /tmp/ztd_surf.csv
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DA_ROOT = HERE.parent
sys.path.insert(0, str(DA_ROOT / "main_code"))
sys.path.insert(0, str(DA_ROOT / "preprocessing"))

DEFAULT_OBS_DIR = ("/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/database/"
                   "fuxi-obs/obs-pkl_qc/convi/surf_ncep")
R_EARTH = 6371.0088          # km
G_0, R_D = 9.80665, 287.05


def haversine_km(lat1, lon1, lat2, lon2):
    """逐点球面距离 [km]（度）。"""
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dp = p2 - p1
    dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default="configs")
    ap.add_argument("--obs-dir", default=DEFAULT_OBS_DIR)
    ap.add_argument("--start", default="2024-05-01")
    ap.add_argument("--end", default="2024-08-16", help="半开区间上界")
    ap.add_argument("--hours", default="0,6,12,18")
    ap.add_argument("--max-km", type=float, default=25.0,
                    help="GNSS 站与 NCEP 站的最大配对距离（默认 25 km）")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import importlib
    import zarr
    cfg = importlib.import_module(args.configs)
    from common import decode_time_axis
    from main.utils.utils import station_geometry
    from ztd_operator import ztd_surface

    # ---- NGL：GNSS 站坐标 + 实测 ZTD --------------------------------------
    iy, ix, h_gnss_cell, station_id = station_geometry(cfg)
    st = pd.read_parquet(Path(cfg.ngl_zarr).parent / "ngl_europe_stations.parquet")
    st["gnss_station_id"] = st["gnss_station_id"].astype(str)
    meta = st.set_index("gnss_station_id").reindex(station_id)
    lat_g, lon_g = meta["lat"].values.astype(float), meta["lon"].values.astype(float)
    h_g = meta["height_m"].values.astype(float)
    print(f"[gnss] {station_id.size} 站（NGL 网格 {iy.size} 格），"
          f"高度 {h_g.mean():.0f}±{h_g.std():.0f} m")

    ng = zarr.open(str(cfg.ngl_zarr), "r")
    ztd_obs = ng["ztd"]
    z_mean = float(ng["ztd_train_mean"][0]); z_std = float(ng["ztd_train_std"][0])
    obs_time = decode_time_axis(Path(cfg.ngl_zarr), "time")
    t0 = obs_time[0]
    step_min = int((obs_time[1] - obs_time[0]).total_seconds() // 60)
    print(f"[ngl] ZTD 时间轴 {obs_time[0]} ~ {obs_time[-1]}（{step_min} 分钟，"
          f"{obs_time.size} 帧）；μ={z_mean:.2f} σ={z_std:.2f} mm")

    hours = {int(v) for v in str(args.hours).split(",")}
    rows, n_file_used, n_file_missing, n_nopair = [], 0, 0, 0
    for t in pd.date_range(args.start, args.end, freq="1h", inclusive="left"):
        if t.hour not in hours:
            continue
        path = Path(args.obs_dir) / f"surf_{t:%Y%m%d%H}.pkl"
        if not path.exists():
            n_file_missing += 1
            continue
        fidx = int((t - t0).total_seconds() // (step_min * 60))
        if fidx < 0 or fidx >= obs_time.size:
            continue
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        n_file_used += 1

        lon_n = (d["lon"].values % 360.0 + 180.0) % 360.0 - 180.0   # 0..360 -> -180..180
        lat_n = d["lat"].values.astype(float)
        p_n = d["p"].values.astype(float)
        slp_n = d["slp"].values.astype(float)
        h_n = d["h"].values.astype(float) / G_0                     # 位势 -> 米
        t2_n = d["t2m"].values.astype(float)
        rh_n = d["r2m"].values.astype(float)
        # p 缺了就用地表气压按测高公式折回来
        need = ~np.isfinite(p_n) & np.isfinite(slp_n) & np.isfinite(h_n) & np.isfinite(t2_n)
        p_n[need] = slp_n[need] * np.exp(-G_0 * h_n[need] / (R_D * t2_n[need]))
        valid_n = (np.isfinite(p_n) & np.isfinite(t2_n) & np.isfinite(rh_n)
                   & np.isfinite(h_n) & np.isfinite(lat_n))
        if not valid_n.any():
            continue

        # ---- 最近邻配对（3D 弦长精确等价于球面距离） ----
        def xyz(lat, lon):
            la, lo = np.deg2rad(lat), np.deg2rad(lon)
            return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)
        Xn = xyz(lat_n[valid_n], lon_n[valid_n])
        Xg = xyz(lat_g, lon_g)
        # 逐 GNSS 站找最近的 NCEP 站：分块算距离（1378 × ~2000）
        best_d = np.full(lat_g.size, np.inf); best_j = np.full(lat_g.size, -1, int)
        idx_n = np.flatnonzero(valid_n)
        for s in range(0, lat_g.size, 256):
            e = min(s + 256, lat_g.size)
            chord = np.linalg.norm(Xg[s:e, None, :] - Xn[None, :, :], axis=-1)
            j = chord.argmin(axis=1)
            dkm = 2 * R_EARTH * np.arcsin(np.clip(chord[np.arange(e - s), j] / 2, 0, 1))
            best_d[s:e], best_j[s:e] = dkm, idx_n[j]
        keep = best_d <= args.max_km
        if not keep.any():
            n_nopair += 1
            continue
        j = best_j[keep]

        # ---- NGL 实测 ZTD [mm] ----
        z_gnss = (np.asarray(ztd_obs[fidx, iy[keep], ix[keep]], "float32") * z_std + z_mean).astype(float)
        # ---- 地面观测算 ZTD [mm] ----
        lat_u, h_u = lat_n[j], h_n[j]
        raw = ztd_surface(t2_n[j], p_n[j], rh_n[j], h_u, lat_u, temp_unit="K")["ZTD_mm"]
        # 把 p 折到 GNSS 站高度（只动静力项），剥离高度不匹配
        p_corr = p_n[j] * np.exp(-G_0 * (h_g[keep] - h_u) / (R_D * t2_n[j]))
        zcorr = ztd_surface(t2_n[j], p_corr, rh_n[j], h_g[keep], lat_g[keep],
                            temp_unit="K")["ZTD_mm"]
        m = np.isfinite(z_gnss) & np.isfinite(raw) & np.isfinite(zcorr)
        if not m.any():
            continue
        rows.append(pd.DataFrame({
            "time": t, "station": station_id[keep][m],
            "lat_gnss": lat_g[keep][m], "lon_gnss": lon_g[keep][m], "h_gnss": h_g[keep][m],
            "lat_ncep": lat_u[m], "lon_ncep": lon_n[j][m], "h_ncep": h_u[m],
            "dist_km": best_d[keep][m],
            "ztd_gnss": z_gnss[m], "ztd_surf_raw": raw[m], "ztd_surf_zcorr": zcorr[m],
            "p_ncep": p_n[j][m], "t2m_ncep": t2_n[j][m], "rh_ncep": rh_n[j][m],
        }))

    if not rows:
        raise SystemExit("没有任何配对 —— 检查 --max-km / 时间窗口 / 观测目录")
    df = pd.concat(rows, ignore_index=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)

    print(f"[run] 用时次 {n_file_used}（缺文件 {n_file_missing}，无配对 {n_nopair}），"
          f"配对记录 {len(df)}，涉及 GNSS 站 {df['station'].nunique()}")
    print(f"[pair] 距离 km：mean {df['dist_km'].mean():.2f}  中位 {df['dist_km'].median():.2f}  "
          f"p90 {df['dist_km'].quantile(.9):.2f}  最大 {df['dist_km'].max():.2f}（阈值 {args.max_km}）")
    dh = df["h_ncep"] - df["h_gnss"]
    print(f"[pair] 高度差（NCEP−GNSS）m：mean {dh.mean():+.1f}  中位 {dh.median():+.1f}  "
          f"|差| 的 p90 {dh.abs().quantile(.9):.0f}  |差|>100 m 占 {100*(dh.abs()>100).mean():.1f}%")

    def stats(a, b):
        d = a - b
        return (int(d.size), a.mean(), b.mean(), d.mean(),
                np.sqrt((d ** 2).mean()), np.abs(d).mean(),
                np.corrcoef(a, b)[0, 1] if d.size > 2 else np.nan)

    print(f"\n=== ZTD（mm）：地面观测算子 − NGL GNSS 实测 ===")
    print(f'{"口径":>10}{"n":>9}{"算子均值":>11}{"GNSS均值":>11}{"Bias":>9}{"RMSE":>9}{"MAE":>9}{"r":>8}')
    for tag, col in (("raw", "ztd_surf_raw"), ("zcorr", "ztd_surf_zcorr")):
        n, mu_a, mu_b, bias, rmse, mae, r = stats(df[col].values, df["ztd_gnss"].values)
        print(f"{tag:>10}{n:>9d}{mu_a:>11.2f}{mu_b:>11.2f}{bias:>9.2f}{rmse:>9.2f}{mae:>9.2f}{r:>8.4f}")

    # 去站点静态偏差（各站减去自己的时间均值）后看距平
    print(f"\n=== 距平（各站减去自身时间均值）===")
    print(f'{"口径":>10}{"n":>9}{"算子距平std":>13}{"GNSS距平std":>13}{"Bias":>9}{"RMSE":>9}{"r":>8}')
    for tag, col in (("raw", "ztd_surf_raw"), ("zcorr", "ztd_surf_zcorr")):
        g = df.groupby("station")
        a = (df[col] - g[col].transform("mean")).values
        b = (df["ztd_gnss"] - g["ztd_gnss"].transform("mean")).values
        d = a - b
        print(f"{tag:>10}{d.size:>9d}{a.std():>13.2f}{b.std():>13.2f}"
              f"{d.mean():>9.2f}{np.sqrt((d**2).mean()):>9.2f}{np.corrcoef(a,b)[0,1]:>8.4f}")
    if args.out:
        print(f"\n逐条记录 -> {args.out}")


if __name__ == "__main__":
    main()
