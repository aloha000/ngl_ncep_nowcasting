#!/usr/bin/env python3
"""用 ERA5 官方 37 层资料 + 方法 E 重算区域 ZTD，样本与上一个结果对齐。

为什么要重算
------------
项目里的 ERA5 store 只有 **13 个气压层**（50~1000 hPa），方法 E 的柱积分就搭在这 13 层的
几何高度上，顶层只到 50 hPa，剩下的用解析顶盖修正补（113.7 mm）。官方下载的 ERA5 有
**37 层**（1~1000 hPa），还带真实表面气压，可以把整个柱积分做得更细。

输入（官方 1h 资料，NetCDF4/HDF5，需要 xarray + netCDF4）
--------------------------------------------------------
``ERA5_ROOT/pressure_level/geopotential/YYYY/YYYYMMDD.nc``   ``z``   37 层位势 [m^2/s^2]
``ERA5_ROOT/pressure_level/temperature/...``                 ``t``   37 层温度 [K]
``ERA5_ROOT/pressure_level/relative_humidity/...``           ``r``   37 层相对湿度 [%]
``ERA5_ROOT/surface_level/surface_pressure/...``             ``sp``  表面气压 [Pa]
``ERA5_ROOT/surface_level/2m_temperature/...``               ``t2m`` 2 m 温度 [K]

计算方案（= 现在的"方法 E"，只把层数从 13 换成 37）
---------------------------------------------------
1. 层高：``z / g`` 得到 37 层的几何高度；层序按气压**升序**（1 hPa 在最前）传给算子，
   与 ``LEV_HPA`` 的通道顺序一致；
2. 地面气压：``sp`` 定义在 ERA5 模式地形高度 ``h_oro`` 上。``h_oro`` 由**地面上方最近的
   真实层**（满足 ``p_j < p_s`` 的最大气压层）按测高公式反推

       h_oro = z(p1) + (R_d*T(p1)/g)*ln(p1/p_s)

   再把气压折到参考高度 ``h_ref``：``p(h_ref) = sp*exp(-g*(h_ref-h_oro)/(R_d*t2m))``。
   站格取 ``h_ref`` = 测站高度（NGL 元数据），其余格点取 ``h_ref = h_oro``（直接用 ``sp``）；
3. 柱积分：调 ``preprocessing/ztd_operator.py`` 的 ``ztd_profile_zdz``（梯形法积分
   ``N_h = k1(p-e)/T``、``N_w = (k2-eps*k1)e/T + k3*e/T^2``，``p >= p_s`` 的层塌到地面
   → 层厚 0），顶盖修正用 ``p_top = 1 hPa``（~2.27 mm；13 层方案是 50 hPa → 113.7 mm）。
   同时算一版只用 13 层的，把"层数"这一项单独剥出来。

用法（**hydro** 环境，见 test/README_era5_37lev.md）
--------------------------------------------------
    python ../test/dump_station_geom.py --out /tmp/geom_1378.npz     # gnss 环境先跑一次
    python ../test/era5_37lev_ztd.py \
        --samples /tmp/zhd_three_ways_v2.csv \
        --geom /tmp/geom_1378.npz \
        --out-csv /tmp/era5_37lev_ztd.csv \
        --out-field /tmp/era5_37lev_ztd_field.npz \
        --workers 16
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
sys.path.insert(0, str(DA_ROOT / "preprocessing"))

from ztd_operator import ztd_profile_zdz  # noqa: E402

ERA5_ROOT = Path("/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/"
                 "database/era5/from_official/1h")

G_0 = 9.80665        # m/s^2
R_D = 287.05         # J/(kg K)
K1_BEVIS = 77.6890   # K/hPa，用于顶盖修正的说明

# 官方 37 个气压层（降序），以及给算子用的升序版本
LEV_DESC = np.array([1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700,
                     650, 600, 550, 500, 450, 400, 350, 300, 250, 225, 200, 175,
                     150, 125, 100, 70, 50, 30, 20, 10, 7, 5, 3, 2, 1], dtype=np.float64)
LEV_ASC = LEV_DESC[::-1]                       # 1 ... 1000 hPa
LEV13 = np.array([50, 100, 150, 200, 250, 300, 400, 500,
                  600, 700, 850, 925, 1000], dtype=np.float64)
IDX13 = np.searchsorted(LEV_ASC, LEV13)        # 13 层在 37 层中的列号

SOURCES = {
    "z": "pressure_level/geopotential",
    "t": "pressure_level/temperature",
    "r": "pressure_level/relative_humidity",
    "sp": "surface_level/surface_pressure",
    "t2m": "surface_level/2m_temperature",
}

_G: dict = {}


def _init(geom: dict) -> None:
    """把 80x120 的网格索引变成"排序去重 + 取回"，每次只读 nc 的一个小 slab。"""
    global _G
    _G = geom
    for ax in ("lat", "lon"):
        idx = geom[f"{ax}_idx"]
        sel = np.sort(np.unique(idx))
        _G[f"{ax}_sel"] = sel
        _G[f"{ax}_inv"] = np.searchsorted(sel, idx)


def _open_day(day: str):
    """打开一天的 5 个文件；同时算出"文件层序 -> 气压升序"的排列。

    ERA5 归档的 pressure_level 是**气压降序**（1000, 975, ..., 1 hPa），而算子要求
    第一层是最顶层（1 hPa），所以必须显式按坐标排序，不能假定文件顺序。
    """
    import xarray as xr

    handles, data = [], {}
    for key, rel in SOURCES.items():
        f = ERA5_ROOT / rel / day[:4] / f"{day}.nc"
        ds = xr.open_dataset(f)
        handles.append(ds)
        da = ds[list(ds.data_vars)[0]]
        perm = None
        if "pressure_level" in da.dims:
            lev = np.asarray(ds["pressure_level"].values, dtype=np.float64)
            perm = np.argsort(lev)                       # 升序
            if lev.size != LEV_ASC.size or not np.allclose(lev[perm], LEV_ASC):
                raise ValueError(f"{f}: 层集合与预期 37 层不符：{lev}")
        elif key in ("sp", "t2m") and da.dims[-2:] != ("latitude", "longitude"):
            raise ValueError(f"{f}: 维度顺序非预期 {da.dims}")
        data[key] = (da, perm)
    return handles, data


def _extract(item, ti: int) -> np.ndarray:
    """取某时刻的 (..., 80, 120) 子区域，按统一网格和气压升序排好。"""
    da, perm = item
    keep = {"valid_time": ti, "latitude": _G["lat_sel"], "longitude": _G["lon_sel"]}
    a = da.isel(**keep).values
    if perm is not None:
        a = np.take(a, perm, axis=0)                     # 层 -> 气压升序
    a = np.take(a, _G["lat_inv"], axis=-2)
    a = np.take(a, _G["lon_inv"], axis=-1)
    return np.asarray(a, dtype=np.float64)


def _surface_geometry(p_s_hpa: np.ndarray, z_lev: np.ndarray, t_lev: np.ndarray) -> np.ndarray:
    """由地面上方最近的真实层反推 ERA5 模式地形高度 h_oro [m]。

    气压随高度递减，``p < p_s`` 的层在地面之上；在升序的 ``LEV_ASC`` 里，
    其中最靠下的一层就是满足条件、列号最大的那个。
    """
    above = LEV_ASC[:, None, None] < p_s_hpa[None]
    col = np.where(above, np.arange(LEV_ASC.size)[:, None, None], -1).max(axis=0)
    valid = col >= 0
    col = np.where(valid, col, 0)
    lev = np.broadcast_to(LEV_ASC[:, None, None], z_lev.shape)
    p1 = np.take_along_axis(lev, col[None], axis=0)[0]
    z1 = np.take_along_axis(z_lev, col[None], axis=0)[0]
    t1 = np.take_along_axis(t_lev, col[None], axis=0)[0]
    h_oro = z1 + (R_D * t1 / G_0) * np.log(p1 / p_s_hpa)
    return np.where(valid, h_oro, np.nan)


def _ztd(prof: dict, h_ref: np.ndarray, lev, idx=None) -> dict:
    """廓线（层轴在 0，形如 (37, 80, 120) 或 (37, N)）-> ZHD/ZWD/ZTD [mm]。

    ``idx`` 给定则只挑出 13 层那一版。
    """
    if idx is None:
        t, r, z, lev_use = prof["t"], prof["r"], prof["z"], lev
    else:
        t, r, z, lev_use = prof["t"][idx], prof["r"][idx], prof["z"][idx], LEV13
    n_lev = int(t.shape[0])
    # 算子要求层轴在最后，且所有数组的前导维要一致 -> 一律摊平成 (N, n_lev)。
    # 注意用 moveaxis 而不是 .T：.T 对 3 维数组会把三个轴全反过来。
    out = ztd_profile_zdz(
        np.asarray(np.moveaxis(t, 0, -1), "f8").reshape(-1, n_lev),
        np.asarray(np.moveaxis(r, 0, -1), "f8").reshape(-1, n_lev),
        np.asarray(prof["t2m"], "f8").reshape(-1),
        np.asarray(prof["p"], "f8").reshape(-1),
        np.asarray(h_ref, "f8").reshape(-1),
        np.asarray(np.moveaxis(z, 0, -1), "f8").reshape(-1, n_lev),
        lev_hpa=lev_use)
    shape = np.asarray(h_ref).shape
    return {k: np.asarray(out[k], "f8").reshape(shape)
            for k in ("ZHD_mm", "ZWD_mm", "ZTD_mm")}


def _worker(task):
    day, times = task
    handles, data = _open_day(day)
    recs = []
    try:
        for ti, iso in times:
            z_lev = _extract(data["z"], ti) / G_0            # (37,80,120) 几何高度
            t_lev = _extract(data["t"], ti)
            r_lev = _extract(data["r"], ti)
            sp = _extract(data["sp"], ti) / 100.0            # Pa -> hPa
            t2m = _extract(data["t2m"], ti)

            h_oro = _surface_geometry(sp, z_lev, t_lev)
            h_oro_safe = np.where(np.isfinite(h_oro), h_oro, 0.0)

            # (a) 区域场：参考高度 = 模式地形高度，即直接用 sp
            prof_f = {"t": t_lev, "r": r_lev, "z": z_lev, "t2m": t2m, "p": sp}
            field = _ztd(prof_f, h_oro_safe, LEV_ASC)

            # (b) 站格：参考高度 = 测站高度
            iy, ix = _G["iy"], _G["ix"]
            h_st = _G["h"]
            p_st = sp[iy, ix] * np.exp(
                -G_0 * (h_st - h_oro[iy, ix]) / (R_D * t2m[iy, ix]))
            prof_s = {"t": t_lev[:, iy, ix], "r": r_lev[:, iy, ix],
                      "z": z_lev[:, iy, ix], "t2m": t2m[iy, ix], "p": p_st}
            recs.append({
                "time": iso,
                "field": {k: np.asarray(field[k], "f4") for k in field},
                "st37": _ztd(prof_s, h_st, LEV_ASC),
                "st13": _ztd(prof_s, h_st, None, idx=IDX13),
                "h_oro": h_oro[iy, ix].astype("f4"),
                "p_sp": sp[iy, ix].astype("f4"),
                "p_st": p_st.astype("f4"),
                "t2m": t2m[iy, ix].astype("f4"),
            })
    finally:
        for ds in handles:
            ds.close()
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=Path, default=Path("/tmp/zhd_three_ways_v2.csv"),
                    help="上一个结果，用来对齐 (time, station) 样本")
    ap.add_argument("--geom", type=Path, default=Path("/tmp/geom_1378.npz"))
    ap.add_argument("--out-csv", type=Path, default=Path("/tmp/era5_37lev_ztd.csv"))
    ap.add_argument("--out-field", type=Path, default=Path("/tmp/era5_37lev_ztd_field.npz"))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    if not args.geom.exists():
        raise SystemExit(f"缺 {args.geom}；先在 gnss 环境跑 test/dump_station_geom.py")

    g = np.load(args.geom, allow_pickle=True)
    geom = {k: g[k] for k in ("iy", "ix", "h", "station_id", "lat_ax", "lon_ax")}
    # 统一网格 -> ERA5 全球 0.25 度网格索引（lat 从 90 往南，lon 0~359.75）
    geom["lat_idx"] = np.rint((90.0 - geom["lat_ax"]) / 0.25).astype(np.int64)
    geom["lon_idx"] = np.rint((geom["lon_ax"] % 360.0) / 0.25).astype(np.int64)
    print(f"[grid] lat {geom['lat_ax'][0]:.2f}..{geom['lat_ax'][-1]:.2f} -> "
          f"{geom['lat_idx'].min()}..{geom['lat_idx'].max()} | lon "
          f"{geom['lon_ax'][0]:.2f}..{geom['lon_ax'][-1]:.2f} -> "
          f"{geom['lon_idx'].min()}..{geom['lon_idx'].max()}")

    times = pd.to_datetime(pd.read_csv(args.samples, usecols=["time"])["time"])
    times = times.drop_duplicates().sort_values()
    print(f"[samples] {len(times)} 个时刻（{times.min()} ~ {times.max()}）")

    tasks = []
    for day, grp in times.groupby(times.dt.strftime("%Y%m%d")):
        tasks.append((day, [(int(iso.hour), iso.strftime("%Y-%m-%dT%H:%M:%S"))
                            for iso in grp]))

    import multiprocessing as mp

    results = []
    with mp.Pool(args.workers, initializer=_init, initargs=(geom,)) as pool:
        for i, recs in enumerate(pool.imap_unordered(_worker, tasks), 1):
            results.extend(recs)
            if i % 10 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} 天", flush=True)
    results.sort(key=lambda r: r["time"])

    keys = ["ZHD_mm", "ZWD_mm", "ZTD_mm"]
    rows = []
    for r in results:
        d = {"time": r["time"], "h_oro": r["h_oro"], "p_sp": r["p_sp"],
             "p_st": r["p_st"], "t2m": r["t2m"]}
        for k in keys:
            short = k.split("_")[0].lower()
            d[f"{short}37"] = r["st37"][k]
            d[f"{short}13"] = r["st13"][k]
        rows.append(pd.DataFrame(d))
    st = pd.concat(rows, ignore_index=True)
    st.insert(1, "st", np.tile(np.arange(geom["iy"].size), len(results)))
    st.insert(2, "station_id", np.tile(geom["station_id"], len(results)))
    st.to_csv(args.out_csv, index=False)
    print(f"[out] {args.out_csv}  ({len(st)} 行)")

    np.savez_compressed(
        args.out_field,
        time=np.asarray([r["time"] for r in results]),
        lat=geom["lat_ax"], lon=geom["lon_ax"],
        **{k.split("_")[0].lower(): np.stack([r["field"][k] for r in results]) for k in keys},
    )
    print(f"[out] {args.out_field}  ({len(results)} 时刻 x 80 x 120)")

    print("\n=== 层数对顶盖修正的影响（说明用） ===")
    for tag, p_top in (("37 层", LEV_ASC[0]), ("13 层", LEV13[0])):
        print(f"  {tag}: p_top = {p_top:.0f} hPa -> 解析顶盖干柱 "
              f"{1e-6 * K1_BEVIS * (R_D / G_0) * p_top * 1000:.2f} mm")
    print(f"\n测站高度 {geom['h'].min():.0f}~{geom['h'].max():.0f} m；"
          f"ERA5 模式地形高度均值 {np.nanmean([np.nanmean(r['h_oro']) for r in results]):.0f} m")
    print(f'{"量":>4}{"37层均值":>12}{"13层均值":>12}{"37-13":>10}{"std":>9}{"max|d|":>9}')
    for tag, c in (("ZHD", "zhd"), ("ZWD", "zwd"), ("ZTD", "ztd")):
        a, b = st[f"{c}37"].to_numpy(), st[f"{c}13"].to_numpy()
        d = a - b
        print(f"{tag:>4}{np.nanmean(a):>12.2f}{np.nanmean(b):>12.2f}"
              f"{np.nanmean(d):>+10.2f}{np.nanstd(d):>9.2f}{np.nanmax(np.abs(d)):>9.1f}")


if __name__ == "__main__":
    main()
