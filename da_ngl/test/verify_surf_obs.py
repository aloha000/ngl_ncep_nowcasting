#!/usr/bin/env python3
"""地表观测独立检验：surf_ncep 地面站 vs ERA5 标签 / FuXi 24h 背景。

用途
----
拿一份**独立于 ERA5** 的地面观测，去检验同化系统里两样东西的真实水平：
  * ERA5 标签（label store，同化目标）
  * FuXi 24h 背景（fuxi store，init = T − fcst_step×6h）
并给出逐变量、逐月的 MAE / Bias / RMSE。

数据
----
观测目录 ``obs-pkl_qc/convi/surf_ncep`` 是**逐小时**的全球地面站 DataFrame
（每时刻一个 pkl，约 1.6 万行），列：

    lat, lon        经纬度（lon 是 0..360）
    p               本站气压 [hPa]
    slp             海平面气压 [hPa]（约六成站点有）
    h               位势 [m²/s²]，z = h / 9.80665 得米
    t2m             2 米气温 [K]
    r2m             2 米相对湿度 [%]
    u10, v10        10 米风 [m/s]

⚠️ **时间覆盖只有 2016-01-01 ~ 2024-08-15**，而本项目的 test split 是
2025-01-01 ~ 2025-10-01 —— 测试集时段没有任何观测可对。所以本脚本实际能检验的是
"有观测的重叠区间"，默认取最接近测试集的 val 期后段（2024-05-01 ~ 2024-08-15）。
需要覆盖更长的窗口可以改用同级的 ``convi/surf``（到 2024-09-21）。

对齐约定（与建库一致）
------
* 只用 00/06/12/18 UTC 四个时次 —— ERA5/FuXi 都是 6 小时间隔；
* ERA5 取 label store 的 T 时刻，FuXi 取 fuxi store 的 ``init = T − fcst_step×6h``；
* 观测站点按**最近邻**落到 80×120 的 0.25° 网格（和 ``common.Region`` 的采样方式相同）；
* 模型侧先反标准化回物理量（``x = x_std·std_era5 + mean_era5``），msl 再 Pa→hPa；
* 轻量合理性过滤（t2m 200~340 K、slp 850~1100 hPa、|风| < 60 m/s），可用
  ``--no-filter`` 关掉。

注意：观测是**站点**、模式是**0.25° 格点平均**，两者的差异里含代表性误差，
所以这里的 MAE 会高于"模式对模式"的误差，适合横向比较 ERA5 / FuXi，不宜当成
模式的绝对误差。

用法
----
    cd da_ngl/main_code
    python ../test/verify_surf_obs.py                                   # 默认窗口
    python ../test/verify_surf_obs.py --start 2023-08-16 --end 2024-08-16
    python ../test/verify_surf_obs.py --split val --out /tmp/surf_val.csv
    python ../test/verify_surf_obs.py --obs-dir <convi/surf>            # 换一份观测源
"""

from __future__ import annotations

import argparse
import importlib
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)

HERE = Path(__file__).resolve().parent
DA_ROOT = HERE.parent
sys.path.insert(0, str(DA_ROOT / "main_code"))
sys.path.insert(0, str(DA_ROOT / "preprocessing"))

DEFAULT_OBS_DIR = ("/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/database/"
                   "fuxi-obs/obs-pkl_qc/convi/surf_ncep")
# 观测源的实际覆盖（用于把请求窗口夹到可用区间）
OBS_COVER = {"surf_ncep": ("2016-01-01", "2024-08-15"),
             "surf": ("2016-01-01", "2024-09-21")}
# 要检验的变量：观测列 -> (store 通道名, 单位换算)
# 要检验的变量：观测列 -> (store 通道名, 模型侧还要除以的比例)
#   msl 通道存的是 Pa，观测 slp 是 hPa，所以模型值要 /100
VARS = (("t2m", "t2m", 1.0), ("slp", "msl", 100.0),
        ("u10", "u10", 1.0), ("v10", "v10", 1.0))


def _coerce(value):
    low = str(value).lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def load_cfg(configs, overrides):
    cfg = importlib.import_module(configs)
    for item in overrides or []:
        key, _, val = str(item).partition("=")
        setattr(cfg, key.strip(), _coerce(val.strip()))
    return cfg


def resolve_window(cfg, args):
    """请求窗口 -> (start, end, 说明)。会夹到观测覆盖区间内。"""
    if args.start and args.end:
        start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
        note = "命令行指定"
    elif args.split in ("train", "val", "test"):
        rng = {"train": cfg.dates_train_range, "val": cfg.dates_val_range,
               "test": cfg.dates_test_range}[args.split]
        start = pd.to_datetime(str(rng[0]), format="%Y%m%d%H")
        end = pd.to_datetime(str(rng[1]), format="%Y%m%d%H")
        note = f"{args.split} split"
    else:                                   # all：观测与 store 的重叠
        start = max(pd.Timestamp("2022-01-01"), pd.Timestamp(args.obs_cover_start))
        end = pd.Timestamp(args.obs_cover_end)
        note = "全部可用重叠区间"
    cover_lo, cover_hi = pd.Timestamp(args.obs_cover_start), pd.Timestamp(args.obs_cover_end)
    clipped = (start, end)
    start, end = max(start, cover_lo), min(end, cover_hi + pd.Timedelta(days=1))
    if start >= end:
        raise SystemExit(
            f"请求窗口 {clipped[0].date()} ~ {clipped[1].date()} 与观测覆盖 "
            f"{cover_lo.date()} ~ {cover_hi.date()} 没有交集 —— 无法检验。\n"
            f"（本项目的 test split 是 2025-01-01 起，这份观测到 2024-08-15 就结束了；"
            f"同级的 convi/surf 也只到 2024-09-21。）")
    if (start, end) != clipped:
        print(f"[warn] 请求窗口被夹到观测覆盖内：{clipped[0].date()} ~ "
              f"{clipped[1].date()} -> {start.date()} ~ {(end - pd.Timedelta(days=1)).date()}")
    return start, end, note


def build_grid_index(cfg, lat, lon):
    """观测站点 -> 80x120 网格下标（最近邻），并返回域内掩膜。"""
    lat_ax = np.asarray(cfg.lat, dtype=float)
    lon_ax = np.asarray(cfg.lon, dtype=float)
    # 观测经度是 0..360；折到 -180..180 才能和以 -5.25 起的目标域比较
    lon360 = (np.asarray(lon, dtype=float) % 360.0 + 180.0) % 360.0 - 180.0
    res = float(lat_ax[1] - lat_ax[0])
    iy = np.clip(np.round((lat - lat_ax[0]) / res).astype(int), 0, lat_ax.size - 1)
    ix = np.clip(np.round((lon360 - lon_ax[0]) / res).astype(int), 0, lon_ax.size - 1)
    inside = ((lat >= lat_ax[0] - res / 2) & (lat <= lat_ax[-1] + res / 2)
              & (lon360 >= lon_ax[0] - res / 2) & (lon360 <= lon_ax[-1] + res / 2))
    return iy, ix, lon360, inside


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default="configs")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="覆盖 configs.py 的开关，例如 --set fcst_step=4")
    ap.add_argument("--obs-dir", default=DEFAULT_OBS_DIR)
    ap.add_argument("--start", default=None, help="YYYY-MM-DD（与 --end 同时给）")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD（半开区间上界）")
    ap.add_argument("--split", default="window", choices=("window", "train", "val", "test", "all"),
                    help="默认 window：2024-05-01~2024-08-15，最接近测试集的有观测区间")
    ap.add_argument("--hours", default="0,6,12,18",
                    help="参与对比的 UTC 时次，逗号分隔（ERA5/FuXi 为 6 小时）")
    ap.add_argument("--no-filter", action="store_true", help="关掉轻量合理性过滤")
    ap.add_argument("--out", type=Path, default=None, help="把逐条记录存成 csv")
    args = ap.parse_args()
    if (args.start is None) != (args.end is None):
        raise SystemExit("--start 与 --end 必须同时给")
    if args.split == "window" and args.start is None:
        args.start, args.end = "2024-05-01", "2024-08-16"

    key = os.path.basename(str(args.obs_dir).rstrip("/"))
    cov = OBS_COVER.get(key)
    if cov is None:                      # 未知观测源：用 --split all 时退化成 store 区间
        cov = ("1900-01-01", "2100-01-01")
        print(f"[warn] 未知观测源 {key}，跳过覆盖区间裁剪")
    args.obs_cover_start, args.obs_cover_end = cov

    cfg = load_cfg(args.configs, args.set)
    import zarr
    from common import CHANNELS, era5_channel_stats
    from main.utils.utils_data import decode_axis

    start, end, note = resolve_window(cfg, args)
    hours = {int(v) for v in str(args.hours).split(",")}
    mean, std = era5_channel_stats()
    idx_of = {name: CHANNELS.index(name) for _, name, _ in VARS}
    print(f"[cfg] label={cfg.label_zarr}\n[cfg] fuxi ={cfg.fuxi_zarr}  "
          f"(fcst_step={cfg.fcst_step} -> lead {int(cfg.fcst_step) * 6} h)")
    print(f"[obs] {args.obs_dir}\n[win] {start.date()} ~ "
          f"{(end - pd.Timedelta(days=1)).date()}（{note}），时次 {sorted(hours)}")

    lab = zarr.open(str(cfg.label_zarr), "r")["label"]
    gf = zarr.open(str(cfg.fuxi_zarr), "r")
    lb_t = decode_axis(cfg.label_zarr, "time")
    init_t = decode_axis(cfg.fuxi_zarr, "init")
    lead = pd.Timedelta(hours=int(cfg.fcst_step) * 6)
    lb_pos = {t: i for i, t in enumerate(lb_t)}
    init_vals = init_t.values
    n_bg_chan = int(gf["z"].shape[2])

    def de_std(block, chan, scale):
        v = block[chan] * np.float32(std[chan]) + np.float32(mean[chan])
        return (v / scale).astype(np.float32) if scale != 1.0 else v

    rows, n_missing, n_used, n_empty = [], 0, 0, 0
    for t in pd.date_range(start, end, freq="6h", inclusive="left"):
        if t.hour not in hours:
            continue
        path = Path(args.obs_dir) / f"surf_{t:%Y%m%d%H}.pkl"
        if not path.exists():
            n_missing += 1
            continue
        with open(path, "rb") as fh:
            obs = pickle.load(fh)
        iy, ix, lon360, inside = build_grid_index(cfg, obs["lat"].values, obs["lon"].values)
        if not inside.any():
            n_empty += 1
            continue
        sub = obs[inside].reset_index(drop=True)
        iy, ix, lon_in = iy[inside], ix[inside], lon360[inside]

        lb_i = lb_pos.get(t)
        bg_pos = int(np.searchsorted(init_vals, (t - lead).to_datetime64()))
        if lb_i is None or bg_pos >= init_vals.size or \
                abs(init_vals[bg_pos] - (t - lead).to_datetime64()) > np.timedelta64(0, "s"):
            continue
        n_used += 1
        era5_blk = np.asarray(lab[lb_i], dtype=np.float32)
        fuxi_blk = np.asarray(gf["z"][bg_pos, 0, :n_bg_chan], dtype=np.float32)

        for obs_col, chan_name, scale in VARS:
            if obs_col not in sub.columns:
                continue
            o = sub[obs_col].values.astype(float)
            e = de_std(era5_blk, idx_of[chan_name], scale)[iy, ix].astype(float)
            f = de_std(fuxi_blk, idx_of[chan_name], scale)[iy, ix].astype(float)
            m = np.isfinite(o) & np.isfinite(e) & np.isfinite(f)
            if not args.no_filter:
                if obs_col == "t2m":
                    m &= (o > 200) & (o < 340)
                elif obs_col == "slp":
                    m &= (o > 850) & (o < 1100)
                else:
                    m &= np.abs(o) < 60
            if not m.any():
                continue
            rows.append(pd.DataFrame({
                "time": t, "var": obs_col, "obs": o[m], "era5": e[m], "fuxi": f[m],
                "lat": sub["lat"].values[m], "lon": lon_in[m],
                "month": t.month, "hour": t.hour}))

    if not rows:
        raise SystemExit("没有任何有效对比记录 —— 检查 --obs-dir / 时间窗口 / 时次")
    df = pd.concat(rows, ignore_index=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)

    print(f"[run] 计划时次 {len(pd.date_range(start, end, freq='6h', inclusive='left'))}，"
          f"缺文件 {n_missing}，域内空 {n_empty}，实际用时次 {n_used}，"
          f"有效记录 {len(df)}，独立站点 {df.groupby(['lat', 'lon']).ngroups}")

    def table(frame, by):
        if by is None:
            groups = [("", frame)]
        else:
            groups = list(frame.groupby(by))
        for label, g in groups:
            o, e, f = g["obs"].values, g["era5"].values, g["fuxi"].values
            me, mf = np.abs(e - o).mean(), np.abs(f - o).mean()
            head = f"{label:>4}" if by else "全部"
            print(f"{head:>10}{len(g):>9d}{o.mean():>11.3f}{e.mean():>11.3f}{f.mean():>11.3f}"
                  f"{me:>10.3f}{mf:>10.3f}{100 * (mf - me) / mf:>8.1f}%"
                  f"{(e - o).mean():>9.3f}{(f - o).mean():>9.3f}"
                  f"{np.sqrt(((e - o) ** 2).mean()):>9.3f}{np.sqrt(((f - o) ** 2).mean()):>9.3f}")

    hdr = (f'{"":>10}{"n":>9}{"obs":>11}{"ERA5":>11}{"FuXi":>11}{"MAE_ERA5":>10}'
           f'{"MAE_FuXi":>10}{"ERA5优":>9}{"Bias_E":>9}{"Bias_F":>9}{"RMSE_E":>9}{"RMSE_F":>9}')
    print("\n=== 总体（域内站点，站点-时次等权）===")
    print(hdr)
    for var, g in df.groupby("var", sort=False):
        print(f"{var:>10}", end="")
        table(g, None)
    print("\n=== 逐月 ===")
    print(hdr)
    for var, g in df.groupby("var", sort=False):
        print(f"--- {var}")
        table(g, "month")
    if args.out:
        print(f"\n逐条记录 -> {args.out}")


if __name__ == "__main__":
    main()
