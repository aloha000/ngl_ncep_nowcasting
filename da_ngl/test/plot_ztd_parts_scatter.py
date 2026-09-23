#!/usr/bin/env python3
"""三个散点图：ZTD / ZHD / ZWD，横轴=观测(NGL)，纵轴=FuXi、ERA5 13L、ERA5 37L。

上排：全部 (站, 时次) 样本的散点（浅色）+ 逐站时间均值（深色点）
下排：逐站时间均值的偏差 (模型 − 观测) 对观测值，用来把 ZHD 那种 0.5% 的系统差看清楚

FuXi 用蓝色、ERA5 13L 用橙色、ERA5 37L 用红色（37 层那一路按 (time, station) 从
``--era5-37l-csv`` 合并进来，样本时刻对不上的点直接缺省不画）；两张图的观测同源（NGL：ZTD=TROTOT，ZHD=ZTD−TRWET，ZWD=TRWET）。
NOTE: 本机无中文字体，图注用英文。

用法（在 da_ngl/main_code 下）
----
    python ../test/plot_ztd_parts_scatter.py --stride 20
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--era5-37l-csv", type=Path,
                    default=DA_ROOT / "dataset" / "era5_37lev_ztd_stations_test102.csv",
                    help="ERA5 37 层方法 E 的结果（era5_37lev_ztd.py 产出）；缺省则不加这一路")
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
        e5 = ztd_profile_zdz(T_lev, R_lev, t2m, p_s, h_st, H_lev)
        rows.append(pd.DataFrame({
            "time": T, "st": np.arange(iy.size),
            "obs_ztd": z_o, "obs_zwd": w_o, "obs_zhd": z_o - w_o,
            "fx_ztd": np.asarray(fz["ztd_fuxi"][i], "float32")[iy, ix],
            "fx_zwd": np.asarray(fz["zwd"][i], "float32")[iy, ix],
            "fx_zhd": np.asarray(fz["zhd"][i], "float32")[iy, ix],
            "e5_ztd": e5["ZTD_mm"], "e5_zwd": e5["ZWD_mm"], "e5_zhd": e5["ZHD_mm"]}))
        if (n + 1) % 20 == 0:
            print(f"  {n+1}/{idx.size}", flush=True)
    df = pd.concat(rows, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])

    # ERA5 37 层（方法 E，官方资料）：按 (time, station) 合进来
    has37 = False
    if args.era5_37l_csv and Path(args.era5_37l_csv).exists():
        e37 = pd.read_csv(args.era5_37l_csv, usecols=["time", "st", "zhd37", "zwd37", "ztd37"])
        e37["time"] = pd.to_datetime(e37["time"])
        df = df.merge(e37.rename(columns={"zhd37": "e37_zhd", "zwd37": "e37_zwd",
                                          "ztd37": "e37_ztd"}),
                      on=["time", "st"], how="left")
        cov = float(df["e37_ztd"].notna().mean())
        print(f"[37L] {args.era5_37l_csv.name}: 覆盖 {100 * cov:.1f}% 的样本"
              f"（{df.loc[df['e37_ztd'].notna(), 'time'].nunique()} / {df['time'].nunique()} 个时刻）")
        has37 = cov > 0
    g = df.groupby("st").mean(numeric_only=True)

    comps = (("ztd", "ZTD"), ("zhd", "ZHD"), ("zwd", "ZWD"))
    colors = {"fx": "tab:blue", "e5": "tab:orange", "e37": "tab:red"}
    names = {"fx": "FuXi 24h", "e5": "ERA5 13L", "e37": "ERA5 37L"}
    sources = ("fx", "e5", "e37") if has37 else ("fx", "e5")
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.6), dpi=130)

    print(f'\n{"分量":>5}{"源":>10}{"逐条 bias":>12}{"RMSE":>9}{"r":>9}'
          f'{"逐站均值 bias":>15}{"RMSE":>9}{"r":>9}')
    for col, (c, lab_) in enumerate(comps):
        o, gx = df[f"obs_{c}"].values, g[f"obs_{c}"].values
        ax = axes[0, col]
        for src in sources:
            v, gv = df[f"{src}_{c}"].values, g[f"{src}_{c}"].values
            m = np.isfinite(o) & np.isfinite(v)
            gm = np.isfinite(gx) & np.isfinite(gv)
            ax.scatter(o[m], v[m], s=1.0, alpha=0.05, color=colors[src], rasterized=True)
            ax.scatter(gx[gm], gv[gm], s=7, alpha=0.55, color=colors[src], lw=0,
                       label=f"{names[src]}（逐站均值）")
            print(f'{lab_:>5}{names[src]:>10}{np.nanmean(v-o):>12.2f}'
                  f'{np.sqrt(np.nanmean((v-o)**2)):>9.2f}{np.corrcoef(v[m],o[m])[0,1]:>9.4f}'
                  f'{np.nanmean(gv-gx):>15.2f}{np.sqrt(np.nanmean((gv-gx)**2)):>9.2f}'
                  f'{np.corrcoef(gv[gm],gx[gm])[0,1]:>9.4f}')
        lo = np.nanmin([o.min()] + [g[f"{s_}_{c}"].values.min() for s_ in sources])
        hi = np.nanmax([o.max()] + [g[f"{s_}_{c}"].values.max() for s_ in sources])
        ax.plot([lo, hi], [lo, hi], "k--", lw=.8, zorder=0)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"observed {lab_} (mm, NGL)")
        ax.set_ylabel(f"model {lab_} (mm)")
        ax.set_title(f"{lab_}: obs vs model (all samples faint, station means solid)")
        ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=.25)

        # 下排：逐站均值的偏差
        ax = axes[1, col]
        for src in sources:
            gv = g[f"{src}_{c}"].values
            gm = np.isfinite(gx) & np.isfinite(gv)
            ax.scatter(gx[gm], (gv-gx)[gm], s=9, alpha=0.6, color=colors[src], lw=0,
                       label=f"{names[src]}  bias {np.nanmean(gv[gm]-gx[gm]):+.2f} mm")
        ax.axhline(0, c="k", lw=.8)
        ax.set_xlabel(f"observed {lab_} (mm)"); ax.set_ylabel(f"model − obs (mm)")
        ax.set_title(f"{lab_}: per-station mean bias"); ax.legend(fontsize=8); ax.grid(alpha=.25)

    fig.suptitle(f"ZTD / ZHD / ZWD: observation (NGL) vs FuXi 24h vs ERA5 13L vs ERA5 37L "
                 f"({args.split} split, stride {args.stride}, {len(df)} samples)",
                 fontsize=12, y=.995)
    fig.tight_layout(rect=(0, 0, 1, .975))
    out = args.out or (DA_ROOT / "plots" / "ztd_parts_scatter.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
