#!/usr/bin/env python3
"""把 ERA5 官方 37 层重算的 ZTD/ZHD/ZWD 和上一个结果放在同一批样本上比较。

参与的来源
----------
======================  ==========================================================
``era5_37``             本次：官方 37 层 + 方法 E（参考面 = 测站高度，真实 ``sp``）
``era5_13``             本次：同上但只用 13 层（剥离"层数"这一项）
``era5_op13``           上一个结果：项目 store 的 13 层 + 方法 E（``p_s`` 由 msl 折出）
``fuxi``                FuXi 24h 背景（13 层，方法 E，``ztd_fuxi_..._zdz.zarr``）
``obs``                 NGL GNSS 观测；ZHD 用 ``ZTD - TRWET`` 反推，不是独立量
======================  ==========================================================

样本 = ``/tmp/zhd_three_ways_v2.csv`` 里的 (time, station) 对（102 个时刻 × ~660 站），
与上一个结果完全对齐。

用法（**gnss** 环境，在 da_ngl/main_code 下）
-------------------------------------------
    python ../test/compare_era5_37lev_ztd.py \
        --new-csv /tmp/era5_37lev_ztd.csv \
        --prev-csv /tmp/zhd_three_ways_v2.csv \
        --out-csv /tmp/era5_37lev_ztd_matched.csv \
        --out-fig ../plots/era5_37lev_ztd_matched.png
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

G_0, R_D = 9.80665, 287.05


def _stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    return {"n": int(np.isfinite(d).sum()), "bias": float(np.mean(d)),
            "rmse": float(np.sqrt(np.mean(d ** 2))), "mae": float(np.mean(np.abs(d))),
            "r": float(np.corrcoef(a, b)[0, 1])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default="configs")
    ap.add_argument("--new-csv", type=Path, default=Path("/tmp/era5_37lev_ztd.csv"))
    ap.add_argument("--prev-csv", type=Path, default=Path("/tmp/zhd_three_ways_v2.csv"))
    ap.add_argument("--out-csv", type=Path, default=Path("/tmp/era5_37lev_ztd_matched.csv"))
    ap.add_argument("--out-fig", type=Path, default=None)
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()

    import importlib
    cfg = importlib.import_module(args.configs)
    from common import CHANNELS, decode_time_axis, era5_channel_stats
    from main.utils import station_geometry
    from ztd_operator import LEV_HPA, ztd_profile_zdz

    iy, ix, h_st, station_id = station_geometry(cfg)
    n_st = iy.size
    prev = pd.read_csv(args.prev_csv)
    prev["time"] = pd.to_datetime(prev["time"])
    new = pd.read_csv(args.new_csv)
    new["time"] = pd.to_datetime(new["time"])
    print(f"[in] 上一个结果 {len(prev)} 行 / 本次结果 {len(new)} 行")

    # ---------- 上一个结果里的 ERA5 13 层（项目 store，方法 E，p_s 由 msl 折出） ----------
    lab = zarr.open(str(cfg.label_zarr), "r")["label"]
    lab_t = decode_time_axis(Path(cfg.label_zarr), "time")
    mean, std = era5_channel_stats()
    OP = ([CHANNELS.index(f"t{int(L)}") for L in LEV_HPA]
          + [CHANNELS.index(f"r{int(L)}") for L in LEV_HPA]
          + [CHANNELS.index("t2m"), CHANNELS.index("msl")]
          + [CHANNELS.index(f"z{int(L)}") for L in LEV_HPA])

    ng = zarr.open(str(cfg.ngl_zarr), "r")
    z_mu, z_sd = float(ng["ztd_train_mean"][0]), float(ng["ztd_train_std"][0])
    w_mu, w_sd = float(ng["zwd_train_mean"][0]), float(ng["zwd_train_std"][0])
    obs_t = decode_time_axis(Path(cfg.ngl_zarr), "time")
    fz = zarr.open(str(cfg.ztd_fuxi_zarr), "r")
    fx_init = decode_time_axis(Path(cfg.ztd_fuxi_zarr), "time")   # = label time axis

    times = np.sort(prev["time"].unique())
    print(f"[obs] {len(times)} 个时刻，{n_st} 个站格")

    recs = []
    for T in times:
        T = pd.Timestamp(T)
        li = int(np.flatnonzero(lab_t.values == T.to_datetime64())[0])
        fi = int((T - obs_t[0]).total_seconds() // 300)
        fidx = int(np.flatnonzero(fx_init.values == T.to_datetime64())[0])

        blk = np.asarray(lab[li, :70], "float32")
        ph = blk[OP] * std[OP][:, None, None] + mean[OP][:, None, None]
        ph = ph[:, iy, ix]
        T_lev, R_lev = ph[:13].T, ph[13:26].T
        t2m, msl = ph[26], ph[27]
        H_lev = (ph[28:41] / G_0).T
        p_s = msl / 100.0 * np.exp(-G_0 * h_st / (R_D * t2m))
        op13 = ztd_profile_zdz(T_lev, R_lev, t2m, p_s, h_st, H_lev,
                               lev_hpa=LEV_HPA)

        recs.append(pd.DataFrame({
            "time": T, "st": np.arange(n_st),
            "sd_zhd_op13": op13["ZHD_mm"], "sd_zwd_op13": op13["ZWD_mm"],
            "sd_ztd_op13": op13["ZTD_mm"], "sd_p_s": p_s,
            "fx_zhd": np.asarray(fz["zhd"][fidx], "f4")[iy, ix],
            "fx_zwd": np.asarray(fz["zwd"][fidx], "f4")[iy, ix],
            "fx_ztd": np.asarray(fz["ztd_fuxi"][fidx], "f4")[iy, ix],
            "obs_ztd": np.asarray(ng["ztd"][fi, iy, ix], "f4") * z_sd + z_mu,
            "obs_zwd": np.asarray(ng["zwd"][fi, iy, ix], "f4") * w_sd + w_mu,
        }))
    ref = pd.concat(recs, ignore_index=True)
    ref["obs_zhd"] = ref["obs_ztd"] - ref["obs_zwd"]

    df = (prev[["time", "st", "dist_km", "h_gnss", "p_s_era5", "zhd_op_era5",
                "zhd_op_fuxi", "zhd_surf", "zhd_gnss"]]
          .merge(new, on=["time", "st"], how="inner")
          .merge(ref, on=["time", "st"], how="inner"))
    n_all = len(df)

    # 闭式 Saastamoinen 参照（和 check_zhd_three_ways.py 一样用格心纬度 + 站高）
    lat_g = np.asarray(cfg.lat, dtype=float)[iy]
    den = (1.0 - 0.00266 * np.cos(2 * np.deg2rad(lat_g))
           - 0.00028 * h_st / 1000.0)
    df["saas_pst"] = 1000.0 * 0.0022768 * df["p_st"].to_numpy(float) / den[df["st"].to_numpy(int)]
    df["saas_prev"] = 1000.0 * 0.0022768 * df["p_s_era5"].to_numpy(float) / den[df["st"].to_numpy(int)]

    # 统一到上一个结果的口径：只保留 NGL 真的有观测的样本
    obs_ok = np.isfinite(df[["obs_ztd", "obs_zwd", "obs_zhd"]].to_numpy(float)).all(axis=1)
    print(f"[merge] 与上一个结果对齐 {n_all} 行；其中 {int(obs_ok.sum())} 行有 NGL 观测"
          f"（= 上一个结果的 57,232 口径），{int((~obs_ok).sum())} 行无观测被剔除")
    df = df[obs_ok].reset_index(drop=True)
    print(f"[basis] 以下所有统计都基于 n={len(df)}"
          f"（{df['time'].nunique()} 时刻 × {df['st'].nunique()} 站）")
    df.to_csv(args.out_csv, index=False)
    print(f"[out] {args.out_csv}")

    # ---------------- 汇总 ----------------
    print("\n" + "=" * 78)
    print("ZTD / ZHD / ZWD 汇总（mm）—— 同一批样本")
    print("=" * 78)
    print(f'{"量":>4}{"来源":>16}{"mean":>10}{"std":>8}{"min":>9}{"max":>9}')
    table = {
        "ZTD": [("obs (NGL GNSS)", "obs_ztd"), ("ERA5 37L (new)", "ztd37"),
                ("ERA5 13L (project store)", "sd_ztd_op13"),
                ("FuXi 24h (13L)", "fx_ztd")],
        "ZHD": [("obs (ZTD-TRWET)", "obs_zhd"), ("ERA5 37L (new)", "zhd37"),
                ("ERA5 13L (project store)", "sd_zhd_op13"),
                ("FuXi 24h (13L)", "fx_zhd"), ("NCEP surf (Saas)", "zhd_surf"),
                ("closed-form Saas (p_st)", "saas_pst")],
        "ZWD": [("obs (TRWET)", "obs_zwd"), ("ERA5 37L (new)", "zwd37"),
                ("ERA5 13L (project store)", "sd_zwd_op13"),
                ("FuXi 24h (13L)", "fx_zwd")],
    }
    for q, rows in table.items():
        for tag, c in rows:
            v = df[c].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            print(f"{q:>4}{tag:>26}{v.mean():>10.2f}{v.std():>8.2f}"
                  f"{v.min():>9.2f}{v.max():>9.2f}")
        print("-" * 78)

    print("\n相对 NGL 观测的偏差（同一样本）")
    print(f'{"量":>4}{"来源":>26}{"n":>8}{"bias":>9}{"rmse":>9}{"mae":>9}{"r":>9}')
    for q, obs in (("ZTD", "obs_ztd"), ("ZHD", "obs_zhd"), ("ZWD", "obs_zwd")):
        for tag, c in table[q][1:]:
            m = np.isfinite(df[obs]) & np.isfinite(df[c])
            s = _stats(df[c].to_numpy(float)[m], df[obs].to_numpy(float)[m])
            print(f"{q:>4}{tag:>26}{s['n']:>8d}{s['bias']:>+9.2f}{s['rmse']:>9.2f}"
                  f"{s['mae']:>9.2f}{s['r']:>9.4f}")
        print("-" * 78)

    print("\n与闭式 Saastamoinen 的差（同 p_s 才可比）")
    for tag, c, r in (("ERA5 37L method E", "zhd37", "saas_pst"),
                      ("上一个结果 13L", "zhd_op_era5", "saas_prev"),
                      ("NCEP 实测气压 Saas", "zhd_surf", "saas_pst")):
        d = df[c].to_numpy(float) - df[r].to_numpy(float)
        print(f"  {tag:>22}: {d.mean():+6.2f} ± {d.std():4.2f} mm")

    d13 = df["zhd13"].to_numpy(float) - df["zhd_op_era5"].to_numpy(float)
    print(f"\n[复现检验] zhd13(官方资料) - zhd_op_era5(项目 store): "
          f"bias {d13.mean():+.2f} mm, RMSE {np.sqrt((d13 ** 2).mean()):.2f} mm"
          f"（两路 13 层一致，差异可全部归给层数）")

    print("\nERA5 新旧方案互比（本次 37L 对上一个结果的 13L）")
    print(f'{"量":>4}{"对比":>34}{"n":>8}{"bias":>9}{"rmse":>9}{"mae":>9}{"r":>9}')
    for q, a, b in (("ZHD", "zhd37", "zhd_op_era5"), ("ZWD", "zwd37", "sd_zwd_op13"),
                    ("ZTD", "ztd37", "sd_ztd_op13")):
        m = np.isfinite(df[a]) & np.isfinite(df[b])
        s = _stats(df[a].to_numpy(float)[m], df[b].to_numpy(float)[m])
        print(f"{q:>4}{a + ' - ' + b:>34}{s['n']:>8d}{s['bias']:>+9.2f}"
              f"{s['rmse']:>9.2f}{s['mae']:>9.2f}{s['r']:>9.4f}")

    print("\n地面气压来源对比（folding 是否影响 ZHD）")
    for tag, c in (("ERA5 sp (official)", "p_sp"), ("ERA5 msl->station (store)", "sd_p_s"),
                   ("ERA5 msl->station (prev csv)", "p_s_era5"),
                   ("ERA5 sp folded to station", "p_st")):
        v = df[c].to_numpy(float)
        print(f"  {tag:>32}: {np.nanmean(v):8.2f} ± {np.nanstd(v):5.2f} hPa")

    if args.no_fig:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2), dpi=140)
    series = [("ERA5 37L (new)", "ztd37", "tab:red"),
              ("ERA5 13L (official)", "ztd13", "tab:blue"),
              ("ERA5 13L (project store)", "sd_ztd_op13", "tab:cyan"),
              ("FuXi 24h (13L)", "fx_ztd", "tab:green")]
    ax = axes[0]
    x = df["obs_ztd"].to_numpy(float)
    for tag, c, col in series:
        y = df[c].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[m], y[m], s=4, alpha=.18, color=col, lw=0, rasterized=True,
                   label=f"{tag}  bias {np.mean(y[m] - x[m]):+.1f} mm")
    lim = [np.nanmin(x), np.nanmax(x)]
    ax.plot(lim, lim, "k--", lw=.9, label="y = x")
    ax.set_xlabel("NGL GNSS observed ZTD (mm)")
    ax.set_ylabel("model ZTD (mm)")
    ax.set_title(f"ZTD, same {len(df)} matched (station,time) samples")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=.25)

    ax = axes[1]
    for tag, c, col in series:
        y = df[c].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        ax.hist((y - x)[m], bins=120, histtype="step", lw=1.3, color=col,
                label=f"{tag}  {np.mean(y[m]-x[m]):+.1f} ± {np.std(y[m]-x[m]):.1f}")
    ax.axvline(0, c="k", lw=.9)
    ax.set_xlabel("model ZTD - obs ZTD (mm)")
    ax.set_ylabel("count")
    ax.set_title("ZTD bias distribution")
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)

    ax = axes[2]
    names, vals, cols = [], [], []
    for q, rows in table.items():
        for tag, c in rows[1:]:
            if q == "ZHD" and c == "zhd_surf":
                continue
            m = np.isfinite(df[f"obs_{q.lower()}"]) & np.isfinite(df[c])
            d = df[c].to_numpy(float)[m] - df[f"obs_{q.lower()}"].to_numpy(float)[m]
            names.append(f"{q}: {tag}")
            vals.append(np.sqrt(np.mean(d ** 2)))
            cols.append({"ZTD": "tab:red", "ZHD": "tab:blue", "ZWD": "tab:green"}[q])
    ax.barh(np.arange(len(names)), vals, color=cols, alpha=.8)
    ax.set_yticks(np.arange(len(names)), names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE vs NGL observation (mm)")
    ax.set_title("RMSE by quantity and source")
    ax.grid(alpha=.25, axis="x")
    fig.tight_layout()
    out = args.out_fig or (DA_ROOT / "plots" / "era5_37lev_ztd_matched.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"\n[fig] {out}")


if __name__ == "__main__":
    main()
