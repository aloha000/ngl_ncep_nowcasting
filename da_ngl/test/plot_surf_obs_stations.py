#!/usr/bin/env python3
"""地表观测站网在 80x120 域内的分布（含与 GNSS ZTD 站网的对比）。

扫描 ``obs-pkl_qc/convi/surf_ncep`` 里若干时次的 pkl，取出落在 0.25° 目标网格
（lat 36.50..56.25, lon -5.25..24.50）内的地面站，画四联图：

  (a) 每个 0.25° 格点的地表站数（log 色标）
  (b) 单个站点散点，颜色 = 该站在窗口内的报告次数（log）
  (c) GNSS ZTD 站网（1378 个站格，来自 station_grid_map.parquet）用于对比
  (d) 每格站数直方图：地表站 vs GNSS 站

NOTE: 这台机器没有中文字体，所有图注用英文。

用法（在 da_ngl/main_code 下）：
    python ../test/plot_surf_obs_stations.py
    python ../test/plot_surf_obs_stations.py --start 2024-05-01 --end 2024-08-16 --hours 0,12
    python ../test/plot_surf_obs_stations.py --out ../plots/surf_stations.png
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

HERE = Path(__file__).resolve().parent
DA_ROOT = HERE.parent
sys.path.insert(0, str(DA_ROOT / "main_code"))
sys.path.insert(0, str(DA_ROOT / "preprocessing"))

DEFAULT_OBS_DIR = ("/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/database/"
                   "fuxi-obs/obs-pkl_qc/convi/surf_ncep")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default="configs")
    ap.add_argument("--obs-dir", default=DEFAULT_OBS_DIR)
    ap.add_argument("--start", default="2024-05-01")
    ap.add_argument("--end", default="2024-08-16", help="半开区间上界")
    ap.add_argument("--hours", default="0,6,12,18")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import importlib
    cfg = importlib.import_module(args.configs)
    lat_ax = np.asarray(cfg.lat, dtype=float)
    lon_ax = np.asarray(cfg.lon, dtype=float)
    res = float(lat_ax[1] - lat_ax[0])
    hours = {int(v) for v in str(args.hours).split(",")}

    # ---- 扫描观测：统计每个站点的报告次数 ----------------------------------
    counts: dict[tuple[float, float], int] = {}
    times = pd.date_range(args.start, args.end, freq="6h", inclusive="left")
    used = missing = 0
    for t in times:
        if t.hour not in hours:
            continue
        path = Path(args.obs_dir) / f"surf_{t:%Y%m%d%H}.pkl"
        if not path.exists():
            missing += 1
            continue
        with open(path, "rb") as fh:
            obs = pickle.load(fh)
        # 观测经度是 0..360，必须先折到 -180..180 再判域，
        # 否则 lon<0 的西侧条带（爱尔兰/英国西部/葡萄牙）会被整条漏掉
        lonv = (obs["lon"].values % 360.0 + 180.0) % 360.0 - 180.0
        inside = ((obs["lat"].values >= lat_ax[0] - res / 2)
                  & (obs["lat"].values <= lat_ax[-1] + res / 2)
                  & (lonv >= lon_ax[0] - res / 2) & (lonv <= lon_ax[-1] + res / 2))
        if not inside.any():
            continue
        used += 1
        key = np.round(np.stack([obs["lat"].values[inside], lonv[inside]], 1), 3)
        for row in key:
            k = (float(row[0]), float(row[1]))
            counts[k] = counts.get(k, 0) + 1

    if not counts:
        raise SystemExit("窗口内没有域内站点 —— 检查 --obs-dir / --start / --end")
    keys = np.array(list(counts.keys()))
    lat_s, lon_s = keys[:, 0], keys[:, 1]
    n_rep = np.array([counts[(float(a), float(b))] for a, b in keys], dtype=float)
    print(f"[obs] 时次 {used}（缺 {missing}），域内独立站点 {lat_s.size}，"
          f"报告次数 中位数 {np.median(n_rep):.0f} 最小 {n_rep.min():.0f} 最大 {n_rep.max():.0f}")

    # 站点落到 0.25° 格点后的计数
    iy = np.clip(np.round((lat_s - lat_ax[0]) / res).astype(int), 0, lat_ax.size - 1)
    ix = np.clip(np.round((lon_s - lon_ax[0]) / res).astype(int), 0, lon_ax.size - 1)
    cell_cnt = np.zeros((lat_ax.size, lon_ax.size), dtype=float)
    np.add.at(cell_cnt, (iy, ix), 1.0)

    # ---- GNSS ZTD 站网（对比用） -------------------------------------------
    gm = pd.read_parquet(Path(cfg.ngl_zarr).parent /
                         "ngl_europe_0p25_80x120_station_grid_map.parquet")
    g_cells = gm[~gm["mask"].astype(bool) & gm["station_id"].notna()]
    giy = np.clip(np.round((g_cells["lat"].values - lat_ax[0]) / res).astype(int),
                  0, lat_ax.size - 1)
    gix = np.clip(np.round((g_cells["lon"].values - lon_ax[0]) / res).astype(int),
                  0, lon_ax.size - 1)
    gnss = np.zeros_like(cell_cnt)
    gnss[giy, gix] = 1.0

    ext = [lon_ax[0] - res / 2, lon_ax[-1] + res / 2,
           lat_ax[0] - res / 2, lat_ax[-1] + res / 2]
    fig, axes = plt.subplots(2, 2, figsize=(17, 9.5), dpi=130)

    ax = axes[0, 0]
    m = np.ma.masked_where(cell_cnt == 0, cell_cnt)
    im = ax.imshow(m, origin="lower", extent=ext, cmap="viridis",
                   norm=LogNorm(vmin=1, vmax=max(cell_cnt.max(), 2)), aspect="auto")
    plt.colorbar(im, ax=ax, fraction=.045, label="surface stations per 0.25 deg cell")
    ax.set_title(f"(a) surface obs (surf_ncep) density, {used} time steps\n"
                 f"{int((cell_cnt > 0).sum())} cells occupied / 9600", fontsize=10)
    ax.set_xlabel("lon"); ax.set_ylabel("lat"); ax.grid(alpha=.25)

    ax = axes[0, 1]
    sc = ax.scatter(lon_s, lat_s, c=np.maximum(n_rep, 1), s=3.5, cmap="magma",
                    norm=LogNorm(vmin=1, vmax=max(n_rep.max(), 2)), alpha=.55, lw=0)
    plt.colorbar(sc, ax=ax, fraction=.045, label="reports per station in window")
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_title(f"(b) individual stations: {lat_s.size} unique sites\n"
                 f"median reports {np.median(n_rep):.0f} / max {n_rep.max():.0f}", fontsize=10)
    ax.set_xlabel("lon"); ax.set_ylabel("lat"); ax.grid(alpha=.25)

    ax = axes[1, 0]
    im = ax.imshow(gnss, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1,
                   aspect="auto", alpha=.85)
    ax.scatter(lon_s, lat_s, s=2.0, c="tab:blue", alpha=.35, lw=0, label="surface obs")
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_title(f"(c) GNSS ZTD station cells (grey, n={int(gnss.sum())}) with\n"
                 f"surface-obs sites overlaid (blue)", fontsize=10)
    ax.set_xlabel("lon"); ax.set_ylabel("lat"); ax.grid(alpha=.25)

    ax = axes[1, 1]
    bins = np.arange(0, max(cell_cnt.max(), 2) + 2) - .5
    ax.hist(cell_cnt[cell_cnt > 0], bins=bins, color="tab:blue", alpha=.8,
            label=f"surface obs ({int((cell_cnt > 0).sum())} cells)")
    ax.hist(gnss[gnss > 0], bins=[-.5, .5, 1.5], color="0.55", alpha=.8,
            label=f"GNSS ZTD ({int(gnss.sum())} cells)")
    ax.set_yscale("log")
    ax.set_xlabel("stations per 0.25 deg cell"); ax.set_ylabel("cells (log)")
    ax.set_title("(d) per-cell station count", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.25, axis="y")

    fig.suptitle(f"Observation networks over the assimilation domain "
                 f"(lat {lat_ax[0]:.2f}..{lat_ax[-1]:.2f}, lon {lon_ax[0]:.2f}..{lon_ax[-1]:.2f}, 0.25 deg)",
                 fontsize=12, y=.98)
    fig.tight_layout(rect=(0, 0, 1, .96))
    out = args.out or (DA_ROOT / "plots" / "obs_station_distribution.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"[out] {out}")
    print(f"[stats] 地表站: {lat_s.size} 个独立站点 / 0.25 度格点中 {int((cell_cnt > 0).sum())} 格有站，"
          f"格内最多 {cell_cnt.max():.0f} 个；GNSS: {int(gnss.sum())} 格（每格 1 站）")
    both = int(((cell_cnt > 0) & (gnss > 0)).sum())
    print(f"[stats] 两类站网同时覆盖的格点 {both}；"
          f"只有地表站 {int(((cell_cnt > 0) & (gnss == 0)).sum())}；只有 GNSS {int(((cell_cnt == 0) & (gnss > 0)).sum())}")


if __name__ == "__main__":
    main()
