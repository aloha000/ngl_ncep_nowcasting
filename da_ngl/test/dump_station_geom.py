#!/usr/bin/env python3
"""导出 1378 个站格的几何信息，供 ERA5 官方资料重算 ZTD 的脚本使用。

必须在 **gnss** conda 环境里跑（需要 zarr 2.x 读项目的 store）：

    cd da_ngl/main_code
    python ../test/dump_station_geom.py --out /tmp/geom_1378.npz

输出字段
--------
``iy, ix``      站格在 80x120 统一网格上的索引（station_geometry 的顺序）
``h``           测站高度 [m]（NGL 元数据的 height_m）
``station_id``  站号，顺序与 iy/ix/h 一致
``lat_true, lon_true``  测站真实经纬度（元数据值，非格心）
``lat_ax, lon_ax``      统一网格的 80 个纬度 / 120 个经度 (0.25 deg)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DA_ROOT = HERE.parent
sys.path.insert(0, str(DA_ROOT / "main_code"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default="configs", help="在 da_ngl/main_code 下运行")
    ap.add_argument("--stations", default=str(DA_ROOT / "dataset" / "ngl_europe_stations.parquet"))
    ap.add_argument("--out", type=Path, default=Path("/tmp/geom_1378.npz"))
    args = ap.parse_args()

    import importlib
    cfg = importlib.import_module(args.configs)
    from main.utils import station_geometry

    iy, ix, h, station_id = station_geometry(cfg)
    st = pd.read_parquet(args.stations)
    st["gnss_station_id"] = st["gnss_station_id"].astype(str)
    meta = st.set_index("gnss_station_id").reindex([str(s) for s in station_id])
    if meta["lat"].isna().any():
        raise SystemExit("有站号在元数据里找不到经纬度")

    np.savez(
        args.out,
        iy=np.asarray(iy), ix=np.asarray(ix), h=np.asarray(h, dtype="f8"),
        station_id=np.asarray([str(s) for s in station_id]),
        lat_true=meta["lat"].to_numpy("f8"), lon_true=meta["lon"].to_numpy("f8"),
        lat_ax=np.asarray(cfg.lat, dtype="f8"), lon_ax=np.asarray(cfg.lon, dtype="f8"),
    )
    print(f"[out] {args.out}: {iy.size} 站，高度 {h.min():.0f}~{h.max():.0f} m，"
          f"网格 {np.asarray(cfg.lat).size}x{np.asarray(cfg.lon).size}")


if __name__ == "__main__":
    main()
