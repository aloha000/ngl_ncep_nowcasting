#!/usr/bin/env python3
"""Map scattered GNSS stations onto a fixed 0.05-deg grid.

For every grid cell:
  * no station            -> mask
  * exactly one station   -> that station represents the cell
  * two or more stations  -> one station is selected randomly

The output is a long-form map (one row per 0.05 x 0.05 grid point) with the
grid-center latitude/longitude, the number of stations in that cell and the
selected station id.  Grid centers are exact multiples of 0.05 degrees.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


STATION_FILE_RE = re.compile(r"^(?P<station>[^./\\]+)\.\d{4}\.trop\.zip$")


def discover_stations(data_root: Path) -> list[str]:
    """Return the sorted station ids found under the ZTD archive tree."""
    station_ids: set[str] = set()
    for path in data_root.rglob("*.trop.zip"):
        match = STATION_FILE_RE.match(path.name)
        if match:
            station_ids.add(match.group("station"))
    return sorted(station_ids)


def station_grid_indices(
    lon: np.ndarray,
    lat: np.ndarray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    res: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest 0.05-deg grid-center indices for each station.

    A station is assigned to the cell whose center is the nearest multiple of
    ``res`` inside the requested range.  Stations outside the range are
    excluded (index == -1).
    """
    lat_idx = np.floor((lat - lat_min) / res + 0.5).astype(np.int64)
    lon_idx = np.floor((lon - lon_min) / res + 0.5).astype(np.int64)
    n_lat = int(round((lat_max - lat_min) / res)) + 1
    n_lon = int(round((lon_max - lon_min) / res)) + 1
    valid = (
        (lat_idx >= 0) & (lat_idx < n_lat) & (lon_idx >= 0) & (lon_idx < n_lon)
    )
    lat_idx = np.where(valid, lat_idx, -1)
    lon_idx = np.where(valid, lon_idx, -1)
    return lat_idx, lon_idx


def build_grid_map(
    station_info: pd.DataFrame,
    data_root: Path,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    res: float,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    info = station_info.copy()
    info["gnss_station_id"] = info["gnss_station_id"].astype(str)

    available = discover_stations(data_root)
    missing_geo = sorted(set(available) - set(info["gnss_station_id"]))
    if missing_geo:
        print(f"[warning] {len(missing_geo)} stations have no coordinates: "
              f"{missing_geo[:10]}")

    st = info[info["gnss_station_id"].isin(available)].copy()
    st = st.drop_duplicates("gnss_station_id").reset_index(drop=True)
    print(f"[data] stations found={len(available)}, with coordinates={len(st)}")

    lat = st["lat"].to_numpy(dtype=np.float64)
    lon = st["lon"].to_numpy(dtype=np.float64)
    li, ci = station_grid_indices(
        lon, lat, lat_min, lat_max, lon_min, lon_max, res
    )
    st["lat_idx"] = li
    st["lon_idx"] = ci
    st = st[(st["lat_idx"] >= 0) & (st["lon_idx"] >= 0)]
    print(f"[data] stations inside grid range={len(st)}")

    n_lat = int(round((lat_max - lat_min) / res)) + 1
    n_lon = int(round((lon_max - lon_min) / res)) + 1
    lat_centers = np.round(lat_min + np.arange(n_lat) * res, 10)
    lon_centers = np.round(lon_min + np.arange(n_lon) * res, 10)

    grouped: dict[tuple[int, int], list[str]] = {}
    for row in st.itertuples(index=False):
        key = (int(row.lat_idx), int(row.lon_idx))
        grouped.setdefault(key, []).append(row.gnss_station_id)

    rng = np.random.default_rng(seed)
    station_grid: list[list[str]] = [[""] * n_lon for _ in range(n_lat)]
    count_grid = np.zeros((n_lat, n_lon), dtype=np.int32)
    for (i, j), ids in grouped.items():
        ids_sorted = sorted(ids)
        count_grid[i, j] = len(ids_sorted)
        station_grid[i][j] = str(rng.choice(ids_sorted))

    lat_2d, lon_2d = np.meshgrid(lat_centers, lon_centers, indexing="ij")
    out = pd.DataFrame(
        {
            "lat": lat_2d.ravel(),
            "lon": lon_2d.ravel(),
            "n_stations": count_grid.ravel(),
            "mask": count_grid.ravel() == 0,
            "station_id": np.asarray(station_grid, dtype=object).ravel(),
        }
    )
    out["station_id"] = out["station_id"].where(out["n_stations"] > 0, None)

    summary = {
        "res_deg": res,
        "lat_centers": lat_centers.tolist(),
        "lon_centers": lon_centers.tolist(),
        "n_grid_points": int(len(out)),
        "n_stations_available": int(len(available)),
        "n_stations_mapped": int((out["n_stations"] > 0).sum()),
        "cells_with_one_station": int((out["n_stations"] == 1).sum()),
        "cells_with_many_stations": int((out["n_stations"] > 1).sum()),
        "masked_cells": int((out["n_stations"] == 0).sum()),
        "seed": seed,
    }
    return out, summary


def main() -> None:
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--station-info",
        type=Path,
        default=here / "dataset" / "ngl_europe_stations.parquet",
        help="parquet with station_id / lat / lon / height_m",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/"
            "ngl_ztd_all_downloaded/data/top10_2022_2026/"
            "Western_and_central_Europe"
        ),
        help="directory tree containing *.YYYY.trop.zip station files",
    )
    parser.add_argument(
        "--output-map",
        type=Path,
        default=here / "dataset" / "ngl_europe_0p05_station_grid_map.parquet",
    )
    parser.add_argument("--lat-min", type=float, default=36.25)
    parser.add_argument("--lat-max", type=float, default=56.50)
    parser.add_argument("--lon-min", type=float, default=-5.25)
    parser.add_argument("--lon-max", type=float, default=24.25)
    parser.add_argument("--res", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=2021)
    args = parser.parse_args()

    station_info = pd.read_parquet(args.station_info)
    required = {"gnss_station_id", "lat", "lon"}
    if not required.issubset(station_info.columns):
        raise SystemExit(
            f"station-info must contain columns {sorted(required)}; "
            f"got {list(station_info.columns)}"
        )

    out, summary = build_grid_map(
        station_info,
        args.data_root,
        args.lat_min,
        args.lat_max,
        args.lon_min,
        args.lon_max,
        args.res,
        args.seed,
    )

    args.output_map.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output_map, index=False)
    print(f"[output] saved {len(out)} grid rows -> {args.output_map}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
