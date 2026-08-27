#!/usr/bin/env python3
"""Build the US surface-station/GNSS sample index.

The script deliberately keeps the expensive intermediate arrays in ``dataset/cache``
so that the Parquet files can be regenerated without rescanning the raw archives.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import os
import re
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0088
SURF_RE = re.compile(r"surf_(\d{10})\.pkl$")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--surf-dir", type=Path, default=root / "surf_ncep")
    p.add_argument("--ngl-dir", type=Path, default=root / "ngl")
    p.add_argument("--output-dir", type=Path, default=root / "dataset")
    p.add_argument("--radius-km", type=float, default=50.0)
    p.add_argument("--max-neighbors", type=int, default=5)
    p.add_argument("--min-neighbors", type=int, default=3)
    p.add_argument("--history-hours", type=int, default=6)
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--row-group-size", type=int, default=250_000)
    p.add_argument("--force", action="store_true", help="discard caches and rescan")
    return p.parse_args()


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required to write Parquet. Install it with: "
            "python -m pip install pyarrow"
        ) from exc
    return pa, pq


def normalize_lon(values):
    return (np.asarray(values, dtype=np.float64) + 180.0) % 360.0 - 180.0


_US_PATHS = None


def _us_paths():
    """Return individual state/island polygons from Bokeh's offline boundaries."""
    global _US_PATHS
    if _US_PATHS is not None:
        return _US_PATHS
    from bokeh.sampledata.us_states import data
    from matplotlib.path import Path as MplPath

    result = []
    for state in data.values():
        lat = np.asarray(state["lats"], dtype=np.float64)
        lon = np.asarray(state["lons"], dtype=np.float64)
        good = np.isfinite(lat) & np.isfinite(lon)
        edges = np.flatnonzero(np.diff(np.r_[False, good, False]))
        for begin, end in edges.reshape(-1, 2):
            if end - begin < 3:
                continue
            xy = np.column_stack((lon[begin:end], lat[begin:end]))
            result.append(
                (xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max(), MplPath(xy))
            )
    _US_PATHS = result
    return result


def points_in_us(lat, lon):
    """50 states + DC; overseas territories are intentionally excluded."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = normalize_lon(lon)
    inside = np.zeros(lat.size, dtype=bool)
    for xmin, xmax, ymin, ymax, polygon in _us_paths():
        candidate = (
            (~inside) & (lon >= xmin) & (lon <= xmax) & (lat >= ymin) & (lat <= ymax)
        )
        if candidate.any():
            inside[candidate] = polygon.contains_points(
                np.column_stack((lon[candidate], lat[candidate])), radius=1e-10
            )
    return inside


def _scan_surface_file(task):
    path, hour_index = task
    frame = pd.read_pickle(path)[["lat", "lon", "h"]]
    keep = points_in_us(frame["lat"].to_numpy(), frame["lon"].to_numpy())
    selected = frame.loc[keep].copy()
    missing_height = int(selected["h"].isna().sum())
    selected = selected.dropna(subset=["lat", "lon", "h"])
    selected["lon"] = normalize_lon(selected["lon"].to_numpy())
    grouped = selected.groupby(["lat", "lon", "h"], sort=False, dropna=False).size()
    rows = [(*map(float, key), min(int(count), 65535)) for key, count in grouped.items()]
    return hour_index, rows, len(frame), int(keep.sum()), missing_height


@dataclass
class ObservationData:
    keys: np.ndarray
    reports: np.ndarray
    start: pd.Timestamp
    end: pd.Timestamp
    source_file_exists: np.ndarray
    stats: dict


def surface_files_and_axis(surf_dir: Path):
    parsed = []
    for path in surf_dir.glob("surf_*.pkl"):
        match = SURF_RE.search(path.name)
        if match:
            parsed.append((pd.to_datetime(match.group(1), format="%Y%m%d%H", utc=True), path))
    if not parsed:
        raise FileNotFoundError(f"no surf_YYYYMMDDHH.pkl files in {surf_dir}")
    parsed.sort()
    start, end = parsed[0][0], parsed[-1][0]
    n_hours = int((end - start).total_seconds() // 3600) + 1
    files = [(str(path), int((time - start).total_seconds() // 3600)) for time, path in parsed]
    return files, start, end, n_hours


def scan_observations(surf_dir: Path, cache_dir: Path, workers: int, force: bool):
    cache = cache_dir / "observations.npz"
    if cache.exists() and not force:
        x = np.load(cache, allow_pickle=False)
        return ObservationData(
            keys=x["keys"],
            reports=x["reports"],
            start=pd.Timestamp(x["start"].item(), tz="UTC"),
            end=pd.Timestamp(x["end"].item(), tz="UTC"),
            source_file_exists=x["source_file_exists"],
            stats=json.loads(str(x["stats"].item())),
        )

    tasks, start, end, n_hours = surface_files_and_axis(surf_dir)
    reports_by_key: dict[tuple[float, float, float], np.ndarray] = {}
    source_exists = np.zeros(n_hours, dtype=bool)
    stats = {"raw_reports": 0, "us_reports": 0, "us_reports_missing_height": 0}

    def consume(results):
        for done, (hour, rows, raw_n, us_n, missing_h) in enumerate(results, 1):
            source_exists[hour] = True
            stats["raw_reports"] += raw_n
            stats["us_reports"] += us_n
            stats["us_reports_missing_height"] += missing_h
            for lat, lon, height, count in rows:
                key = (lat, lon, height)
                values = reports_by_key.get(key)
                if values is None:
                    values = np.zeros(n_hours, dtype=np.uint16)
                    reports_by_key[key] = values
                values[hour] = count
            if done % 1000 == 0 or done == len(tasks):
                print(
                    f"[surface] {done}/{len(tasks)} files; "
                    f"{len(reports_by_key)} unique target stations",
                    flush=True,
                )

    if workers == 1:
        consume(map(_scan_surface_file, tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            consume(pool.map(_scan_surface_file, tasks, chunksize=8))

    ordered = sorted(reports_by_key)
    keys = np.asarray(ordered, dtype=np.float64)
    reports = np.stack([reports_by_key[key] for key in ordered])
    stats["n_target_stations"] = len(ordered)
    stats["source_files"] = len(tasks)
    stats["source_hours_missing"] = int((~source_exists).sum())
    np.savez_compressed(
        cache,
        keys=keys,
        reports=reports,
        start=np.array(start.tz_localize(None).isoformat()),
        end=np.array(end.tz_localize(None).isoformat()),
        source_file_exists=source_exists,
        stats=np.array(json.dumps(stats)),
    )
    return ObservationData(keys, reports, start, end, source_exists, stats)


def _dms(degrees, minutes, seconds):
    sign = -1.0 if float(degrees) < 0 else 1.0
    return sign * (abs(float(degrees)) + float(minutes) / 60 + float(seconds) / 3600)


def _read_ngl_site(task):
    station_id, archives = task
    for archive in archives:
        try:
            with zipfile.ZipFile(archive) as outer:
                names = outer.namelist()
                if not names:
                    continue
                with gzip.GzipFile(fileobj=io.BytesIO(outer.read(names[0]))) as inner:
                    in_site = False
                    for raw in inner:
                        line = raw.decode("ascii", errors="replace").strip()
                        if line == "+SITE/ID":
                            in_site = True
                        elif line == "-SITE/ID":
                            break
                        elif in_site and line and not line.startswith("*"):
                            fields = line.split()
                            if len(fields) >= 11:
                                lon = _dms(*fields[-7:-4])
                                lat = _dms(*fields[-4:-1])
                                height = float(fields[-1])
                                return station_id, lat, float(normalize_lon([lon])[0]), height
        except (OSError, ValueError, zipfile.BadZipFile, gzip.BadGzipFile):
            continue
    return station_id, np.nan, np.nan, np.nan


def scan_ngl_catalog(ngl_dir: Path, cache_dir: Path, workers: int, force: bool):
    cache = cache_dir / "ngl_stations.csv"
    if cache.exists() and not force:
        return pd.read_csv(cache)
    tasks = []
    for directory in sorted(path for path in ngl_dir.iterdir() if path.is_dir()):
        archives = sorted(directory.glob("*.trop.zip"), reverse=True)
        if archives:
            tasks.append((directory.name, [str(x) for x in archives]))
    rows = []
    if workers == 1:
        results = map(_read_ngl_site, tasks)
        for done, row in enumerate(results, 1):
            rows.append(row)
            if done % 500 == 0 or done == len(tasks):
                print(f"[gnss-catalog] {done}/{len(tasks)} stations", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = pool.map(_read_ngl_site, tasks, chunksize=8)
            for done, row in enumerate(results, 1):
                rows.append(row)
                if done % 500 == 0 or done == len(tasks):
                    print(f"[gnss-catalog] {done}/{len(tasks)} stations", flush=True)
    result = pd.DataFrame(rows, columns=["gnss_station_id", "lat", "lon", "height_m"])
    result.to_csv(cache, index=False)
    return result


def initial_bearing(lat1, lon1, lat2, lon2):
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    delta = np.radians(lon2 - lon1)
    y = np.sin(delta) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta)
    return float((np.degrees(np.arctan2(y, x)) + 360.0) % 360.0)


def build_neighbors(obs: ObservationData, ngl: pd.DataFrame, radius_km: float, max_neighbors: int):
    from sklearn.neighbors import BallTree

    valid_ngl = ngl.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    tree = BallTree(np.radians(valid_ngl[["lat", "lon"]].to_numpy()), metric="haversine")
    target_xy = np.radians(obs.keys[:, :2])
    indices, distances = tree.query_radius(
        target_xy, r=radius_km / EARTH_RADIUS_KM, return_distance=True, sort_results=True
    )
    rows = []
    selected: list[list[str]] = []
    counts = np.zeros(len(obs.keys), dtype=np.uint16)
    for target_idx, (idx, dist) in enumerate(zip(indices, distances)):
        counts[target_idx] = min(len(idx), 65535)
        chosen = []
        target_lat, target_lon, target_h = obs.keys[target_idx]
        for rank, (ngl_idx, angular_distance) in enumerate(zip(idx[:max_neighbors], dist[:max_neighbors]), 1):
            station = valid_ngl.iloc[int(ngl_idx)]
            chosen.append(station.gnss_station_id)
            rows.append(
                {
                    "target_index": target_idx,
                    "gnss_station_id": station.gnss_station_id,
                    "rank": rank,
                    "distance_km": float(angular_distance * EARTH_RADIUS_KM),
                    "bearing_deg": initial_bearing(target_lat, target_lon, station.lat, station.lon),
                    "height_difference_m": float(station.height_m - target_h),
                }
            )
        selected.append(chosen)
    return pd.DataFrame(rows), selected, counts


def _parse_epoch(epoch: str):
    year, day, seconds = map(int, epoch.split(":"))
    year += 2000 if year < 80 else 1900
    return pd.Timestamp(year=year, month=1, day=1, tz="UTC") + pd.Timedelta(
        days=day - 1, seconds=seconds
    )


def _ngl_station_validity(task):
    station_id, station_dir, start_iso, end_iso, history_hours = task
    start, end = pd.Timestamp(start_iso), pd.Timestamp(end_iso)
    n_hours = int((end - start).total_seconds() // 3600) + 1
    raw_valid = np.zeros(n_hours, dtype=np.uint8)
    years = range(start.year, end.year + 1)
    for year in years:
        archive = Path(station_dir) / f"{station_id}.{year}.trop.zip"
        if not archive.exists():
            continue
        try:
            with zipfile.ZipFile(archive) as outer:
                for name in outer.namelist():
                    try:
                        with gzip.GzipFile(fileobj=io.BytesIO(outer.read(name))) as inner:
                            in_solution = False
                            for raw in inner:
                                line = raw.decode("ascii", errors="replace").strip()
                                if line == "+TROP/SOLUTION":
                                    in_solution = True
                                    continue
                                if line == "-TROP/SOLUTION":
                                    break
                                if not in_solution or not line or line.startswith("*"):
                                    continue
                                fields = line.split()
                                if len(fields) < 5:
                                    continue
                                seconds = int(fields[1].rsplit(":", 1)[1])
                                if seconds % 3600:
                                    continue
                                ztd, zwd = float(fields[2]), float(fields[4])
                                if not (math.isfinite(ztd) and math.isfinite(zwd)):
                                    continue
                                time = _parse_epoch(fields[1])
                                hour = int((time - start).total_seconds() // 3600)
                                if 0 <= hour < n_hours:
                                    raw_valid[hour] = 1
                    except (OSError, ValueError, gzip.BadGzipFile):
                        continue
        except (OSError, zipfile.BadZipFile):
            continue
    width = history_hours + 1
    cumulative = np.r_[0, np.cumsum(raw_valid, dtype=np.int64)]
    window_sum = cumulative[width:] - cumulative[:-width]
    valid = np.zeros(n_hours, dtype=bool)
    valid[width - 1 :] = window_sum == width
    return station_id, np.packbits(valid)


def build_gnss_validity(
    selected, ngl_dir: Path, start: pd.Timestamp, end: pd.Timestamp, history_hours: int,
    cache_dir: Path, workers: int, force: bool,
):
    cache = cache_dir / "gnss_validity.npz"
    needed = sorted({station for stations in selected for station in stations})
    n_hours = int((end - start).total_seconds() // 3600) + 1
    if cache.exists() and not force:
        x = np.load(cache, allow_pickle=False)
        cached_ids = x["station_ids"].astype(str).tolist()
        if cached_ids == needed:
            return needed, np.unpackbits(x["packed"], axis=1, count=n_hours).astype(bool)
    tasks = [
        (station, str(ngl_dir / station), start.isoformat(), end.isoformat(), history_hours)
        for station in needed
    ]
    rows = []
    if workers == 1:
        results = map(_ngl_station_validity, tasks)
        for done, row in enumerate(results, 1):
            rows.append(row)
            if done % 50 == 0 or done == len(tasks):
                print(f"[gnss-validity] {done}/{len(tasks)} stations", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = pool.map(_ngl_station_validity, tasks, chunksize=1)
            for done, row in enumerate(results, 1):
                rows.append(row)
                if done % 50 == 0 or done == len(tasks):
                    print(f"[gnss-validity] {done}/{len(tasks)} stations", flush=True)
    row_map = dict(rows)
    packed = np.stack([row_map[x] for x in needed]) if needed else np.empty((0, 0), np.uint8)
    np.savez_compressed(cache, station_ids=np.asarray(needed), packed=packed)
    return needed, np.unpackbits(packed, axis=1, count=n_hours).astype(bool)


def write_outputs(args, obs, ngl, neighbors, selected, nearby_counts, gnss_ids, gnss_valid):
    pa, pq = _require_pyarrow()
    out = args.output_dir
    station_ids = np.asarray([f"ncep_{i + 1:05d}" for i in range(len(obs.keys))])
    first = np.argmax(obs.reports > 0, axis=1)
    last = obs.reports.shape[1] - 1 - np.argmax((obs.reports > 0)[:, ::-1], axis=1)

    station_table = pd.DataFrame(
        {
            "target_station_id": station_ids,
            "lat": obs.keys[:, 0],
            "lon": obs.keys[:, 1],
            "height_m": obs.keys[:, 2],
            "first_report_utc": obs.start + pd.to_timedelta(first, unit="h"),
            "last_report_utc": obs.start + pd.to_timedelta(last, unit="h"),
            "n_reports_total": obs.reports.sum(axis=1, dtype=np.uint64),
            "n_gnss_50km": nearby_counts,
        }
    )
    pq.write_table(pa.Table.from_pandas(station_table, preserve_index=False), out / "target_stations.parquet")
    ngl_out = ngl.copy()
    pq.write_table(pa.Table.from_pandas(ngl_out, preserve_index=False), out / "gnss_stations.parquet")

    neighbors = neighbors.copy()
    neighbors.insert(0, "target_station_id", station_ids[neighbors.pop("target_index").to_numpy()])
    pq.write_table(pa.Table.from_pandas(neighbors, preserve_index=False), out / "target_gnss_neighbors.parquet")

    id_to_valid = {station: gnss_valid[i] for i, station in enumerate(gnss_ids)}
    sample_path = out / "sample_index.parquet"
    writer = None
    sample_id = 0
    reason_values = np.asarray(
        [
            "", "missing_target_file", "fewer_than_3_gnss_within_50km",
            "missing_target_file;fewer_than_3_gnss_within_50km",
            "fewer_than_3_valid_gnss", "missing_target_file;fewer_than_3_valid_gnss",
        ], dtype=object,
    )
    try:
        for i, target_id in enumerate(station_ids):
            begin, finish = int(first[i]), int(last[i]) + 1
            for chunk_begin in range(begin, finish, args.row_group_size):
                chunk_end = min(finish, chunk_begin + args.row_group_size)
                hour_idx = np.arange(chunk_begin, chunk_end)
                report_count = obs.reports[i, chunk_begin:chunk_end]
                target_exists = report_count > 0
                static_ok = nearby_counts[i] >= args.min_neighbors
                if selected[i]:
                    valid_count = np.sum(
                        np.stack([id_to_valid[x][chunk_begin:chunk_end] for x in selected[i]]), axis=0
                    ).astype(np.uint8)
                else:
                    valid_count = np.zeros(chunk_end - chunk_begin, dtype=np.uint8)
                input_valid = static_ok & (valid_count >= args.min_neighbors)
                if not static_ok:
                    reason_code = np.where(target_exists, 2, 3)
                else:
                    reason_code = np.where(
                        input_valid, np.where(target_exists, 0, 1), np.where(target_exists, 4, 5)
                    )
                n = chunk_end - chunk_begin
                frame = pd.DataFrame(
                    {
                        "sample_id": np.arange(sample_id, sample_id + n, dtype=np.uint64),
                        "target_station_id": target_id,
                        "time_utc": obs.start + pd.to_timedelta(hour_idx, unit="h"),
                        "target_source_exists": target_exists.astype(np.uint8),
                        "n_reports": report_count,
                        "n_gnss_50km": np.full(n, nearby_counts[i], dtype=np.uint16),
                        "n_gnss_valid": valid_count,
                        "input_valid": input_valid.astype(np.uint8),
                        "split": "train",
                        "exclusion_reason": reason_values[reason_code],
                    }
                )
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(sample_path, table.schema, compression="zstd")
                writer.write_table(table, row_group_size=args.row_group_size)
                sample_id += n
            if (i + 1) % 100 == 0 or i + 1 == len(station_ids):
                print(
                    f"[sample-index] {i + 1}/{len(station_ids)} stations; "
                    f"{sample_id} rows",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.close()
    return station_table, sample_id


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    obs = scan_observations(args.surf_dir, cache_dir, args.workers, args.force)
    ngl = scan_ngl_catalog(args.ngl_dir, cache_dir, args.workers, args.force)
    neighbors, selected, nearby_counts = build_neighbors(
        obs, ngl, args.radius_km, args.max_neighbors
    )
    gnss_ids, gnss_valid = build_gnss_validity(
        selected, args.ngl_dir, obs.start, obs.end, args.history_hours,
        cache_dir, args.workers, args.force,
    )
    station_table, n_samples = write_outputs(
        args, obs, ngl, neighbors, selected, nearby_counts, gnss_ids, gnss_valid
    )
    metadata = {
        "coverage": "50 US states and District of Columbia; overseas territories excluded",
        "candidate_rule": "hourly from each station's first through last report, inclusive",
        "station_identity": "exact normalized longitude, latitude, and finite elevation",
        "missing_elevation_reports_excluded": obs.stats["us_reports_missing_height"],
        "gnss_station_valid_rule": (
            f"finite ZTD/ZWD at all {args.history_hours + 1} hourly timestamps from T-"
            f"{args.history_hours} through T"
        ),
        "radius_km": args.radius_km,
        "max_neighbors": args.max_neighbors,
        "min_neighbors": args.min_neighbors,
        "split_rule": "all train (temporal split not yet specified)",
        "start_utc": obs.start.isoformat(),
        "end_utc": obs.end.isoformat(),
        "n_target_stations": len(station_table),
        "n_samples": n_samples,
        "observation_scan": obs.stats,
    }
    (args.output_dir / "sample_index_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
