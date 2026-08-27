#!/usr/bin/env python3
"""Convert NGL TRO archives to one hourly, restartable Zarr store."""

from __future__ import annotations

import argparse
import gzip
import io
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ngl-dir", type=Path, default=root / "ngl")
    parser.add_argument("--output", type=Path, default=root / "dataset" / "ngl_hourly.zarr")
    parser.add_argument("--start", default="2017-12-31T00:00:00Z")
    parser.add_argument("--end", default="2024-09-01T00:00:00Z")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true", help="replace an existing store")
    return parser.parse_args()


def parse_epoch(epoch: str) -> np.datetime64:
    year, day, seconds = map(int, epoch.split(":"))
    year += 2000 if year < 80 else 1900
    date = np.datetime64(f"{year:04d}-01-01T00", "h")
    return date + np.timedelta64(day - 1, "D") + np.timedelta64(seconds, "s")


def parse_station(task):
    station_id, station_dir, start_ns, end_ns, n_hours = task
    start = np.datetime64(start_ns, "ns")
    end = np.datetime64(end_ns, "ns")
    ztd = np.full(n_hours, np.nan, dtype=np.float32)
    zwd = np.full(n_hours, np.nan, dtype=np.float32)
    hourly_records = 0
    for year in range(pd.Timestamp(start_ns).year, pd.Timestamp(end_ns).year + 1):
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
                                if seconds % 3600 != 0:
                                    continue
                                time = parse_epoch(fields[1])
                                if time < start or time > end:
                                    continue
                                total, wet = float(fields[2]), float(fields[4])
                                hour = int((time - start) / np.timedelta64(1, "h"))
                                ztd[hour] = total
                                zwd[hour] = wet
                                hourly_records += 1
                    except (OSError, gzip.BadGzipFile):
                        continue
        except (OSError, zipfile.BadZipFile):
            continue
    return station_id, ztd, zwd, hourly_records


def create_store(path: Path, station_ids, times, force: bool):
    if path.exists() and force:
        import shutil

        shutil.rmtree(path)
    if path.exists():
        group = zarr.open_group(str(path), mode="a")
        expected_start = np.datetime_as_string(times[0], unit="s")
        expected_end = np.datetime_as_string(times[-1], unit="s")
        if (
            group.attrs.get("start_utc") != expected_start
            or group.attrs.get("end_utc") != expected_end
            or group["station"].shape[0] != len(station_ids)
            or group["time"].shape[0] != len(times)
        ):
            raise RuntimeError("existing store axes differ; rerun with --force")
        if group["station"][:].astype(str).tolist() != list(station_ids):
            raise RuntimeError("existing station order differs; rerun with --force")
        return group

    path.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(path), mode="w")
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    n_time, n_station = len(times), len(station_ids)
    station_width = max(map(len, station_ids))
    group.create_dataset("time", data=times, chunks=(n_time,), compressor=compressor)
    group.create_dataset(
        "station", data=np.asarray(station_ids, dtype=f"<U{station_width}"),
        chunks=(n_station,), compressor=compressor,
    )
    for variable in ("ztd", "zwd"):
        array = group.create_dataset(
            variable,
            shape=(n_time, n_station),
            chunks=(min(8760, n_time), 1),
            dtype="f4",
            fill_value=np.nan,
            compressor=compressor,
        )
        array.attrs.update({"_ARRAY_DIMENSIONS": ["time", "station"], "units": "mm"})
    group["time"].attrs.update(
        {"_ARRAY_DIMENSIONS": ["time"], "standard_name": "time", "timezone": "UTC"}
    )
    group["station"].attrs.update({"_ARRAY_DIMENSIONS": ["station"]})
    complete = group.create_dataset(
        "_station_complete", shape=(n_station,), chunks=(n_station,), dtype="bool", fill_value=False
    )
    complete.attrs["description"] = "restart marker; removed after successful completion"
    group.attrs.update(
        {
            "title": "NGL hourly zenith tropospheric delays",
            "start_utc": np.datetime_as_string(times[0], unit="s"),
            "end_utc": np.datetime_as_string(times[-1], unit="s"),
            "time_sampling": "1 hour; exact UTC hour records only",
            "source_fields": "TROTOT -> ztd; TRWET -> zwd",
            "units": "millimetres",
            "complete": False,
        }
    )
    return group


def main():
    args = parse_args()
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if start.tzinfo is None or end.tzinfo is None:
        raise SystemExit("--start and --end must include a timezone")
    start, end = start.tz_convert("UTC"), end.tz_convert("UTC")
    if end < start:
        raise SystemExit("--end must not precede --start")
    times = pd.date_range(start, end, freq="h").tz_localize(None).to_numpy(dtype="datetime64[ns]")
    station_ids = sorted(path.name for path in args.ngl_dir.iterdir() if path.is_dir())
    station_ids = [
        station for station in station_ids if any((args.ngl_dir / station).glob("*.trop.zip"))
    ]
    group = create_store(args.output, station_ids, times, args.force)
    if "_station_complete" not in group:
        if group.attrs.get("complete"):
            print(f"already complete: {args.output}")
            return
        raise RuntimeError("incomplete store has no restart marker; rerun with --force")

    completed = group["_station_complete"][:]
    pending_indices = np.flatnonzero(~completed)
    tasks = [
        (
            station_ids[i], str(args.ngl_dir / station_ids[i]), start.to_datetime64(),
            end.to_datetime64(),
            len(times),
        )
        for i in pending_indices
    ]
    station_to_index = {station: i for i, station in enumerate(station_ids)}
    total_hourly_records = int(group.attrs.get("n_hourly_records", 0))

    if args.workers == 1:
        results = map(parse_station, tasks)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=args.workers)
        results = pool.map(parse_station, tasks, chunksize=1)
    try:
        for done, (station, ztd, zwd, hourly_records) in enumerate(results, 1):
            index = station_to_index[station]
            group["ztd"][:, index] = ztd
            group["zwd"][:, index] = zwd
            group["_station_complete"][index] = True
            total_hourly_records += hourly_records
            # Persist counters together with every restart marker so resumed runs
            # cannot under-count the last partial progress block.
            group.attrs.update(
                {
                    "n_hourly_records": total_hourly_records,
                }
            )
            if done % 50 == 0 or done == len(tasks):
                print(
                    f"[ngl-zarr] {completed.sum() + done}/{len(station_ids)} stations; "
                    f"{total_hourly_records} hourly records",
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.shutdown()

    if bool(group["_station_complete"][:].all()):
        del group["_station_complete"]
        group.attrs.update(
            {
                "complete": True,
                "n_stations": len(station_ids),
                "n_times": len(times),
                "n_hourly_records": total_hourly_records,
            }
        )
        zarr.consolidate_metadata(str(args.output))
    print(group.tree())


if __name__ == "__main__":
    main()
