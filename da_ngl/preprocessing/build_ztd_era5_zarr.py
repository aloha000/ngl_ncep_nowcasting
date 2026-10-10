#!/usr/bin/env python3
"""Generate ERA5 station-height ZTD with the FuXi builder's 13-level method E.

Uses label[time,channel,lat,lon], its ERA5 mean/std, and the exact station
heights saved in the FuXi ZTD store. No forecast lead shift is applied.
Output: zhd/zwd/ztd_era5 in mm, NaN outside station cells, on label valid times.
Only t/r/z on 13 pressure levels plus t2m/msl enter the operator; tp is unused.

    python preprocessing/build_ztd_era5_zarr.py --mean-file /path/mean_era5.npy \
        --std-file /path/std_era5.npy

Default: all label times. For a small check use --start 2025-01-01
--max-times 8 --out /tmp/era5_ztd_smoke.zarr. Existing outputs are not overwritten.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc

from common import MEAN_STD_DIR, SOURCE_CHANNELS, finalize_store
from ztd_operator import G_0, LEV_HPA, R_D, ztd_profile_zdz

ROOT = Path(__file__).resolve().parents[1]


def station_geometry(label, geometry):
    for name in ('lat', 'lon'):
        if not np.array_equal(label[name][:], geometry[name][:]):
            raise ValueError(f'ERA5 and FuXi geometry {name} coordinates differ')
    station = np.asarray(geometry['station'][:]).astype(str)
    mask = np.asarray(geometry['mask'][:], dtype=bool)
    iy, ix = np.where(~mask)
    ids = station[iy, ix]
    height_ids = geometry['station'].attrs.get('station_id', [])
    heights = np.asarray(geometry['height_m'][:], dtype=np.float64)
    if len(height_ids) != len(heights) or len(set(height_ids)) != len(heights):
        raise ValueError('FuXi height_m needs unique station_id metadata')
    if len(set(ids)) != len(ids) or (ids == '').any():
        raise ValueError('Occupied station IDs must be unique and nonempty')
    height_lookup = dict(zip(height_ids, heights))
    h = np.array([height_lookup[s] for s in ids])
    if not np.isfinite(h).all():
        raise ValueError('Non-finite GNSS station heights')
    return iy, ix, ids, h, mask, station


def compute_block(label, indices, channels, mean, std, iy, ix, heights):
    # Match build_ztd_fuxi_zarr._block, operating only on station cells.
    raw = np.asarray(label.oindex[indices, channels, :, :])[:, :, iy, ix]
    physical = mean[None, :, None] + std[None, :, None] * raw
    nlev = len(LEV_HPA)
    temperature = physical[:, :nlev].transpose(0, 2, 1)
    humidity = physical[:, nlev:2*nlev].transpose(0, 2, 1)
    t2m, msl = physical[:, 2*nlev], physical[:, 2*nlev+1]
    level_heights = (physical[:, 2*nlev+2:] / G_0).transpose(0, 2, 1)
    h = np.broadcast_to(heights[None, :], t2m.shape)
    ps = msl / 100.0 * np.exp(-G_0 * h / (R_D * t2m))
    result = ztd_profile_zdz(temperature, humidity, t2m, ps, h, level_heights)
    return [result[name] for name in ('ZHD_mm', 'ZWD_mm', 'ZTD_mm')]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--label-zarr', type=Path, default=ROOT/'dataset/label_europe_0p25.zarr')
    parser.add_argument('--geometry-zarr', type=Path, default=ROOT/'dataset/ztd_fuxi_europe_0p25_24h_zdz.zarr')
    parser.add_argument('--mean-file', type=Path, default=MEAN_STD_DIR/'mean_era5.npy')
    parser.add_argument('--std-file', type=Path, default=MEAN_STD_DIR/'std_era5.npy')
    parser.add_argument('--out', type=Path, default=ROOT/'dataset/ztd_era5_europe_0p25_zdz.zarr')
    parser.add_argument('--start', help='inclusive UTC start, default all label times')
    parser.add_argument('--end', help='exclusive UTC end')
    parser.add_argument('--max-times', type=int, default=0)
    parser.add_argument('--block', type=int, default=24)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.block < 1 or args.workers < 1 or args.max_times < 0:
        parser.error('block/workers must be positive; max-times must be nonnegative')
    if args.out.exists():
        parser.error(f'Output already exists: {args.out}; use a new --out path')
    for path in (args.mean_file, args.std_file):
        if not path.is_file():
            parser.error(f'Missing normalization file: {path}; specify --mean-file / --std-file')
    label = zarr.open_group(str(args.label_zarr), mode='r')
    geometry = zarr.open_group(str(args.geometry_zarr), mode='r')
    iy, ix, ids, heights, mask, station = station_geometry(label, geometry)
    names = ([f't{int(p)}' for p in LEV_HPA] + [f'r{int(p)}' for p in LEV_HPA]
             + ['t2m', 'msl'] + [f'z{int(p)}' for p in LEV_HPA])
    source_names = list(np.asarray(label['channel'][:]).astype(str))
    channels = [source_names.index(name) for name in names]
    stats_indices = [SOURCE_CHANNELS.index(name) for name in names]
    mean = np.asarray(np.load(args.mean_file, allow_pickle=False), dtype=np.float64).reshape(-1)[stats_indices]
    std = np.asarray(np.load(args.std_file, allow_pickle=False), dtype=np.float64).reshape(-1)[stats_indices]
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError('Invalid ERA5 mean/std')
    unit, separator, origin = label['time'].attrs['units'].partition(' since ')
    units = {'hours': 'h', 'minutes': 'min', 'seconds': 's', 'days': 'D'}
    if not separator or unit not in units:
        raise ValueError('Unsupported label time units')
    times = pd.DatetimeIndex(pd.to_datetime(origin, utc=True)
                            + pd.to_timedelta(label['time'][:], unit=units[unit]))
    if not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError('Label time coordinates must be unique and increasing')
    keep = np.ones(len(times), dtype=bool)
    if args.start:
        keep &= times >= pd.to_datetime(args.start, utc=True)
    if args.end:
        keep &= times < pd.to_datetime(args.end, utc=True)
    selected = np.flatnonzero(keep)
    if args.max_times:
        selected = selected[:args.max_times]
    if not len(selected):
        raise ValueError('No selected ERA5 times')
    expected_shape = (len(times), len(source_names), *mask.shape)
    if label['label'].shape != expected_shape:
        raise ValueError(f'Expected label shape {expected_shape}')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    output = zarr.open_group(str(args.out), mode='w-', zarr_version=2)
    output.attrs.update(dict(
        title='ERA5 station-height ZTD from 13 pressure levels', build_complete=False,
        operator='ztd_profile_zdz (method E), same as build_ztd_fuxi_zarr.py',
        label_source=str(args.label_zarr.resolve()), geometry_source=str(args.geometry_zarr.resolve()),
        mean_file=str(args.mean_file.resolve()), std_file=str(args.std_file.resolve()),
        operator_channels=names, operator_mean=mean.tolist(), operator_std=std.tolist(),
        levels_hpa=LEV_HPA.astype(int).tolist(), units='mm',
        heights='GNSS heights from FuXi height_m, joined by station ID',
        surface_pressure='msl/100 * exp(-g*height_m/(Rd*t2m)), hPa',
        time_semantics='ERA5 analysis valid time; no lead shift', stations=len(ids),
        fill_value='NaN outside station cells or where inputs are invalid'))
    create = getattr(output, 'create_array', output.create_dataset)

    def coordinate(name, data, dimensions, attrs=None):
        data = np.asarray(data)
        a = create(name, shape=data.shape, dtype=data.dtype, chunks=data.shape, fill_value=None)
        a[:] = data
        a.attrs.update({'_ARRAY_DIMENSIONS': dimensions, **(attrs or {})})
        return a

    coordinate('time', np.asarray(label['time'][:])[selected], ['time'], dict(label['time'].attrs))
    for name in ('lat', 'lon'):
        coordinate(name, label[name][:], [name], dict(label[name].attrs))
    coordinate('mask', mask, ['lat', 'lon'])
    coordinate('station', station, ['lat', 'lon'], {'station_id': ids.tolist()})
    coordinate('height_m', heights.astype('f4'), ['station_index'], {'units': 'm'})
    shape = (len(selected), *mask.shape)
    arrays = []
    for name in ('zhd', 'zwd', 'ztd_era5'):
        a = create(name, shape=shape, chunks=(min(args.block, len(selected)), *mask.shape),
                   dtype='f4', fill_value=np.nan,
                   compressor=Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE))
        a.attrs.update({'_ARRAY_DIMENSIONS': ['time', 'lat', 'lon'], 'units': 'mm'})
        arrays.append(a)
    blocks = [(i, min(i+args.block, len(selected))) for i in range(0, len(selected), args.block)]
    print(f'[build] {len(selected)} times x {len(ids)} stations; {args.workers} workers', flush=True)
    # Bounded queue: at most workers blocks in flight, a single Zarr writer.
    finite = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {}

        def submit(k):
            begin, end = blocks[k]
            pending[k] = pool.submit(compute_block, label['label'], selected[begin:end],
                                     channels, mean, std, iy, ix, heights)

        for k in range(min(args.workers, len(blocks))):
            submit(k)
        for k, (begin, end) in enumerate(blocks):
            values = pending.pop(k).result()
            if k+args.workers < len(blocks):
                submit(k+args.workers)
            for a, value in zip(arrays, values):
                field = np.full((end-begin, *mask.shape), np.nan, dtype='f4')
                field[:, iy, ix] = value
                a[begin:end] = field
            finite += int(np.isfinite(values[2]).sum())
            print(f'[build] {end}/{len(selected)} times; finite station ZTD={finite:,}', flush=True)
    output.attrs.update({'build_complete': True, 'finite_station_ztd': finite})
    finalize_store(args.out)
    print(f'[out] {args.out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
