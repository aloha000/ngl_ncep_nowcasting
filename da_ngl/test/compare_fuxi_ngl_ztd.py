#!/usr/bin/env python3
"""直接读取 Zarr，比较最近站点格的 FuXi / ERA5 ZTD 与 NGL ZTD（mm）。

NGL Zarr 已将每个保留的 GNSS 站映射到最近的 0.25° 格心；原始站点
经纬度不在 Zarr 中。本脚本用该格心寻找 FuXi 最近格点，并要求两格
站号一致（FuXi 算子使用该站高度）。grid_distance_km 是两格心距离，
不是原始 GNSS 站到格心的距离。不会用邻站的值补缺测。

FuXi 的 time 是有效时刻，直接与 NGL 的同一 UTC 时刻精确配对，
不再加减 24 小时。NGL 通过 store 内 mean/std 反标准化为 mm。
默认测试时段为 [2025-01-01, 2025-10-01)，无需 GPU 或训练配置。
ERA5 由 preprocessing/build_ztd_era5_zarr.py 生成（13 层方法 E）；
默认只在三方同时有效的站点-时刻上统计两种模式，保证样本一致。
--fuxi-only 恢复仅 FuXi/NGL 对比。

用法（任意工作目录）：
    python test/compare_fuxi_ngl_ztd.py --max-times 8 --out-dir /tmp/ztd_smoke
    python test/compare_fuxi_ngl_ztd.py
    python test/compare_fuxi_ngl_ztd.py --start 2022-01-01 --end 2025-10-01

依赖：numpy pandas zarr matplotlib（--no-figure 时不需要 matplotlib）。
输出：pairs.csv、station_metrics.csv、summary.json、comparison.png。
误差统一定义为模式 - NGL；全局指标按共同有效的站点-时间样本加权。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# The project's gnss environment lacks Intel's OpenMP symbols when matplotlib
# loads MKL. Use GNU threading by default; preserve an explicit user choice.
os.environ.setdefault('MKL_THREADING_LAYER', 'GNU')

import numpy as np
import pandas as pd
import zarr

ROOT = Path(__file__).resolve().parents[1]


def time_axis(group) -> pd.DatetimeIndex:
    """Decode the numeric CF UTC time coordinates used by these stores."""
    axis = group['time']
    units = str(axis.attrs.get('units', ''))
    unit, separator, origin = units.partition(' since ')
    codes = {'seconds': 's', 'minutes': 'min', 'hours': 'h', 'days': 'D'}
    if not separator or unit not in codes:
        raise ValueError(f'Unsupported time units: {units!r}')
    values = np.asarray(axis[:])
    if not np.isfinite(values).all():
        raise ValueError('Non-finite time coordinates')
    result = pd.DatetimeIndex(pd.Timestamp(origin, tz='UTC')
                              + pd.to_timedelta(values, unit=codes[unit]))
    if not result.is_unique or not result.is_monotonic_increasing:
        raise ValueError('Time coordinates must be unique and increasing')
    return result


def nearest_cells(obs, fuxi, source='FuXi') -> pd.DataFrame:
    """Use the existing station-to-nearest-cell map, checking station identity."""
    oy, ox = np.where(~np.asarray(obs['mask'][:], dtype=bool))
    ids = np.asarray(obs['station'][:]).astype(str)[oy, ox]
    if (ids == '').any() or len(np.unique(ids)) != len(ids):
        raise ValueError('Occupied NGL cells must have unique, nonempty station IDs')
    olat, olon = np.asarray(obs['lat'][:])[oy], np.asarray(obs['lon'][:])[ox]
    flat, flon = np.asarray(fuxi['lat'][:]), np.asarray(fuxi['lon'][:])
    # Same rectilinear nearest-coordinate convention as map_stations_to_grid.py.
    fy = np.abs(olat[:, None] - flat[None, :]).argmin(axis=1)
    lon_delta = (olon[:, None] - flon[None, :] + 180) % 360 - 180
    fx = np.abs(lon_delta).argmin(axis=1)
    fids = np.asarray(fuxi['station'][:]).astype(str)[fy, fx]
    bad = (fids != ids) | np.asarray(fuxi['mask'][:], dtype=bool)[fy, fx]
    if bad.any():
        raise ValueError(f'{bad.sum()} nearest {source} cells belong to different/no '
                         f'stations, e.g. {ids[bad][:5].tolist()}; '
                         'cannot compare a station-height ZTD from another station')
    lat1, lat2 = np.radians(olat), np.radians(flat[fy])
    dlon = np.radians((flon[fx] - olon + 180) % 360 - 180)
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    distance = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return pd.DataFrame(dict(station_id=ids, obs_iy=oy, obs_ix=ox,
                             obs_grid_lat=olat, obs_grid_lon=olon,
                             fuxi_iy=fy, fuxi_ix=fx,
                             fuxi_grid_lat=flat[fy], fuxi_grid_lon=flon[fx],
                             grid_distance_km=distance))


def moments(obs: np.ndarray, model: np.ndarray) -> np.ndarray:
    """One sufficient-statistics row per station; ignore either-source NaNs/inf."""
    valid = np.isfinite(obs) & np.isfinite(model)
    x, y = np.where(valid, obs, 0), np.where(valid, model, 0)
    error = y - x
    return np.stack([valid.sum(0), error.sum(0), np.abs(error).sum(0),
                     (error**2).sum(0), x.sum(0), y.sum(0),
                     (x*x).sum(0), (y*y).sum(0), (x*y).sum(0)], axis=1)


def metrics(row: np.ndarray) -> dict:
    n, error, absolute, squared, sx, sy, sxx, syy, sxy = row
    if not n:
        return dict(n=0, bias_mm=None, mae_mm=None, rmse_mm=None, correlation=None)
    vx, vy = max(0, sxx - sx*sx/n), max(0, syy - sy*sy/n)
    corr = (sxy - sx*sy/n) / np.sqrt(vx*vy) if n > 1 and vx > 1e-8 and vy > 1e-8 else None
    return dict(n=int(n), bias_mm=float(error/n), mae_mm=float(absolute/n),
                rmse_mm=float(np.sqrt(squared/n)),
                correlation=None if corr is None else float(np.clip(corr, -1, 1)))


def plot_comparison(sample: np.ndarray, overalls: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    obs = sample[:, 1]
    fig, axes = plt.subplots(len(overalls), 2, figsize=(11, 4.5*len(overalls)), dpi=150, squeeze=False)
    low, high = sample[:, 1:].min(), sample[:, 1:].max()
    errors = sample[:, 2:] - obs[:, None]
    bins = np.linspace(errors.min()-1e-6, errors.max()+1e-6, 101)
    for i, (source, overall) in enumerate(overalls.items()):
        model = sample[:, i+2]
        name = {'fuxi': 'FuXi', 'era5': 'ERA5 13L'}[source]
        ax, hist = axes[i]
        ax.hexbin(obs, model, gridsize=65, mincnt=1, bins='log', cmap='viridis')
        ax.plot([low, high], [low, high], 'k--', lw=1)
        ax.set(xlabel='NGL ZTD (mm)', ylabel=f'{name} ZTD (mm)',
               xlim=(low, high), ylim=(low, high), title=f'{name} vs NGL')
        hist.hist(model-obs, bins=bins, color='steelblue' if i == 0 else 'darkorange')
        hist.axvline(0, color='k', linestyle='--', lw=1)
        hist.set(xlabel=f'{name} - NGL (mm)', ylabel='Sample count',
                 title=f"All pairs: n={overall['n']:,}, bias={overall['bias_mm']:.3f} mm\n"
                       f"MAE={overall['mae_mm']:.3f}, RMSE={overall['rmse_mm']:.3f} mm")
    fig.suptitle(f'Common valid station-time samples; plot sample: {len(sample):,}')
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fuxi-zarr', type=Path, default=ROOT/'dataset/ztd_fuxi_europe_0p25_24h_zdz.zarr')
    parser.add_argument('--obs-zarr', type=Path, default=ROOT/'dataset/ngl_europe_0p25_5min.zarr')
    parser.add_argument('--era5-zarr', type=Path, default=ROOT/'dataset/ztd_era5_europe_0p25_zdz.zarr')
    parser.add_argument('--fuxi-only', action='store_true', help='disable ERA5 comparison')
    parser.add_argument('--start', default='2025-01-01', help='inclusive UTC start')
    parser.add_argument('--end', default='2025-10-01', help='exclusive UTC end')
    parser.add_argument('--max-times', type=int, default=0, help='use first N matched times; 0 = all')
    parser.add_argument('--block-size', type=int, default=64, help='time frames per read block')
    parser.add_argument('--out-dir', type=Path, help='default test/fuxi_era5_ngl_ztd_comparison')
    parser.add_argument('--no-figure', action='store_true')
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = ROOT/'test'/('fuxi_ngl_ztd_comparison' if args.fuxi_only else 'fuxi_era5_ngl_ztd_comparison')
    if args.max_times < 0 or args.block_size < 1:
        parser.error('--max-times must be >= 0 and --block-size must be >= 1')
    start, end = pd.to_datetime(args.start, utc=True), pd.to_datetime(args.end, utc=True)
    if start >= end:
        parser.error('--start must be before --end')

    fuxi = zarr.open_group(str(args.fuxi_zarr), mode='r')
    obs = zarr.open_group(str(args.obs_zarr), mode='r')
    era5 = None
    if not args.fuxi_only:
        if not args.era5_zarr.exists():
            parser.error(f'Missing ERA5 ZTD: {args.era5_zarr}; run preprocessing/build_ztd_era5_zarr.py first, '
                         'or use --fuxi-only')
        era5 = zarr.open_group(str(args.era5_zarr), mode='r')
        if era5.attrs.get('build_complete') is not True:
            raise ValueError('ERA5 ZTD build is incomplete')
    groups = [(fuxi, 'ztd_fuxi'), (obs, 'ztd')]
    if era5 is not None:
        groups.append((era5, 'ztd_era5'))
    for group, variable in groups:
        shape = tuple(group[name].shape[0] for name in ('time', 'lat', 'lon'))
        if group[variable].shape != shape:
            raise ValueError(f'{variable} must have shape [time, lat, lon]')
    if fuxi['ztd_fuxi'].attrs.get('units') != 'mm':
        raise ValueError('Expected FuXi ztd_fuxi units mm')
    if era5 is not None and era5['ztd_era5'].attrs.get('units') != 'mm':
        raise ValueError('Expected ERA5 ztd_era5 units mm')
    if obs['ztd'].attrs.get('units') != '1':
        raise ValueError('Expected standardized NGL ztd (units=1)')
    mean, std = float(obs['ztd_train_mean'][0]), float(obs['ztd_train_std'][0])
    if not np.isfinite([mean, std]).all() or std <= 0:
        raise ValueError('Invalid NGL normalization statistics')
    mapping = nearest_cells(obs, fuxi)
    if era5 is not None:
        emap = nearest_cells(obs, era5, source='ERA5')
        assert np.array_equal(mapping.station_id, emap.station_id)
        for name in ('fuxi_iy', 'fuxi_ix', 'fuxi_grid_lat', 'fuxi_grid_lon', 'grid_distance_km'):
            target = name.replace('fuxi', 'era5') if name.startswith('fuxi') else 'era5_grid_distance_km'
            mapping[target] = emap[name]
    if mapping.empty:
        raise ValueError('No occupied station cells')
    ft, ot = time_axis(fuxi), time_axis(obs)
    selected = np.flatnonzero((ft >= start) & (ft < end))
    obs_indices = ot.get_indexer(ft[selected])
    missing_times = int((obs_indices < 0).sum())
    keep = obs_indices >= 0
    era_indices = np.zeros(len(selected), dtype=int)
    missing_era_times = 0
    if era5 is not None:
        era_indices = time_axis(era5).get_indexer(ft[selected])
        missing_era_times = int((era_indices < 0).sum())
        keep &= era_indices >= 0
    selected, obs_indices = selected[keep], obs_indices[keep]
    era_indices = era_indices[keep]
    if args.max_times:
        selected, obs_indices = selected[:args.max_times], obs_indices[:args.max_times]
        era_indices = era_indices[:args.max_times]
    if not len(selected):
        raise ValueError('No exact matching times in requested interval')

    print(f'[match] {len(mapping)} stations, {len(selected)} times; '
          f'{missing_times} times without NGL, {missing_era_times} without ERA5', flush=True)
    print(f'[units] NGL mm = stored_value * {std} + {mean}', flush=True)
    print('[space] existing nearest-station grid mapping; station IDs verified', flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sources = ['fuxi'] if era5 is None else ['fuxi', 'era5']
    totals = {source: np.zeros((len(mapping), 9), dtype=np.float64) for source in sources}
    oy, ox = mapping.obs_iy.to_numpy(), mapping.obs_ix.to_numpy()
    fy, fx = mapping.fuxi_iy.to_numpy(), mapping.fuxi_ix.to_numpy()
    if era5 is not None:
        ey, ex = mapping.era5_iy.to_numpy(), mapping.era5_ix.to_numpy()
    sample = np.empty((0, 2+len(sources)))
    rng = np.random.default_rng(2026)
    counters = dict(candidate_pairs=0, invalid_obs=0, invalid_fuxi=0, invalid_either=0)
    if era5 is not None:
        counters['invalid_era5'] = 0
    # Stream output and accumulate metrics so full multi-year comparisons fit in memory.
    with (args.out_dir/'pairs.csv').open('w', encoding='utf-8', newline='') as handle:
        header = True
        for offset in range(0, len(selected), args.block_size):
            fi, oi = selected[offset:offset+args.block_size], obs_indices[offset:offset+args.block_size]
            # Slice full spatial planes, then paired numpy indexing; Zarr oindex
            # with separate iy/ix arrays would form an unintended outer product.
            model = np.asarray(fuxi['ztd_fuxi'].oindex[fi, :, :], dtype=np.float64)[:, fy, fx]
            observed = np.asarray(obs['ztd'].oindex[oi, :, :], dtype=np.float64)[:, oy, ox]
            observed = observed * std + mean
            valid = np.isfinite(model) & np.isfinite(observed)
            models = {'fuxi': model}
            if era5 is not None:
                ei = era_indices[offset:offset+args.block_size]
                models['era5'] = np.asarray(era5['ztd_era5'].oindex[ei, :, :], dtype=np.float64)[:, ey, ex]
                valid &= np.isfinite(models['era5'])
                counters['invalid_era5'] += int((~np.isfinite(models['era5'])).sum())
            for source, values in models.items():
                totals[source] += moments(np.where(valid, observed, np.nan), values)
            counters['candidate_pairs'] += int(valid.size)
            counters['invalid_obs'] += int((~np.isfinite(observed)).sum())
            counters['invalid_fuxi'] += int((~np.isfinite(model)).sum())
            counters['invalid_either'] += int((~valid).sum())
            ti, si = np.where(valid)
            pairs = mapping.iloc[si].reset_index(drop=True).copy()
            pairs.insert(0, 'time_utc', ft[fi[ti]].strftime('%Y-%m-%dT%H:%M:%SZ'))
            pairs['obs_ztd_mm'], pairs['fuxi_ztd_mm'] = observed[valid], model[valid]
            pairs['error_mm'] = model[valid] - observed[valid]
            for source, values in models.items():
                pairs[f'{source}_ztd_mm'] = values[valid]
                pairs[f'{source}_error_mm'] = values[valid] - observed[valid]
            pairs.to_csv(handle, index=False, header=header, float_format='%.8f')
            header = False
            if not args.no_figure:
                # Priority reservoir: uniform bounded sample for plotting only.
                batch = np.column_stack([rng.random(len(ti)), observed[valid],
                                         *[values[valid] for values in models.values()]])
                sample = np.concatenate([sample, batch])
                if len(sample) > 50000:
                    sample = sample[np.argpartition(sample[:, 0], 49999)[:50000]]
            print(f'[read] {min(offset+len(fi), len(selected))}/{len(selected)} times; '
                  f'common valid pairs={int(totals["fuxi"][:, 0].sum()):,}', flush=True)

    overalls = {source: metrics(total.sum(0)) for source, total in totals.items()}
    station_metrics = pd.concat([
        pd.concat([mapping, pd.DataFrame([metrics(row) for row in totals[source]])], axis=1).assign(source=source)
        for source in sources], ignore_index=True)
    station_metrics.to_csv(args.out_dir/'station_metrics.csv', index=False)
    summary = dict(
        fuxi_zarr=str(args.fuxi_zarr.resolve()), obs_zarr=str(args.obs_zarr.resolve()),
        requested_start=str(start), requested_end_exclusive=str(end),
        matched_start=str(ft[selected[0]]), matched_end=str(ft[selected[-1]]),
        matched_times=len(selected), max_times=args.max_times,
        stations=len(mapping), stations_with_valid_pairs=int((totals['fuxi'][:, 0] > 0).sum()),
        unmatched_fuxi_times=missing_times, **counters,
        unmatched_era5_times=missing_era_times,
        era5_zarr=None if era5 is None else str(args.era5_zarr.resolve()),
        era5_operator=None if era5 is None else era5.attrs.get('operator'),
        valid_sample_policy='All selected sources finite on the same station and exact UTC time',
        obs_mean_mm=mean, obs_std_mm=std,
        matching='Exact UTC valid time; existing NGL nearest-grid station assignment; '
                 'nearest model grid coordinates with identical station ID',
        distance_note='grid_distance_km is grid-center to grid-center, not station to grid',
        fuxi_background_metadata=fuxi.attrs.get('background'),
        error_definition='model - NGL (mm); error_mm remains an alias of fuxi_error_mm',
        **{f'{source}_vs_ngl': value for source, value in overalls.items()})
    with (args.out_dir/'summary.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(overalls, indent=2), flush=True)
    if not overalls['fuxi']['n']:
        raise SystemExit('No finite matched pairs; see summary.json for missing counts')
    if not args.no_figure:
        plot_comparison(sample, overalls, args.out_dir/'comparison.png')
    print(f'[out] {args.out_dir.resolve()}')


if __name__ == '__main__':
    main()
