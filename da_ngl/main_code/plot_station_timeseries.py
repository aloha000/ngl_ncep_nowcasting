#!/usr/bin/env python3
"""Compare best-checkpoint analyses, ERA5 and FuXi-24h at random NGL station cells.

Run from main_code; --stats-dir must contain the original mean_era5/std_era5.npy.
PNG/PDF plots, physical-unit CSVs, normalized station predictions and provenance
are saved in a new output directory. Existing prediction files are not overwritten.
The test split currently overlaps validation; these are analyses, not free forecasts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('MKL_THREADING_LAYER', 'GNU')
os.environ['WANDB_MODE'] = 'disabled'

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import torch
import zarr

from eval_results import (EvaluationDataset, evaluation_metadata, load_model,
                          pick_checkpoint, set_eval_precision)
from main.utils import apply_overrides, run_dir
from main.utils.utils_data import LABEL_IDX

LEVELS = (850, 700, 500)
SOURCES = ('ERA5', 'FuXi-24h', 'Analysis')
COLORS = ('#222222', '#e58b27', '#1976c9')


def physical_variables(values, channels, mean, std):
    """Input (time, station, channel); output physical-unit scalar series.

    ERA5 r is percent, z is geopotential (m2/s2); convert z / g0 to gpm.
    Do not clip RH or any output: clipping would hide model errors.
    """
    field = np.asarray(values, dtype=np.float64) * std + mean
    ci = {name: index for index, name in enumerate(channels)}
    out = {'msl': field[..., ci['msl']] / 100.0}
    for level in LEVELS:
        out[f't{level}'] = field[..., ci[f't{level}']] - 273.15
        out[f'r{level}'] = field[..., ci[f'r{level}']]
        for var in ('u', 'v'):
            out[f'{var}{level}'] = field[..., ci[f'{var}{level}']]
        out[f'wind{level}'] = np.hypot(out[f'u{level}'], out[f'v{level}'])
        out[f'z{level}'] = field[..., ci[f'z{level}']] / 9.80665
    return out


def variable_unit(name):
    if name == 'msl':
        return 'hPa'
    return {'t': 'degC', 'r': '%', 'u': 'm/s', 'v': 'm/s', 'w': 'm/s', 'z': 'gpm'}[name[0]]


def metric_rows(physical, stations):
    rows = []
    for si, station in enumerate(stations):
        for var in physical['ERA5']:
            target = physical['ERA5'][var][:, si]
            row = dict(station_id=station['station_id'], variable=var,
                       unit=variable_unit(var), n_times=len(target))
            for source, prefix in (('FuXi-24h', 'fuxi'), ('Analysis', 'analysis')):
                error = physical[source][var][:, si] - target
                row[f'{prefix}_mae'] = float(np.abs(error).mean())
                row[f'{prefix}_rmse'] = float(np.sqrt(np.square(error).mean()))
                row[f'{prefix}_bias'] = float(error.mean())
            row['mae_improvement_pct'] = (100 * (1 - row['analysis_mae'] / row['fuxi_mae'])
                                           if row['fuxi_mae'] else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def plot_station(times, physical, station, si, select, window_name, out_dir, pdf):
    """13 panels per station: MSL and 4 quantities x 3 pressure levels."""
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(18, 14), layout='constrained')
    grid = fig.add_gridspec(5, 3, height_ratios=[1.0, 1, 1, 1, 1])
    panels = [(fig.add_subplot(grid[0, :]), 'msl', 'Mean sea-level pressure', 'hPa')]
    for row, (prefix, title, unit) in enumerate((
            ('t', 'Temperature', 'degC'), ('r', 'Relative humidity', '%'),
            ('wind', 'Wind speed', 'm/s'), ('z', 'Geopotential height', 'gpm')), start=1):
        for col, level in enumerate(LEVELS):
            panels.append((fig.add_subplot(grid[row, col]), f'{prefix}{level}',
                           f'{title} | {level} hPa', unit))
    x = times[select]
    for pi, (ax, var, title, unit) in enumerate(panels):
        for source, color in zip(SOURCES, COLORS):
            ax.plot(x, physical[source][var][select, si], label=source, color=color,
                    lw=0.85 if window_name == 'full' else 1.25,
                    alpha=0.85, ls='--' if source == 'FuXi-24h' else '-')
        target = physical['ERA5'][var][select, si]
        eb = float(np.abs(physical['FuXi-24h'][var][select, si] - target).mean())
        ea = float(np.abs(physical['Analysis'][var][select, si] - target).mean())
        ax.set_title(f'{title}   |   MAE: FuXi {eb:.2f}, Analysis {ea:.2f}', fontsize=10)
        ax.set_ylabel(unit)
        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.grid(alpha=.22)
        ax.margins(x=.01)
        if pi == 0:
            ax.legend(loc='upper right', ncol=3)
    fig.suptitle(
        f"{station['station_id']} | grid {station['grid_lat']:.2f} N, {station['grid_lon']:.2f} E | "
        f"{x[0]:%Y-%m-%d %H} to {x[-1]:%Y-%m-%d %H} UTC | {len(x)} samples\n"
        'Best checkpoint: observation-conditioned analysis vs ERA5 / FuXi-24h; '
        '6-hourly valid times; test dates overlap validation', fontsize=13)
    name = f"{station['station_id']}_{window_name}"
    fig.savefig(out_dir / f'{name}.png', dpi=150)
    fig.savefig(out_dir / f'{name}.pdf')
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--configs', default='configs')
    ap.add_argument('--model-id', dest='model_id', required=True)
    ap.add_argument('--checkpoint')
    ap.add_argument('--stats-dir', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path)
    ap.add_argument('--seed', type=int, default=20261015)
    ap.add_argument('--n-stations', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--num-workers', type=int, default=0)
    ap.add_argument('--precision', choices=['fp16', 'fp32'], default='fp16')
    ap.add_argument('--zoom-start', default='2025-01-01')
    ap.add_argument('--zoom-days', type=int, default=14)
    ap.add_argument('--set', action='append', default=[])
    args = ap.parse_args()
    args.split, args.max_batches = 'test', 0
    if args.batch_size <= 0 or args.num_workers < 0 or args.n_stations <= 0 or args.zoom_days <= 0:
        ap.error('Invalid batch size / workers / station count / zoom days')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required by the project preprocessing functions')
    torch.set_num_threads(4)
    torch.manual_seed(2000)
    cfg = importlib.import_module(args.configs)
    apply_overrides(cfg, [f'model_id={args.model_id}'] + args.set)
    from train_FSDP import process_bg, process_obs

    mean = np.load(args.stats_dir / 'mean_era5.npy').reshape(-1).astype(np.float64)
    std = np.load(args.stats_dir / 'std_era5.npy').reshape(-1).astype(np.float64)
    if mean.shape != (70,) or std.shape != (70,) or not np.isfinite(mean).all() or not (std > 0).all():
        raise ValueError('Expected finite 70-channel ERA5 mean/std, std > 0')
    ds = EvaluationDataset(
        era5_dir=cfg.era5_dir, fcst_dir=cfg.fcst_dir, fcst_step=cfg.fcst_step,
        obs_dir=cfg.obs_dir, obs_frames=cfg.obs_frames, obs_stat_dir=cfg.obs_stat_dir,
        obs_channum=cfg.obs_channum, dates_range=cfg.dates_test_range,
        obs_frame_minutes=cfg.obs_frame_minutes, grid_hw=cfg.grid_hw,
        add_fuxi_ztd=cfg.add_fuxi_ztd, ztd_fuxi_zarr=cfg.ztd_fuxi_zarr)
    obs_store = zarr.open(str(cfg.obs_dir), mode='r')
    for axis in ('lat', 'lon'):
        np.testing.assert_allclose(obs_store[axis][:], getattr(cfg, axis), atol=1e-6)
        np.testing.assert_allclose(ds.read_era5.store[axis][:], getattr(cfg, axis), atol=1e-6)
        np.testing.assert_allclose(ds.read_fcst.store[axis][:], getattr(cfg, axis), atol=1e-6)
    cells = np.argwhere(~np.asarray(obs_store['mask'][:], dtype=bool))
    if args.n_stations > len(cells):
        raise ValueError('More requested stations than station cells')
    selected = cells[np.random.default_rng(args.seed).choice(len(cells), args.n_stations, replace=False)]
    station_names = obs_store['station'][:]
    stations = [dict(station_id=str(station_names[y, x]), iy=int(y), ix=int(x),
                     grid_lat=float(cfg.lat[y]), grid_lon=float(cfg.lon[x])) for y, x in selected]
    print('[stations]', stations, flush=True)
    channels = [str(c) for c in ds.read_era5.store['channel'][:][LABEL_IDX]]
    expected = ([f'{v}{p}' for v in ('z', 't', 'u', 'v', 'r')
                 for p in (50,100,150,200,250,300,400,500,600,700,850,925,1000)]
                + ['t2m', 'u10', 'v10', 'msl', 'era5_tp'])
    if channels != expected:
        raise ValueError('Unexpected ERA5 channel order; cannot apply supplied 70-channel statistics')
    out_dir = args.out_dir or Path(run_dir(cfg)) / f'station_timeseries_test_seed{args.seed}'
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise FileExistsError(f'{out_dir} is nonempty; select a fresh --out-dir')
    ckpt = pick_checkpoint(cfg, args.checkpoint)
    model = load_model(cfg, ckpt, 'cuda:0')
    dtype = set_eval_precision(model, args.precision)
    mean_std_dict = ds.read_obs.mean_std_dict
    loader_options = ({'multiprocessing_context': 'forkserver', 'prefetch_factor': 1}
                      if args.num_workers else {})
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers, **loader_options)
    iy, ix = [torch.as_tensor(selected[:, i], device='cuda:0') for i in (0, 1)]
    parts = {name: [] for name in SOURCES}
    count, start_time = 0, time.monotonic()
    with torch.inference_mode():
        for bi, (bf, be, bo) in enumerate(loader):
            bg = process_bg(bf, 0, cfg.grid_hw).to(dtype=dtype)
            label = process_bg(be, 0, cfg.grid_hw).to(dtype=dtype)
            obs = process_obs(bo, mean_std_dict, 0, cfg).to(dtype=dtype)
            output = model(bg, obs)
            for name, field in zip(SOURCES, (label, bg, output)):
                points = field[:, 0, :, iy, ix].permute(0, 2, 1).float().cpu().numpy()
                if not np.isfinite(points).all():
                    raise ValueError(f'Non-finite station values in {name}, batch {bi}')
                parts[name].append(points)
            count += len(bf)
            if bi == 0 or (bi + 1) % 10 == 0 or count == len(ds):
                elapsed = time.monotonic() - start_time
                print(f'[infer] {count}/{len(ds)} times, {elapsed:.1f}s, '
                      f'ETA {elapsed / count * (len(ds) - count):.1f}s', flush=True)
    values = {name: np.concatenate(parts[name]) for name in SOURCES}
    assert count == len(ds) and ds.dates.is_unique
    np.savez_compressed(out_dir / 'station_predictions_normalized.npz',
                        times=ds.dates.values, channels=np.array(channels),
                        station_ids=np.array([s['station_id'] for s in stations]),
                        grid_indices=selected, era5=values['ERA5'], fuxi=values['FuXi-24h'],
                        analysis=values['Analysis'], mean=mean, std=std)
    physical = {name: physical_variables(array, channels, mean, std) for name, array in values.items()}
    frames = []
    for si, station in enumerate(stations):
        row = {'time_utc': ds.dates, **{key: [value] * count for key, value in station.items()}}
        for name, prefix in zip(SOURCES, ('era5', 'fuxi', 'analysis')):
            for variable, data in physical[name].items():
                row[f'{variable}_{prefix}'] = data[:, si]
        frames.append(pd.DataFrame(row))
    pd.concat(frames, ignore_index=True).to_csv(out_dir / 'station_timeseries_physical.csv', index=False)
    pd.DataFrame(stations).to_csv(out_dir / 'selected_stations.csv', index=False)
    metrics = metric_rows(physical, stations)
    metrics.to_csv(out_dir / 'station_metrics.csv', index=False)
    metadata = evaluation_metadata(cfg, args, ds, model, mean_std_dict, ckpt, count)
    metadata.update({
        'checkpoint': str(ckpt.resolve()), 'station_seed': args.seed, 'stations': stations,
        'station_selection': 'uniform without replacement over occupied NGL grid cells',
        'comparison': 'same valid time; analysis uses contemporaneous observations; FuXi init=T-24h',
        'reference': 'ERA5 label store, not independent station meteorological observations',
        'test_overlaps_validation': list(cfg.dates_test_range) == list(cfg.dates_val_range),
        'units': {v: variable_unit(v) for v in physical['ERA5']},
        'stats_dir': str(args.stats_dir.resolve()),
        'stats_sha256': {name: hashlib.sha256((args.stats_dir / name).read_bytes()).hexdigest()
                         for name in ('mean_era5.npy', 'std_era5.npy')},
        'plot_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'zoom_start': args.zoom_start, 'zoom_days': args.zoom_days,
        'zoom_selection': 'fixed calendar window, not selected based on model performance',
        'notes': ['No smoothing or clipping; raw 6-hourly values.',
                  'Station coordinates refer to grid centres, not exact GNSS antenna locations.',
                  'Pressure levels are not masked below terrain; interpret high-elevation sites cautiously.'],
    })
    (out_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    zoom_start = pd.Timestamp(args.zoom_start)
    zoom = (ds.dates >= zoom_start) & (ds.dates < zoom_start + pd.Timedelta(days=args.zoom_days))
    if not zoom.any():
        raise ValueError('Zoom window does not overlap the test set; predictions already saved')
    with PdfPages(out_dir / 'all_stations_timeseries.pdf') as pdf:
        for si, station in enumerate(stations):
            for window, select in (('full', np.ones(count, dtype=bool)), ('zoom14d', zoom)):
                plot_station(ds.dates, physical, station, si, select, window, out_dir, pdf)
            print(f'[plot] {station["station_id"]} done', flush=True)
    fig, ax = plt.subplots(figsize=(9, 6), layout='constrained')
    ax.scatter(np.asarray(cfg.lon)[cells[:, 1]], np.asarray(cfg.lat)[cells[:, 0]],
               s=3, color='#c5cbd1', label='NGL station grid cells')
    for station in stations:
        ax.scatter(station['grid_lon'], station['grid_lat'], s=55, color='#1976c9')
        ax.annotate(station['station_id'], (station['grid_lon'], station['grid_lat']),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    ax.set(xlabel='Longitude (deg E)', ylabel='Latitude (deg N)',
           title=f'5 randomly selected NGL station grid cells | seed {args.seed}')
    ax.set_aspect(1 / np.cos(np.deg2rad(46)))
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(out_dir / 'selected_stations.png', dpi=160)
    plt.close(fig)
    print('[out]', out_dir.resolve(), flush=True)
    print(metrics[metrics.variable == 'msl'].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
