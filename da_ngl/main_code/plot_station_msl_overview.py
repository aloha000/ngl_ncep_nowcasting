#!/usr/bin/env python3
"""Make a five-station MSL overview from plot_station_timeseries.py's CSV."""
import argparse
import os
from pathlib import Path

os.environ.setdefault('MKL_THREADING_LAYER', 'GNU')
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_dir', type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.result_dir / 'station_timeseries_physical.csv',
                        parse_dates=['time_utc'])
    stations = frame.station_id.unique()
    for window in ('full', 'zoom14d'):
        data = frame
        if window == 'zoom14d':
            start = frame.time_utc.min()
            data = frame[(frame.time_utc >= start) &
                         (frame.time_utc < start + pd.Timedelta(days=14))]
        fig, axes = plt.subplots(len(stations), 2, figsize=(19, 12),
                                 squeeze=False, layout='constrained')
        for row, station in enumerate(stations):
            block = data[data.station_id == station].sort_values('time_utc')
            ax, error_ax = axes[row]
            for name, label, color, style in (
                    ('era5', 'ERA5', '#222222', '-'),
                    ('fuxi', 'FuXi-24h', '#e58b27', '--'),
                    ('analysis', 'Analysis', '#1976c9', '-')):
                ax.plot(block.time_utc, block[f'msl_{name}'], color=color,
                        label=label, lw=.9, ls=style)
                if name != 'era5':
                    error = block[f'msl_{name}'] - block.msl_era5
                    error_ax.plot(block.time_utc, error, color=color, lw=.85,
                                  label=f'{label}: MAE {error.abs().mean():.3f} hPa')
            error_ax.axhline(0, color='black', lw=.6)
            ax.set_title(f'{station} | MSL | grid {block.grid_lat.iloc[0]:.2f} N, '
                         f'{block.grid_lon.iloc[0]:.2f} E', fontsize=11)
            error_ax.set_title(f'{station} | signed error relative to ERA5', fontsize=11)
            ax.legend(ncol=3, fontsize=8, loc='upper right')
            error_ax.legend(ncol=2, fontsize=8, loc='upper right')
            for panel in (ax, error_ax):
                panel.set_ylabel('hPa')
                panel.grid(alpha=.22)
                loc = mdates.AutoDateLocator(minticks=4, maxticks=8)
                panel.xaxis.set_major_locator(loc)
                panel.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
                panel.margins(x=.01)
        fig.suptitle('Mean sea-level pressure | 5 random NGL station grid cells | 6-hourly UTC\n'
                     f'{data.time_utc.min():%Y-%m-%d %H} to {data.time_utc.max():%Y-%m-%d %H} | '
                     'Analysis uses contemporaneous observations; test overlaps validation', fontsize=14)
        for extension in ('png', 'pdf'):
            path = args.result_dir / f'msl_5stations_{window}.{extension}'
            fig.savefig(path, dpi=150)
            print(path, flush=True)
        plt.close(fig)


if __name__ == '__main__':
    main()
