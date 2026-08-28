# US surface/GNSS sample index

`build_sample_index.py` builds these files in `dataset/`:

- `sample_index.parquet`: one target-station/hour candidate per row;
- `target_stations.parquet`: normalized target coordinates and active spans;
- `gnss_stations.parquet`: coordinates parsed from NGL TRO headers;
- `target_gnss_neighbors.parquet`: nearest five NGL stations within 50 km,
  including distance, initial bearing, and height difference;
- `sample_index_metadata.json`: exact processing rules and counts.

Only the 50 US states and Washington, DC are included. Alaska and Hawaii are
included; overseas territories are not. State boundaries come from Bokeh's
offline `us_states` sample data.

A GNSS station is valid at time T only when finite ZTD and ZWD occur at each of
the seven hourly timestamps T-6, ..., T. An input is valid when at least three
of the selected nearest stations satisfy that rule. Reports lacking elevation
cannot meet the requested `(latitude, longitude, elevation)` station identity
and are recorded in metadata rather than assigned an ambiguous station ID.

Run:

```bash
python preprocessing/build_sample_index.py --workers 8
```

The expensive scan caches are retained under `dataset/cache/`. Use `--force`
only when raw data or processing rules have changed. `pyarrow`, `scikit-learn`,
`bokeh`, and `matplotlib` are required.

## Hourly NGL Zarr

Build the exact-hour ZTD/ZWD store with:

```bash
python preprocessing/build_ngl_hourly_zarr.py --workers 8
```

## 5-minute NGL Zarr

Identical to the hourly build except the time sampling is 5 minutes:

```bash
python preprocessing/build_ngl_5min_zarr.py --workers 8
```

The default output is `dataset/ngl_5min.zarr`; records are kept only at exact
UTC 5-minute epochs (seconds % 300 == 0), the time axis spans the same
inclusive interval, and the store has ~12x the hourly time steps (~701k). The
same restart-marker machinery applies.

The default inclusive UTC interval is `2017-12-31 00:00` through
`2024-09-01 00:00`. The resulting `dataset/ngl_hourly.zarr` has dimensions
`(time, station)` and float32 `ztd`/`zwd` variables in the original NGL unit
of millimetres. Interrupted runs are restartable without `--force`.
Values are copied from the NGL `TROTOT` and `TRWET` fields without physical
range filtering; in particular, negative ZWD values are preserved.

## Hourly NCEP Zarr

Build the exact-hour NCEP surface store with:

```bash
python preprocessing/build_ncep_hourly_zarr.py --workers 8
```

The default inclusive UTC interval is 2017-12-31 21:00 through 2024-08-15 20:00
(the span of the surf_*.pkl files). The resulting `dataset/ncep_hourly.zarr` has
dimensions (time, station) and float32 `p`/`slp`/`t2m`/`r2m`/`u10`/`v10` variables.
Stations follow `dataset/target_stations.parquet` (ncep_00001 ... ncep_02331),
matched by exact (lat, lon, h). No interpolation is applied: missing source
hours or stations are NaN. For duplicate rows sharing a key, each variable
drops one maximum and one minimum and averages the remaining finite values;
rows with `u10 == v10 == 0` are excluded from the wind averages, and an
all-NaN `slp` stays NaN. Runs are restartable without `--force`.
