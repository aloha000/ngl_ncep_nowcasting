# GNSS → NCEP nowcasting with iTransformer

## Task

Nowcasting: use **5-minute** NGL zenith tropospheric delays (**ZTD** / **ZWD**) of
the nearest GNSS stations over the window **T-2h .. T** (25 five-minute steps) to
predict the NCEP surface observations (**p, slp, t2m, r2m, u10, v10**) at time **T**
for each target surface station. The NCEP targets stay hourly.

Model: `Time-Series-Library/models/iTransformer.py` (iTransformer, adapted,
see below).

## Temporal splits (UTC)

| Split | Range |
| --- | --- |
| train | `2018-01-01T00:00` ≤ T < `2023-11-01T00:00` |
| val | `2023-11-01T00:00` ≤ T < `2024-03-01T00:00` |
| test | `2024-03-01T00:00` ≤ T ≤ last available hour (`2024-08-15T20:00`) |

Boundaries are configurable via `--train-start/--train-end/--val-start/--val-end/--test-start/--test-end`.

## Input / output layout

Per sample:

- Input `x`: `(seq_len, 10)` float32, where `seq_len = window_hours*60/ngl_step_minutes + 1` (default 25)
  - channels `0..9`: `ztd`, `zwd` of the up-to-5 nearest GNSS stations (rank 1..5,
    from `dataset/target_gnss_neighbors.parquet`), zero-padded when a station is
    missing from the input window.
- Time marks `x_mark` (optional): `time_encoding: none` passes no marks
  (`x_mark=None`, 10 tokens); `hour_sincos` adds `[sin(2πh/24), cos(2πh/24)]`
  (2 channels); `sincos` adds sin/cos pairs for hour/day-of-week/day-of-month/
  day-of-year (8 channels); `linear` uses TSL timeF features at `model.time_freq`
  resolution (`5min`: minute-of-hour/hour/day-of-week/day-of-month/day-of-year).
- Spatial encoding `x_geo` (when `model.spatial_enc: true`): `(max_neighbors, n_geo + static_dim)` per-sample.
  The first columns are `[dE_km, dN_km, dU_m, ngl_h_m]` plus optional target/GNSS lat/lon;
  const.nc static GNSS features are appended after them. The full vector is normalized with
  train-pair statistics, zeroed for invalid/missing neighbors, embedded by a small MLP, and
  added to that neighbor's ztd/zwd tokens before the encoder.
- Target-station feature `x_tgt` (when `model.target_h_feat: true`): per-sample target height
  plus target const.nc static features, normalized with train-station statistics and concatenated
  to the token outputs right before the `separate_output` linear layer. Kept out of the input
  variates on purpose: per-variate instance normalization would zero out any channel that
  is constant across the window.
- Target `y`: `(1, 6)` NCEP variables at T, z-scored with train-split statistics
  (disable with `--no-target-scale`).

Samples are kept only when ≥ `--min-valid-neighbors` (default 3) neighbors are
valid and all 6 target values are finite.

## Files

| Path | Purpose |
| --- | --- |
| `nowcasting/train_iTransformer_nowcast.py` | compatibility training entry point; delegates to `nowcasting/main_code/main.py` |
| `nowcasting/main_code/` | modular training implementation (`config.py`, `dataset.py`, `model.py`, `train.py`, `plot.py`, `main.py`) |
| `nowcasting/test/` | experiment/test plotting and grid-inference scripts |
| `nowcasting/config.yaml` | all run parameters (data paths, splits, sampling, model, training, run) |
| `nowcasting/EXPERIMENTS.md` | experiment log and conclusions (2026-08-27 ~ 08-31) |
| `nowcasting/outputs/<setting>/checkpoint.pth` | best checkpoint (by val loss) |
| `nowcasting/outputs/gnss_nowcast_s1915_off0_h6_dm128_el2_nh4_df512_sp_thf/` | current best run: spatial encoding + target height before output head |
| `nowcasting/outputs/<setting>/config_used.yaml` | effective config of the run (reproducibility) |
| `nowcasting/outputs/<setting>/test_predictions.npz` | preds/trues (physical units) and normalized copies |
| `nowcasting/outputs/<setting>/test_metrics.json` | per-variable MAE/MSE/RMSE on the test split |
| `nowcasting/outputs/<setting>/loss_curve.json` / `loss_curve.png` | per-epoch train/val loss + curve |
| `nowcasting/outputs/<setting>/test_scatter.png` | pred vs truth scatter (per variable, R²) |
| `nowcasting/outputs/<setting>/test_error_hist.png` | prediction error histograms |
| `nowcasting/outputs/<setting>/test_timeseries.png` | pred vs truth time-series snippet |
| `Time-Series-Library/models/iTransformer.py` | model (adapted, see below) |

## Model adaptation

`iTransformer.py` keeps its original behaviour by default and gains two optional
features. First: when `configs.separate_output` is set (our script sets it), a final
`nn.Linear(enc_in, c_out)` maps the per-input-variate projections to the target
variates, and the non-stationary de-normalization is skipped because the output
channels are not the input channels. Second: when `configs.spatial_enc` is set,
a small MLP (`Linear(n_geo→d_model)→GELU→Linear(d_model→d_model)`) embeds the
per-neighbor ENU/height/static vector and the result is added to that neighbor's
ztd/zwd tokens before the encoder; the output head and token layout are
unchanged. Third: when `configs.target_h_feat` is set, the per-sample target-station
features are concatenated to the token outputs just before the
`separate_output` linear layer (its input becomes `enc_in + target_feat_dim`), so the final layer learns a
direct per-variable linear height term. This is the current best configuration
(best val 0.706 at epoch 5; test overall RMSE 8.15). All other tasks/runs are unchanged.

## How to run

All parameters live in `nowcasting/config.yaml` (data paths, temporal splits,
station/hour sampling, model, training, run). Use the `gnss` conda environment
(torch + CUDA + zarr):

```bash
conda activate gnss
# or
/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/cfff_linan/anaconda3/anaconda3/envs/gnss/bin/python \
    nowcasting/train_iTransformer_nowcast.py --help

# default run (uses nowcasting/config.yaml)
python nowcasting/train_iTransformer_nowcast.py

# use another config file
python nowcasting/train_iTransformer_nowcast.py --config my_config.yaml

# override individual keys from the command line
python nowcasting/train_iTransformer_nowcast.py --set stations=128 --set epochs=20

# train on ALL usable stations (stations <= 0 means "all", from station_offset onward)
# currently ~1915 stations meet min_valid_neighbors=3; ~36M train hours with hour_stride=1
python nowcasting/train_iTransformer_nowcast.py --set stations=0 --set hour_stride=6
```

The effective (merged) config is printed at startup and saved as
`config_used.yaml` next to the checkpoint. Unknown config keys or `--set` keys
raise an error so typos are caught early.

Environment note: the `gnss` env had `zarr` missing and an incompatible
`numcodecs`; it now has `zarr==2.18.2` + `numcodecs==0.12.1` (matches the
version pair used to build the stores). `pyarrow`/`pandas` were already present.

## I/O notes (this filesystem is slow)

- The NGL store is chunked `(8760, 1)` (~52k tiny column chunks): we read the
  neighbor columns directly per station — fast for a station subset, but a full
  array read is very slow. `--load-full-arrays` exists but is off by default.
- The NCEP store is chunked `(48, 2331)`: per-column reads are slow, so the
  script loads the 6 variables row-major into RAM once (~3.3 GB; `--no-load-full-ncep` to disable).
- Use `--station-offset` to shift the selected stations and `--stations` to
  control the subset; `--hour-stride` subsamples train hours (val/test default
  to every hour).

## Next steps / known simplifications

- Tune values in `config.yaml` (d_model, layers, lr schedule, more stations/epochs).
- Save prediction provenance (station id + UTC per sample).
- Filter on `dataset/sample_index.parquet` when regenerated, and/or per-variable
  loss weighting to balance variable scales.
- Add input feature engineering (bearing/distance/height-difference embeddings,
  NCEP history channels) and inverse/denormalization handling in the model.
- Run focused ablations for the new `target_h_feat` head feature if attribution is needed
  (for example target height only, spatial ENU only, or removing `ngl_h_m`).
