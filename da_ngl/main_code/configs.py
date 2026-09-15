"""Config for the GNSS/FuXi -> ERA5 assimilation run (adapted train_packet).

Everything the training script needs lives here; override by editing this file
or by importing a different module with ``--configs``.
"""

import numpy as np

from main.model import mae  # noqa: F401  (mse is available too)

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
DATASET_DIR = '/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/linan/linan_dev/gnss/da_ngl/dataset'

work_dir = '/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/linan/linan_dev/gnss/da_ngl/main_code/work_dir'

# --------------------------------------------------------------------------
# run identity / output layout
# --------------------------------------------------------------------------
# One run = one folder  {results_dir}/{model_id}_{exp_tag}/
# holding    the val-best checkpoint  {model_id}_{arch_tag}.pth
#            the plots + metrics.csv/json + summary.json
# and one log  {log_dir}/{model_id}_{arch_tag}.log
model_id = 'model'          # 入参: --model_id ; prefixes the run folder and files
exp_tag = None              # 实验配置 ; None -> auto from the data/setup knobs
arch_tag = None             # 模型配置 ; None -> auto from the network shape
results_dir = None         # None -> {work_dir}/results (kept lazy so a
                           # config that overrides work_dir stays self-contained)
log_dir = '/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/linan/linan_dev/gnss/da_ngl/logs'

fuxi_zarr = f'{DATASET_DIR}/fuxi_europe_0p25.zarr'
ngl_zarr = f'{DATASET_DIR}/ngl_europe_0p25_5min.zarr'
label_zarr = f'{DATASET_DIR}/label_europe_0p25.zarr'

# --------------------------------------------------------------------------
# grid (must match the stores: 80 x 120, 0.25 deg)
# --------------------------------------------------------------------------
lat = np.round(np.arange(36.50, 56.25 + 1e-9, 0.25), 6)     # 80
lon = np.round(np.arange(-5.25, 24.50 + 1e-9, 0.25), 6)     # 120
grid_hw = (lat.size, lon.size)

# --------------------------------------------------------------------------
# sample layout
# --------------------------------------------------------------------------
# Background (FuXi) convention, same as the reference ``read_bg``:
#   init = T - fcst_step * 6 h ;  step = fcst_step * 6 h
# The fuxi store only keeps lead 6 h, so fcst_step must be 1.
fcst_step = 1
obs_frame_minutes = 5        # NGL native sampling
obs_frames = 73              # 73 x 5 min = the 6 hours ending at T
obs_end_offset_minutes = 0   # window end relative to the valid time T
obs_add_mask = True          # per-frame validity mask channel
obs_add_latlon = True        # append lat / lon channels

# Observation channel set fed to the network:
#   'absolute' -- the NGL ZTD as stored (standardised), i.e. the old behaviour
#   'residual' -- obs - H(FuXi), the innovation built with the ZTD operator
#                 (preprocessing/ztd_operator.py + build_ztd_fuxi_zarr.py)
#   'both'     -- two channels: absolute ZTD + innovation
obs_mode = 'absolute'
obs_res_scale_mm = 15.0      # innovation is divided by this (~its own std) so
                             # it enters the net at O(1) like the other inputs
ztd_fuxi_zarr = f'{DATASET_DIR}/ztd_fuxi_europe_0p25_6h.zarr'

# Which tp the *training* target uses.  The label store holds both:
#   channel 69 = IMERG tp   (independent precipitation observation)
#   channel 70 = ERA5 tp    (ERA5's own tp, consistent with channels 0..68)
# plot_results.py always reports the ERA5-vs-IMERG disagreement and scores
# the model against whichever tp it was trained on.
tp_label_source = 'era5'     # 'imerg' or 'era5'

# "No GNSS" ablation: replace the ZTD values by 0 (= the training-mean ZTD,
# which is also what a cell without a station carries) after the validity mask
# has been built, so the network sees the station geometry but no ZTD signal.
# Consumed by process_obs(), hence it applies to training and to
# plot_results.py alike.
zero_obs = False

# --------------------------------------------------------------------------
# date ranges (6-hourly, half-open [start, end); format YYYYMMDDHH)
# --------------------------------------------------------------------------
dates_train_range = ['2022010100', '2024050100']
dates_val_range = ['2024050100', '2025010100']
dates_test_range = ['2025010100', '2025100100']

# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
model_bg_chans = 69          # FuXi channels (tp dropped)
model_out_chans = 70         # ERA5 69 + IMERG tp (residual only on the first 69)
# The label store carries a 71st channel ``era5_tp`` (ERA5's own tp) that is
# only used for evaluation/plots -- training reads just the first 70.
label_n_chans = 70
model_obs_chans = {'absolute': 1, 'residual': 1, 'both': 2}[obs_mode] \
                  + int(obs_add_mask) + 2 * int(obs_add_latlon)   # ztd/innov (+mask, lat, lon)
model_obs_frames = obs_frames
model_embed_dim = 128
model_depth = (2, 2, 2)
model_dropout_rate = 0       # kept for parity with the original config

resume_model = None
pre_model = None

# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
rand_seed = 2000
num_iteration = 20000        # total optimisation steps (per rank)
num_epochs = 60              # safety cap; training also stops at num_iteration
batch_size = 2
num_workers = 8
prefetch_factor = 3
persistent_workers = True
multiprocessing_context = 'forkserver'
pin_memory = False
amp = True                   # fp16 mixed precision + ShardedGradScaler
save_interval = 1
min_delta = 0.0              # val loss must improve by this much to (re)save
log_interval = 100           # iterations between progress lines (0 = silent)
log_batch_loss = False       # True: log the loss of *every* batch (rank 0) and
                             # save the per-iteration curve to iter_loss.npy

loss_fn = mae()              # latitude-weighted, NaN-safe
# early_stop = {'patience': 5, 'min_delta': 1e-4}

# --------------------------------------------------------------------------
# optimizer
# --------------------------------------------------------------------------
opt_type = 'AdamW'
learning_rate = 1e-4
weight_decay = 0.1

warmup = True
start_lr = 1e-8
stop_lr = 1e-4
warmup_rate = 0.05

scheduler = 'CosineAnnealingLR'
step_size = 2                # only used by StepLR / MultiStepLR
