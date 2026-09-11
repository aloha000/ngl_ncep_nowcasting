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
obs_frames = 25              # 25 x 5 min = the two hours ending at T
obs_end_offset_minutes = 0   # window end relative to the valid time T
obs_add_mask = True          # per-frame validity mask channel
obs_add_latlon = True        # append lat / lon channels

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
model_obs_chans = 1 + int(obs_add_mask) + 2 * int(obs_add_latlon)   # ztd (+mask, lat, lon)
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
