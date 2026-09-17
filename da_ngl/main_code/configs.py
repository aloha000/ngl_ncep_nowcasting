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

fuxi_zarr = f'{DATASET_DIR}/fuxi_europe_0p25_24h_70ch.zarr'   # lead 24 h, +FuXi tp
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
# The 24 h store keeps only lead 24 h, so fcst_step must be 4.
fcst_step = 4
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
obs_mode = 'both'
obs_res_scale_mm = 15.0      # innovation is divided by this (~its own std) so
                             # it enters the net at O(1) like the other inputs
ztd_fuxi_zarr = f'{DATASET_DIR}/ztd_fuxi_europe_0p25_24h.zarr'   # must match fcst_step

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
# observation consistency + per-station de-biasing (HANDOFF 11.8 items 1 + 2)
# --------------------------------------------------------------------------
# Item 1 -- observation-consistency term.  Every other term in the loss is a
# regression against ERA5; none of them says *in which direction* the analysis
# should move towards the GNSS ZTD, and the network settled on an almost
# identity map.  This adds
#     J = J_label + lambda_obs * MAE_{valid station cells} |H(x_a) - obs'| / sigma_o
# with H the differentiable ZTD operator (main/model/ztd_torch.py) applied to
# the analysis, obs' the observed ZTD at the valid time T [mm], and sigma_o the
# operator + representativeness error.  11.5 measured std(obs - H(ERA5)) = 10.9 mm,
# so sigma_o = 11 mm.  lambda_obs = 0 disables the term.  Note only the ratio
# matters: the effective weight is lambda_obs / sigma_o per mm, and 0.2/11 mm
# makes the term ~0.15, i.e. about as large as the label MAE (~0.126 at 24 h).
lambda_obs = 0.2
obs_sigma_o_mm = 11.0
# Item 2 -- per-station static bias of the innovation
#     b_s = mean over the *train* split of ( obs_mm - H(bg)_mm ),  1378 values
# built by preprocessing/build_obs_debias.py.  The innovation the network sees
# becomes (obs - b_s) - H(bg) and the consistency-loss target is obs - b_s, so
# the network no longer has to spend its capacity on 1378 constant offsets
# (the absolute ZTD is ~85 % station-static variance).  b_s depends on the
# lead, hence one cached file per lead.
obs_debias = True
obs_debias_file = None       # None -> dataset/obs_debias_lead{fcst_step*6}h.npz
# Evaluate the consistency term as H*(x_a) = ZHD(x_bg) + ZWD(x_a): the
# hydrostatic delay (and the column geometry: p_s, which levels are below
# ground, the surface node) comes from the background, so no gradient reaches
# msl and only the thermodynamic column (t / r / t2m) can be moved.
# Rationale: ZHD is ~90 % of ZTD and only mirrors surface pressure -- the
# information in a GNSS ZTD is in the wet delay -- while an unfrozen operator
# leaves the cheap lever "buy ZTD fit by nudging msl", which cost the oc0.2 run
# -121 % on msl at the station cells (+127 % of its net MAE change, i.e. all of
# it) even though r500..r1000 and z850 gained 1-2 %.  At x_a = x_bg the frozen
# and unfrozen forms agree exactly, so this changes only the gradient path.
obs_freeze_zhd = False

# --------------------------------------------------------------------------
# date ranges (6-hourly, half-open [start, end); format YYYYMMDDHH)
# --------------------------------------------------------------------------
dates_train_range = ['2022010100', '2024050100']
dates_val_range = ['2024050100', '2025010100']
dates_test_range = ['2025010100', '2025100100']

# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
# Should the network see FuXi's own precipitation as a *background* for the tp
# channel?  Roughly: False -> bg = 69 state channels and tp is predicted from
# scratch; True -> bg = 70 channels (state + FuXi tp), the residual
# ``out = decoder + bg`` then covers tp too, and the background-only baseline
# becomes a 70-channel number that is directly comparable with the model.
# True requires a FuXi store built with ``--with-tp`` (70 channels), e.g.
#   dataset/fuxi_europe_0p25_24h_70ch.zarr
include_fuxi_tp = True
model_bg_chans = 69 + int(include_fuxi_tp)   # kept in sync at run time from cfg
model_out_chans = 70         # ERA5 69 + tp (residual on bg_chans channels)
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
num_iteration = 25000        # total optimisation steps (per rank)
num_epochs = 45              # safety cap; training also stops at num_iteration
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
# --------------------------------------------------------------------------
# spatial weighting of the loss (station cells get their own weight)
# --------------------------------------------------------------------------
# 1378 of the 9600 grid cells carry a GNSS station; the other 8222 have no
# observation at all, so their residual is unpredictable and the loss can only
# push the network back towards the background there -- which it then does
# everywhere (the convolutions are shared).  Set loss_nostation_weight < 1 to
# down-weight them.  (1.0, 1.0) reproduces the historical behaviour exactly.
# The mask is the NGL store's static ``mask`` (True = *no* station) and its
# polarity is verified at run time.  NOTE: plot_results.py keeps reporting the
# *unweighted* metric on purpose, so runs with different weights stay comparable.
loss_station_weight = 1.0
loss_nostation_weight = 1.0
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
