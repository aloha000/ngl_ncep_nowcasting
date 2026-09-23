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
results_dir = '/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/linan/linan_dev/gnss/da_ngl/main_code/work_dir/results/stage_two'         
                        
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

fcst_step = 4
obs_frame_minutes = 5        # NGL native sampling
obs_frames = 73              # 73 x 5 min = the 6 hours ending at T
obs_end_offset_minutes = 0   # window end relative to the valid time T
obs_add_mask = True          # per-frame validity mask channel
obs_add_latlon = True        # append lat / lon channels

# Observation channel set fed to the network:
#   'absolute' -- the NGL ZTD as stored (standardised), i.e. the old behaviour
#   'residual' -- obs - H(FuXi), the innovation built with the ZTD operator
#   'both'     -- two channels: absolute ZTD + innovation
obs_mode = 'both'
# train-split mean/std stored in the NGL zarr (ztd_train_mean / ztd_train_std).
#     ztd_norm  = (obs_mm     - mean_ngl) / std_ngl      <- the absolute channel
#     fuxi_norm = (H(FuXi)_mm - mean_ngl) / std_ngl
#     innovation = ztd_norm - fuxi_norm

obs_res_scale_mm = None  # 如果是None或0，则obs和H(Fuxi)都用 ztd_train_std 归一化；否则innovation=(obs-H(Fuxi)) / obs_res_scale_mm，绝对通道仍然用 ztd_train_std 归一化。

ztd_fuxi_zarr = f'{DATASET_DIR}/ztd_fuxi_europe_0p25_24h_zdz.zarr'   # must match fcst_step

tp_label_source = 'era5'     # 'imerg' or 'era5'

zero_obs = False      # "No GNSS" ablation: replace the ZTD values by 0 (= the training-mean ZTD,

# --------------------------------------------------------------------------
# observation consistency + per-station de-biasing (HANDOFF 11.8 items 1 + 2)
# --------------------------------------------------------------------------

#     J = J_label + lambda_obs * MAE_{valid station cells} |H(x_a) - obs'| / sigma_o
lambda_obs = 0.2

lambda_obs_domain_compensation = True  # 2026-09-21: 观测一致性损失只在站点格上算，且每个站点格的权重按其覆盖的格子数补偿（否则大站点格的权重过大）。
obs_sigma_o_mm = 11.0           # std(obs-H(era5)) H()的逐站标准差，和lambda_obs一起决定了观测一致性损失的权重。 


obs_debias = True     # 每个站算 mean(ztd-H(fuxi))
# H(bg)算子变化的话，就要重新算 obs_debias_file（dataset/obs_debias_lead24h_zdz.npz）；
obs_debias_file = f'{DATASET_DIR}/obs_debias_lead24h_zdz.npz'

obs_freeze_zhd = False
freeze_msl = False
# --- 2026-09-21：把"改分析场"按通道标价，避免网络只用 msl（方法 E 下还有 z）去买 ZTD 拟合 ---
#     J_B = increment_penalty_mu * mean_{c<69, 站点格} |x_a - x_b|_c / sigma_b,c
# sigma_b,c 是背景误差的逐通道标准差（dataset/bg_err_std.npz，train 段统计）。
# 实测（站点格）：z 族 0.026~0.067、msl 0.066、t 族 0.12、r 族 0.38~0.52 ——
# 除以 sigma_b 后，动 z/msl 比动 r 贵 6~10 倍，修正会被推向热力与湿度廓线。
# 0 = 关闭（旧行为）。
increment_penalty_mu = 0.0
increment_penalty_file = None          # None -> dataset/bg_err_std.npz
increment_penalty_mode = 'station'     # 'station'（只在站点格统计）或 'all'
# 方法 E 的层高与地面气压都来自网络输出（z / msl）；置 True 则改为取背景并 detach，
# 即"柱几何是给定的"，只让热力/湿度廓线带梯度（等价于把两个廉价出口一起堵住）。
obs_freeze_geometry = False

# --------------------------------------------------------------------------
# date ranges (6-hourly, half-open [start, end); format YYYYMMDDHH)
# --------------------------------------------------------------------------
dates_train_range = ['2022010100', '2024123100']
dates_val_range = ['2025010100', '2025100100']
dates_test_range = ['2025010100', '2025100100']

# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
include_fuxi_tp = True
model_bg_chans = 69 + int(include_fuxi_tp)   # kept in sync at run time from cfg
model_out_chans = 70         # ERA5 69 + tp (residual on bg_chans channels)
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
num_epochs = 40              # safety cap; training also stops at num_iteration
batch_size = 2
num_workers = 8
prefetch_factor = 3
persistent_workers = True
multiprocessing_context = 'forkserver'
pin_memory = False
amp = True                   # fp16 mixed precision + ShardedGradScaler
save_interval = 1
min_delta = 0.0              # val loss must improve by this much to (re)save
log_interval = 10            # iterations between progress lines (0 = silent)
log_batch_loss = False       # True: log the loss of *every* batch (rank 0) and
                             # save the per-iteration curve to iter_loss.npy

# 标签损失的空间加权：True = 历史行为（cos lat，按均值归一化）；False = 全 1 权重，
# 退化成普通逐格点等权 MAE。关掉之后 exp_tag 会多一个 `_nolat`，并且
# loss_domain_weight_share 会自动改用纯格点比例（否则 lambda_obs 的域补偿会失准）。
lat_weight = True
loss_fn = mae(lat_weight=lat_weight)   # NaN-safe；纬度加权由上面的 lat_weight 控制

loss_station_weight = 1.0
loss_nostation_weight = 0.0   # 0 = cells outside the region carry no weight

loss_domain = 'station_halo'
loss_halo_cells = 3
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
