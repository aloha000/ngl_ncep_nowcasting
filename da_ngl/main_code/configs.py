"""FuXi -> ERA5 同化训练（参照 xuxiaoze/for_zrx/train_packet 的最小改动版）。

数据全部是欧域 0.25 度、80x120 的 zarr store：

    label（对应参考版的 era5） dataset/label_europe_0p25.zarr
                               label[time, 71, 80, 120]，训练只用前 70 个（69 状态 + ERA5 自己的 tp）
    fcst （背景）              dataset/fuxi_europe_0p25_24h_70ch.zarr
                               z[init, step=24h, 70, 80, 120]
    obs  （ZTD）               dataset/ngl_europe_0p25_5min.zarr
                               ztd[frame, 80, 120]（已按训练段 mean/std 标准化）

只喂 fuxi / era5 / ztd 三样，不做 innovation（没有 H(背景) 那一项）。
"""
import numpy as np

from main.model import mse, mae

work_dir = '/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/linan/linan_dev/gnss/da_ngl/main_code/work_dir/results/stage_three'

# 一次运行一个目录：所有产物都落在 {work_dir}/{model_id}/ 下。
# 做不同实验（比如观测置零）时改这个，就不会互相覆盖。
model_id = 'lead24h_allarea'  

# 消融开关：True 时把 ZTD 通道整片置零（= 训练均值），网络只看到站点几何/掩膜，
zero_obs = False

# 标签损失的格点掩膜：
#   'none'          全域（参考版行为）
#   'station_halo'  只监督"有 ZTD 站的格点 + 周围 loss_halo_cells 格"（方形核）
loss_mask = 'none'
loss_halo_cells = 3

DATASET_DIR = ('/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/linan/linan_dev/'
               'gnss/da_ngl/dataset')

era5_dir = f'{DATASET_DIR}/label_europe_0p25.zarr'
fcst_dir = f'{DATASET_DIR}/fuxi_europe_0p25_24h_70ch.zarr'
# 背景的 lead（小时）：T 时刻的背景 = T - fcst_step 起报、取 lead fcst_step 的那一帧。
# 这个 store 里只有 lead 24 h。
fcst_step = 24

obs_dir = f'{DATASET_DIR}/ngl_europe_0p25_5min.zarr'
obs_stat_dir = obs_dir
obs_channum = 1               # 观测只有 ZTD 一路
obs_frames = 25               # 25 x 5 min = 分析时刻前 2 小时（原为 73=6 小时）
obs_frame_minutes = 5         # NGL 原生采样间隔



# %% 格点（process_obs 用它拼 sin(lat)/cos(lon) 侧通道）
lat = np.round(np.arange(36.50, 56.25 + 1e-9, 0.25), 6)     # 80
lon = np.round(np.arange(-5.25, 24.50 + 1e-9, 0.25), 6)     # 120
grid_hw = (lat.size, lon.size)

# %% Data Setting（6 小时一次）
dates_train_range = ['2022010100', '2024123100']
dates_val_range = ['2025010100', '2025100100']
dates_test_range = ['2025010100', '2025100100']

# %% Model Setting
model_bg_chans = 70            # FuXi 70 通道（69 状态 + FuXi tp）
model_obs_chans = obs_channum + 2   # ZTD + sin(lat) + cos(lon)
model_obs_frames = obs_frames
model_embed_dim = 256
model_depth = (2, 2, 2)

model_dropout_rate = 0
resume_model = None
pre_model = None

# %% Train Setting
rand_seed = 2000
num_teration = 8000
batch_size = 2
num_workers = 8

prefetch_factor = 3
persistent_workers = True
multiprocessing_context = "forkserver"
pin_memory = False

save_interval = 10
loss_fn = mae(lat=lat)     # 纬度加权用本网格的 80 个纬度

early_stop = {'patience': 5, 'min_delta': 0.00001}

# %% Optimizer Setting
opt_type = 'AdamW'  # [Adam, AdamW, SGD]
learning_rate = 1e-4
weight_decay = 0.1

warmup = True
if warmup:
    start_lr = 1e-8
    stop_lr = 1e-4
    warmup_rate = 0.05

scheduler = 'CosineAnnealingLR'  # [StepLR, MultiStepLR, CosineAnnealingLR]
if scheduler == 'StepLR' or scheduler == 'MultiStepLR':
    step_size = 2
