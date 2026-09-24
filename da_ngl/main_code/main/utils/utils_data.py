"""数据管道：FuXi(背景) / ERA5(label) / NGL ZTD(观测)。

与参考版 xuxiaoze/for_zrx/train_packet/main/utils/utils_data.py **结构完全一致**，
只把三个 reader 的取数部分从"逐日期的 nc 文件"换成项目里的 zarr store
（训练环境里没有 xarray / dask，所以直接用 zarr 读）：

    参考版                               本版
    read_era5  xr.open_zarr(dir).data    label_europe_0p25.zarr 的 label[time, 70, 80, 120]
    read_fcst  {date}_{step}.nc          fuxi_europe_0p25_24h_70ch.zarr 的 z[init=T-24h, 24h]
    read_obs   逐小时 {date}.nc 的 z      ngl_europe_0p25_5min.zarr 的 ztd[frame, 80, 120]

三个 reader 的对外接口（``prepare_data(date) -> Tensor | None``）没变，所以
``MyDataset`` / ``build_dataloader`` / ``train_FSDP.py`` 的流程都不用动。
"""
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
import numpy as np
import time
import os
import zarr


# label store 的 71 个通道 = ERA5 69 个状态 + IMERG tp(69) + ERA5 tp(70)。
# 训练要"纯 ERA5" 70 个通道：前 69 个 + ERA5 自己的 tp。
LABEL_IDX = np.array(list(range(69)) + [70], dtype=int)


def decode_time(store, name):
    """把 CF 风格的 int64 时间轴解成 DatetimeIndex（unit/origin 写在 attrs.units 里）。"""
    arr = store[name]
    units = str(dict(arr.attrs).get('units', 'hours since 2022-01-01 00:00:00'))
    unit, origin = [s.strip() for s in units.split('since')]
    code = {'hour': 'h', 'hours': 'h', 'minute': 'min', 'minutes': 'min',
            'second': 's', 'seconds': 's', 'day': 'D', 'days': 'D'}[unit.lower()]
    return pd.DatetimeIndex(pd.Timestamp(origin) + pd.to_timedelta(np.asarray(arr[:]), unit=code))


def build_dataloader(world_size,
                     rank,
                     era5_dir,
                     fcst_dir,
                     fcst_step,
                     obs_dir,
                     obs_frames,
                     obs_stat_dir,
                     obs_channum,
                     dates_range,
                     batch_size,
                     num_workers,
                     persistent_workers=True,
                     prefetch_factor=5,
                     multiprocessing_context="forkserver",
                     pin_memory=False,
                     shuffle=True,
                     obs_frame_minutes=5,
                     grid_hw=(80, 120),
                     seed=2000):
    dataset = MyDataset(era5_dir=era5_dir,
                        fcst_dir=fcst_dir,
                        fcst_step=fcst_step,
                        obs_dir=obs_dir,
                        obs_frames=obs_frames,
                        obs_stat_dir=obs_stat_dir,
                        obs_channum=obs_channum,
                        dates_range=dates_range,
                        obs_frame_minutes=obs_frame_minutes,
                        grid_hw=grid_hw,
                        seed=seed)
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
    else:
        ind_ls = list(range(len(dataset)))
        if shuffle:
            sampler = SubsetRandomSampler(ind_ls)
        else:
            sampler = SequentialSampler(dataset)
    if persistent_workers:
        dataloader = DataLoader(dataset=dataset,
                                sampler=sampler,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                prefetch_factor=prefetch_factor,
                                persistent_workers=persistent_workers,
                                multiprocessing_context=multiprocessing_context,
                                pin_memory=pin_memory)
    else:
        dataloader = DataLoader(dataset=dataset,
                                sampler=sampler,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=pin_memory)
    return dataloader


class read_era5:
    """label store -> (1, C, H, W)。

    对应参考版的 ``read_era5``（``xr.open_zarr(...).data``），这里读 ``label`` 变量，
    并把 71 通道裁成"纯 ERA5"的 70 个（见 ``LABEL_IDX``）。
    """

    def __init__(self, data_dir):
        self.store = zarr.open(str(data_dir), 'r')
        self.dates = decode_time(self.store, 'time')
        self.channum = int(self.store['label'].shape[1])
        self.channel = [str(c) for c in self.store['channel'][:]]

    def prepare_data(self, date):
        i = self.dates.get_indexer([date])[0]
        if i < 0:
            return None
        data = np.asarray(self.store['label'][int(i)], dtype='float32')[LABEL_IDX]
        return torch.from_numpy(data).unsqueeze(0)


class read_fcst:
    """FuXi store -> (1, C, H, W)；背景 = date - fcst_step 小时起报、取 lead fcst_step。

    对应参考版的 ``read_fcst``（文件名里带 lead，按日期取那个文件）。本 store 只有一个
    lead，按 ``init`` 轴取第 ``fcst_step`` 小时前的那一帧。
    """

    def __init__(self, data_dir, fcst_step):
        self.store = zarr.open(str(data_dir), 'r')
        self.dates = decode_time(self.store, 'init')
        steps = np.asarray(self.store['step'][:], dtype=int)
        if int(fcst_step) not in steps:
            raise ValueError(f'{data_dir}: 没有 lead {fcst_step} h，store 里只有 {list(steps)}')
        self.step_index = int(np.where(steps == int(fcst_step))[0][0])
        self.fcst_step = int(fcst_step)
        self.channum = int(self.store['z'].shape[2])

    def prepare_data(self, date):
        init = pd.Timestamp(date) - pd.Timedelta(hours=self.fcst_step)
        i = self.dates.get_indexer([init])[0]
        if i < 0:
            return None
        data = np.asarray(self.store['z'][int(i), self.step_index], dtype='float32')
        return torch.from_numpy(data).unsqueeze(0)


class read_obs:
    """NGL ZTD store -> (frames, C, H, W)，返回的是标准化前的原始 ZTD（单位 mm）。

    对应参考版的 ``read_obs``：同样在最后拼成多帧张量、缺帧填 NaN，标准化统计照样放在
    ``mean_std_dict['instrument']`` 里，由 ``process_obs`` 使用。

    与参考版唯一的语义差别：参考版的窗口以分析时刻为中心
    （``date - frames/2`` ~ ``date + frames/2 - 1``），本版取**分析时刻之前
    ``obs_frames`` 帧**（73 x 5 min = 前 6 小时）——73 是奇数、且这里是 5 分钟帧，
    居中会落到非整 5 分钟的时刻上。
    """

    def __init__(self, data_dir, stat_dir, data_frames, data_channum, drop_rate=0.0,
                 frame_minutes=5, grid_hw=(80, 120)):
        self.drop_rate = drop_rate
        self.data_frames = int(data_frames)
        self.data_channum = int(data_channum)
        self.frame_minutes = int(frame_minutes)
        self.grid_hw = tuple(int(v) for v in grid_hw)
        self.store = zarr.open(str(data_dir), 'r')
        self.dates = decode_time(self.store, 'time')

        instrument_mean = np.asarray(self.store['ztd_train_mean'][:]).reshape(-1)
        instrument_mean = torch.from_numpy(np.nan_to_num(instrument_mean))
        instrument_mean = instrument_mean.unsqueeze(0).float()

        instrument_std = np.asarray(self.store['ztd_train_std'][:]).reshape(-1)
        instrument_std = torch.from_numpy(np.nan_to_num(instrument_std))
        instrument_std = instrument_std.unsqueeze(0).float()

        self.mean_std_dict = {
            'instrument': [instrument_mean, instrument_std],
        }
        # 注意：store 里的 ztd 已经是标准化值，而参考版的契约是
        # "read_obs 返回物理量(mm) + mean_std_dict 给出标准化统计"，
        # 由 process_obs 做 (x-mean)/std。所以这里先反标准化回 mm，
        # 否则 process_obs 会二次标准化，把观测压成一个常数。
        self.z_mu = float(np.nan_to_num(instrument_mean.reshape(-1)[0]))
        self.z_sd = float(np.nan_to_num(instrument_std.reshape(-1)[0]))

    def prepare_data(self, date):
        end = self.dates.get_indexer([date])[0]
        if end < 0:
            return None
        idx = np.arange(end - self.data_frames + 1, end + 1)
        obs = []
        kk = 0
        for i in idx:
            if 0 <= i < len(self.dates):
                # 标准化值 -> mm（NaN 原样保留）
                arr = np.asarray(self.store['ztd'][int(i)], dtype='float32') * self.z_sd + self.z_mu
                # 每帧 (C, H, W) -> (1, C, H, W)，与参考版逐帧 nc 的形状对齐
                data = torch.from_numpy(arr).reshape(1, self.data_channum, *self.grid_hw)
                kk += 1
            else:
                data = torch.full((1, self.data_channum, *self.grid_hw), float('nan'))
            obs.append(data)
        if kk == 0:
            return None
        obs = torch.concat(obs, dim=0)
        return obs


class MyDataset(Dataset):
    def __init__(self,
                 era5_dir,
                 fcst_dir,
                 fcst_step,
                 obs_dir,
                 obs_frames,
                 obs_stat_dir,
                 obs_channum,
                 dates_range,
                 obs_frame_minutes=5,
                 grid_hw=(80, 120),
                 seed=2000):
        start_date = pd.to_datetime(dates_range[0], format="%Y%m%d%H")
        end_date = pd.to_datetime(dates_range[1], format="%Y%m%d%H")
        self.dates = pd.date_range(start_date, end_date, freq=f"6h")
        # 固定随机种子：某个时刻缺数据时（例如 test 区间末尾超出 label store 范围）
        # __getitem__ 会"随机换一个时刻"，这里用固定种子的 RNG，保证每次运行换到的
        # 是同一个时刻 —— 否则每次评估的样本集合都不一样，结果无法复现。
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self.read_era5 = read_era5(era5_dir)
        self.read_fcst = read_fcst(fcst_dir, fcst_step)
        self.read_obs = read_obs(obs_dir, obs_stat_dir, obs_frames, obs_channum,
                                 frame_minutes=obs_frame_minutes, grid_hw=grid_hw)

        print(f'[Dataset Time] ||| {self.dates[0]} ~ {self.dates[-1]}')
        print(f'[Dataset Num] ||| {len(self.dates)}')

    def __len__(self):
        return len(self.dates)

    def _rand_another(self):
        # 固定种子（self._rng 在 __init__ 里用 seed 初始化）：换到的时刻每次都一样
        return int(self._rng.integers(len(self.dates)))

    def __getitem__(self, idx):
        while True:
            data_fcst, data_era5, data_obs = self.prepare_data(idx)
            if any(var is None for var in [data_fcst, data_era5, data_obs]):
                idx = self._rand_another()
                continue
            return data_fcst, data_era5, data_obs

    def prepare_data(self, idx):
        date = self.dates[idx]

        readtime_era5 = time.perf_counter()
        data_era5 = self.read_era5.prepare_data(date)

        readtime_fcst = time.perf_counter()
        data_fcst = self.read_fcst.prepare_data(date)

        readtime_obs = time.perf_counter()
        data_obs = self.read_obs.prepare_data(date)

        readtime_end = time.perf_counter()

        if any(var is None for var in [data_fcst, data_era5, data_obs]):
            return None, None, None

        return torch.nan_to_num(data_fcst), \
               torch.nan_to_num(data_era5), \
               data_obs
