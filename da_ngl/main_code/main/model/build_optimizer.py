"""Optimiser / scheduler / losses (adapted from the train_packet).

The original losses hard-coded ``torch.linspace(90, -90, 720)`` for the
latitude weighting.  Our data is a 80x120 regional crop, so the weights are
built from the real regional latitude axis and the loss is NaN-safe (the ERA5
label has a few missing cells).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, StepLR

# unified grid: lat 36.50..56.25 (80), lon -5.25..24.50 (120)
GRID_LAT = np.round(np.arange(36.50, 56.25 + 1e-9, 0.25), 6)

__all__ = ["build_optimizer", "EarlyStopping", "WarmupScheduler", "mae", "mse",
           "ObsConsistencyLoss", "IncrementPenalty", "GRID_LAT"]


def build_optimizer(opt_type, learning_rate, weight_decay, model, scheduler,
                    step_size=None, T_max=None):
    if opt_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif opt_type == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif opt_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f'unknown opt_type {opt_type}')

    if scheduler == 'StepLR':
        scheduler = StepLR(optimizer, step_size=step_size, gamma=0.2)
    elif scheduler == 'MultiStepLR':
        scheduler = MultiStepLR(optimizer, milestones=step_size, gamma=0.2)
    elif scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max)
    else:
        scheduler = None
    return optimizer, scheduler


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class WarmupScheduler:
    def __init__(self, optimizer, start_lr=1e-8, stop_lr=1e-3, warmup_steps=10000):
        super().__init__()
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.stop_lr = stop_lr
        self.steps = float(warmup_steps)
        self.count = 0.0

    def __call__(self):
        if self.count < self.steps and self.start_lr < self.stop_lr:
            self.count += 1
            next_lr = (self.count / self.steps) * (self.stop_lr - self.start_lr) + self.start_lr
            self.optimizer.param_groups[0]['lr'] = next_lr
            return True
        return False


class _LatWeightedLoss(nn.Module):
    """Latitude-weighted (cos lat, mean-normalised) MAE / MSE over all channels.

    ``lat_weight=False`` 关掉纬度加权（权重全 1，退化成逐格点等权）。

    An optional per-cell weight map (``set_cell_weight``) is multiplied on top of
    the latitude weight -- used to down-weight the 8222 grid cells that carry no
    GNSS station, where the background error is unpredictable and the loss only
    pushes the output back towards the background.
    """

    def __init__(self, lat=None, kind='mae', ignore_nan=True, lat_weight=True):
        super().__init__()
        self.lat_deg = np.asarray(GRID_LAT if lat is None else lat, dtype=np.float32)
        self.lat_weight = bool(lat_weight)
        self.register_buffer('wlat', self._lat_weights(self.lat_deg, self.lat_weight))
        self.register_buffer('cellw', torch.ones(1, 1, 1, 1))
        self.has_cellw = False
        self.kind = kind
        self.ignore_nan = ignore_nan

    @staticmethod
    def _lat_weights(lat_deg, lat_weight: bool = True):
        """(1, 1, H, 1) 的纬度权重。

        ``lat_weight=False`` 时权重恒为 1，除以均值后还是 1 -- 加权平均就退化成
        普通的逐格点等权 MAE，等于把纬度加权关掉。
        """
        lat = np.asarray(lat_deg, dtype=np.float32).reshape(-1)
        if lat_weight:
            w = torch.cos(torch.deg2rad(torch.as_tensor(lat, dtype=torch.float32)))
        else:
            w = torch.ones(lat.size, dtype=torch.float32)
        return (w / w.mean()).view(1, 1, -1, 1)

    def set_lat_weight(self, on: bool):
        """开关纬度加权。

        configs.py 是在 import 时就把 ``loss_fn`` 建好的，而 ``--set lat_weight=...``
        在那之后才生效，所以由 train_FSDP.init_dist 调这个把实例同步过来。
        """
        self.lat_weight = bool(on)
        self.register_buffer('wlat', self._lat_weights(self.lat_deg, self.lat_weight))
        return self

    def set_cell_weight(self, weight):
        """(H, W) array of non-negative cell weights; all ones reproduces the
        historical behaviour exactly (the buffer is dropped so the maths below is
        bit-identical to the unweighted case)."""
        w = np.asarray(weight, dtype=np.float32)
        if w.ndim != 2:
            raise ValueError(f'cell weight must be (H, W), got shape {w.shape}')
        if np.allclose(w, 1.0):
            self.cellw = torch.ones(1, 1, 1, 1)
            self.has_cellw = False
            return self
        self.cellw = torch.as_tensor(w).view(1, 1, *w.shape)
        self.has_cellw = True
        return self

    def forward(self, outputs, labels):
        outputs = outputs.float()
        labels = labels.float()
        w = self.wlat.to(outputs.device)
        if self.has_cellw:
            w = w * self.cellw.to(outputs.device)
        if self.ignore_nan:
            valid = torch.isfinite(labels)
            # fill the missing labels *before* differencing: NaN * 0 is NaN
            diff = outputs - torch.nan_to_num(labels)
            err = diff.abs() if self.kind == 'mae' else diff ** 2
            err = torch.nan_to_num(err)
            m = valid.float()
            denom = (w * m).sum().clamp_min(1e-6)
            return ((err * w) * m).sum() / denom
        diff = outputs - labels
        err = diff.abs() if self.kind == 'mae' else diff ** 2
        return (err * w).mean()


class mae(_LatWeightedLoss):
    def __init__(self, lat=None, ignore_nan=True, lat_weight=True):
        super().__init__(lat, 'mae', ignore_nan, lat_weight)


class mse(_LatWeightedLoss):
    def __init__(self, lat=None, ignore_nan=True, lat_weight=True):
        super().__init__(lat, 'mse', ignore_nan, lat_weight)


class ObsConsistencyLoss(nn.Module):
    """Observation-consistency term (HANDOFF 11.8 item 1).

        J_o = lambda * mean_{valid station cells} | H(x_analysis) - obs' | / sigma_o

    ``H`` is the differentiable ZTD observation operator
    (:class:`main.model.ztd_torch.StationZTD`) applied to the analysis the
    network emits (optionally with its hydrostatic part frozen at the
    background, cf. ``obs_freeze_zhd``); ``obs'`` is the observed GNSS ZTD at the valid time ``T``
    [mm], optionally de-biased per station (item 2).  Everything else in the
    loss is a regression against ERA5 -- this is the only term that says in
    *which direction* the analysis should move towards the observations.

    Only cells with a finite observation and a finite ``H`` enter the mean, and
    the difference is taken on ``nan_to_num`` copies: ``where(valid, x, 0)`` on
    an already-NaN ``x`` is fine forward but its gradient is ``0 * nan = nan``.
    """

    def __init__(self, operator, weight=1.0, sigma_o_mm=11.0, bias=None):
        super().__init__()
        self.operator = operator
        self.weight = float(weight)
        self.sigma_o_mm = float(sigma_o_mm)
        self.register_buffer('bias', None)          # replaceable buffer, so the
        if bias is not None:                        # map follows .to(device)
            self.register_buffer('bias', torch.as_tensor(np.asarray(bias, dtype=np.float32)))
        self.last_mae_mm = float('nan')

    def forward(self, analysis, obs_mm, background=None):
        # ``background`` is only needed (and only used) when the operator runs in
        # the freeze-ZHD mode, see StationZTD
        ztd = self.operator(analysis.float(),
                            None if background is None else background.float())

        iy, ix = self.operator.cells
        target = obs_mm[:, iy, ix].float()                              # (B, n_cell) mm
        if self.bias is not None:
            target = target - self.bias
        valid = torch.isfinite(ztd) & torch.isfinite(target)
        err = (torch.nan_to_num(ztd) - torch.nan_to_num(target)).abs()
        err = torch.where(valid, err, torch.zeros_like(err))
        mae = err.sum() / valid.sum().clamp_min(1)
        self.last_mae_mm = float(mae.detach())
        return self.weight * mae / self.sigma_o_mm


class IncrementPenalty(nn.Module):
    """逐通道增量惩罚（对角 B 的软约束，HANDOFF 12.8 item 1b）。

        J_B = mu * mean_{c in 状态通道} ( mean_{cells} |x_a - x_b|_c ) / sigma_b,c

    ``sigma_b,c`` 是**背景误差**的逐通道标准差（``dataset/bg_err_std.npz``，
    由 ``preprocessing/build_bg_err_std.py`` 在 train 段统计）。因为 store 已经用
    气候态 std 标准化过，若直接用气候态 std 会让每个通道都等于 1、失去区分度；
    用背景误差 std 才能表达"这个通道动这么多算不算多"。

    实测（站点格）：z 族 0.026~0.067、msl 0.066、t 族 0.12、r 族 0.38~0.52。
    也就是说 msl 和 z 是"最便宜"的两个出口——正是网络拿来买 ZTD 拟合的杠杆。
    除以 sigma_b,c 之后，动 1 个单位的 z/msl 比动 1 个单位的 r 贵 6~10 倍，
    于是观测一致性要求的修正会被推向热力与湿度廓线。

    ``mask``（(H,W) 或 None）限制统计的格点；默认由 train_FSDP 传站点格掩膜，
    因为 ZTD 约束只在站格上起作用。
    """

    def __init__(self, sigma_b, weight=1.0, channel_slice=slice(0, 69), mask=None):
        super().__init__()
        sig = torch.as_tensor(np.asarray(sigma_b, dtype=np.float32))
        self.register_buffer("sigma", sig)
        self.weight = float(weight)
        self.ch0 = 0 if channel_slice.start is None else int(channel_slice.start)
        self.ch1 = int(channel_slice.stop)
        if mask is None:
            self.mask = None
        else:
            self.register_buffer("mask", torch.as_tensor(np.asarray(mask, dtype=np.float32)))
        self.last = float("nan")

    def forward(self, analysis, background):
        a = analysis.float()
        b = background.float()
        if a.dim() == 5:
            a = a[:, 0]
            b = b[:, 0]
        d = (a[:, self.ch0:self.ch1] - b[:, self.ch0:self.ch1]).abs()
        if self.mask is not None:
            m = self.mask.view(1, 1, *self.mask.shape).to(d.device)
            d = d * m
            # 每个通道的样本数 = batch × 掩膜内格点数（2026-09-21 修正：原来漏了 batch 维，
            # 导致惩罚被低估约 batch 倍；batch=2 时实测小 33.8 倍）
            denom = d.shape[0] * m.sum()
        else:
            denom = d.shape[0] * d.shape[2] * d.shape[3]
        per_chan = d.sum(dim=(0, 2, 3)) / denom.clamp_min(1.0)
        sig = self.sigma[self.ch0:self.ch1].to(per_chan.device).clamp_min(1e-6)
        val = (per_chan / sig).mean()
        self.last = float(val.detach())
        return self.weight * val
