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

__all__ = ["build_optimizer", "EarlyStopping", "WarmupScheduler", "mae", "mse", "GRID_LAT"]


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
    """Latitude-weighted (cos lat, mean-normalised) MAE / MSE over all channels."""

    def __init__(self, lat=None, kind='mae', ignore_nan=True):
        super().__init__()
        lat = GRID_LAT if lat is None else np.asarray(lat, dtype=np.float32)
        w = torch.cos(torch.deg2rad(torch.as_tensor(lat, dtype=torch.float32)))
        w = w / w.mean()
        self.register_buffer('wlat', w.view(1, 1, -1, 1))
        self.kind = kind
        self.ignore_nan = ignore_nan

    def forward(self, outputs, labels):
        outputs = outputs.float()
        labels = labels.float()
        w = self.wlat.to(outputs.device)
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
    def __init__(self, lat=None, ignore_nan=True):
        super().__init__(lat, 'mae', ignore_nan)


class mse(_LatWeightedLoss):
    def __init__(self, lat=None, ignore_nan=True):
        super().__init__(lat, 'mse', ignore_nan)
