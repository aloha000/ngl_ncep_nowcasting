import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR
from torch import optim
from torch import nn


def build_optimizer(opt_type, learning_rate, weight_decay, model, scheduler, step_size=None, T_max=None):
    if opt_type == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=learning_rate,
                               weight_decay=weight_decay)
    elif opt_type == 'AdamW':
        optimizer = optim.AdamW(model.parameters(),
                                lr=learning_rate,
                                weight_decay=weight_decay)
    elif opt_type == 'SGD':
        optimizer = optim.SGD(model.parameters(),
                              lr=learning_rate,
                              weight_decay=weight_decay)
    # 设置学习率调整策略
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
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
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
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


class WarmupScheduler:
    def __init__(self, optimizer, start_lr=1e-8, stop_lr=1e-3, warmup_steps=10000):
        super(WarmupScheduler, self).__init__()
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.stop_lr = stop_lr
        self.steps = float(warmup_steps)
        self.count = 0.

    def __call__(self):
        if self.count < self.steps and self.start_lr < self.stop_lr:
            self.count += 1
            next_lr = (self.count / self.steps) * (self.stop_lr - self.start_lr) + self.start_lr
            self.optimizer.param_groups[0]['lr'] = next_lr
            return True
        else:
            return False


class mse(nn.Module):
    def __init__(self, obs_rect=None, lat=None):
        super(mse, self).__init__()
        # 参考版写死 720 个纬度（全球 0.25 度）；本版允许传入实际网格的纬度
        self.lat = (torch.linspace(90, -90, 720) if lat is None
                    else torch.as_tensor(np.asarray(lat, dtype='float32')))
        self.obs_rect = obs_rect
        if self.obs_rect is not None:
            self.lat = self.lat[self.obs_rect[0]:self.obs_rect[1]]
        # 逐格掩膜（0/1）：只在这些格点上算 loss，其它格点既不进分子也不进分母。
        # 由 set_cell_mask() 设置，默认 None（= 全域，和参考版完全一致）。
        self.cell_mask = None

    def set_cell_mask(self, mask):
        '''设置 (H, W) 的 0/1 掩膜；None 表示全域。

        例如"只监督 ZTD 站点周围 3 格"，就是传入站点格点膨胀 3 格后的布尔图。
        '''
        if mask is None:
            self.cell_mask = None
            return self
        m = torch.as_tensor(np.asarray(mask, dtype='float32'))
        if m.ndim != 2:
            raise ValueError('cell_mask 必须是 (H, W)，收到 ' + str(tuple(m.shape)))
        if self.obs_rect is not None:      # 与 obs_rect 一起用时按同样的窗口裁
            m = m[self.obs_rect[0]:self.obs_rect[1], self.obs_rect[2]:self.obs_rect[3]]
        self.cell_mask = m
        return self

    def forward(self, outputs, labels):
        if self.obs_rect is not None:
            outputs = outputs[:, :, self.obs_rect[0]:self.obs_rect[1], self.obs_rect[2]:self.obs_rect[3]]
            labels = labels[:, :, self.obs_rect[0]:self.obs_rect[1], self.obs_rect[2]:self.obs_rect[3]]
        
        self.lat = self.lat.to(outputs.device)
        wlat = torch.cos(torch.deg2rad(self.lat))[:, None]
        wlat /= wlat.mean()
        if self.cell_mask is None:
            error = (outputs - labels) ** 2 * wlat  # B C H W
            return error.mean()
        # 掩膜版：按 (纬度权重 x 掩膜) 加权平均。
        # 分母要乘上 B 和 C，才是"每个 (样本, 通道) 一个加权平均"再对 B、C 取平均，
        # 这样掩膜全 1 时与全域口径的数值完全一致（漏掉 B*C 会让 loss 大 B*C 倍）。
        w = wlat * self.cell_mask.to(outputs.device).view(1, 1, *self.cell_mask.shape)
        error = (outputs - labels) ** 2 * w
        # 归一化：numel/(H*W) = B*T*C（4 维/5 维都适用），再乘掩膜内的纬度权重和；
        # 这样掩膜全 1 时与全域口径数值完全一致。
        n_bc = error.numel() // (error.shape[-2] * error.shape[-1])
        denom = n_bc * w.sum()
        return error.sum() / denom.clamp_min(1e-6)


class mae(nn.Module):
    def __init__(self, obs_rect=None, lat=None):
        super(mae, self).__init__()
        # 参考版写死 720 个纬度（全球 0.25 度）；本版允许传入实际网格的纬度
        self.lat = (torch.linspace(90, -90, 720) if lat is None
                    else torch.as_tensor(np.asarray(lat, dtype='float32')))
        self.obs_rect = obs_rect
        if self.obs_rect is not None:
            self.lat = self.lat[self.obs_rect[0]:self.obs_rect[1]]
        # 逐格掩膜（0/1）：只在这些格点上算 loss，其它格点既不进分子也不进分母。
        # 由 set_cell_mask() 设置，默认 None（= 全域，和参考版完全一致）。
        self.cell_mask = None

    def set_cell_mask(self, mask):
        '''设置 (H, W) 的 0/1 掩膜；None 表示全域。

        例如"只监督 ZTD 站点周围 3 格"，就是传入站点格点膨胀 3 格后的布尔图。
        '''
        if mask is None:
            self.cell_mask = None
            return self
        m = torch.as_tensor(np.asarray(mask, dtype='float32'))
        if m.ndim != 2:
            raise ValueError('cell_mask 必须是 (H, W)，收到 ' + str(tuple(m.shape)))
        if self.obs_rect is not None:      # 与 obs_rect 一起用时按同样的窗口裁
            m = m[self.obs_rect[0]:self.obs_rect[1], self.obs_rect[2]:self.obs_rect[3]]
        self.cell_mask = m
        return self
        
    def forward(self, outputs, labels):
        if self.obs_rect is not None:
            outputs = outputs[:, :, self.obs_rect[0]:self.obs_rect[1], self.obs_rect[2]:self.obs_rect[3]]
            labels = labels[:, :, self.obs_rect[0]:self.obs_rect[1], self.obs_rect[2]:self.obs_rect[3]]
        
        self.lat = self.lat.to(outputs.device)
        wlat = torch.cos(torch.deg2rad(self.lat))[:, None]
        wlat /= wlat.mean()
        # wlat=1
        if self.cell_mask is None:
            error = torch.abs(outputs - labels) * wlat  # B C H W
            return error.mean()
        # 掩膜版：按 (纬度权重 x 掩膜) 加权平均。
        # 分母要乘上 B 和 C，才是"每个 (样本, 通道) 一个加权平均"再对 B、C 取平均，
        # 这样掩膜全 1 时与全域口径的数值完全一致（漏掉 B*C 会让 loss 大 B*C 倍）。
        w = wlat * self.cell_mask.to(outputs.device).view(1, 1, *self.cell_mask.shape)
        error = torch.abs(outputs - labels) * w
        # 归一化：numel/(H*W) = B*T*C（4 维/5 维都适用），再乘掩膜内的纬度权重和；
        # 这样掩膜全 1 时与全域口径数值完全一致。
        n_bc = error.numel() // (error.shape[-2] * error.shape[-1])
        denom = n_bc * w.sum()
        return error.sum() / denom.clamp_min(1e-6)