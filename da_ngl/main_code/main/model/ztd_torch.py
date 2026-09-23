"""Differentiable ZTD observation operator (torch) on the station cells.

``preprocessing/ztd_operator.py`` implements this physics in numpy and is what
produced the ``ztd_fuxi_*`` stores.  This module mirrors
``ztd_profile_surface`` (validated against those stores by
``preprocessing/check_ztd_torch.py``) but runs on the analysis field the network
emits -- which lives in the ERA5 standardised space -- and is differentiable, so
the observation-consistency loss can pull the analysis towards the GNSS ZTD.

Conventions, identical to the numpy version:

* ``t``/``r`` are the 13 ERA5 pressure levels, 50 hPa first, 1000 hPa last
* the store carries ``msl`` in Pa (divided by 100 here)
* heights are the GNSS station heights [m] -- a 100 m error is already ~12 hPa
  of surface pressure, so ETOPO is deliberately *not* used
* ``ZHD = 0.0022768 p_s / (1 - 0.00266 cos 2phi - 0.00028 h_km)``
* the bottom node is the station surface: ``t2m`` and ``e_sfc = RH_low/100 es``
* levels at or below ``p_s`` collapse onto ``p_s`` (zero thickness), so the
  model's below-ground extrapolation cannot leak into the column

NaN in (NaN out): a cell whose inputs are missing yields NaN, which the loss
masks off rather than silently treating as a zero error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# the physics constants and the channel order live in preprocessing/ so the
# numpy builder and this torch operator cannot drift apart
_PREPROC = Path(__file__).resolve().parents[3] / "preprocessing"
if str(_PREPROC) not in sys.path:
    sys.path.insert(0, str(_PREPROC))

from common import CHANNELS, era5_channel_stats  # noqa: E402
from ztd_operator import (EPS_MW_MD, G_0, K1_BEVIS, K2_BEVIS, K2_PRIME, K3,  # noqa: E402
                          K3_BEVIS, LEV_HPA, R_D, ZHD_COEF)

__all__ = ["vapor_pressure_torch", "zhd_zwd_torch", "ztd_profile_surface_torch",
           "StationZTD", "OP_CHANNELS", "LEV_HPA"]

# the 28 state channels the operator reads, in the order it wants them
T_LEVEL_CHANNELS = [CHANNELS.index(f"t{int(L)}") for L in LEV_HPA]      # 13..25
R_LEVEL_CHANNELS = [CHANNELS.index(f"r{int(L)}") for L in LEV_HPA]      # 52..64
Z_LEVEL_CHANNELS = [CHANNELS.index(f"z{int(L)}") for L in LEV_HPA]      # 0..12
T2M_CHANNEL, MSL_CHANNEL = CHANNELS.index("t2m"), CHANNELS.index("msl")  # 65, 68
# 方法 E（默认）：13 层 t + 13 层 r + t2m + msl + 13 层 z（位势 -> 层高）
OP_CHANNELS = (T_LEVEL_CHANNELS + R_LEVEL_CHANNELS + [T2M_CHANNEL, MSL_CHANNEL]
               + Z_LEVEL_CHANNELS)

N_LEV = len(LEV_HPA)
_LEV_TENSOR = np.asarray(LEV_HPA, dtype=np.float64)


def vapor_pressure_torch(temp_k, rh_percent, over_ice: bool = False):
    """Water-vapour pressure ``e`` [hPa] from ``T`` [K] and ``RH`` [%]."""
    tc = temp_k - 273.15
    if over_ice:
        es = 6.1078 * torch.exp(22.44294 * tc / (tc + 272.4406))
    else:
        es = 6.1078 * torch.exp(17.269 * tc / (tc + 237.3))
    return rh_percent * 0.01 * es


def zhd_torch(p_s_hpa, lat_deg, height_m):
    """Saastamoinen zenith hydrostatic delay [m]."""
    h_km = height_m / 1000.0
    denom = 1.0 - 0.00266 * torch.cos(2.0 * torch.deg2rad(lat_deg)) - 0.00028 * h_km
    return ZHD_COEF * p_s_hpa / denom


def zhd_zwd_torch(t_lev, r_lev, t2m, p_s, height_m, lat_deg,
                  lev_hpa=None, over_ice: bool = False):
    """``(ZHD, ZWD)`` [m]; torch mirror of ``ztd_operator.ztd_profile_surface``.

    ``t_lev`` / ``r_lev`` are ``(..., 13)`` in ``lev_hpa`` order (50 hPa first);
    ``p_s`` is the station surface pressure [hPa].  The two parts are returned
    separately so ``StationZTD(freeze_zhd=True)`` can keep the hydrostatic term
    (and the whole column geometry) from the background and only let the wet
    part carry gradient into the analysis.
    """
    p = torch.as_tensor(
        np.asarray(LEV_HPA if lev_hpa is None else lev_hpa, dtype=np.float64),
        dtype=torch.float32, device=p_s.device)

    # lowest level still above ground (last True along the level axis); columns
    # with no valid level -- NaN surface pressure, i.e. the masked cells -- fall
    # back to the lowest level, exactly like the numpy version
    valid = p < p_s.unsqueeze(-1)
    rev = valid.flip(-1).to(torch.int64).argmax(dim=-1)
    idx = torch.where(valid.any(dim=-1), N_LEV - 1 - rev,
                      torch.full_like(rev, N_LEV - 1))
    rh_low = torch.gather(r_lev, -1, idx.unsqueeze(-1)).squeeze(-1)

    e_sfc = vapor_pressure_torch(t2m, rh_low, over_ice)
    e_lev = vapor_pressure_torch(t_lev, r_lev, over_ice)

    # collapse every level at or below the surface onto p_s, then append the
    # surface itself as the bottom node
    below = p >= p_s.unsqueeze(-1)
    p_eff = torch.where(below, p_s.unsqueeze(-1), p)
    t_eff = torch.where(below, t2m.unsqueeze(-1), t_lev)
    e_eff = torch.where(below, e_sfc.unsqueeze(-1), e_lev)

    P = torch.cat([p_eff, p_s.unsqueeze(-1)], dim=-1)
    T = torch.cat([t_eff, t2m.unsqueeze(-1)], dim=-1)
    E = torch.cat([e_eff, e_sfc.unsqueeze(-1)], dim=-1)

    integrand = K2_PRIME * E / P + K3 * E / (T * P)
    dp = P[..., 1:] - P[..., :-1]
    trap = 0.5 * (integrand[..., 1:] + integrand[..., :-1]) * dp
    zwd = 1e-6 * (R_D / G_0) * trap.sum(dim=-1)
    return zhd_torch(p_s, lat_deg, height_m), zwd


def ztd_profile_surface_torch(t_lev, r_lev, t2m, p_s, height_m, lat_deg,
                              lev_hpa=None, over_ice: bool = False):
    """Column ZTD [m] = ZHD + ZWD; the single-call form of the operator."""
    zhd, zwd = zhd_zwd_torch(t_lev, r_lev, t2m, p_s, height_m, lat_deg,
                             lev_hpa=lev_hpa, over_ice=over_ice)
    return zhd + zwd


def zdz_torch(t_lev, r_lev, t2m, p_s, height_m, z_lev_m,
              lev_hpa=None, over_ice: bool = False,
              k1=K1_BEVIS, k2=K2_BEVIS, k3=K3_BEVIS, eps=EPS_MW_MD,
              top_correction: bool = True):
    """方法 E 的可微实现：在几何高度上梯形积分，返回 ``(ZHD, ZWD)`` [m]。

    与 ``preprocessing/ztd_operator.ztd_profile_zdz`` 逐项对应：

    * ``p >= p_s`` 的层塌缩到地面（高度置 h_s、值换成地面值），层厚归零；
    * 节点按 "地面 + 由低到高的 13 层" 排列；
    * ``N_h = k1(p-e)/T``、``N_w = (k2-eps*k1)e/T + k3 e/T^2``，梯形积分；
    * 加上 ``p_top`` 以上的干空气柱解析项 ``1e-6*k1*(R_d/g)*p_top``。

    形状：``t_lev/r_lev/z_lev_m`` 为 ``(..., n_lev)``，其余为 ``(...)``。
    """
    p = torch.as_tensor(np.asarray(LEV_HPA if lev_hpa is None else lev_hpa, dtype=np.float64),
                        dtype=t_lev.dtype, device=t_lev.device)
    n_lev = p.numel()

    valid = p < p_s.unsqueeze(-1)                     # (..., n_lev) 地面之上
    rev = valid.flip(-1).to(torch.int64).argmax(dim=-1)
    idx = torch.where(valid.any(dim=-1), n_lev - 1 - rev, torch.full_like(rev, n_lev - 1))
    rh_low = torch.gather(r_lev, -1, idx.unsqueeze(-1)).squeeze(-1)

    e_sfc = vapor_pressure_torch(t2m, rh_low, over_ice)
    e_lev = vapor_pressure_torch(t_lev, r_lev, over_ice)

    below = ~valid
    p_eff = torch.where(below, p_s.unsqueeze(-1), p)
    t_eff = torch.where(below, t2m.unsqueeze(-1), t_lev)
    e_eff = torch.where(below, e_sfc.unsqueeze(-1), e_lev)
    h_eff = torch.where(below, height_m.unsqueeze(-1), z_lev_m)

    flip = lambda a: a.flip(-1)
    P = torch.cat([p_s.unsqueeze(-1), flip(p_eff)], dim=-1)
    T = torch.cat([t2m.unsqueeze(-1), flip(t_eff)], dim=-1)
    E = torch.cat([e_sfc.unsqueeze(-1), flip(e_eff)], dim=-1)
    H = torch.cat([height_m.unsqueeze(-1), flip(h_eff)], dim=-1)

    N_h = k1 * (P - E) / T
    N_w = (k2 - eps * k1) * (E / T) + k3 * (E / T ** 2)
    dh = (H[..., 1:] - H[..., :-1]).clamp_min(0)
    zhd = 1e-6 * ((N_h[..., 1:] + N_h[..., :-1]) * 0.5 * dh).sum(dim=-1)
    zwd = 1e-6 * ((N_w[..., 1:] + N_w[..., :-1]) * 0.5 * dh).sum(dim=-1)
    if top_correction:
        zhd = zhd + 1e-6 * k1 * (R_D / G_0) * float(np.asarray(lev_hpa if lev_hpa is not None
                                                               else LEV_HPA)[0])
    return zhd, zwd


class StationZTD(nn.Module):
    """``H``: analysis field -> GNSS ZTD [mm] at the 1378 station cells.

    ``analysis`` is ``(B, 1, C, H, W)`` or ``(B, C, H, W)`` in the ERA5
    standardised space; only the first 69 (state) channels are read, so the
    extra ``tp`` channel an ``out_chans = 70`` model emits is ignored.

    ``freeze_zhd`` (config ``obs_freeze_zhd``) evaluates

        H*(x_a) = ZHD(x_bg) + ZWD(x_a)

    i.e. the hydrostatic delay -- and with it the column geometry: the surface
    pressure, which levels sit below ground, and the ``p_s`` of the surface node
    -- comes from the *background*, leaving only the thermodynamic column
    (``t`` / ``r`` / ``t2m``) for the analysis to move.  Two reasons:

    * physics -- ``ZHD`` is 90 % of the delay and only reflects surface
      pressure; "GNSS ZTD tells us about the moisture column" is the honest
      reading, and it is what the linear-DA baseline exploited (it improved
      r500..r925 and z850, nothing hydrostatic).
    * the unfrozen operator leaves a cheap lever: ``ZHD = 2.2768 mm/hPa``, so
      the network can buy millimetres of ZTD fit by nudging ``msl``.  In the
      ``oc0.2`` run that wrecked ``msl`` (-121 % at the station cells, +127 % of
      the net MAE change) while the humidity channels still gained 1-2 %.

    At ``x_a = x_bg`` the frozen and unfrozen forms agree exactly, so switching
    the flag changes only *which* channels the observation can move.
    """

    def __init__(self, cfg, freeze_zhd=None, freeze_geometry=None):
        super().__init__()
        from ..utils.utils import station_geometry

        iy, ix, height_m, station_id = station_geometry(cfg)
        mean, std = era5_channel_stats()
        mean = np.asarray(mean[:69], dtype=np.float64)[OP_CHANNELS]
        std = np.asarray(std[:69], dtype=np.float64)[OP_CHANNELS]
        lat_axis = np.asarray(cfg.lat, dtype=np.float64)

        self.n_state = 69
        self.n_cell = int(iy.size)
        self.freeze_zhd = (bool(getattr(cfg, 'obs_freeze_zhd', False))
                           if freeze_zhd is None else bool(freeze_zhd))
        # freeze_geometry: 层高与地面气压都取自背景（detach），只让热力/湿度廓线带梯度。
        # 方法 E 的层高来自 z 通道，而 z 是网络输出之一，若不冻结就等于开了一个新杠杆。
        # 兼容旧名：obs_freeze_zhd=True 等价于冻结几何，freeze_zhd 参数同理。
        fg = freeze_geometry
        if fg is None:
            fg = getattr(cfg, 'obs_freeze_geometry', None)
        if fg is None:
            fg = getattr(cfg, 'obs_freeze_zhd', False)
        if freeze_zhd is True:
            fg = True
        self.freeze_geometry = bool(fg)
        self.station_id = [str(s) for s in station_id]
        self.register_buffer("iy", torch.as_tensor(iy, dtype=torch.long))
        self.register_buffer("ix", torch.as_tensor(ix, dtype=torch.long))
        self.register_buffer("height_m", torch.as_tensor(height_m, dtype=torch.float32))
        self.register_buffer("lat_deg", torch.as_tensor(lat_axis[iy], dtype=torch.float32))
        self.register_buffer("chan_index", torch.as_tensor(OP_CHANNELS, dtype=torch.long))
        self.register_buffer("op_mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("op_std", torch.as_tensor(std, dtype=torch.float32))

    @property
    def cells(self):
        """(H, W) long indices of the station cells, for masking other fields."""
        return self.iy, self.ix

    def _physics(self, field):
        """Standardised field -> the 28 physical channels at the station cells."""
        if field.dim() == 5:
            field = field[:, 0]
        x = field.float()
        if x.shape[1] > self.n_state:
            x = x[:, :self.n_state]
        # (B, C, H, W) -> (B, 28, H, W) -> (B, 28, n_cell)
        sub = x[:, self.chan_index][:, :, self.iy, self.ix]
        return self.op_mean.view(1, -1, 1) + self.op_std.view(1, -1, 1) * sub

    def _surface_pressure(self, ph, h):
        """Hypsometric reduction of ``msl`` to the station height [hPa]."""
        t2m, msl = ph[:, 2 * N_LEV], ph[:, 2 * N_LEV + 1]       # (B, n_cell)
        return msl / 100.0 * torch.exp(-G_0 * h / (R_D * t2m))

    def forward(self, analysis, background=None):
        """方法 E：ZTD [mm] = 几何高度梯形积分(ZHD) + 湿项积分 + 顶层干柱修正。"""
        ph = self._physics(analysis)
        t_lev = ph[:, :N_LEV].transpose(1, 2)                   # (B, n_cell, 13)
        r_lev = ph[:, N_LEV:2 * N_LEV].transpose(1, 2)
        t2m = ph[:, 2 * N_LEV]
        z_lev = ph[:, 2 * N_LEV + 2:].transpose(1, 2) / G_0     # 位势 -> 层高 [m]

        # h 要扩到 batch 维，后面要跟 (B, n_cell, n_lev) 做 cat
        h = self.height_m.view(1, -1).expand(ph.shape[0], -1)
        p_s = self._surface_pressure(ph, h)

        if self.freeze_geometry:
            if background is None:
                raise ValueError('StationZTD(freeze_geometry=True) needs ``background``')
            ph_bg = self._physics(background)
            t2m_bg = ph_bg[:, 2 * N_LEV]
            msl_bg = ph_bg[:, 2 * N_LEV + 1]
            # 层高与地面气压都来自背景，且 detached：柱几何是给定的，
            # 观测一致性不允许通过改 z 或 msl 去"买"ZTD 拟合。
            z_lev = (ph_bg[:, 2 * N_LEV + 2:].transpose(1, 2) / G_0).detach()
            p_s = (msl_bg / 100.0 * torch.exp(-G_0 * h / (R_D * t2m_bg))).detach()

        zhd, zwd = zdz_torch(t_lev, r_lev, t2m, p_s, h, z_lev)
        return (zhd + zwd) * 1000.0
