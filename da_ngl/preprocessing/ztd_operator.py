#!/usr/bin/env python3
"""ZTD observation operator: meteorological fields -> zenith delays.

Two ways to get a total zenith delay out of the model fields:

1. ``ztd_surface``  -- the classic surface formulation (Saastamoinen + Tetens),
   exactly the form used in the project's reference snippet:

       es = 6.1078 * exp(17.269 * Tc / (Tc + 237.3))        [hPa, Tetens]
       e  = RH/100 * es                                     [hPa]
       ZHD = 0.0022768 * p_s / (1 - 0.00266 cos2phi - 0.00028 h_km)
       ZWD = 0.002277 * (1255 / T + 0.05) * e
       ZTD = ZHD + ZWD

   It needs *surface* temperature, pressure and relative humidity.

2. ``ztd_profile``  -- the column form, used for the FuXi background because the
   pressure-level store carries ``t`` and ``r`` on 13 levels but neither 2 m
   humidity nor surface pressure:

       ZHD = 0.0022768 * p_s / (1 - 0.00266 cos2phi - 0.00028 h_km)
       ZWD = 1e-6 * (R_d/g) * int [ k2'' * e/p + k3 * e/(T p) ] dp
           = 1e-6 * int ( k2'' e/T + k3 e/T^2 ) dz        (hydrostatic dz)

   with Bevis (1994) refractivity constants k2'' = 16.52 K/hPa, k3 = 3.776e5
   K^2/hPa, R_d = 287.05 J/(kg K), g = 9.80665 m/s^2.

Conventions / units (kept explicit because they are easy to get wrong):
  T          K      (pass ``temp_unit='C'`` for Celsius)
  p, msl     hPa    (the ERA5/FuXi ``msl`` channel is in Pa -- divide by 100)
  RH         %      (relative humidity)
  height     m      (station height or ETOPO2 cell elevation)
  lat        deg
  ZHD/ZWD/ZTD  m    (``*_mm`` variants return mm)

Water-vapour pressure ``e`` is always in hPa; ``es`` is saturated over water by
default (``over_ice=True`` switches to the ice formula, which matters below
0 C -- ERA5's ``r`` is archived with respect to water, so the default matches).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = [
    "saturation_vapor_pressure", "vapor_pressure",
    "surface_pressure_from_msl", "ztd_surface", "ztd_profile",
    "ztd_profile_surface", "ztd_profile_zdz", "zhd_top_correction",
    "elevation_from_etopo", "lev_pressure_hpa",
]

# ERA5 pressure levels carried by the store, in the same order as the channels
LEV_HPA = np.array([50, 100, 150, 200, 250, 300, 400, 500,
                    600, 700, 850, 925, 1000], dtype=np.float64)

K2_PRIME = 16.52        # K/hPa   (Bevis et al. 1994)
K3 = 3.776e5            # K^2/hPa
R_D = 287.05            # J/(kg K)
G_0 = 9.80665           # m/s^2
ZHD_COEF = 0.0022768    # m/hPa  (Saastamoinen)

ETOPO_PATH = Path("/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/"
                  "xuxiaoze/shape/ETOPO2v2c_f4.nc")


def saturation_vapor_pressure(temp_c, over_ice: bool = False):
    """Saturated water-vapour pressure ``es`` [hPa] from Celsius temperature.

    Tetens (over water).  ``over_ice=True`` uses the ice coefficients; the two
    differ by up to ~10 ``%`` around -20 C and are identical at 0 C.
    """
    tc = np.asarray(temp_c, dtype=np.float64)
    if over_ice:
        return 6.1078 * np.exp(22.44294 * tc / (tc + 272.4406))
    return 6.1078 * np.exp(17.269 * tc / (tc + 237.3))


def vapor_pressure(temp, humidity, temp_unit: str = "K", over_ice: bool = False):
    """Water-vapour pressure ``e`` [hPa] from temperature, RH [%]."""
    t = np.asarray(temp, dtype=np.float64)
    rh = np.asarray(humidity, dtype=np.float64)
    unit = temp_unit.upper()
    if unit == "K":
        tc = t - 273.15
    elif unit in ("C", "DEGC"):
        tc = t
    else:
        raise ValueError("temp_unit must be ''K'' or ''C''")
    return rh / 100.0 * saturation_vapor_pressure(tc, over_ice=over_ice)


def surface_pressure_from_msl(msl, height_m, temp_k, msl_unit: str = "hPa"):
    """Reduce mean-sea-level pressure to station pressure (hypsometric).

    ``p_s = msl * exp(-g h / (R_d T_v))`` with ``T_v ~ T`` (moisture ignored).
    Sanity: h = 100 m, T = 288 K -> p_s = 0.988 * msl (~12 hPa lower).
    """
    p_msl = np.asarray(msl, dtype=np.float64)
    if msl_unit.lower() == "pa":
        p_msl = p_msl / 100.0
    h = np.asarray(height_m, dtype=np.float64)
    t = np.asarray(temp_k, dtype=np.float64)
    return p_msl * np.exp(-G_0 * h / (R_D * t))


def _zhd(p_s_hpa, lat_deg, height_m):
    """Saastamoinen zenith hydrostatic delay [m]."""
    lat = np.asarray(lat_deg, dtype=np.float64)
    h_km = np.asarray(height_m, dtype=np.float64) / 1000.0
    denom = 1.0 - 0.00266 * np.cos(2.0 * np.deg2rad(lat)) - 0.00028 * h_km
    return ZHD_COEF * np.asarray(p_s_hpa, dtype=np.float64) / denom


def ztd_surface(temp, pressure_hpa, humidity_percent, height_m, latitude_deg,
                temp_unit: str = "K", over_ice: bool = False):
    """Classic surface ZTD [m] -- the reference implementation.

    Returns a dict with ``ZHD``, ``ZWD``, ``ZTD`` (all metres) plus ``ZHD_mm``,
    ``ZWD_mm``, ``ZTD_mm`` and ``e_hpa``.
    """
    t = np.asarray(temp, dtype=np.float64)
    tk = t + 273.15 if temp_unit.upper() in ("C", "DEGC") else t
    e = vapor_pressure(t, humidity_percent, temp_unit=temp_unit, over_ice=over_ice)
    zhd = _zhd(pressure_hpa, latitude_deg, height_m)
    zwd = 0.002277 * (1255.0 / tk + 0.05) * e
    ztd = zhd + zwd
    return {"ZHD": zhd, "ZWD": zwd, "ZTD": ztd, "e_hpa": e,
            "ZHD_mm": zhd * 1000.0, "ZWD_mm": zwd * 1000.0, "ZTD_mm": ztd * 1000.0}


def ztd_profile(t_levels, r_levels, sfc_pressure_hpa, height_m, latitude_deg,
                lev_hpa=LEV_HPA, over_ice: bool = False, p_top_hpa: float = None):
    """Column ZTD [m] from pressure-level ``t`` [K] / ``r`` [%] profiles.

    ``t_levels`` / ``r_levels`` are (..., n_lev) arrays ordered like ``lev_hpa``
    (50 hPa first, 1000 hPa last).  ZHD uses the surface pressure; ZWD integrates
    the wet refractivity over the column with hydrostatic layer thickness.

    ``p_top_hpa`` (default: the lowest pressure in ``lev_hpa``) lets the dry part
    of the column be ignored above that pressure, which is what the surface ZHD
    already accounts for.
    """
    t = np.asarray(t_levels, dtype=np.float64)
    r = np.asarray(r_levels, dtype=np.float64)
    p = np.asarray(lev_hpa, dtype=np.float64)
    assert t.shape[-1] == p.size == r.shape[-1], (t.shape, r.shape, p.size)

    e = vapor_pressure(t, r, temp_unit="K", over_ice=over_ice)          # hPa
    integrand = K2_PRIME * e / p + K3 * e / (t * p)                     # per hPa
    # ZWD = 1e-6 * (R_d/g) * int integrand dp, trapezoid over the level axis,
    # integrated from the surface (largest p) upwards.
    dp = np.diff(p)                                                     # ascending p
    trap = 0.5 * (integrand[..., 1:] + integrand[..., :-1]) * dp
    zwd = 1e-6 * (R_D / G_0) * np.sum(trap, axis=-1)
    zhd = _zhd(sfc_pressure_hpa, latitude_deg, height_m)
    return {"ZHD": zhd, "ZWD": zwd, "ZTD": zhd + zwd, "e_hpa": e,
            "ZHD_mm": zhd * 1000.0, "ZWD_mm": zwd * 1000.0, "ZTD_mm": (zhd + zwd) * 1000.0}


def ztd_profile_surface(t_levels, r_levels, t2m, sfc_pressure_hpa, height_m,
                        latitude_deg, lev_hpa=LEV_HPA, over_ice: bool = False,
                        surface_rh: str = "lowest"):
    """Column ZTD [m] with the station surface spliced in as the bottom node.

    Why this exists: the store stops at 1000 hPa, which is *below ground* for any
    station above ~110 m (the mean station pressure in this network is ~975 hPa),
    so a plain trapezoid over 50..1000 hPa integrates a slice of the model's
    below-ground extrapolation and never sees the 2 m state.  Here

    * every level at or below the surface pressure is collapsed onto ``p_s`` and
      its values replaced by the surface state, so those layers get zero
      thickness and the model's under-ground values cannot leak upward, and
    * the bottom node is the surface itself: ``t2m`` for temperature and
      ``e_sfc = RH_low/100 * es(t2m)`` for vapour pressure, where ``RH_low`` is
      the relative humidity of the lowest level still above ground (that is the
      only humidity information available -- the store has no 2 m humidity).

    ``surface_rh``: "lowest" (default) or "fixed_1000" (use the 1000 hPa RH).
    """
    t = np.asarray(t_levels, dtype=np.float64)
    r = np.asarray(r_levels, dtype=np.float64)
    p = np.asarray(lev_hpa, dtype=np.float64)
    p_s = np.asarray(sfc_pressure_hpa, dtype=np.float64)
    ts = np.asarray(t2m, dtype=np.float64)
    assert t.shape[-1] == p.size == r.shape[-1], (t.shape, r.shape, p.size)

    # lowest level still above ground: last True along the level axis
    valid = p < p_s[..., None]                       # (..., n_lev), p ascending
    rev = valid[..., ::-1].argmax(axis=-1)
    # columns with no valid level (NaN surface pressure, e.g. no station) fall
    # back to the lowest level -- their inputs are NaN, so the output is NaN too
    idx = np.where(valid.any(axis=-1), p.size - 1 - rev, p.size - 1)
    take = lambda a: np.take_along_axis(a, idx[..., None], axis=-1)[..., 0]
    rh_low = take(r) if surface_rh == "lowest" else r[..., -1]

    e_sfc = vapor_pressure(ts, rh_low, temp_unit="K", over_ice=over_ice)   # hPa
    e_lev = vapor_pressure(t, r, temp_unit="K", over_ice=over_ice)

    below = p >= p_s[..., None]                      # collapse these onto p_s
    p_eff = np.where(below, p_s[..., None], p)
    t_eff = np.where(below, ts[..., None], t)
    e_eff = np.where(below, e_sfc[..., None], e_lev)

    P = np.concatenate([p_eff, p_s[..., None]], axis=-1)      # ascending, ends at p_s
    T = np.concatenate([t_eff, ts[..., None]], axis=-1)
    E = np.concatenate([e_eff, e_sfc[..., None]], axis=-1)

    integrand = K2_PRIME * E / P + K3 * E / (T * P)           # per hPa
    dp = np.diff(P)
    zwd = 1e-6 * (R_D / G_0) * np.sum(0.5 * (integrand[..., 1:] + integrand[..., :-1])
                                      * dp, axis=-1)
    zhd = _zhd(p_s, latitude_deg, height_m)
    return {"ZHD": zhd, "ZWD": zwd, "ZTD": zhd + zwd, "e_hpa": e_lev,
            "e_sfc_hpa": e_sfc, "rh_surface": rh_low,
            "ZHD_mm": zhd * 1000.0, "ZWD_mm": zwd * 1000.0, "ZTD_mm": (zhd + zwd) * 1000.0}


# ---------------------------------------------------------------------------
# 方法 E（2026-09-21 起成为默认实现）：在几何高度上梯形积分
#   与 ztd_profile_surface（气压坐标）在数学上等价，差别只在积分坐标与格式：
#   * 层高直接取 store 的位势通道 z* [m^2/s^2] ÷ g，不做静力重建；
#   * 用梯形法（而不是"下层值×层厚"），13 层粗廓线上误差从 ~10% 降到 ~2%；
#   * 补上 p_top(50 hPa) 以上那截干空气柱的解析贡献 ~113.7 mm；
#   * 常数用 Bevis 1994 的 k1/k2/k3，k2' = k2 - (M_w/M_d)k1 = 22.98。
# 与观测（NGL GNSS ZTD）对比：RMSE 12.0~12.4 mm，优于旧实现的 14.1~14.4 mm。
# ---------------------------------------------------------------------------
K1_BEVIS = 77.6890      # K/hPa
K2_BEVIS = 71.2952      # K/hPa
K3_BEVIS = 375463.0     # K^2/hPa
EPS_MW_MD = 0.62198     # M_w / M_d


def zhd_top_correction(p_top_hpa, k1=K1_BEVIS):
    """p_top 以上那截干空气柱的解析贡献 [m]（约 0.00227 * p_top）。"""
    return 1e-6 * k1 * (R_D / G_0) * float(p_top_hpa)


def ztd_profile_zdz(t_levels, r_levels, t2m, sfc_pressure_hpa, height_m,
                    z_levels_m, lev_hpa=LEV_HPA, over_ice=False,
                    surface_rh="lowest", k1=K1_BEVIS, k2=K2_BEVIS,
                    k3=K3_BEVIS, eps=EPS_MW_MD, top_correction=True):
    """几何高度梯形积分版 ZTD [m]，返回与 :func:`ztd_profile_surface` 同结构的 dict。

    参数
    ----
    t_levels, r_levels : (..., n_lev)  各气压层的温度 [K] / 相对湿度 [%]
    t2m                : (...)          2 米温度 [K]
    sfc_pressure_hpa   : (...)          地面气压 [hPa]（由 msl 按测高公式折算）
    height_m           : (...)          测站高度 [m]
    z_levels_m         : (..., n_lev)   各气压层的位势高度 [m]（store 的 z* ÷ g）

    做法
    ----
    1. ``p >= p_s`` 的层塌缩到地面（高度置为 h_s、值换成地面值），层厚随之归零，
       模式的地下外推不会进入积分；
    2. 节点按"地面 + 由低到高的 13 层"排列，用梯形法对
       ``N_h = k1(p-e)/T`` 与 ``N_w = (k2-eps*k1)e/T + k3 e/T^2`` 积分；
    3. 再加上 ``p_top`` 以上的干空气柱解析修正。
    """
    t = np.asarray(t_levels, dtype=np.float64)
    r = np.asarray(r_levels, dtype=np.float64)
    p = np.asarray(lev_hpa, dtype=np.float64)
    p_s = np.asarray(sfc_pressure_hpa, dtype=np.float64)
    ts = np.asarray(t2m, dtype=np.float64)
    hs = np.asarray(height_m, dtype=np.float64)
    hz = np.asarray(z_levels_m, dtype=np.float64)
    assert t.shape[-1] == r.shape[-1] == hz.shape[-1] == p.size, \
        (t.shape, r.shape, hz.shape, p.size)

    valid = p < p_s[..., None]
    rev = valid[..., ::-1].argmax(axis=-1)
    idx = np.where(valid.any(axis=-1), p.size - 1 - rev, p.size - 1)
    take = lambda a: np.take_along_axis(a, idx[..., None], axis=-1)[..., 0]
    rh_low = take(r) if surface_rh == "lowest" else r[..., -1]

    e_sfc = vapor_pressure(ts, rh_low, temp_unit="K", over_ice=over_ice)
    e_lev = vapor_pressure(t, r, temp_unit="K", over_ice=over_ice)

    below = p >= p_s[..., None]
    p_eff = np.where(below, p_s[..., None], p)
    t_eff = np.where(below, ts[..., None], t)
    e_eff = np.where(below, e_sfc[..., None], e_lev)
    # 地下层的"高度"置为地面高度 -> 层厚 0（NaN 也一并换成地面值，避免 NaN 传播）
    h_eff = np.where(below, hs[..., None], hz)

    # 由下往上：地面节点 + 反序的 13 层
    P = np.concatenate([p_s[..., None], p_eff[..., ::-1]], axis=-1)
    T = np.concatenate([ts[..., None], t_eff[..., ::-1]], axis=-1)
    E = np.concatenate([e_sfc[..., None], e_eff[..., ::-1]], axis=-1)
    H = np.concatenate([hs[..., None], h_eff[..., ::-1]], axis=-1)

    N_h = k1 * (P - E) / T
    N_w = (k2 - eps * k1) * (E / T) + k3 * (E / T ** 2)
    dh = np.maximum(np.diff(H, axis=-1), 0.0)
    zhd = 1e-6 * np.sum(0.5 * (N_h[..., 1:] + N_h[..., :-1]) * dh, axis=-1)
    zwd = 1e-6 * np.sum(0.5 * (N_w[..., 1:] + N_w[..., :-1]) * dh, axis=-1)
    if top_correction:
        zhd = zhd + zhd_top_correction(p[0], k1)

    return {"ZHD": zhd, "ZWD": zwd, "ZTD": zhd + zwd, "e_hpa": e_lev,
            "e_sfc_hpa": e_sfc, "rh_surface": rh_low,
            "ZHD_mm": zhd * 1000.0, "ZWD_mm": zwd * 1000.0, "ZTD_mm": (zhd + zwd) * 1000.0}


def lev_pressure_hpa():
    """The 13 ERA5 pressure levels in the store's channel order."""
    return LEV_HPA.copy()


_ETOPO_CACHE = {}


def _etopo_handle(path=ETOPO_PATH):
    """(z variable, nx, ny) -- opened once with mmap, kept open."""
    key = str(path)
    if key not in _ETOPO_CACHE:
        from scipy.io import netcdf_file

        f = netcdf_file(key, "r", mmap=True)
        z = f.variables["z"]
        _ETOPO_CACHE[key] = (z, z.shape[1], z.shape[0])       # (ny, nx)
    return _ETOPO_CACHE[key]


def elevation_from_etopo(lat, lon, path=ETOPO_PATH):
    """Nearest ETOPO2v2 cell elevation [m] for scalar/array lat/lon (degrees).

    The file is 2-arc-minute with *cell centred* coordinates: cell (iy, ix)
    covers lat = -90 + iy*res ... , so the index is round((lat + 90)/res - 0.5)
    with the exact resolution 180/ny -- do **not** use the float32 ``y``/``x``
    spacing, its rounding error shifts the lookup by half a cell.  ``z`` is
    (ny, nx) float32, north-up (y ascending), so no latitude flip is needed.
    """
    z, nx, ny = _etopo_handle(path)
    rx, ry = 360.0 / nx, 180.0 / ny
    la = np.atleast_1d(np.asarray(lat, dtype=np.float64))
    lo = np.atleast_1d(np.asarray(lon, dtype=np.float64))
    iy = np.clip(np.round((la + 90.0) / ry - 0.5), 0, ny - 1).astype(int)
    ix = np.clip(np.round((lo + 180.0) / rx - 0.5), 0, nx - 1).astype(int)
    cache = {}
    out = np.empty(iy.size, dtype=np.float64)
    for k in range(iy.size):                      # scalar reads; the European
        key = (int(iy[k]), int(ix[k]))            # sub-block stays page-cached
        if key not in cache:
            cache[key] = float(z[key[0], key[1]])
        out[k] = cache[key]
    return out if (np.ndim(lat) or np.ndim(lon)) else float(out[0])
