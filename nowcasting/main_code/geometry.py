from __future__ import annotations

import numpy as np

_WGS84_A = 6378137.0
_WGS84_E2 = 0.0066943799901413165  # WGS84 (f = 1/298.257223563)


def _ecef(lat_deg: float, lon_deg: float, h_m: float) -> tuple[float, float, float]:
    """Geodetic (WGS84) -> geocentric ECEF."""
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    n = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * np.sin(lat) ** 2)
    x = (n + h_m) * np.cos(lat) * np.cos(lon)
    y = (n + h_m) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - _WGS84_E2) + h_m) * np.sin(lat)
    return x, y, z


def _enu(t_info: tuple[float, float, float], g_info: tuple[float, float, float]) -> tuple[float, float, float]:
    """Neighbor position in the target's local ENU frame (target = origin).

    Returns (dE_km, dN_km, dU_m); dU uses the station metadata heights directly
    (consistent with target_gnss_neighbors.height_difference_m).
    """
    tx, ty, tz = _ecef(*t_info)
    gx, gy, gz = _ecef(*g_info)
    dx, dy, dz = gx - tx, gy - ty, gz - tz
    lat, lon = np.deg2rad(t_info[0]), np.deg2rad(t_info[1])
    slat, clat = np.sin(lat), np.cos(lat)
    slon, clon = np.sin(lon), np.cos(lon)
    e = -slon * dx + clon * dy
    n = -slat * clon * dx - slat * slon * dy + clat * dz
    du = g_info[2] - t_info[2]
    return float(e / 1000.0), float(n / 1000.0), float(du)
