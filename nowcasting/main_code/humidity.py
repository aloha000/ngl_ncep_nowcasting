"""Humidity conversions used for observed surface targets."""

from __future__ import annotations

import numpy as np


def specific_humidity_from_t_rh_p(t_k, rh_percent, p_pa):
    """Return IFS Cy48r1 mixed-phase specific humidity in kg kg-1.

    Parameters are air temperature in K, relative humidity in percent, and
    surface pressure in Pa.  The saturation-vapour-pressure calculation uses
    the water/ice blending specified for IFS Cy48r1.  NaN inputs propagate to
    the result.
    """
    t_k = np.asarray(t_k, dtype=np.float64)
    rh_percent = np.asarray(rh_percent, dtype=np.float64)
    p_pa = np.asarray(p_pa, dtype=np.float64)

    t0 = 273.16
    t_ice = 250.16
    a1, a3_w, a4_w = 611.21, 17.502, 32.19
    alpha = np.where(
        t_k >= t0,
        1.0,
        np.where(t_k <= t_ice, 0.0, ((t_k - t_ice) / (t0 - t_ice)) ** 2),
    )
    esat_w = a1 * np.exp(a3_w * (t_k - t0) / (t_k - a4_w))
    esat_i = a1 * np.exp(22.587 * (t_k - t0) / (t_k + 0.7))
    esat = alpha * esat_w + (1.0 - alpha) * esat_i
    e = rh_percent / 100.0 * esat
    with np.errstate(divide="ignore", invalid="ignore"):
        return 0.622 * e / (p_pa - 0.378 * e)
