#!/usr/bin/env python3
"""Plot ERA5 (CONUS, 0.1-deg regridded) for the grid-forecast times, using the
same per-time/per-variable colorbar limits as the forecast maps (1-99 percentiles
of the forecast field over covered cells).

Channel mapping to the six nowcast variables:
  slp <- msl (hPa), t2m <- t2m, u10 <- u10, v10 <- v10,
  p   <- derived from msl + DEM height + t2m (hydrostatic approximation),
  r2m <- RH at 1000 hPa (fallback 925 hPa) [approximation].
"""

from __future__ import annotations

import numpy as np
import netCDF4
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ERA5_DIR = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/xuxiaoze/data_prep/era5_1h"
RUN_OUT = ("/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/linan/linan_dev/gnss/"
           "nowcasting/outputs/gnss_nowcast_s1915_off0_h6_dm128_el2_nh4_df512_sp_thf")
TARGET_TIMES = ["2024-03-15T00:00", "2024-05-20T12:00",
                "2024-07-04T18:00", "2024-08-10T06:00"]
VARS = ["p", "slp", "t2m", "r2m", "u10", "v10"]
G, R = 9.80665, 287.05

d = np.load(f"{RUN_OUT}/grid_predictions.npz")
lat_c = d["lat"]
lon_c = d["lon"]
preds = d["preds"]          # (T, 260, 600, 6)
valid = d["valid"]          # (260, 600)
heights = d["height"]       # (260, 600)


def forecast_limits(k: int, j: int):
    field = np.where(valid, preds[k, :, :, j], np.nan)
    return np.nanpercentile(field, [1, 99])


def read_era5_conus(fname: str):
    """Return dict of 0.25-deg CONUS fields (lat ascending), plus lat/lon."""
    ds = netCDF4.Dataset(fname)
    ch = [str(c) for c in ds.variables["channel"][:]]
    lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
    lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
    z = ds.variables["z"]

    lat_asc = np.flip(lat)                       # ascending
    lat_i = np.flatnonzero((lat_asc >= 24.0) & (lat_asc <= 50.0))
    lon0, lon1 = 235.0, 295.0                    # -125 .. -65 in 0..360
    lon_i = np.flatnonzero((lon >= lon0) & (lon <= lon1))

    def get(name):
        i = ch.index(name)
        a = np.asarray(z[0, i, :, :], dtype=np.float64)  # (lat, lon) original order
        a = np.flip(a, axis=0)                            # ascending lat
        return a[lat_i[0]:lat_i[-1] + 1, lon_i[0]:lon_i[-1] + 1]

    msl = get("msl") / 100.0                      # Pa -> hPa
    t2m = get("t2m")                              # K
    u10 = get("u10")
    v10 = get("v10")
    r1000 = get("r1000")
    r925 = get("r925")
    r2m = np.where(np.isfinite(r1000), r1000, r925)
    ds.close()
    return dict(msl=msl, t2m=t2m, u10=u10, v10=v10, r2m=r2m,
                lat=lat_asc[lat_i[0]:lat_i[-1] + 1], lon=lon[lon_i[0]:lon_i[-1] + 1])


def regrid(fld, lat_src, lon_src):
    interp = RegularGridInterpolator((lat_src, lon_src), fld,
                                     bounds_error=False, fill_value=np.nan)
    LON, LAT = np.meshgrid(lon_c + 360.0, lat_c)   # target lon in 0..360 like ERA5
    return interp(np.stack([LAT.ravel(), LON.ravel()], axis=-1)).reshape(LAT.shape)


def main():
    lat_edges = np.concatenate([[lat_c[0] - 0.05], lat_c + 0.05])
    lon_edges = np.concatenate([[lon_c[0] - 0.05], lon_c + 0.05])

    for k, tstr in enumerate(TARGET_TIMES):
        stamp = f"{tstr[:10].replace('-', '')}{tstr[11:13]}"
        fname = f"{ERA5_DIR}/{stamp}.nc"
        e = read_era5_conus(fname)

        # derived surface pressure (hydrostatic inversion of msl at station height)
        LON, LAT = np.meshgrid(lon_c, lat_c)
        hgt = heights                                # already on the 0.1-deg grid
        msl = regrid(e["msl"], e["lat"], e["lon"])
        t2m = regrid(e["t2m"], e["lat"], e["lon"])
        p_der = msl * np.exp(-G * hgt / (R * (t2m + 0.0065 * hgt / 2)))   # hPa
        fields = {
            "p": p_der,
            "slp": msl,
            "t2m": t2m,
            "r2m": regrid(e["r2m"], e["lat"], e["lon"]),
            "u10": regrid(e["u10"], e["lat"], e["lon"]),
            "v10": regrid(e["v10"], e["lat"], e["lon"]),
        }
        # identical holes to the forecast panel: keep ERA5 only where the
        # forecast has a finite value at this time (GNSS coverage + window valid)
        fields = {v: np.where(np.isfinite(preds[k, :, :, j]), f, np.nan)
                  for j, (v, f) in enumerate(fields.items())}

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        for j, var in enumerate(VARS):
            ax = axes.flat[j]
            lo, hi = forecast_limits(k, j)            # same colorbar as forecast
            pm = ax.pcolormesh(lon_edges, lat_edges, np.ma.masked_invalid(fields[var]),
                               cmap="turbo", vmin=lo, vmax=hi, shading="flat")
            ax.set_title(var, fontsize=12)
            ax.set_aspect(1.0 / np.cos(np.deg2rad(35.0)))
            fig.colorbar(pm, ax=ax, shrink=0.8)
            ax.tick_params(labelsize=8)
        note = ("p: msl+DEM+t2m hydrostatic approx | r2m: RH@1000hPa (925 fallback) | "
                "masked to forecast GNSS coverage")
        fig.suptitle(f"ERA5  {tstr} UTC  (0.1-deg regrid, same colorbar + same mask as forecast)\n{note}",
                     fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        out = f"{RUN_OUT}/era5_{stamp}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"[save] {out}", flush=True)


if __name__ == "__main__":
    main()
