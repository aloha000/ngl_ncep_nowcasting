#!/usr/bin/env python3
"""0.1-deg CONUS grid nowcasting inference (Route B).

For every 0.1-deg grid cell (treated as a virtual target station):
  - find the up-to-5 nearest NGL GNSS stations within 50 km (ranked by distance),
  - build the same inputs as training (25-step 5-min ZTD/ZWD window +
    per-neighbor ENU/height/static encoding + target height/static features + linear
    hour time marks),
  - run the trained iTransformer checkpoint, denormalize with training stats.

Heights come from a local ETOPO2v2 global relief NetCDF (2-arc-min; each 0.1-deg
cell height = mean of the 3x3 ETOPO cells around the cell centre).

Outputs (per run):
  grid_pred_<utc>.png     2x3 panel maps of the six NCEP variables
  grid_predictions.npz    lat/lon/heights + (n_times, n_lat, n_lon, 6) preds
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
TSL_ROOT = ROOT / "Time-Series-Library"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TSL_ROOT))

import zarr  # noqa: E402
import netCDF4  # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402
from scipy.io import netcdf_file  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from nowcasting.main_code.config import load_config  # noqa: E402
from nowcasting.main_code.constants import NCEP_VARS  # noqa: E402
from nowcasting.main_code.geometry import _enu  # noqa: E402
from nowcasting.main_code.model import Model, make_model_config  # noqa: E402
from utils.timefeatures import time_features  # noqa: E402

# ----------------------------------------------------------------------------
# Config (edit here)
# ----------------------------------------------------------------------------
CONFIG_YAML = ROOT / "nowcasting/config.yaml"
RUN_OUT = ROOT / "nowcasting/outputs/gnss_nowcast_s1915_off0_h6_dm128_el2_nh4_df512_sp_thf"
ETOPO_NC = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/xuxiaoze/shape/ETOPO2v2c_f4.nc"

# CONUS 0.1-deg grid (cell centres)
LAT0, LAT1, DLAT = 24.0, 50.0, 0.1
LON0, LON1, DLON = -125.0, -65.0, 0.1

# Target hours (UTC, must exist in the NCEP hourly store)
TARGET_TIMES = [
    "2024-03-15T00:00",
    "2024-05-20T12:00",
    "2024-07-04T18:00",
    "2024-08-10T06:00",
]

BATCH = 512
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MIN_VALID_NEIGHBORS = 3
RADIUS_M = 50_000.0
EARTH_R = 6_371_000.0


def lat_lon_xy(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Equirectangular local x/y in metres (adequate for 50-km radius queries)."""
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    x = EARTH_R * np.cos(lat_r) * lon_r
    y = EARTH_R * lat_r
    return np.stack([x, y], axis=1)


def lon_to_360(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) % 360.0


def nearest_indices(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if axis[0] <= axis[-1]:
        pos = np.searchsorted(axis, values)
        left = np.clip(pos - 1, 0, len(axis) - 1)
        right = np.clip(pos, 0, len(axis) - 1)
    else:
        rev = axis[::-1]
        pos = np.searchsorted(rev, values)
        left = len(axis) - 1 - np.clip(pos, 0, len(axis) - 1)
        right = len(axis) - 1 - np.clip(pos - 1, 0, len(axis) - 1)
    choose_right = np.abs(axis[right] - values) < np.abs(axis[left] - values)
    return np.where(choose_right, right, left).astype(np.int64)


def make_interpolator(lat: np.ndarray, lon: np.ndarray, field: np.ndarray) -> RegularGridInterpolator:
    if lat[0] > lat[-1]:
        lat_use = lat[::-1]
        field_use = field[::-1, :]
    else:
        lat_use = lat
        field_use = field
    return RegularGridInterpolator(
        (lat_use, lon), field_use, method="linear", bounds_error=False, fill_value=np.nan
    )


def sample_grid_static(const_nc: Path, grid_lat: np.ndarray, grid_lon: np.ndarray,
                       expected_channels: np.ndarray, onehot: np.ndarray) -> np.ndarray:
    with netCDF4.Dataset(const_nc) as ds:
        channels = np.asarray([str(c) for c in ds.variables["channel"][:]], dtype=object).astype(str)
        if channels.tolist() != expected_channels.tolist():
            raise SystemExit("const.nc static channels do not match training station_static_const.npz")
        lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
        x = np.asarray(ds.variables["x"][:], dtype=np.float32)
    st_lon = lon_to_360(grid_lon)
    pts = np.stack([grid_lat.astype(np.float64), st_lon], axis=1)
    li = nearest_indices(lat, grid_lat)
    oi = nearest_indices(lon, st_lon)
    values = np.empty((len(grid_lat), x.shape[0]), dtype=np.float32)
    for c in range(x.shape[0]):
        if onehot[c]:
            values[:, c] = x[c, li, oi]
        else:
            values[:, c] = make_interpolator(lat, lon, x[c])(pts).astype(np.float32)
    values[~np.isfinite(values)] = 0.0
    return values


def load_dem() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ETOPO2 x/lon, y/lat, z (lat,lon) heights."""
    print("[dem] loading ETOPO2 ...", flush=True)
    f = netcdf_file(ETOPO_NC, mmap=False)
    x = np.asarray(f.variables["x"][:], dtype=np.float64)
    y = np.asarray(f.variables["y"][:], dtype=np.float64)
    z = np.asarray(f.variables["z"][:], dtype=np.float64)
    f.close()
    return x, y, z


def build_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """0.1-deg cell-centre lat/lon + ETOPO2 3x3-mean height (m)."""
    lat_c = np.arange(LAT0 + DLAT / 2, LAT1, DLAT)
    lon_c = np.arange(LON0 + DLON / 2, LON1, DLON)
    lons, lats = np.meshgrid(lon_c, lat_c)  # (n_lat, n_lon)

    x, y, z = load_dem()
    dx, dy = x[1] - x[0], y[1] - y[0]
    iy = np.clip(np.round((lats - y[0]) / dy).astype(np.int64), 1, len(y) - 2)
    ix = np.clip(np.round((lons - x[0]) / dx).astype(np.int64), 1, len(x) - 2)
    # mean of the 3x3 ETOPO cells around each grid-cell centre
    h = (
        z[iy, ix] + z[iy - 1, ix] + z[iy + 1, ix]
        + z[iy, ix - 1] + z[iy, ix + 1]
        + z[iy - 1, ix - 1] + z[iy - 1, ix + 1]
        + z[iy + 1, ix - 1] + z[iy + 1, ix + 1]
    ) / 9.0
    print(f"[dem] grid {lat_c.size}x{lon_c.size} cells; "
          f"h min/mean/max = {h.min():.0f}/{h.mean():.0f}/{h.max():.0f} m", flush=True)
    return lat_c, lon_c, h.astype(np.float32)


def main() -> None:
    args = load_config(CONFIG_YAML, [])
    args.seq_len = int(args.window_hours * 60 // args.ngl_step_minutes) + 1
    args.n_time_features = 4  # time_encoding=linear, freq=h (fixed by config)
    scal = np.load(RUN_OUT / "scalers.npz", allow_pickle=True)
    geo_mean = scal["geo_mean"].astype(np.float32)
    geo_std = scal["geo_std"].astype(np.float32)
    target_feat_mean = scal["target_feat_mean"].astype(np.float32) if "target_feat_mean" in scal.files else None
    target_feat_std = scal["target_feat_std"].astype(np.float32) if "target_feat_std" in scal.files else None
    th_mean = float(scal["target_h_mean"])
    th_std = float(scal["target_h_std"])
    ym = scal["ym"].astype(np.float32)                  # (6,)
    ys = scal["ys"].astype(np.float32)                  # (6,)
    args.n_geo_total = int(len(geo_mean))
    if args.target_h_feat:
        args.target_feat_dim = int(len(target_feat_mean)) if target_feat_mean is not None else 1
    cfg = make_model_config(args)
    device = torch.device(DEVICE)

    print("[model] loading checkpoint ...", flush=True)
    ckpt = torch.load(RUN_OUT / "checkpoint.pth", map_location=device, weights_only=False)
    model = Model(cfg).float().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print("[model] loaded", flush=True)
    print(f"[scalers] geo_dim={len(geo_mean)} target_feat_dim={cfg.target_feat_dim} "
          f"target_h mean={th_mean:.1f} std={th_std:.1f} | ym={ym.tolist()} ys={ys.tolist()}",
          flush=True)

    # --- grid + DEM ---------------------------------------------------------
    lat_c, lon_c, h_grid = build_grid()
    n_lat, n_lon = h_grid.shape
    flat = np.stack([lats := np.broadcast_to(lat_c[:, None], h_grid.shape),
                     np.broadcast_to(lon_c[None, :], h_grid.shape)], axis=-1).reshape(-1, 2)
    lat_flat = flat[:, 0]
    lon_flat = flat[:, 1]
    h_flat = h_grid.reshape(-1)
    n_cells = len(lat_flat)
    print(f"[grid] {n_cells} cells ({n_lat}x{n_lon})", flush=True)

    # --- neighbors ----------------------------------------------------------
    print("[nb] loading NGL stations ...", flush=True)
    gst = pd.read_parquet(args.gnss_stations_parquet)
    g_lat = gst["lat"].to_numpy(np.float64)
    g_lon = gst["lon"].to_numpy(np.float64)
    g_h = gst["height_m"].to_numpy(np.float64)
    static = np.load(args.station_static_npz, allow_pickle=True)
    static_channels = np.asarray(static["channels"], dtype=object).astype(str)
    static_onehot = np.asarray(static["onehot_mask"], dtype=bool)
    static_dim = int(len(static_channels))
    base_n_geo = int(args.n_geo)
    use_static = cfg.n_geo == base_n_geo + static_dim
    if cfg.n_geo not in (base_n_geo, base_n_geo + static_dim):
        raise SystemExit(f"checkpoint expects n_geo={cfg.n_geo}, but config/static imply {base_n_geo} or {base_n_geo + static_dim}")
    g_static = {str(sid): static["gnss_static"][i].astype(np.float32)
                for i, sid in enumerate(static["gnss_station_ids"])}
    good = np.isfinite(g_lat) & np.isfinite(g_lon) & np.isfinite(g_h)
    gidx = np.flatnonzero(good)
    print(f"[nb] NGL stations: {len(g_lat)} total, {len(gidx)} usable", flush=True)
    tree = cKDTree(lat_lon_xy(g_lat[good], g_lon[good]))
    dists, idxs = tree.query(lat_lon_xy(lat_flat, lon_flat), k=6,
                             distance_upper_bound=RADIUS_M)
    # dists/idxs: (C, 6) sorted by distance; k=6 because neighbour may be the cell itself.
    # idxs are positions inside the filtered array; map back to original parquet rows.
    n_neigh = 5
    nb_idx = np.full((n_cells, n_neigh), -1, dtype=np.int64)
    nb_dist = np.full((n_cells, n_neigh), np.inf, dtype=np.float64)
    for j in range(n_neigh):
        ok = np.isfinite(dists[:, j])
        nb_idx[ok, j] = gidx[np.minimum(idxs[ok, j], len(gidx) - 1)]
        nb_dist[ok, j] = dists[ok, j]
    nb_count = (nb_idx >= 0).sum(axis=1)
    covered = nb_count >= MIN_VALID_NEIGHBORS
    print(f"[nb] cells covered (>= {MIN_VALID_NEIGHBORS} NGL within 50 km): "
          f"{covered.sum()}/{n_cells} ({100.0*covered.mean():.1f}%)", flush=True)

    # per-neighbour geometry/static features (same construction as training)
    x_geo = np.zeros((n_cells, n_neigh, cfg.n_geo), dtype=np.float32)
    for c in range(n_cells):
        t_info = (float(lat_flat[c]), float(lon_flat[c]), float(h_flat[c]))
        for j in range(n_neigh):
            gi = nb_idx[c, j]
            if gi < 0:
                continue
            g_info = (float(g_lat[gi]), float(g_lon[gi]), float(g_h[gi]))
            de, dn, du = _enu(t_info, g_info)
            if base_n_geo == 4:
                geo_values = (de, dn, du, g_h[gi])
            elif base_n_geo == 8:
                geo_values = (de, dn, du, g_h[gi], lat_flat[c], lon_flat[c], g_lat[gi], g_lon[gi])
            else:
                raise SystemExit(f"unsupported n_geo={base_n_geo}; use 4 or 8")
            x_geo[c, j, :base_n_geo] = geo_values
            if use_static:
                sid = str(gst["gnss_station_id"].iloc[int(gi)])
                x_geo[c, j, base_n_geo:] = g_static.get(sid, np.zeros(static_dim, dtype=np.float32))
    x_geo = (x_geo - geo_mean) / geo_std
    x_geo[~np.isfinite(x_geo)] = 0.0
    if cfg.target_feat_dim > 1:
        const_nc = Path(str(static["const_nc"].item() if np.ndim(static["const_nc"]) == 0 else static["const_nc"]))
        target_static = sample_grid_static(const_nc, lat_flat, lon_flat, static_channels, static_onehot)
        x_tgt_raw = np.concatenate([h_flat.astype(np.float32)[:, None], target_static], axis=1)
        x_tgt = ((x_tgt_raw - target_feat_mean) / target_feat_std).astype(np.float32)
    else:
        x_tgt = ((h_flat.astype(np.float32) - th_mean) / th_std).astype(np.float32)[:, None]

    # --- NGL data for the needed hours (small span read, no full-array load) -
    print("[data] opening NGL store ...", flush=True)
    ngl = zarr.open(args.ngl_zarr, mode="r")
    ngl_time = pd.DatetimeIndex(pd.to_datetime(np.asarray(ngl["time"][:])))
    if ngl_time.tz is None:
        ngl_time = ngl_time.tz_localize("UTC")
    ncep = zarr.open(args.ncep_zarr, mode="r")
    ncep_time = pd.DatetimeIndex(pd.to_datetime(np.asarray(ncep["time"][:])))
    if ncep_time.tz is None:
        ncep_time = ncep_time.tz_localize("UTC")
    ngl_col = {str(s): i for i, s in enumerate(np.asarray(ngl["station"][:]))}

    step_min = int(args.ngl_step_minutes)
    sph = 60 // step_min
    win = args.seq_len
    offset = int((ncep_time[0] - ngl_time[0]).total_seconds() // 60 // step_min)

    targets = [pd.Timestamp(t, tz="UTC") for t in TARGET_TIMES]
    t_idx = []
    for ts in targets:
        m = np.flatnonzero(ncep_time == ts)
        if len(m) != 1:
            raise SystemExit(f"target time {ts} not found in NCEP store")
        t_idx.append(int(m[0]))
    print(f"[data] target hours at NCEP indices {t_idx} (offset={offset}, win={win})", flush=True)

    # unique neighbour columns needed
    need_cols = np.unique(nb_idx[nb_idx >= 0])
    need_store = np.array([ngl_col[str(gst["gnss_station_id"].iloc[int(i)])]
                           for i in need_cols], dtype=np.int64)
    col_pos = {int(sc): p for p, sc in enumerate(need_cols)}
    print(f"[data] unique neighbours needed: {len(need_cols)}", flush=True)

    t0_all = min(t * sph + offset - (win - 1) for t in t_idx)
    t1_all = max(t * sph + offset + 1 for t in t_idx)
    print(f"[data] reading NGL span [{t0_all}, {t1_all}) ...", flush=True)
    ts0 = time.time()
    ztd_all = np.asarray(ngl["ztd"][t0_all:t1_all, need_store], dtype=np.float32)
    zwd_all = np.asarray(ngl["zwd"][t0_all:t1_all, need_store], dtype=np.float32)
    print(f"[data] span read {ztd_all.shape} in {time.time()-ts0:.1f}s", flush=True)

    # time marks (linear, freq=h) over the loaded span
    mark_all = time_features(ngl_time[t0_all:t1_all], freq=args.time_freq).T.astype(np.float32)

    # per-cell neighbour positions inside the loaded span's column layout
    cell_col_pos = np.full((n_cells, n_neigh), 0, dtype=np.int64)
    for j in range(n_neigh):
        ok = nb_idx[:, j] >= 0
        cell_col_pos[ok, j] = np.fromiter(
            (col_pos[int(i)] for i in nb_idx[ok, j]), dtype=np.int64, count=int(ok.sum()))

    # --- inference ----------------------------------------------------------
    n_chan = cfg.enc_in
    n_times = len(t_idx)
    preds = np.full((n_times, n_lat, n_lon, len(NCEP_VARS)), np.nan, dtype=np.float32)
    valid_map = np.full((n_lat, n_lon), False, dtype=bool)
    valid_map.reshape(-1)[:] = covered

    for k, t in enumerate(t_idx):
        g0 = t * sph + offset - (win - 1)
        g1 = g0 + win
        sl0, sl1 = g0 - t0_all, g1 - t0_all
        wz = np.isfinite(ztd_all[sl0:sl1]) & np.isfinite(zwd_all[sl0:sl1])  # (25, U)
        win_valid = wz.all(axis=0)                                          # (U,)
        cell_valid = np.zeros((n_cells, n_neigh), dtype=bool)
        for j in range(n_neigh):
            ok = nb_idx[:, j] >= 0
            cell_valid[ok, j] = win_valid[cell_col_pos[ok, j]]
        n_valid = cell_valid.sum(axis=1)
        keep = covered & (n_valid >= MIN_VALID_NEIGHBORS)
        print(f"[{TARGET_TIMES[k]}] valid cells: {keep.sum()}/{n_cells} "
              f"({100.0*keep.mean():.1f}%)", flush=True)

        x = np.zeros((n_cells, win, n_chan), dtype=np.float32)
        for j in range(n_neigh):
            pos = cell_col_pos[:, j]
            ok = cell_valid[:, j]
            zt = ztd_all[sl0:sl1, pos]; zw = zwd_all[sl0:sl1, pos]
            x[:, :, 2 * j] = np.where(ok[None, :], zt, 0.0).T
            x[:, :, 2 * j + 1] = np.where(ok[None, :], zw, 0.0).T
        xm = np.broadcast_to(mark_all[sl0:sl1], (n_cells, win, mark_all.shape[1])).copy()
        xg = x_geo * cell_valid[:, :, None]
        xt = x_tgt

        idx_keep = np.flatnonzero(keep)
        pred_norm = np.full((n_cells, len(NCEP_VARS)), np.nan, dtype=np.float32)
        dec_inp = torch.zeros(BATCH, cfg.pred_len, cfg.c_out, device=device)
        if cfg.n_time_features > 0:
            y_mark_np = np.broadcast_to(mark_all[g1 - 1:g1], (n_cells, cfg.pred_len, cfg.n_time_features)).copy()
        else:
            y_mark_np = np.zeros((n_cells, cfg.pred_len, 0), dtype=np.float32)
        t0 = time.time()
        with torch.no_grad():
            for b0 in range(0, len(idx_keep), BATCH):
                bi = idx_keep[b0:b0 + BATCH]
                bx = torch.from_numpy(x[bi]).float().to(device)
                bxm = torch.from_numpy(xm[bi]).float().to(device)
                bxg = torch.from_numpy(xg[bi]).float().to(device)
                bxt = torch.from_numpy(xt[bi]).float().to(device)
                bym = torch.from_numpy(y_mark_np[bi]).float().to(device) if cfg.n_time_features > 0 else None
                out = model(bx, bxm, dec_inp[:len(bi)], bym, x_geo=bxg, x_tgt=bxt)
                pred_norm[bi] = out[:, -1, :].cpu().numpy()
        preds[k].reshape(-1, len(NCEP_VARS))[:] = pred_norm * ys + ym
        print(f"  inference done in {time.time()-t0:.1f}s", flush=True)

    # --- save ---------------------------------------------------------------
    out_npz = RUN_OUT / "grid_predictions.npz"
    np.savez(out_npz, lat=lat_c, lon=lon_c, height=h_grid,
             times=np.asarray(TARGET_TIMES), preds=preds, valid=valid_map)
    print(f"[save] {out_npz} preds={preds.shape}", flush=True)

    # --- plots ---------------------------------------------------------------
    lat_edges = np.concatenate([[lat_c[0] - DLAT / 2], lat_c + DLAT / 2])
    lon_edges = np.concatenate([[lon_c[0] - DLON / 2], lon_c + DLON / 2])
    for k, tstr in enumerate(TARGET_TIMES):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        for j, var in enumerate(NCEP_VARS):
            ax = axes.flat[j]
            field = preds[k, :, :, j]
            field = np.where(valid_map, field, np.nan)
            lo, hi = np.nanpercentile(field, [1, 99])
            pm = ax.pcolormesh(lon_edges, lat_edges, np.ma.masked_invalid(field),
                               cmap="turbo", vmin=lo, vmax=hi, shading="flat")
            ax.set_title(var, fontsize=12)
            ax.set_aspect(1.0 / np.cos(np.deg2rad(35.0)))
            fig.colorbar(pm, ax=ax, shrink=0.8)
            ax.tick_params(labelsize=8)
        fig.suptitle(f"0.1-deg grid nowcast  {tstr} UTC  (Route B, GNSS->NCEP)", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        png = RUN_OUT / f"grid_pred_{tstr.replace(':', '').replace('-', '')}.png"
        fig.savefig(png, dpi=130)
        plt.close(fig)
        print(f"[plot] {png}", flush=True)


if __name__ == "__main__":
    main()
