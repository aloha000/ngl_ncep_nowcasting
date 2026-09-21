#!/usr/bin/env python3
"""NGL stations on a Europe map (Natural Earth 50m coastlines as the basemap).

No cartopy/geopandas on this machine, so the Natural Earth shapefiles are read
directly with a ~40-line ESRI shapefile reader (polygon type 5 / polyline
type 3 are both just "parts + points" arrays).

Shown: land/sea basemap, the 36.5-56.25 N / 5.25 W-24.5 E assimilation domain,
the 1870 raw NGL stations and the 1378 grid cells that hold one of them.

Usage (from da_ngl/test):
    python plot_stations_on_map.py
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from matplotlib.path import Path as MplPath
from matplotlib.colors import ListedColormap
from matplotlib.patches import PathPatch, Rectangle

HERE = Path(__file__).resolve().parent
DA = HERE.parent
NE = Path("/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/xuxiaoze/shape/coastline")


def read_shapefile(shp: Path):
    """Yield one *list of rings* per record; each ring is (n, 2) lon/lat.

    Types 3 (polyline) and 5 (polygon) share the layout: shape type, box,
    numParts, numPoints, part offsets, then the points.  Keeping the rings of a
    record together matters for polygons: outer ring + holes must go into one
    compound path so the holes punch through.
    """
    with shp.open("rb") as fh:
        fh.seek(100)                                  # skip the 100-byte header
        while True:
            head = fh.read(8)
            if len(head) < 8:
                return
            length = struct.unpack(">i", head[4:])[0] * 2
            body = fh.read(length)
            if len(body) < length:
                return
            if struct.unpack("<i", body[:4])[0] == 0:  # null shape
                continue
            n_parts, n_pts = struct.unpack("<2i", body[36:44])
            parts = struct.unpack(f"<{n_parts}i", body[44:44 + 4 * n_parts])
            off = 44 + 4 * n_parts
            pts = np.frombuffer(body, dtype="<f8", count=2 * n_pts, offset=off)
            pts = pts.reshape(n_pts, 2)
            yield [pts[parts[k]: parts[k + 1] if k + 1 < n_parts else n_pts]
                   for k in range(n_parts)]


def land_patch(rings, **kw):
    """One compound path per shapefile record, so holes keep their winding."""
    verts, codes = [], []
    for r in rings:
        verts.append(r)
        codes.append(np.r_[MplPath.MOVETO, np.full(len(r) - 1, MplPath.LINETO)])
    return PathPatch(MplPath(np.vstack(verts), np.concatenate(codes)), **kw)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=DA / "dataset")
    ap.add_argument("--output", type=Path, default=None,
                    help="default: plots/ngl_stations_europe_map[_points].png")
    ap.add_argument("--extent", type=float, nargs=4,
                    default=[-12.0, 31.0, 33.5, 59.5],
                    metavar=("LON0", "LON1", "LAT0", "LAT1"))
    ap.add_argument("--halo", type=int, default=3,
                    help="cells of blank padding kept around the station mask "
                         "(the training region); 0 = station mask only")
    ap.add_argument("--style", choices=("cells", "points"), default="cells",
                    help="cells: the 0.25 deg cells that hold a station "
                         "(gridded distribution); points: raw station dots")
    args = ap.parse_args()
    if args.output is None:
        suffix = "" if args.style == "cells" else "_points"
        args.output = DA / "plots" / f"ngl_stations_europe_map{suffix}.png"

    gm = pd.read_parquet(args.dataset / "ngl_europe_0p25_80x120_station_grid_map.parquet")
    st = pd.read_parquet(args.dataset / "ngl_europe_stations.parquet")
    cells = gm[~gm["mask"].astype(bool)]
    rep_ids = set(cells["station_id"].astype(str))
    st = st.assign(chosen=st["gnss_station_id"].astype(str).isin(rep_ids))

    lat_axis = np.unique(gm["lat"].values)
    lon_axis = np.unique(gm["lon"].values)
    occupied = np.zeros((lat_axis.size, lon_axis.size), bool)
    iy = np.searchsorted(lat_axis, cells["lat"].values)
    ix = np.searchsorted(lon_axis, cells["lon"].values)
    occupied[iy, ix] = True
    # training region = station cells + a blank halo on every side
    region = (occupied.copy() if args.halo <= 0 else
              ndi.binary_dilation(occupied, structure=np.ones(
                  (2 * args.halo + 1, 2 * args.halo + 1), bool)))

    lon0, lon1, lat0, lat1 = args.extent
    fig, ax = plt.subplots(figsize=(13, 11.5), dpi=130)
    ax.set_facecolor("#cfe0f0")

    n_land = 0
    for rings in _records(NE / "ne_50m_land" / "ne_50m_land.shp", lon0, lon1, lat0, lat1):
        ax.add_patch(land_patch(rings, facecolor="#f4f1ea", edgecolor="#9e9889",
                                linewidth=0.5, zorder=1))
        n_land += 1
    for rings in _records(NE / "ne_50m_lakes" / "ne_50m_lakes.shp", lon0, lon1, lat0, lat1):
        ax.add_patch(land_patch(rings, facecolor="#cfe0f0", edgecolor="#7fa4c8",
                                linewidth=0.4, zorder=2))

    # the training region: filled halo + a real outline (not a rectangle)
    ax.pcolormesh(np.r_[lon_axis - 0.125, lon_axis[-1] + 0.125],
                  np.r_[lat_axis - 0.125, lat_axis[-1] + 0.125],
                  np.ma.masked_where(~region, region.astype(float)),
                  cmap=ListedColormap(["#8fb8de"]), alpha=0.45,
                  shading="flat", zorder=3)
    ax.contour(lon_axis, lat_axis, region.astype(float), levels=[0.5],
               colors="#123a63", linewidths=2.0, zorder=6)
    ax.add_patch(Rectangle((-5.25, 36.5), 24.5 + 5.25, 56.25 - 36.5, fill=False,
                           edgecolor="#888888", linewidth=0.8, linestyle=":",
                           zorder=4))
    ax.plot([], [], color="#123a63", lw=2.0,
            label=f"training region: station mask + {args.halo}-cell halo "
                  f"({region.sum()} cells = {100 * region.sum() / occupied.size:.0f}%)")
    ax.plot([], [], color="#888888", ls=":", lw=1.0,
            label="old rectangular domain (80$\\times$120 @ 0.25$^\\circ$)")

    drop = st[~st["chosen"]]
    keep = st[st["chosen"]]

    if args.style == "cells":
        # the gridded product: every 0.25 deg cell that holds at least one
        # station, drawn on its true footprint and coloured by how many raw
        # stations were collapsed into it
        lat_axis = np.unique(gm["lat"].values)
        lon_axis = np.unique(gm["lon"].values)
        dy = abs(lat_axis[1] - lat_axis[0]) / 2
        dx = abs(lon_axis[1] - lon_axis[0]) / 2
        le = np.r_[lat_axis - dy, lat_axis[-1] + dy]
        xe = np.r_[lon_axis - dx, lon_axis[-1] + dx]
        counts = np.full((lat_axis.size, lon_axis.size), np.nan)
        iy = np.searchsorted(lat_axis, cells["lat"].values)
        ix = np.searchsorted(lon_axis, cells["lon"].values)
        counts[iy, ix] = cells["n_stations"].values
        pm = ax.pcolormesh(xe, le, np.ma.masked_invalid(counts),
                           cmap="turbo", vmin=0.5, vmax=12.5,
                           edgecolors="#ffffff", linewidth=0.15,
                           shading="flat", zorder=5)
        cb = fig.colorbar(pm, ax=ax, pad=0.015, fraction=0.035,
                          ticks=[1, 2, 3, 4, 6, 9, 12])
        cb.set_label("raw NGL stations per 0.25$^\\circ$ cell")
        ax.scatter(st["lon"], st["lat"], s=3, color="#333333", alpha=0.45,
                   linewidths=0, zorder=7,
                   label=f"all {len(st)} raw NGL stations")
        title_kind = ("gridded station distribution: "
                      f"{len(cells)} cells of the 80$\\times$120 lattice")
    else:
        ax.scatter(drop["lon"], drop["lat"], s=9, color="#d62728", alpha=0.75,
                   linewidths=0, zorder=6,
                   label=f"NGL stations not used ({len(drop)})")
        ax.scatter(keep["lon"], keep["lat"], s=11, color="#17408b", alpha=0.9,
                   linewidths=0, zorder=7,
                   label="the 1378 cell representatives")
        ax.scatter(cells["lon"], cells["lat"], s=90, facecolors="none",
                   edgecolors="#00a0a0", linewidths=0.7, zorder=8,
                   label=f"grid cells with a station ({len(cells)})")
        title_kind = "raw stations and the cells they were assigned to"

    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0, lat1)
    ax.set_aspect(1.0 / np.cos(np.radians(0.5 * (lat0 + lat1))))
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")
    ax.grid(alpha=0.18, ls=":", zorder=0)
    ax.set_title("NGL GNSS stations on the 0.25$^\\circ$ assimilation grid\n"
                 f"{title_kind}  |  {len(st)} raw stations -> {len(cells)} cells "
                 f"({len(drop)} stations share a cell with another)",
                 fontsize=13)
    leg = ax.legend(loc="lower left", fontsize=10, framealpha=0.95,
                    markerscale=3 if args.style == "cells" else 1)
    leg.set_zorder(10)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=130, bbox_inches="tight")
    print(f"land polygons drawn: {n_land}")
    print(f"saved -> {args.output}")


def _records(shp: Path, lon0, lon1, lat0, lat1, margin=2.0):
    """Records whose bounding box touches the window (cheap filter)."""
    for rings in read_shapefile(shp):
        box = np.vstack(rings)
        if box[:, 0].max() < lon0 - margin or box[:, 0].min() > lon1 + margin:
            continue
        if box[:, 1].max() < lat0 - margin or box[:, 1].min() > lat1 + margin:
            continue
        yield rings


if __name__ == "__main__":
    main()
