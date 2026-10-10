#!/usr/bin/env python3
"""Check the torch ZTD operator against the numpy-built ``ztd_fuxi_*`` store.

``main/model/ztd_torch.py`` reimplements ``ztd_operator.ztd_profile_surface`` so
the observation-consistency loss can differentiate through it.  The two must
agree to float32 noise, otherwise the network would be pushed towards an
analysis whose implied ZTD differs from the one the innovation was built with.

This feeds the *same* FuXi background the assimilation uses (standardised, at
``init = T - lead``) into ``StationZTD`` and compares the result with the stored
``ztd_fuxi`` at the station cells.  Any mismatch in the channel order, the
de-standardisation, the station cells or the physics shows up here.

Usage (from da_ngl/main_code, or anywhere -- paths come from the configs):
    python ../preprocessing/check_ztd_torch.py --samples 8
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "main_code"))
sys.path.insert(0, str(HERE))

from common import DATASET_DIR, decode_time_axis  # noqa: E402
from main.model import StationZTD  # noqa: E402


def apply_overrides(configs, items):
    cfg = importlib.import_module(configs)
    for item in items:
        key, _, value = item.partition("=")
        for cast in (int, float):
            try:
                value = cast(value)
                break
            except ValueError:
                pass
        setattr(cfg, key.strip(), value)
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", default="configs")
    ap.add_argument("--ztd-store", type=Path, default=None,
                    help="default: dataset/ztd_fuxi_europe_0p25_{lead}h.zarr")
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    cfg = apply_overrides(args.configs, args.set)
    lead = int(cfg.fcst_step) * 6
    # 默认跟 configs 走（方法 E 之后库名带 _zdz），避免比到旧实现产出的库
    ztd_path = args.ztd_store or Path(getattr(cfg, "ztd_fuxi_zarr",
                                             Path(DATASET_DIR) / f"ztd_fuxi_europe_0p25_{lead}h.zarr"))

    gf = zarr.open(str(cfg.fuxi_zarr), "r")
    zf = zarr.open(str(ztd_path), "r")
    steps = np.asarray(gf["step"][:])
    if lead not in steps:
        raise SystemExit(f"{cfg.fuxi_zarr} has no lead {lead} h (steps={steps.tolist()})")
    lead_idx = int(np.where(steps == lead)[0][0])

    init_t = decode_time_axis(Path(cfg.fuxi_zarr), "init")
    lab_t = decode_time_axis(ztd_path, "time")
    init_step = init_t[1] - init_t[0]

    op = StationZTD(cfg)
    iy, ix = op.cells
    iy, ix = iy.numpy(), ix.numpy()

    # the store flattens its stations in the same order the grid map does
    stored_ids = np.asarray(zf["station"].attrs["station_id"]).astype(str)
    if stored_ids.size == len(op.station_id) and not np.array_equal(stored_ids, op.station_id):
        print("WARNING: station order in the store differs from the grid map")

    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(lab_t.size, size=min(args.samples, lab_t.size),
                              replace=False))
    print(f"[check] {ztd_path.name}: lead {lead} h, {pick.size} of {lab_t.size} times")
    worst = 0.0
    for k in pick:
        bg_i = int((lab_t[k] - pd.Timedelta(hours=lead) - init_t[0]) / init_step)
        bg = np.asarray(gf["z"][bg_i, lead_idx, :69], dtype=np.float32)
        h_torch = op(torch.from_numpy(bg)[None, None]).detach().numpy().reshape(-1)
        ref = np.asarray(zf["ztd_fuxi"][k], dtype=np.float32)[iy, ix]
        ok = np.isfinite(ref) & np.isfinite(h_torch)
        if ok.sum() == 0:
            print(f"  {lab_t[k]}: no finite reference cells -- skipped")
            continue
        d = np.abs(h_torch[ok] - ref[ok])
        worst = max(worst, float(d.max()))
        print(f"  {lab_t[k]}  n={ok.sum():5d}  max|d|={d.max():.4f} mm  "
              f"mean|d|={d.mean():.5f} mm  (ref {ref[ok].mean():.1f} mm)")
    print(f"[check] worst |torch - numpy| = {worst:.4f} mm "
          f"{'OK' if worst < 0.05 else '*** FAIL ***'}")

    ok_grad = grad_check(cfg, bg)
    return 0 if (worst < 0.05 and ok_grad) else 1


def grad_check(cfg, bg_np):
    """Frozen-ZHD mode: same value, but no gradient path into msl.

    ``H*(x_a) = ZHD(x_bg) + ZWD(x_a)`` must (a) reproduce the unfrozen operator
    exactly at ``x_a = x_bg`` -- otherwise the de-biased target, which was built
    from the full ``H(bg)``, would be off -- and (b) give ``msl`` exactly zero
    sensitivity, while the thermodynamic channels keep theirs.
    """
    x0 = torch.from_numpy(bg_np)[None, None]
    plain = StationZTD(cfg, freeze_zhd=False)
    frozen = StationZTD(cfg, freeze_zhd=True)

    with torch.no_grad():
        z_plain = plain(x0)
        z_frozen = frozen(x0, x0)
    dmax = float((z_plain - z_frozen).abs().max())
    print(f"\n[grad ] value at x=x_bg: max|frozen - plain| = {dmax:.6f} mm "
          f"{'OK' if dmax < 1e-3 else '*** FAIL ***'}")

    def chan_grad(op, with_bg):
        x = torch.from_numpy(bg_np)[None].clone().requires_grad_(True)
        op(x, x if with_bg else None).sum().backward()
        return x.grad.abs().sum(dim=(0, 2, 3)).numpy()   # (C,) total |dH/dx_c|

    g_plain, g_frozen = chan_grad(plain, False), chan_grad(frozen, True)
    from common import CHANNELS
    rows = ["msl", "t2m", "t850", "t500", "r1000", "r850", "r700", "r500",
            "z850", "u850", "v850"]
    print(f"{'chan':>6} {'|dH/dx| plain':>14} {'|dH/dx| frozen':>15}")
    for nm in rows:
        c = CHANNELS.index(nm)
        print(f"{nm:>6} {g_plain[c]:14.1f} {g_frozen[c]:15.1f}")
    fam = lambda g, p: sum(g[CHANNELS.index(nm)] for nm in CHANNELS if nm.startswith(p))
    print(f"{'family':>6} {'plain':>14} {'frozen':>15}")
    for pre, lbl in (("t", "t*"), ("r", "r*"), ("z", "z*"), ("u", "u*"), ("v", "v*")):
        print(f"{lbl:>6} {fam(g_plain, pre):14.1f} {fam(g_frozen, pre):15.1f}")
    print(f"{'total':>6} {g_plain.sum():14.1f} {g_frozen.sum():15.1f}")
    zop = sum(g_plain[CHANNELS.index(nm)] for nm in CHANNELS
              if nm.startswith(("z", "u", "v")))
    zop_f = sum(g_frozen[CHANNELS.index(nm)] for nm in CHANNELS
                if nm.startswith(("z", "u", "v")))
    print(f"z* (method E 下算子会读 z 当层高；frozen 时应为 0): "
          f"plain {zop:.1f}, frozen {zop_f:.1f}")
    q = lambda g, nm: g[CHANNELS.index(nm)]
    print(f"u*/v* (算子始终不读): plain {sum(q(g_plain, nm) for nm in CHANNELS if nm.startswith(('u','v'))):.1f}")

    ok = (dmax < 1e-3 and g_frozen[CHANNELS.index("msl")] == 0.0
          and g_plain[CHANNELS.index("msl")] > 0.0
          and g_frozen[CHANNELS.index("r700")] > 0.0
          and g_frozen[CHANNELS.index("z850")] == 0.0)
    print(f"[grad ] msl sensitivity {g_plain[CHANNELS.index('msl')]:.1f} -> "
          f"{g_frozen[CHANNELS.index('msl')]:.1f} "
          f"{'OK' if ok else '*** FAIL ***'}")
    return ok


if __name__ == "__main__":
    raise SystemExit(main())
