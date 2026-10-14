from __future__ import annotations

import json
import random
import time

import numpy as np
import torch

from .constants import ROOT  # noqa: F401 - imports set up Time-Series-Library path
from .config import build_cli_parser, load_config, resolve_device
from .dataset import NowcastData
from .model import make_model_config
from .train import test, train
from utils.timefeatures import time_features_from_frequency_str


def main():
    cli = build_cli_parser().parse_args()
    args = load_config(cli.config, cli.set)
    args.seq_len = int(args.window_hours * 60 // args.ngl_step_minutes) + 1
    if args.time_encoding == "none":
        args.n_time_features = 0
    elif args.time_encoding == "linear":
        args.n_time_features = len(time_features_from_frequency_str(args.time_freq))
    elif args.time_encoding == "hour_sincos":
        args.n_time_features = 2
    elif args.time_encoding == "sincos":
        args.n_time_features = 8  # hour/day-of-week/day-of-month/day-of-year, each sin+cos
    else:
        raise SystemExit(f"unknown time_encoding: {args.time_encoding!r} (choose none|linear|hour_sincos|sincos)")
    print("effective config:")
    print(json.dumps(vars(args), indent=2, default=str))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"device={device}  zarr ngl={args.ngl_zarr}  ncep={args.ncep_zarr}")

    t0 = time.time()
    data = NowcastData(args)
    print(f"data index built in {time.time() - t0:.1f}s")

    cfg = make_model_config(args)
    model, out_dir = train(args, data, device, cfg)
    test(args, data, model, cfg, out_dir, device)
    print(f"outputs saved under {out_dir}")


if __name__ == "__main__":
    main()
