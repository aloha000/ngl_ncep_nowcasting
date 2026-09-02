#!/usr/bin/env python3
"""Compatibility entry point for GNSS -> NCEP nowcasting training.

Implementation lives in ``nowcasting/main_code``. Existing commands can keep
using this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nowcasting.main_code.config import build_cli_parser, load_config, parse_utc, resolve_device
from nowcasting.main_code.constants import NCEP_VARS, NGL_VARS, ROOT, TSL_ROOT
from nowcasting.main_code.dataset import GNSSNowcastDataset, NowcastData, make_loader
from nowcasting.main_code.geometry import _ecef, _enu
from nowcasting.main_code.model import Model, make_model_config
from nowcasting.main_code.plot import plot_loss_curve, plot_test_analysis
from nowcasting.main_code.train import evaluate, test, train
from nowcasting.main_code.main import main

__all__ = [
    "ROOT", "TSL_ROOT", "NCEP_VARS", "NGL_VARS",
    "parse_utc", "build_cli_parser", "load_config", "resolve_device",
    "_ecef", "_enu", "NowcastData", "GNSSNowcastDataset", "make_loader",
    "Model", "make_model_config", "plot_loss_curve", "plot_test_analysis",
    "evaluate", "train", "test", "main",
]


if __name__ == "__main__":
    main()
