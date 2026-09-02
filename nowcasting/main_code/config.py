from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml

from .constants import ROOT

def parse_utc(value: str):
    if not value:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


# Run parameters come exclusively from the YAML config (nowcasting/config.yaml);
# the script holds no value defaults. Missing/unknown keys are reported at startup.
REQUIRED_KEYS = {
    # data
    "ngl_zarr", "ncep_zarr", "neighbors_parquet", "target_stations_parquet",
    "gnss_stations_parquet", "ngl_step_minutes",
    # temporal splits
    "train_start", "train_end", "val_start", "val_end", "test_start", "test_end",
    # station/time sampling
    "stations", "station_offset", "max_neighbors", "min_valid_neighbors",
    "hour_stride", "val_stride", "test_stride", "load_full_arrays", "load_full_ncep",
    # model
    "window_hours", "pred_len", "time_encoding", "time_freq", "spatial_enc", "n_geo", "spatial_mlp_hidden",
    "target_h_feat",
    "d_model", "n_heads", "e_layers", "d_ff", "dropout", "activation",
    # training
    "epochs", "batch_size", "learning_rate", "patience", "num_workers", "seed",
    "target_scale",
    # run
    "device", "model_id", "out_root",
}

PATH_KEYS = ("ngl_zarr", "ncep_zarr", "neighbors_parquet",
             "target_stations_parquet", "gnss_stations_parquet", "out_root")


def _flatten(mapping: dict) -> dict:
    """Flatten nested YAML sections to leaf key names (e.g. data.ngl_zarr -> ngl_zarr)."""
    out = {}
    for key, value in mapping.items():
        if isinstance(value, dict):
            out.update(_flatten(value))
        else:
            out[key] = value
    return out


def coerce_value(text: str):
    if text in ("true", "True"):
        return True
    if text in ("false", "False"):
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def load_config(config_path: Path, overrides: list[str]) -> argparse.Namespace:
    if not config_path.exists():
        raise SystemExit(f"config file not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    flat = _flatten(raw)

    unknown = sorted(set(flat) - REQUIRED_KEYS)
    if unknown:
        raise SystemExit(f"unknown config key(s): {', '.join(unknown)}")

    for spec in overrides:
        if "=" not in spec:
            raise SystemExit(f"--set expects KEY=VALUE, got: {spec!r}")
        key, value = spec.split("=", 1)
        if key not in REQUIRED_KEYS:
            raise SystemExit(f"unknown config key in --set: {key!r}")
        flat[key] = coerce_value(value)

    missing = sorted(REQUIRED_KEYS - set(flat))
    if missing:
        raise SystemExit(
            "missing config key(s); add them to the YAML config: " + ", ".join(missing)
        )

    args = argparse.Namespace(**flat)
    for key in PATH_KEYS:
        path = Path(getattr(args, key))
        setattr(args, key, path if path.is_absolute() else ROOT / path)
    return args


def build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="All run parameters live in the YAML config (default: nowcasting/config.yaml). "
               "Override any key with --set KEY=VALUE (repeatable).",
    )
    p.add_argument("--config", type=Path, default=ROOT / "nowcasting/config.yaml",
                   help="path to the YAML config file")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config key, e.g. --set stations=128 --set epochs=20")
    return p


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit(f"device {name} requested but CUDA is not available")
        return torch.device(name)
    return torch.device("cpu")
