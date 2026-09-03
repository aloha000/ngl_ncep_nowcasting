from __future__ import annotations

import argparse

from .constants import NCEP_VARS
from models.iTransformer import Model


def make_model_config(args: argparse.Namespace) -> argparse.Namespace:
    cfg = argparse.Namespace()
    cfg.task_name = "long_term_forecast"
    cfg.seq_len = args.seq_len
    cfg.pred_len = args.pred_len
    cfg.enc_in = 2 * args.max_neighbors
    cfg.c_out = len(NCEP_VARS)
    cfg.d_model = args.d_model
    cfg.n_heads = args.n_heads
    cfg.e_layers = args.e_layers
    cfg.d_ff = args.d_ff
    cfg.dropout = args.dropout
    cfg.embed = "timeF"
    cfg.freq = args.time_freq
    cfg.n_time_features = int(args.n_time_features)
    cfg.decoder_time_feat = cfg.n_time_features > 0
    cfg.decoder_time_dim = cfg.n_time_features
    cfg.activation = args.activation
    cfg.factor = 1
    cfg.separate_output = True
    cfg.spatial_enc = bool(args.spatial_enc)
    cfg.n_geo = int(getattr(args, "n_geo_total", args.n_geo))
    cfg.spatial_mlp_hidden = int(args.spatial_mlp_hidden)
    cfg.target_h_feat = bool(args.target_h_feat)
    cfg.target_feat_dim = int(getattr(args, "target_feat_dim", 1 if cfg.target_h_feat else 0))
    cfg.max_neighbors = int(args.max_neighbors)
    return cfg
