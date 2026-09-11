from __future__ import annotations

import argparse

from .constants import ERA5_BACKGROUND_CHANNELS, NCEP_VARS, NGL_VARS
from models.iTransformer import Model


def make_model_config(args: argparse.Namespace) -> argparse.Namespace:
    cfg = argparse.Namespace()
    cfg.task_name = "long_term_forecast"
    cfg.seq_len = args.seq_len
    cfg.pred_len = args.pred_len
    cfg.n_geo = int(args.n_geo)
    # ZTD is the only input token. Per-neighbour static and ERA5
    # features are concatenated onto their token embeddings before the encoder.
    cfg.enc_in = len(NGL_VARS) * args.max_neighbors
    cfg.spatial_feature_dim = int(getattr(args, "n_geo_total", args.n_geo)) if args.spatial_enc else 0
    cfg.era5_dim = len(ERA5_BACKGROUND_CHANNELS) if args.use_era5 else 0
    cfg.encoder_d_model = args.d_model + cfg.spatial_feature_dim + cfg.era5_dim
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
    # No spatial/ERA5 encoder MLP is used.
    cfg.target_h_feat = bool(args.target_h_feat)
    cfg.target_feat_dim = int(getattr(args, "target_feat_dim", 1 if cfg.target_h_feat else 0))
    cfg.max_neighbors = int(args.max_neighbors)
    return cfg
