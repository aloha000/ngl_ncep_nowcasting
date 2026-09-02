import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # Nowcasting-style tasks may map input variates (e.g. NGL ZTD/ZWD) to a
        # different set of output variates (e.g. NCEP surface variables). When
        # enabled via configs.separate_output, a final linear layer maps the
        # input-token projections to the target variates.
        self.separate_output = getattr(configs, "separate_output", False)
        # Optional target-station feature (e.g. z-scored absolute height of the
        # target station): a per-sample scalar concatenated to the token outputs
        # right before the separate_output linear layer. Kept out of the input
        # variates on purpose: per-variate instance normalization would zero out
        # any channel that is constant across the window.
        self.target_h_feat = bool(getattr(configs, "target_h_feat", False))
        self.target_feat_dim = int(getattr(configs, "target_feat_dim", 1 if self.target_h_feat else 0))
        # Optional per-neighbor spatial (positional) encoding: geometry of each
        # GNSS neighbor relative to the target station (ENU + heights) is
        # embedded and added to that neighbor's input tokens before the encoder.
        self.spatial_enc = bool(getattr(configs, "spatial_enc", False))
        self.max_neighbors = int(getattr(configs, "max_neighbors", 0))
        self.channels_per_neighbor = (int(getattr(configs, "enc_in", 0)) // self.max_neighbors) if self.max_neighbors else 3
        if self.spatial_enc:
            n_geo = int(getattr(configs, "n_geo", 5))
            mlp_hidden = int(getattr(configs, "spatial_mlp_hidden", configs.d_model))
            self.spatial_embed = nn.Sequential(
                nn.Linear(n_geo, mlp_hidden),
                nn.GELU(),
                nn.Linear(mlp_hidden, configs.d_model),
            )
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
            if self.separate_output:
                out_in = configs.enc_in + (self.target_feat_dim if self.target_h_feat else 0)
                self.output_layer = nn.Linear(out_in, configs.c_out, bias=True)
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_geo=None, x_tgt=None):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        if self.spatial_enc and x_geo is not None:
            # x_geo: (B, max_neighbors, n_geo) -> (B, max_neighbors, d_model);
            # each neighbor contributes the same embedding to its ztd/zwd/mask tokens.
            emb = self.spatial_embed(x_geo)
            emb = emb.unsqueeze(2).expand(-1, -1, self.channels_per_neighbor, -1)
            emb = emb.reshape(emb.shape[0], -1, emb.shape[-1])
            enc_out[:, : emb.shape[1]] = enc_out[:, : emb.shape[1]] + emb
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        if self.separate_output:
            # Input and output variates differ; map input tokens to the target
            # variates. De-normalization is skipped because the output channels
            # do not correspond to the normalized input channels.
            if self.target_h_feat and x_tgt is not None:
                # x_tgt: (B, target_feat_dim) z-scored target-station features.
                dec_out = torch.cat(
                    [dec_out, x_tgt.unsqueeze(1).expand(-1, dec_out.shape[1], -1)], dim=-1
                )
            dec_out = self.output_layer(dec_out)
        else:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, x_geo=None, x_tgt=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, x_geo=x_geo, x_tgt=x_tgt)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
