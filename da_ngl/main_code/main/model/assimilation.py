"""Assimilation network (adapted from ``xuxiaoze/for_zrx/train_packet``).

Background field (FuXi) + observations (a window of 5-minute GNSS ZTD) are
fused into an analysis that is trained directly against ERA5.

Differences from the original packet
-----------------------------------
* ``out_chans`` is configurable.  The decoder predicts ``out_chans`` channels
  but the residual ``out = decoder(...) + bg`` is only applied to the first
  ``bg_chans`` channels, because the label carries one extra channel (IMERG
  ``tp``) that has no FuXi background and therefore cannot be a residual.
* ``EnhanceStack`` input channels follow ``out_chans``.
* ``EncoderBlock``'s trailing LayerNorm uses ``out_chans`` (the original used
  ``embed_dim``, which happened to be equal in every call site).
* the stale ``__main__`` self-test (wrong ``forward`` signature) was dropped.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = ["AssimilationNetv6", "get_parameter_number"]


def pad_replicate(x, pad_h, pad_w):
    """Replicate-pad the last two dims (F.pad only accepts 4D for replicate)."""
    b, t, c, h, w = x.shape
    x = x.reshape(b * t, c, h, w)
    x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
    return x.reshape(b, t, c, h + pad_h, w + pad_w)


class ln_norm(nn.Module):
    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim, eps=eps)

    def forward(self, x):
        # (n c h w) -> (n h w c)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_chans, embed_dim, out_chans, kernel_size=3, act=True):
        super().__init__()
        self.block = self.build_2D_block(in_chans, embed_dim, out_chans, kernel_size, act)

    def build_2D_block(self, in_chans, embed_dim, out_chans, kernel_size, act=True):
        stage = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=(2, 2), stride=(2, 2), padding=0),
            ln_norm(embed_dim, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(embed_dim, out_chans, kernel_size=(kernel_size, kernel_size),
                      stride=(1, 1), padding=1),
        )
        if act is True:
            stage.add_module('4', ln_norm(out_chans, eps=1e-6))
            stage.add_module('5', nn.SiLU())
        return stage

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_chans=256, embed_dim=256, out_chans=256, kernel_size=3, act=True):
        super().__init__()
        self.block = self.build_2D_block(in_chans, embed_dim, out_chans, kernel_size, act)

    def build_2D_block(self, in_chans, embed_dim, out_chans, kernel_size, act=True):
        stage = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=(kernel_size, kernel_size),
                      stride=(1, 1), padding=1),
            ln_norm(embed_dim, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(embed_dim, out_chans * 4, kernel_size=(kernel_size, kernel_size),
                      stride=(1, 1), padding=1),
            nn.PixelShuffle(2),
        )
        if act is True:
            stage.add_module('5', ln_norm(out_chans, eps=1e-6))
            stage.add_module('6', nn.SiLU())
        return stage

    def forward(self, x):
        return self.block(x)


class FussionNetv2(nn.Module):
    """Two-scale fusion of the background / observation / side branches."""

    def __init__(self, in_chans_bg=70, in_chans_obs=8, in_chans_side=7, embed_dim=256):
        super().__init__()
        self.downsample_0 = EncoderBlock(in_chans_bg + in_chans_obs + in_chans_side,
                                         embed_dim, embed_dim, act=True)
        self.downsample_1 = EncoderBlock(embed_dim, embed_dim * 2, embed_dim * 2, act=True)

        self.upsample_bg_1 = DecoderBlock(embed_dim * 2, embed_dim * 2, embed_dim, act=True)
        self.upsample_bg_0 = DecoderBlock(embed_dim * 2, embed_dim, in_chans_bg, act=False)

        self.upsample_obs_1 = DecoderBlock(embed_dim * 2, embed_dim * 2, embed_dim, act=True)
        self.upsample_obs_0 = DecoderBlock(embed_dim * 2, embed_dim, in_chans_obs, act=False)

        self.upsample_side_1 = DecoderBlock(embed_dim * 2, embed_dim * 2, embed_dim, act=True)
        self.upsample_side_0 = DecoderBlock(embed_dim * 2, embed_dim, in_chans_side, act=True)

    def forward(self, bg, obs, side_info):
        h = torch.concat([bg, obs, side_info], dim=1)
        h0 = self.downsample_0(h)
        h1 = self.downsample_1(h0)

        h0_bg = self.upsample_bg_1(h1)
        h0_bg = torch.concat([h0_bg, h0], dim=1)
        out_bg = self.upsample_bg_0(h0_bg)
        bg = bg + out_bg

        h0_obs = self.upsample_obs_1(h1)
        h0_obs = torch.concat([h0_obs, h0], dim=1)
        out_obs = self.upsample_obs_0(h0_obs)
        obs = obs + out_obs

        h0_side = self.upsample_side_1(h1)
        h0_side = torch.concat([h0_side, h0], dim=1)
        out_side = self.upsample_side_0(h0_side)
        return bg, obs, out_side


class FussionStackv2(nn.Module):
    def __init__(self, in_chans_bg=70, in_chans_obs=8, in_chans_side=7,
                 embed_dim=256, depth=1, module=FussionNetv2):
        super().__init__()
        self.module_list = nn.ModuleList()
        for _ in range(depth):
            self.module_list.append(
                module(in_chans_bg, in_chans_obs, in_chans_side, embed_dim))

    def forward(self, bg, obs, side_info):
        for m in self.module_list:
            bg, obs, side_info = m(bg, obs, side_info)
        return bg, obs, side_info


class AssimilationBlockv2(nn.Module):
    def __init__(self, bg_encoder=None, obs_encoder=None, side_encoder=None, fussion=None):
        super().__init__()
        self.bg_encoder = bg_encoder
        self.obs_encoder = obs_encoder
        self.side_encoder = side_encoder
        self.fussion = fussion

    def forward(self, bg, obs, side_info):
        bg = self.bg_encoder(bg)
        obs = self.obs_encoder(obs)
        side_info = self.side_encoder(side_info)
        bg, obs, side_info = self.fussion(bg, obs, side_info)
        return bg, obs, side_info


class EnhanceStack(nn.Module):
    def __init__(self, in_chans=70, embed_dim=256):
        super().__init__()
        self.encoder0 = EncoderBlock(in_chans=in_chans, embed_dim=embed_dim,
                                     out_chans=embed_dim, act=True)
        self.encoder1 = EncoderBlock(in_chans=embed_dim, embed_dim=embed_dim * 2,
                                     out_chans=embed_dim * 2, act=True)
        self.decoder1 = DecoderBlock(in_chans=embed_dim * 2, embed_dim=embed_dim * 2,
                                     out_chans=embed_dim, act=True)
        self.decoder0 = DecoderBlock(in_chans=embed_dim * 2, embed_dim=embed_dim,
                                     out_chans=in_chans, act=False)

    def forward(self, x):
        x0 = self.encoder0(x)
        x1 = self.encoder1(x0)
        x2 = self.decoder1(x1)
        x2 = torch.concat([x0, x2], dim=1)
        return self.decoder0(x2)


class AssimilationNetv6(nn.Module):
    """``bg`` (B,1,bg_chans,H,W) + ``obs`` (B,T,obs_chans,H,W) -> (B,1,out_chans,H,W).

    ``H`` and ``W`` must be divisible by 4 (two stride-2 downsamplings).
    """

    def __init__(self, bg_chans=69, obs_chans=4, obs_frames=25, out_chans=70,
                 embed_dim=256, depth=(1, 1, 1), pad_multiple=16,
                 freeze_msl=False):
        super().__init__()
        self.bg_chans = bg_chans
        self.out_chans = out_chans
        # Hard constraint on msl (HANDOFF 12.8 item 1a): channel 68 of the
        # analysis is forced back to the background value, so the network cannot
        # buy a better ZTD fit by moving surface pressure.  d(ZTD)/d(msl) is the
        # largest sensitivity in the observation operator while msl has the
        # smallest background error, and msl carried 127 % of the net MAE change.
        self.freeze_msl = bool(freeze_msl)
        # every fussion stage halving twice needs H, W divisible by 16; smaller
        # inputs are padded (replicate) and the output is cropped back
        self.pad_multiple = int(pad_multiple)
        side_chans = obs_chans * obs_frames + bg_chans
        obs_total = obs_chans * obs_frames

        self.block_0 = AssimilationBlockv2(
            bg_encoder=EncoderBlock(in_chans=bg_chans, embed_dim=embed_dim,
                                    out_chans=embed_dim, act=True),
            obs_encoder=EncoderBlock(in_chans=obs_total, embed_dim=embed_dim,
                                     out_chans=embed_dim, act=True),
            side_encoder=EncoderBlock(in_chans=side_chans, embed_dim=embed_dim,
                                      out_chans=embed_dim, act=True),
            fussion=FussionStackv2(in_chans_bg=embed_dim, in_chans_obs=embed_dim,
                                   in_chans_side=embed_dim, embed_dim=embed_dim,
                                   depth=depth[0], module=FussionNetv2))

        self.block_1 = AssimilationBlockv2(
            bg_encoder=EncoderBlock(in_chans=embed_dim, embed_dim=embed_dim * 2,
                                    out_chans=embed_dim * 2, act=True),
            obs_encoder=EncoderBlock(in_chans=embed_dim, embed_dim=embed_dim * 2,
                                     out_chans=embed_dim * 2, act=True),
            side_encoder=EncoderBlock(in_chans=embed_dim, embed_dim=embed_dim * 2,
                                      out_chans=embed_dim * 2, act=True),
            fussion=FussionStackv2(in_chans_bg=embed_dim * 2, in_chans_obs=embed_dim * 2,
                                   in_chans_side=embed_dim * 2, embed_dim=embed_dim,
                                   depth=depth[1], module=FussionNetv2))

        self.block_2 = AssimilationBlockv2(
            bg_encoder=DecoderBlock(in_chans=embed_dim * 2, embed_dim=embed_dim * 2,
                                    out_chans=embed_dim, act=True),
            obs_encoder=DecoderBlock(in_chans=embed_dim * 2, embed_dim=embed_dim * 2,
                                     out_chans=embed_dim, act=True),
            side_encoder=DecoderBlock(in_chans=embed_dim * 2, embed_dim=embed_dim * 2,
                                      out_chans=embed_dim, act=True),
            fussion=FussionStackv2(in_chans_bg=embed_dim, in_chans_obs=embed_dim,
                                   in_chans_side=embed_dim, embed_dim=embed_dim,
                                   depth=depth[2], module=FussionNetv2))

        self.enhance1 = EnhanceStack(in_chans=out_chans, embed_dim=embed_dim)
        self.enhance2 = EnhanceStack(in_chans=out_chans, embed_dim=embed_dim)
        self.decoder = DecoderBlock(in_chans=embed_dim * 2, embed_dim=embed_dim,
                                    out_chans=out_chans, act=False)

    def forward(self, bg, obs):
        # bg: (b, t=1, c, h, w) ; obs: (b, t=frames, c, h, w)
        h_in, w_in = bg.shape[-2], bg.shape[-1]
        pad_h = (-h_in) % self.pad_multiple
        pad_w = (-w_in) % self.pad_multiple
        if pad_h or pad_w:
            bg = pad_replicate(bg, pad_h, pad_w)
            obs = pad_replicate(obs, pad_h, pad_w)

        bg = rearrange(bg, 'b t c h w -> b (t c) h w')
        bg_cp = bg.clone()

        obs_data = rearrange(obs, 'b t c h w -> b (t c) h w')
        side_info = torch.concat([bg, obs_data], dim=1)

        bg0, obs_data, side_info = self.block_0(bg, obs_data, side_info)
        bg1, obs_data, side_info = self.block_1(bg0, obs_data, side_info)
        bg2, obs_data, side_info = self.block_2(bg1, obs_data, side_info)

        bg2 = torch.concat([bg0, bg2], dim=1)
        out = self.decoder(bg2)
        # residual only on the channels that have a background
        out = out + torch.nn.functional.pad(
            bg_cp, (0, 0, 0, 0, 0, self.out_chans - self.bg_chans))

        out = self.enhance1(out) + out
        out = self.enhance2(out) + out
        if self.freeze_msl and self.out_chans > 68 and self.bg_chans > 68:
            # out (b, c, h, w) <-> bg_cp (b, bg_chans, h, w); channel 68 = msl
            out = torch.cat(
                [out[:, :68], bg_cp[:, 68:69].to(out.dtype), out[:, 69:]], dim=1)
        out = rearrange(out, 'b (t c) h w -> b t c h w', t=1, c=self.out_chans)
        if pad_h or pad_w:
            out = out[..., :h_in, :w_in]
        return out


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total(MB)': total_num * 4 / 1024 ** 2,
            'Trainable(MB)': trainable_num * 4 / 1024 ** 2,
            'Total': total_num, 'Trainable': trainable_num}
