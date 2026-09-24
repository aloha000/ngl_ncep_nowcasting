import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

__all__ = ["AssimilationNetv6"]


class ln_norm(nn.Module):
    def __init__(
            self,
            embed_dim,
            eps=1e-5,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim, eps=eps)

    def forward(self, x):
        # x c h w -> x h w c
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class EncoderBlock(nn.Module):
    def __init__(self,
                 in_chans,
                 embed_dim,
                 out_chans,
                 kernel_size=3,
                 act=True,
                 ):
        super().__init__()
        self.block = self.build_2D_block(in_chans, embed_dim, out_chans, kernel_size, act)

    def build_2D_block(self, in_chans, embed_dim, out_chans, kernel_size, act=True):
        stage = nn.Sequential(
            nn.Conv2d(
                in_chans, embed_dim,
                kernel_size=(2, 2),
                stride=(2, 2),
                padding=0,
            ),
            ln_norm(embed_dim, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(
                embed_dim, out_chans,
                kernel_size=(kernel_size, kernel_size),
                stride=(1, 1),
                padding=1,
            ),
        )
        if act is True:
            stage.add_module('4', ln_norm(embed_dim, eps=1e-6))
            stage.add_module('5', nn.SiLU())
        return stage

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self,
                 in_chans=256,
                 embed_dim=256,
                 out_chans=256,
                 kernel_size=3,
                 act=True,
                 ):
        super().__init__()
        self.block = self.build_2D_block(in_chans, embed_dim, out_chans, kernel_size, act)

    def build_2D_block(self, in_chans, embed_dim, out_chans, kernel_size, act=True):
        stage = nn.Sequential(
            nn.Conv2d(
                in_chans, embed_dim,
                kernel_size=(kernel_size, kernel_size),
                stride=(1, 1),
                padding=1,
            ),
            ln_norm(embed_dim, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(
                embed_dim, out_chans * 4,
                kernel_size=(kernel_size, kernel_size),
                stride=(1, 1),
                padding=1,
            ),
            nn.PixelShuffle(2),
        )
        if act is True:
            stage.add_module('5', ln_norm(out_chans, eps=1e-6))
            stage.add_module('6', nn.SiLU())
        return stage

    def forward(self, x):
        return self.block(x)


class FussionNetv2(nn.Module):
    def __init__(self,
                 in_chans_bg=70,
                 in_chans_obs=8,
                 in_chans_side=7,
                 embed_dim=256,
                 ):
        super().__init__()
        self.downsample_0 = EncoderBlock(in_chans_bg + in_chans_obs + in_chans_side, embed_dim, embed_dim, act=True)
        self.downsample_1 = EncoderBlock(embed_dim, embed_dim * 2, embed_dim * 2, act=True)

        self.upsample_bg_1 = DecoderBlock(embed_dim * 2, embed_dim * 2, embed_dim, act=True)
        self.upsample_bg_0 = DecoderBlock(embed_dim * 2, embed_dim, in_chans_bg, act=False)

        self.upsample_obs_1 = DecoderBlock(embed_dim * 2, embed_dim * 2, embed_dim, act=True)
        self.upsample_obs_0 = DecoderBlock(embed_dim * 2, embed_dim, in_chans_obs, act=False)

        self.upsample_side_1 = DecoderBlock(embed_dim * 2, embed_dim * 2, embed_dim, act=True)
        self.upsample_side_0 = DecoderBlock(embed_dim * 2, embed_dim, in_chans_side, act=True)

    def forward(self, bg, obs, side_info):
        # bg, obs (n c h w)
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
    def __init__(self,
                 in_chans_bg=70,
                 in_chans_obs=8,
                 in_chans_side=7,
                 embed_dim=256,
                 depth=1,
                 module=FussionNetv2,
                 ):
        super().__init__()

        self.module_list = nn.ModuleList()
        for d in range(0, depth):
            self.module_list.append(module(in_chans_bg, in_chans_obs, in_chans_side, embed_dim))

    def forward(self, bg, obs, side_info):
        for m in self.module_list:
            bg, obs, side_info = m(bg, obs, side_info)
        return bg, obs, side_info


class AssimilationBlockv2(nn.Module):
    def __init__(self,
                 bg_encoder=None,
                 obs_encoder=None,
                 side_encoder=None,
                 fussion=None,
                 ):
        super().__init__()
        self.bg_encoder = bg_encoder
        self.obs_encoder = obs_encoder
        self.side_encoder = side_encoder
        self.fussion = fussion
        pass

    def forward(self, bg, obs, side_info):
        # bg.ndim (n c h w),
        #
        bg = self.bg_encoder(bg)
        obs = self.obs_encoder(obs)
        side_info = self.side_encoder(side_info)

        bg, obs, side_info = self.fussion(bg, obs, side_info)
        return bg, obs, side_info


class EnhanceStack(nn.Module):
    def __init__(self,
                 in_chans=70,
                 embed_dim=256,
                 ):
        super().__init__()
        self.encoder0 = EncoderBlock(in_chans=in_chans, embed_dim=embed_dim, out_chans=embed_dim, act=True)

        self.encoder1 = EncoderBlock(in_chans=embed_dim, embed_dim=embed_dim * 2, out_chans=embed_dim * 2, act=True)

        self.decoder1 = DecoderBlock(in_chans=embed_dim * 2, embed_dim=embed_dim * 2, out_chans=embed_dim, act=True)

        self.decoder0 = DecoderBlock(in_chans=embed_dim * 2, embed_dim=embed_dim, out_chans=in_chans, act=False)

    def forward(self, x):
        x0 = self.encoder0(x)
        x1 = self.encoder1(x0)
        x2 = self.decoder1(x1)
        x2 = torch.concat([x0, x2], dim=1)
        out = self.decoder0(x2)
        return out


class AssimilationNetv6(nn.Module):
    def __init__(self,
                 bg_chans=70,
                 obs_chans=8,
                 obs_frames=8,
                 embed_dim=256,
                 depth=(1, 1, 1),
                 pad_multiple=16,
                 ):
        super().__init__()
        # 参考版的输入是 720x1440（H/W 都是 16 的倍数）。每级 fussion 里连着两次
        # stride-2 下采样再上采样，要求尺寸能被 4 整除；欧域网格 80x120 的 W=120
        # 不满足（120/4=30，30 不是 4 的倍数），所以先用 replicate 补齐、
        # 出来再裁回原尺寸。
        self.pad_multiple = int(pad_multiple)
        side_chans = obs_chans * obs_frames + bg_chans
        obs_chans = obs_chans * obs_frames

        self.block_0 = AssimilationBlockv2(
            bg_encoder=EncoderBlock(in_chans=bg_chans, embed_dim=embed_dim, out_chans=embed_dim, act=True),
            obs_encoder=EncoderBlock(in_chans=obs_chans, embed_dim=embed_dim, out_chans=embed_dim, act=True),
            side_encoder=EncoderBlock(in_chans=side_chans, embed_dim=embed_dim, out_chans=embed_dim, act=True),
            fussion=FussionStackv2(in_chans_bg=embed_dim, in_chans_obs=embed_dim, in_chans_side=embed_dim,
                                   embed_dim=embed_dim, depth=depth[0], module=FussionNetv2))

        self.block_1 = AssimilationBlockv2(
            bg_encoder=EncoderBlock(in_chans=embed_dim, embed_dim=embed_dim * 2, out_chans=embed_dim * 2, act=True),
            obs_encoder=EncoderBlock(in_chans=embed_dim, embed_dim=embed_dim * 2, out_chans=embed_dim * 2, act=True),
            side_encoder=EncoderBlock(in_chans=embed_dim, embed_dim=embed_dim * 2, out_chans=embed_dim * 2, act=True),
            fussion=FussionStackv2(in_chans_bg=embed_dim * 2, in_chans_obs=embed_dim * 2, in_chans_side=embed_dim * 2,
                                   embed_dim=embed_dim, depth=depth[1], module=FussionNetv2))

        self.block_2 = AssimilationBlockv2(
            bg_encoder=DecoderBlock(in_chans=embed_dim * 2, embed_dim=embed_dim * 2, out_chans=embed_dim, act=True),
            obs_encoder=DecoderBlock(in_chans=embed_dim * 2, embed_dim=embed_dim * 2, out_chans=embed_dim, act=True),
            side_encoder=DecoderBlock(in_chans=embed_dim * 2, embed_dim=embed_dim * 2, out_chans=embed_dim, act=True),
            fussion=FussionStackv2(in_chans_bg=embed_dim, in_chans_obs=embed_dim, in_chans_side=embed_dim,
                                   embed_dim=embed_dim, depth=depth[2], module=FussionNetv2))

        self.enhance1 = EnhanceStack(in_chans=bg_chans, embed_dim=embed_dim)
        self.enhance2 = EnhanceStack(in_chans=bg_chans, embed_dim=embed_dim)
        self.decoder = DecoderBlock(in_chans=embed_dim * 2, embed_dim=embed_dim, out_chans=bg_chans, act=False)

    def forward(self, bg, obs):
        H, W = int(bg.shape[-2]), int(bg.shape[-1])
        ph, pw = (-H) % self.pad_multiple, (-W) % self.pad_multiple
        if not (ph or pw):
            return self._forward(bg, obs)
        B, T, C = int(obs.shape[0]), int(obs.shape[1]), int(obs.shape[2])
        bg_f = rearrange(bg, 'b t c h w -> b (t c) h w')
        obs_f = rearrange(obs, 'b t c h w -> b (t c) h w')
        bg_p = F.pad(bg_f, (0, pw, 0, ph), mode='replicate').unsqueeze(1)
        obs_p = F.pad(obs_f, (0, pw, 0, ph), mode='replicate')
        obs_p = obs_p.reshape(B, T, C, H + ph, W + pw)
        out = self._forward(bg_p, obs_p)
        return out[..., :H, :W]

    def _forward(self, bg, obs):
        bg = rearrange(bg, 'b t c h w -> b (t c) h w')
        bg_cp = bg.clone()
        
        obs_data = rearrange(obs, 'b t c h w -> b (t c) h w')
        side_info = torch.concat([bg, obs_data], dim=1)
        # n c h w
        bg0, obs_data, side_info = self.block_0(bg, obs_data, side_info)
        bg1, obs_data, side_info = self.block_1(bg0, obs_data, side_info)
        bg2, obs_data, side_info = self.block_2(bg1, obs_data, side_info)
        bg2 = torch.concat([bg0, bg2], dim=1)
        out = self.decoder(bg2)
        out = out + bg_cp
        
        out = self.enhance1(out) + out
        out = self.enhance2(out) + out
        out = rearrange(out, 'b (t c) h w -> b t c h w', t=1)
        return out


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num * 4 / 1024, 'Trainable': trainable_num * 4 / 1024}


def process_bg(data, hw=(720, 1440)):
    B, T, C, H, W = data.shape
    data = data.reshape(B, T * C, H, W)
    output = F.interpolate(
        data,
        size=hw,
        mode="bilinear",
        align_corners=False
    )
    output = output.reshape(B, T, C, hw[0], hw[1])
    return output


if __name__ == '__main__':
    bg = torch.rand((1, 1, 100, 721, 1440)).cuda()
    obs = torch.rand((1, 6, 42, 720, 1440)).cuda()
    mask = torch.rand((1, 6, 40, 720, 1440)).cuda()
    bg = process_bg(bg, hw=(720, 1440))
    model = AssimilationNetv6(bg_chans=100,
                              obs_chans=42,
                              obs_frames=6,
                              embed_dim=256,
                              depth=(1, 1, 1)).cuda()
    x = model(bg, obs, mask)
    print(x.shape)
    print(get_parameter_number(model))

