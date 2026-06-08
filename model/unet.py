"""
U-Net噪声预测网络
基于论文: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码，将时间步t编码为向量"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResBlock(nn.Module):
    """残差块，包含时间嵌入"""

    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch),
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        t_emb = self.time_mlp(t_emb)[:, :, None, None]
        h = h + t_emb
        h = self.conv2(h)
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """自注意力块"""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.q(h).view(B, C, -1)
        k = self.k(h).view(B, C, -1)
        v = self.v(h).view(B, C, -1)

        scale = C ** -0.5
        attn = torch.bmm(q.transpose(1, 2), k) * scale
        attn = F.softmax(attn, dim=-1)
        h = torch.bmm(v, attn.transpose(1, 2)).view(B, C, H, W)
        h = self.proj(h)
        return x + h


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, use_attention=False):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, time_emb_dim)
        self.attn = AttentionBlock(out_ch) if use_attention else nn.Identity()
        self.downsample = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t_emb):
        x = self.res(x, t_emb)
        x = self.attn(x)
        x = self.downsample(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, use_attention=False):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
        )
        self.res = ResBlock(in_ch + out_ch, out_ch, time_emb_dim)
        self.attn = AttentionBlock(out_ch) if use_attention else nn.Identity()

    def forward(self, x, skip, t_emb):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, t_emb)
        x = self.attn(x)
        return x


class UNet(nn.Module):
    """U-Net噪声预测网络

    编码器-解码器架构，包含正弦时间编码、残差块、自注意力机制、下采样与上采样。
    输入为带噪声的图像x_t和时间步t，输出为预测的噪声epsilon_theta。

    Args:
        in_channels: 输入通道数（MNIST灰度图为1）
        base_channels: 基础通道数，控制网络宽度
        time_emb_dim: 时间嵌入维度
    """

    def __init__(self, in_channels=1, base_channels=64, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # 编码器
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.down1 = DownBlock(base_channels, base_channels, time_emb_dim)
        self.down2 = DownBlock(base_channels, base_channels * 2, time_emb_dim, use_attention=True)
        self.down3 = DownBlock(base_channels * 2, base_channels * 4, time_emb_dim, use_attention=True)

        # 中间层
        self.mid_res1 = ResBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        self.mid_attn = AttentionBlock(base_channels * 4)
        self.mid_res2 = ResBlock(base_channels * 4, base_channels * 4, time_emb_dim)

        # 解码器
        self.up3 = UpBlock(base_channels * 4, base_channels * 2, time_emb_dim, use_attention=True)
        self.up2 = UpBlock(base_channels * 2, base_channels, time_emb_dim, use_attention=True)
        self.up1 = UpBlock(base_channels, base_channels, time_emb_dim)

        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        # 编码
        h1 = self.conv_in(x)
        h2 = self.down1(h1, t_emb)
        h3 = self.down2(h2, t_emb)
        h4 = self.down3(h3, t_emb)

        # 中间
        h = self.mid_res1(h4, t_emb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, t_emb)

        # 解码
        h = self.up3(h, h3, t_emb)
        h = self.up2(h, h2, t_emb)
        h = self.up1(h, h1, t_emb)

        return self.conv_out(h)
