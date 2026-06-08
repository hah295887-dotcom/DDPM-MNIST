"""
高斯扩散过程
实现DDPM的前向加噪和反向去噪过程
"""

import torch
import torch.nn.functional as F


class GaussianDiffusion:
    """高斯扩散过程，管理前向加噪和反向去噪

    使用线性噪声调度：beta_t从beta_start线性增长到beta_end。

    Args:
        timesteps: 扩散步数T
        beta_start: 噪声调度起始值
        beta_end: 噪声调度终止值
        device: 计算设备
    """

    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.timesteps = timesteps
        self.device = device

        # 线性beta调度
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # 前向过程 q(x_t | x_{t-1}) 所需
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # 后向过程 q(x_{t-1} | x_t, x_0) 所需
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_0, t, noise=None):
        """前向过程: 从x_0直接采样x_t

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

        Args:
            x_0: 原始图像
            t: 时间步
            noise: 噪声（默认随机生成）
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]

        # 扩展维度以匹配图像形状 [B, 1, 1, 1]
        sqrt_alpha = sqrt_alpha.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = sqrt_one_minus_alpha.view(-1, 1, 1, 1)

        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def p_losses(self, model, x_0, t):
        """计算训练损失: 模型预测噪声与真实噪声的MSE

        L = ||epsilon - epsilon_theta(x_t, t)||^2

        Args:
            model: U-Net噪声预测网络
            x_0: 原始图像
            t: 时间步
        """
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise=noise)
        predicted_noise = model(x_t, t)
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def p_sample(self, model, x_t, t):
        """单步反向去噪: 从x_t采样x_{t-1}

        Args:
            model: U-Net噪声预测网络
            x_t: 当前时刻的噪声图像
            t: 时间步
        """
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recip_alphas_t = (1.0 / torch.sqrt(self.alphas[t])).view(-1, 1, 1, 1)

        # 模型预测噪声
        predicted_noise = model(x_t, t)

        # 计算均值
        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if (t == 0).any():
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t].view(-1, 1, 1, 1)
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        """完整反向采样: 从纯噪声逐步生成图像

        从x_T ~ N(0, I)开始，逐步去噪到x_0。

        Args:
            model: U-Net噪声预测网络
            shape: 生成图像的形状 (B, C, H, W)

        Returns:
            imgs: 采样过程中关键步骤的图像列表
        """
        device = self.device
        img = torch.randn(shape, device=device)
        imgs = []

        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t_batch)
            if t % 100 == 0 or t == self.timesteps - 1:
                imgs.append(img.cpu())

        imgs.append(img.cpu())
        return imgs
