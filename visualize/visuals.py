"""
生成DDPM论文所需的核心可视化结果
用法: python -m visualize.visuals
"""

import torch
import torchvision
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

from model import UNet, GaussianDiffusion


def set_chinese_font():
    """设置中文字体"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


def tensor_to_numpy(tensor):
    """将tensor转为numpy图像"""
    img = tensor.cpu().numpy()
    img = (img + 1) / 2  # [-1,1] -> [0,1]
    img = np.clip(img, 0, 1)
    img = img.transpose(0, 2, 3, 1)  # (B,C,H,W) -> (B,H,W,C)
    if img.shape[-1] == 1:
        img = img.squeeze(-1)
    return img


def generate_samples(model, diffusion, device, n_samples=64):
    """生成样本图像"""
    model.eval()
    with torch.no_grad():
        shape = (n_samples, 1, 32, 32)
        samples_list = diffusion.p_sample_loop(model, shape)
        samples = samples_list[-1].to(device)
    return samples


def save_denoise_process(model, diffusion, device, save_dir):
    """保存去噪过程可视化"""
    model.eval()
    n_steps_to_show = 10
    timesteps = list(range(0, 1000, 1000 // n_steps_to_show))

    with torch.no_grad():
        img = torch.randn(1, 1, 32, 32, device=device)
        imgs = [img.clone()]

        for t in reversed(range(1000)):
            img = diffusion.p_sample(model, img, torch.full((1,), t, device=device, dtype=torch.long))
            if t in timesteps:
                imgs.append(img.clone())

    fig, axes = plt.subplots(1, len(imgs), figsize=(20, 2.5))
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        np_img = tensor_to_numpy(img)[0]
        ax.imshow(np_img, cmap='gray')
        ax.axis('off')
        if i == 0:
            ax.set_title('噪声', fontsize=10)
        elif i == len(imgs) - 1:
            ax.set_title('生成', fontsize=10)
        else:
            ax.set_title(f't={1000 - i * 100}', fontsize=9)

    plt.suptitle('DDPM去噪过程', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'denoise_process.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"去噪过程图已保存: {os.path.join(save_dir, 'denoise_process.png')}")


def save_generated_grid(model, diffusion, device, save_dir, n_samples=64):
    """保存生成样本网格"""
    samples = generate_samples(model, diffusion, device, n_samples)
    np_samples = tensor_to_numpy(samples)

    grid_size = int(np.sqrt(n_samples))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            axes[i, j].imshow(np_samples[idx], cmap='gray')
            axes[i, j].axis('off')

    plt.suptitle('DDPM生成的手写数字', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'generated_grid.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"生成样本网格已保存: {os.path.join(save_dir, 'generated_grid.png')}")


def save_noise_schedule(diffusion, save_dir):
    """保存噪声调度可视化"""
    t = np.arange(1000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(t, diffusion.betas.cpu().numpy(), 'b-', linewidth=1.5)
    ax1.set_xlabel('时间步 t', fontsize=12)
    ax1.set_ylabel('β_t', fontsize=12)
    ax1.set_title('噪声调度 β_t', fontsize=13)
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, diffusion.alphas_cumprod.cpu().numpy(), 'r-', linewidth=1.5)
    ax2.set_xlabel('时间步 t', fontsize=12)
    ax2.set_ylabel('ᾱ_t', fontsize=12)
    ax2.set_title('累积乘积 ᾱ_t', fontsize=13)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'noise_schedule.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"噪声调度图已保存: {os.path.join(save_dir, 'noise_schedule.png')}")


def save_forward_process(diffusion, save_dir, device):
    """保存前向加噪过程可视化"""
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(32),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5], [0.5])
    ])
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    x0 = dataset[0][0].unsqueeze(0).to(device)

    timesteps_to_show = [0, 50, 100, 200, 400, 600, 800, 999]

    fig, axes = plt.subplots(1, len(timesteps_to_show), figsize=(16, 2.5))
    for i, (ax, t) in enumerate(zip(axes, timesteps_to_show)):
        t_tensor = torch.tensor([t], device=device)
        x_t = diffusion.q_sample(x0, t_tensor)
        np_img = tensor_to_numpy(x_t)[0]
        ax.imshow(np_img, cmap='gray')
        ax.axis('off')
        ax.set_title(f't={t}', fontsize=10)

    plt.suptitle('前向加噪过程', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'forward_process.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"前向加噪过程图已保存: {os.path.join(save_dir, 'forward_process.png')}")


def save_training_progress(save_dir):
    """保存训练进度对比图"""
    sample_dir = os.path.join('output', 'samples')
    epochs = [1, 5, 10, 20, 30, 40, 50]

    fig, axes = plt.subplots(1, len(epochs), figsize=(16, 3))
    for i, (ax, epoch) in enumerate(zip(axes, epochs)):
        img_path = os.path.join(sample_dir, f'epoch_{epoch}.png')
        if os.path.exists(img_path):
            img = plt.imread(img_path)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, f'Epoch {epoch}\n(未找到)', ha='center', va='center', fontsize=9)
        ax.axis('off')
        ax.set_title(f'Epoch {epoch}', fontsize=10)

    plt.suptitle('训练过程中生成样本的变化', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_progress.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"训练进度对比图已保存: {os.path.join(save_dir, 'training_progress.png')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='output/checkpoint_epoch_50.pt',
                        help='模型checkpoint路径')
    parser.add_argument('--output_dir', type=str, default='output/visuals',
                        help='可视化结果保存目录')
    args = parser.parse_args()

    set_chinese_font()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = UNet(in_channels=1, base_channels=32).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"模型加载成功，训练到第 {checkpoint['epoch']} 个epoch")

    diffusion = GaussianDiffusion(timesteps=1000, device=device)

    print("\n=== 生成可视化结果 ===\n")

    print("[1/5] 生成去噪过程图...")
    save_denoise_process(model, diffusion, device, args.output_dir)

    print("[2/5] 生成样本网格图...")
    save_generated_grid(model, diffusion, device, args.output_dir)

    print("[3/5] 生成噪声调度图...")
    save_noise_schedule(diffusion, args.output_dir)

    print("[4/5] 生成前向加噪过程图...")
    save_forward_process(diffusion, args.output_dir, device)

    print("[5/5] 生成训练进度对比图...")
    save_training_progress(args.output_dir)

    print("\n=== 所有可视化结果生成完毕！===")
    print(f"保存目录: {args.output_dir}")


if __name__ == '__main__':
    main()
