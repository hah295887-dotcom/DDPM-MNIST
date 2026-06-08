"""
DDPM训练脚本
用法: python -m train.train --epochs 50
"""

import os
import math
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from torch.optim import Adam
from tqdm import tqdm

from model import UNet, GaussianDiffusion


def train(args):
    """训练DDPM模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'samples'), exist_ok=True)

    # 数据集
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # 归一化到[-1, 1]
    ])
    dataset = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    # 模型
    model = UNet(in_channels=1, base_channels=args.base_channels).to(device)
    diffusion = GaussianDiffusion(timesteps=args.timesteps, device=device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 训练
    print('开始训练...')
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')

        for batch, _ in pbar:
            batch = batch.to(device)
            t = torch.randint(0, args.timesteps, (batch.shape[0],), device=device)

            loss = diffusion.p_losses(model, batch, t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}')

        # 保存采样图像
        if (epoch + 1) % args.sample_interval == 0 or epoch == 0:
            sample_and_save(model, diffusion, device, args, epoch + 1)

        # 保存模型
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt'))
            print(f'模型已保存: checkpoint_epoch_{epoch+1}.pt')

    # 最终采样
    sample_and_save(model, diffusion, device, args, args.epochs, num_samples=64)

    print('训练完成！')


@torch.no_grad()
def sample_and_save(model, diffusion, device, args, epoch, num_samples=16):
    """采样并保存生成图像"""
    model.eval()
    shape = (num_samples, 1, args.image_size, args.image_size)
    imgs = diffusion.p_sample_loop(model, shape)

    # 保存最终生成结果
    final_imgs = imgs[-1]
    final_imgs = (final_imgs + 1) * 0.5  # [-1,1] -> [0,1]
    final_imgs = final_imgs.clamp(0, 1)
    grid = make_grid(final_imgs, nrow=int(math.sqrt(num_samples)), padding=2)
    save_path = os.path.join(args.output_dir, 'samples', f'epoch_{epoch}.png')
    save_image(grid, save_path)
    print(f'采样图像已保存: {save_path}')

    # 保存去噪过程
    if len(imgs) > 1:
        denoise_imgs = []
        for img in imgs:
            img = (img + 1) * 0.5
            img = img.clamp(0, 1)
            denoise_imgs.append(img[:8])  # 取前8张
        denoise_imgs = torch.cat(denoise_imgs, dim=0)
        grid_denoise = make_grid(denoise_imgs, nrow=8, padding=2)
        save_path_denoise = os.path.join(args.output_dir, 'samples', f'denoise_epoch_{epoch}.png')
        save_image(grid_denoise, save_path_denoise)


@torch.no_grad()
def generate(args):
    """从已训练模型生成图像"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet(in_channels=1, base_channels=args.base_channels).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    diffusion = GaussianDiffusion(timesteps=args.timesteps, device=device)

    os.makedirs(args.output_dir, exist_ok=True)

    # 生成图像
    shape = (args.num_samples, 1, args.image_size, args.image_size)
    imgs = diffusion.p_sample_loop(model, shape)

    final_imgs = imgs[-1]
    final_imgs = (final_imgs + 1) * 0.5
    final_imgs = final_imgs.clamp(0, 1)
    grid = make_grid(final_imgs, nrow=int(math.sqrt(args.num_samples)), padding=2)
    save_path = os.path.join(args.output_dir, 'generated.png')
    save_image(grid, save_path)
    print(f'生成图像已保存: {save_path}')

    # 保存去噪过程
    denoise_imgs = []
    for img in imgs:
        img = (img + 1) * 0.5
        img = img.clamp(0, 1)
        denoise_imgs.append(img[:8])
    denoise_imgs = torch.cat(denoise_imgs, dim=0)
    grid_denoise = make_grid(denoise_imgs, nrow=8, padding=2)
    save_path_denoise = os.path.join(args.output_dir, 'denoise_process.png')
    save_image(grid_denoise, save_path_denoise)
    print(f'去噪过程已保存: {save_path_denoise}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DDPM训练 - MNIST手写数字生成')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批大小')
    parser.add_argument('--lr', type=float, default=2e-4, help='学习率')
    parser.add_argument('--timesteps', type=int, default=1000, help='扩散步数')
    parser.add_argument('--image_size', type=int, default=32, help='图像大小')
    parser.add_argument('--base_channels', type=int, default=32, help='U-Net基础通道数')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据集目录')
    parser.add_argument('--output_dir', type=str, default='./output', help='输出目录')
    parser.add_argument('--sample_interval', type=int, default=5, help='采样间隔(轮)')
    parser.add_argument('--save_interval', type=int, default=10, help='保存间隔(轮)')
    args = parser.parse_args()
    train(args)
