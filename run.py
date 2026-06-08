"""
DDPM-MNIST: 基于去噪扩散概率模型的手写数字生成
入口脚本，兼容旧版命令行用法
"""

import argparse
import sys
import os

# 确保项目根目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import UNet, GaussianDiffusion
from train.train import train, generate


def main():
    parser = argparse.ArgumentParser(description='DDPM复现 - MNIST手写数字生成')
    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # 训练参数
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    train_parser.add_argument('--batch_size', type=int, default=64, help='批大小')
    train_parser.add_argument('--lr', type=float, default=2e-4, help='学习率')
    train_parser.add_argument('--timesteps', type=int, default=1000, help='扩散步数')
    train_parser.add_argument('--image_size', type=int, default=32, help='图像大小')
    train_parser.add_argument('--base_channels', type=int, default=32, help='U-Net基础通道数')
    train_parser.add_argument('--data_dir', type=str, default='./data', help='数据集目录')
    train_parser.add_argument('--output_dir', type=str, default='./output', help='输出目录')
    train_parser.add_argument('--sample_interval', type=int, default=5, help='采样间隔(轮)')
    train_parser.add_argument('--save_interval', type=int, default=10, help='保存间隔(轮)')

    # 生成参数
    gen_parser = subparsers.add_parser('generate', help='从模型生成图像')
    gen_parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    gen_parser.add_argument('--timesteps', type=int, default=1000, help='扩散步数')
    gen_parser.add_argument('--image_size', type=int, default=32, help='图像大小')
    gen_parser.add_argument('--base_channels', type=int, default=32, help='U-Net基础通道数')
    gen_parser.add_argument('--num_samples', type=int, default=64, help='生成图像数量')
    gen_parser.add_argument('--output_dir', type=str, default='./output', help='输出目录')

    args = parser.parse_args()

    if args.command == 'train':
        train(args)
    elif args.command == 'generate':
        generate(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
