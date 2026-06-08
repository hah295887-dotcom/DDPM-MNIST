"""
DDPM生成脚本
用法: python -m train.generate --checkpoint output/checkpoint_epoch_50.pt
"""

import argparse
from train.train import generate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DDPM生成 - 从已训练模型生成图像')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--timesteps', type=int, default=1000, help='扩散步数')
    parser.add_argument('--image_size', type=int, default=32, help='图像大小')
    parser.add_argument('--base_channels', type=int, default=32, help='U-Net基础通道数')
    parser.add_argument('--num_samples', type=int, default=64, help='生成图像数量')
    parser.add_argument('--output_dir', type=str, default='./output', help='输出目录')
    args = parser.parse_args()
    generate(args)
