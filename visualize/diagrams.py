"""
生成论文所需的高质量原理图和架构图
用法: python -m visualize.diagrams
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os


def set_chinese_font():
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


def save_diffusion_process_diagram(save_dir):
    """DDPM前向/反向过程流程图"""
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.set_xlim(-1, 19)
    ax.set_ylim(-1, 8)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # 前向过程（上方）
    ax.text(9, 7.3, '前向扩散过程（逐步加噪）',
            fontsize=20, ha='center', fontweight='bold', color='#1565C0')

    positions = [1.5, 4.5, 7.5, 11, 14, 17]
    labels = ['$x_0$', '$x_1$', '$x_2$', '$x_{T-1}$', '$x_T$']
    colors_fwd = ['#2E7D32', '#558B2F', '#9E9D24', '#F57F17', '#E65100']

    for i in range(len(labels)):
        pos = positions[i]
        circle = plt.Circle((pos, 5.5), 0.65, facecolor=colors_fwd[i], alpha=0.85,
                            edgecolor='#424242', linewidth=2)
        ax.add_patch(circle)
        ax.text(pos, 5.5, labels[i], fontsize=18, ha='center', va='center',
                fontweight='bold', color='white')

    for i in range(len(labels) - 1):
        x1 = positions[i] + 0.75
        x2 = positions[i + 1] - 0.75
        ax.annotate('', xy=(x2, 5.5), xytext=(x1, 5.5),
                    arrowprops=dict(arrowstyle='->', color='#1565C0', lw=3))

    ax.text(9.25, 5.5, '$\\cdots$', fontsize=28, ha='center', va='center', color='#616161')

    ax.text(1.5, 6.5, '清晰图像', fontsize=16, ha='center', color='#2E7D32', fontweight='bold')
    ax.text(17, 6.5, '纯噪声', fontsize=16, ha='center', color='#E65100', fontweight='bold')
    ax.text(9.25, 4.5, r'加噪：$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon$',
            fontsize=14, ha='center', color='#1565C0',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=1.5))

    # 反向过程（下方）
    ax.text(9, 3.3, '反向去噪过程（逐步恢复）',
            fontsize=20, ha='center', fontweight='bold', color='#C62828')

    colors_bwd = list(reversed(colors_fwd))

    for i in range(len(labels)):
        pos = positions[i]
        circle = plt.Circle((pos, 1.5), 0.65, facecolor=colors_bwd[i], alpha=0.85,
                            edgecolor='#424242', linewidth=2)
        ax.add_patch(circle)
        ax.text(pos, 1.5, labels[i], fontsize=18, ha='center', va='center',
                fontweight='bold', color='white')

    for i in range(len(labels) - 1):
        x1 = positions[i + 1] - 0.75
        x2 = positions[i] + 0.75
        ax.annotate('', xy=(x2, 1.5), xytext=(x1, 1.5),
                    arrowprops=dict(arrowstyle='->', color='#C62828', lw=3))

    ax.text(9.25, 1.5, '$\\cdots$', fontsize=28, ha='center', va='center', color='#616161')

    ax.text(1.5, 0.3, '生成图像', fontsize=16, ha='center', color='#2E7D32', fontweight='bold')
    ax.text(17, 0.3, '纯噪声', fontsize=16, ha='center', color='#E65100', fontweight='bold')
    ax.text(9.25, -0.5, r'去噪：$\epsilon_\theta(x_t, t)$ 预测噪声，逐步恢复图像',
            fontsize=14, ha='center', color='#C62828',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='#C62828', linewidth=1.5))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'diffusion_process_diagram.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"扩散过程流程图已保存: {os.path.join(save_dir, 'diffusion_process_diagram.png')}")


def save_unet_architecture(save_dir):
    """U-Net网络结构图"""
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(-2, 20)
    ax.set_ylim(-1, 10.5)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    ax.text(9, 10, 'U-Net 噪声预测网络结构', fontsize=22, ha='center', fontweight='bold')

    encoder_blocks = [
        (3, 8.5, 'Conv In', '1→32', '#4CAF50', 2.4),
        (3, 7.0, 'Down 1', '32→32', '#43A047', 2.4),
        (3, 5.5, 'Down 2', '32→64', '#2E7D32', 2.4),
        (3, 4.0, 'Down 3', '64→128', '#1B5E20', 2.4),
    ]

    mid_block = (9, 4.0, 'Mid', '128→128', '#1565C0', 2.4)

    decoder_blocks = [
        (15, 4.0, 'Up 3', '128→64', '#C62828', 2.4),
        (15, 5.5, 'Up 2', '64→32', '#D32F2F', 2.4),
        (15, 7.0, 'Up 1', '32→32', '#E53935', 2.4),
        (15, 8.5, 'Conv Out', '32→1', '#EF9A9A', 2.4),
    ]

    def draw_block(x, y, title, detail, color, width, has_attn=False):
        box = FancyBboxPatch((x - width/2, y - 0.5), width, 1.0,
                             boxstyle="round,pad=0.15",
                             facecolor=color, alpha=0.85,
                             edgecolor='#424242', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y + 0.15, title, fontsize=16, ha='center', va='center',
                fontweight='bold', color='white')
        ax.text(x, y - 0.2, detail, fontsize=13, ha='center', va='center',
                color='white', alpha=0.95)
        if has_attn:
            ax.text(x + width/2 + 0.3, y, 'Attn', fontsize=11, ha='left', va='center',
                    color='#FF6F00', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='#FFF3E0', edgecolor='#FF6F00', linewidth=1))

    attn_enc = [False, False, True, True]
    for i, (x, y, title, detail, color, width) in enumerate(encoder_blocks):
        draw_block(x, y, title, detail, color, width, has_attn=attn_enc[i])

    for i in range(len(encoder_blocks) - 1):
        ax.annotate('', xy=(3, encoder_blocks[i+1][1] + 0.55),
                    xytext=(3, encoder_blocks[i][1] - 0.55),
                    arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=2.5))

    x, y, title, detail, color, width = mid_block
    draw_block(x, y, title, detail, color, width, has_attn=True)

    ax.annotate('', xy=(9 - 1.3, 4.0), xytext=(3 + 1.3, 4.0),
                arrowprops=dict(arrowstyle='->', color='#37474F', lw=3))

    ax.annotate('', xy=(15 - 1.3, 4.0), xytext=(9 + 1.3, 4.0),
                arrowprops=dict(arrowstyle='->', color='#37474F', lw=3))

    attn_dec = [True, True, False, False]
    for i, (x, y, title, detail, color, width) in enumerate(decoder_blocks):
        draw_block(x, y, title, detail, color, width, has_attn=attn_dec[i])

    for i in range(len(decoder_blocks) - 1):
        ax.annotate('', xy=(15, decoder_blocks[i+1][1] - 0.55),
                    xytext=(15, decoder_blocks[i][1] + 0.55),
                    arrowprops=dict(arrowstyle='->', color='#C62828', lw=2.5))

    skip_pairs = [(1, 2), (2, 1), (3, 0)]
    for ei, di in skip_pairs:
        ey = encoder_blocks[ei][1]
        dy = decoder_blocks[di][1]
        ax.annotate('', xy=(15 - 1.3, dy), xytext=(3 + 1.3, ey),
                    arrowprops=dict(arrowstyle='->', color='#7B1FA2', lw=2.5,
                                   linestyle='dashed',
                                   connectionstyle='arc3,rad=-0.25'))

    ax.text(9, 6.5, 'Skip Connection', fontsize=16, ha='center',
            color='#7B1FA2', fontweight='bold', style='italic')

    time_box = FancyBboxPatch((7, 9.0), 4, 0.8,
                              boxstyle="round,pad=0.15",
                              facecolor='#FF8F00', alpha=0.9,
                              edgecolor='#424242', linewidth=2)
    ax.add_patch(time_box)
    ax.text(9, 9.4, 'Time Embedding', fontsize=16, ha='center', va='center',
            fontweight='bold', color='white')

    for block_list, x_pos in [(encoder_blocks, 3), (decoder_blocks, 15)]:
        for _, y, _, _, _, _ in block_list:
            ax.annotate('', xy=(x_pos, y + 0.55), xytext=(9, 9.0),
                        arrowprops=dict(arrowstyle='->', color='#FF8F00', lw=1.5,
                                       linestyle='dotted'))
    ax.annotate('', xy=(9, 4.55), xytext=(9, 9.0),
                arrowprops=dict(arrowstyle='->', color='#FF8F00', lw=1.5,
                               linestyle='dotted'))

    ax.text(3, 9.5, '输入: $x_t$', fontsize=18, ha='center',
            fontweight='bold', color='#1565C0')
    ax.text(15, 9.5, '输出: $\\epsilon_\\theta$', fontsize=18, ha='center',
            fontweight='bold', color='#C62828')

    legend_elements = [
        mpatches.Patch(facecolor='#4CAF50', alpha=0.85, edgecolor='#424242', linewidth=1.5, label='编码器'),
        mpatches.Patch(facecolor='#1565C0', alpha=0.85, edgecolor='#424242', linewidth=1.5, label='中间层'),
        mpatches.Patch(facecolor='#C62828', alpha=0.85, edgecolor='#424242', linewidth=1.5, label='解码器'),
        mpatches.Patch(facecolor='#FF8F00', alpha=0.9, edgecolor='#424242', linewidth=1.5, label='时间编码'),
        mpatches.Patch(facecolor='#7B1FA2', alpha=0.5, edgecolor='#424242', linewidth=1.5, label='跳跃连接'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=14,
              frameon=True, fancybox=True, shadow=True,
              edgecolor='#BDBDBD', facecolor='#FAFAFA')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'unet_architecture.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"U-Net架构图已保存: {os.path.join(save_dir, 'unet_architecture.png')}")


def save_training_pipeline(save_dir):
    """训练流程图"""
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.set_xlim(-1, 19)
    ax.set_ylim(-0.5, 6)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    ax.text(9, 5.5, 'DDPM 训练流程', fontsize=22, ha='center', fontweight='bold')

    steps = [
        (2, 3.0, '采样数据', '$x_0, t, \\epsilon$', '#4CAF50', 2.8),
        (6, 3.0, '前向加噪', '$x_t = \\sqrt{\\bar{\\alpha}_t}x_0$\n$+ \\sqrt{1-\\bar{\\alpha}_t}\\epsilon$', '#FF9800', 3.2),
        (10, 3.0, 'U-Net预测', '$\\hat{\\epsilon} = \\epsilon_\\theta(x_t, t)$', '#7B1FA2', 2.8),
        (14, 3.0, '计算损失', '$L = ||\\epsilon - \\hat{\\epsilon}||^2$', '#C62828', 2.8),
        (17.5, 3.0, '更新参数', '梯度下降', '#455A64', 2.2),
    ]

    for i, (x, y, title, detail, color, width) in enumerate(steps):
        shadow = FancyBboxPatch((x - width/2 + 0.06, y - 0.65 - 0.06), width, 1.3,
                                boxstyle="round,pad=0.15",
                                facecolor='#9E9E9E', alpha=0.15)
        ax.add_patch(shadow)
        box = FancyBboxPatch((x - width/2, y - 0.65), width, 1.3,
                             boxstyle="round,pad=0.15",
                             facecolor=color, alpha=0.88,
                             edgecolor='#424242', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y + 0.25, title, fontsize=16, ha='center', va='center',
                fontweight='bold', color='white')
        ax.text(x, y - 0.2, detail, fontsize=12, ha='center', va='center',
                color='white', alpha=0.95)

    for i in range(len(steps) - 1):
        x1 = steps[i][0] + steps[i][5]/2 + 0.1
        x2 = steps[i+1][0] - steps[i+1][5]/2 - 0.1
        ax.annotate('', xy=(x2, 3.0), xytext=(x1, 3.0),
                    arrowprops=dict(arrowstyle='->', color='#37474F', lw=3))

    for i, (x, y, _, _, color, width) in enumerate(steps):
        circle = plt.Circle((x - width/2 + 0.25, y + 0.65 - 0.15), 0.22,
                            facecolor='white', edgecolor=color, linewidth=2)
        ax.add_patch(circle)
        ax.text(x - width/2 + 0.25, y + 0.65 - 0.15, str(i+1), fontsize=13, ha='center', va='center',
                fontweight='bold', color=color)

    ax.annotate('', xy=(2, 3.0 - 0.7), xytext=(17.5, 3.0 - 0.7),
                arrowprops=dict(arrowstyle='->', color='#455A64', lw=3,
                               linestyle='dashed',
                               connectionstyle='arc3,rad=0.25'))
    ax.text(9.75, 1.2, '重复训练直至收敛', fontsize=16, ha='center',
            color='#455A64', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ECEFF1', edgecolor='#455A64', linewidth=1.5))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_pipeline.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"训练流程图已保存: {os.path.join(save_dir, 'training_pipeline.png')}")


def main():
    set_chinese_font()
    save_dir = 'output/visuals'
    os.makedirs(save_dir, exist_ok=True)

    print("=== 生成高质量原理图和架构图 ===\n")

    print("[1/3] 生成扩散过程流程图...")
    save_diffusion_process_diagram(save_dir)

    print("[2/3] 生成U-Net架构图...")
    save_unet_architecture(save_dir)

    print("[3/3] 生成训练流程图...")
    save_training_pipeline(save_dir)

    print("\n=== 所有高质量原理图生成完毕！===")
    print(f"保存目录: {save_dir}")


if __name__ == '__main__':
    main()
