# DDPM-MNIST: Denoising Diffusion Probabilistic Models on MNIST

A PyTorch implementation of **Denoising Diffusion Probabilistic Models** (Ho et al., NeurIPS 2020) for handwritten digit generation on the MNIST dataset.

## Project Structure

```
DDPM-MNIST/
├── run.py                      # Entry point (backward compatible CLI)
├── README.md
│
├── model/
│   ├── __init__.py
│   ├── unet.py                 # U-Net noise prediction network
│   └── diffusion.py            # Gaussian diffusion process (forward & reverse)
│
├── train/
│   ├── __init__.py
│   ├── train.py                # Training & sampling functions
│   └── generate.py             # Generation script (from checkpoint)
│
├── visualize/
│   ├── __init__.py
│   ├── visuals.py              # Core visualizations (denoising, grid, schedules)
│   ├── visuals_extra.py        # Supplementary visualizations (loss, distribution, errors)
│   └── diagrams.py             # High-quality diagrams (process, architecture, pipeline)
│
├── data/                       # MNIST dataset (auto-downloaded)
└── output/                     # Training outputs
    ├── checkpoint_epoch_*.pt   # Model checkpoints
    └── samples/                # Generated samples per epoch
```

## Installation

**Prerequisites:** Python 3.8+, PyTorch 1.12+

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/DDPM-MNIST.git
cd DDPM-MNIST

# 2. Install PyTorch (GPU version recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 3. Install other dependencies
pip install tqdm matplotlib
```

## Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| Python | >=3.8 | Runtime |
| PyTorch | >=1.12 | Core framework |
| torchvision | >=0.13 | Dataset & image utilities |
| tqdm | >=4.60 | Progress bar |
| matplotlib | >=3.5 | Visualization |

## Training

Run from the project root:

```bash
python run.py train --epochs 50
```

Or using the module interface:

```bash
python -m train.train --epochs 50
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--batch_size` | 64 | Batch size |
| `--lr` | 2e-4 | Learning rate |
| `--timesteps` | 1000 | Number of diffusion steps T |
| `--image_size` | 32 | Image resolution |
| `--base_channels` | 32 | U-Net base channel count |
| `--data_dir` | ./data | Dataset directory |
| `--output_dir` | ./output | Output directory |
| `--sample_interval` | 5 | Save samples every N epochs |
| `--save_interval` | 10 | Save checkpoint every N epochs |

Training uses:
- **Adam** optimizer (lr=2e-4)
- **CosineAnnealing** learning rate scheduler
- **MSE** loss between predicted and true noise

## Generation

Generate images from a trained checkpoint:

```bash
python run.py generate --checkpoint output/checkpoint_epoch_50.pt
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | *(required)* | Path to model checkpoint |
| `--num_samples` | 64 | Number of images to generate |
| `--timesteps` | 1000 | Number of diffusion steps |
| `--image_size` | 32 | Image resolution |
| `--output_dir` | ./output | Output directory |

## Visualization

### Core visualizations

```bash
python -m visualize.visuals --checkpoint output/checkpoint_epoch_50.pt
```

Generates: denoising process, sample grid, noise schedule, forward process, training progress.

### Supplementary visualizations

```bash
python -m visualize.visuals_extra
```

Generates: MNIST distribution, loss curve, error case analysis, preprocessing comparison.

### Diagrams (for paper)

```bash
python -m visualize.diagrams
```

Generates: diffusion process diagram, U-Net architecture, training pipeline.

All outputs are saved to `output/visuals/`.

## Model Architecture

The U-Net noise prediction network consists of:

1. **Sinusoidal Position Embeddings** — Encodes timestep t into a vector
2. **Encoder** — 3 downsampling blocks with ResBlock + optional Self-Attention
3. **Middle Block** — ResBlock + Self-Attention + ResBlock
4. **Decoder** — 3 upsampling blocks with skip connections from encoder
5. **Output Layer** — GroupNorm + SiLU + Conv2d

Key design choices:
- **ResBlock**: GroupNorm → SiLU → Conv → Time MLP injection → GroupNorm → SiLU → Conv + residual
- **AttentionBlock**: Applied at 32×32 and 16×16 resolutions
- **Skip Connections**: Concatenate encoder features to decoder (U-Net style)

## Experimental Results

After 50 epochs of training on MNIST:

| Metric | Value |
|--------|-------|
| Initial Loss | ~0.45 |
| Final Loss | ~0.03 |
| Training Time | ~30-60 min (RTX 4050) |
| Image Resolution | 32×32 |
| Diffusion Steps | 1000 |

## References

- Ho J, Jain A, Abbeel P. Denoising Diffusion Probabilistic Models[C]. NeurIPS, 2020.
- Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation[C]. MICCAI, 2015.
- LeCun Y, et al. Gradient-based Learning Applied to Document Recognition[J]. Proceedings of the IEEE, 1998.

## License

This project is released under the MIT License.
