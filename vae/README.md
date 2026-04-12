# Variational Autoencoder (VAE) Template

## Overview

The Variational Autoencoder (VAE) template is designed for **unsupervised learning and generative modeling** tasks. It demonstrates how to:
- Build a VAE architecture with encoder and decoder networks
- Implement the ELBO (Evidence Lower Bound) loss with reconstruction and KL divergence terms
- Train using the FENN framework with validation
- Generate new samples from the learned latent space
- Reconstruct images from the learned representations

This template uses the MNIST dataset for digit generation and reconstruction, but can be adapted for other image datasets.

## What is a VAE?

A Variational Autoencoder (VAE) is a generative model that learns to encode data into a latent space and decode it back to reconstruct the original data. Unlike standard autoencoders, VAEs use a probabilistic approach that enables meaningful sampling and interpolation in the latent space.

**ELBO Objective:**
```
ELBO loss = reconstruction loss + β * KL divergence
```

- **Reconstruction term**: Encourages the decoder to accurately reconstruct the input from the latent representation
- **KL term**: Regularizes `q(z|x)` (learned posterior) towards the prior `p(z)=N(0, I)` (standard Gaussian)
- **β parameter** (default `1.0`): Controls the balance between reconstruction quality and latent space regularity
  - β = 1.0: Standard VAE (balanced)
  - β > 1.0: β-VAE (more regularization, disentangled representations)
  - β < 1.0: Less regularization, better reconstruction

## Expected Input Shape

MNIST images are tensors of shape `(B, 1, 28, 28)` where:
- `B` = batch size
- `1` = grayscale channel
- `28 × 28` = image dimensions (784 pixels total)

Internally, the template flattens them to `(B, 784)` for the MLP VAE architecture.

## Architecture

### VAE Model

A minimal MLP-based Variational Autoencoder consisting of an encoder and decoder:

```
Encoder Path:
Input (28×28=784)
    ↓
Linear(784 → 400) → ReLU
    ↓
Linear(400 → 20)  [mu - mean]
Linear(400 → 20)  [logvar - log variance]
    ↓
Reparameterization: z = mu + exp(0.5*logvar) * epsilon

Decoder Path:
z (latent vector, 20-dim)
    ↓
Linear(20 → 400) → ReLU
    ↓
Linear(400 → 784)  [reconstruction logits]
    ↓
Output (28×28=784, sigmoid for binary pixels)
```

**Architecture Details:**
- **Encoder**: Flattens 28×28 MNIST images to 784-dim vectors, compresses to 20-dim latent space
- **Latent Space**: 20-dimensional continuous representation with Gaussian prior
- **Decoder**: Reconstructs 784-dim vectors from latent codes, reshaped to 28×28 images
- **Reparameterization Trick**: Enables differentiable sampling from the learned distribution

### VAE Loss Function (ELBO)

The loss combines two terms:

```
ELBO = Reconstruction Loss + β * KL Divergence

Reconstruction Loss: Binary Cross-Entropy between reconstructed and original images
KL Divergence: Kullback-Leibler divergence between q(z|x) and p(z)
                where q(z|x) = N(mu, exp(logvar)) and p(z) = N(0, I)
Beta Parameter: Controls the balance between reconstruction quality and latent space regularity
```

**Loss Terms:**
- **Reconstruction Loss**: Penalizes differences between input and reconstructed images
- **KL Divergence**: Encourages the learned posterior to match the prior (standard Gaussian)
- **Beta**: Hyperparameter to weight KL term (β=1.0 for standard VAE, β>1 for β-VAE)

## Configuration

The template uses YAML configuration for hyperparameters:

### Example fenn.yaml

```yaml
project: vae-mnist

logger:
  dir: logger

general:
  device: cuda

model:
  hidden_dim: 400      # Hidden layer dimension
  z_dim: 20            # Latent space dimension

train:
  seed: 42
  batch: 128
  epochs: 30
  lr: 1e-3
  beta: 1.0            # KL divergence weight

test:
  batch: 128

dataset:
  dir: outputs/data
```

### Configuration Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `general.device` | str | Device to use (cpu, cuda) | cuda |
| `model.hidden_dim` | int | Hidden layer dimension | 400 |
| `model.z_dim` | int | Latent space dimension | 20 |
| `train.seed` | int | Random seed for reproducibility | 42 |
| `train.batch` | int | Batch size for training | 128 |
| `train.epochs` | int | Number of training epochs | 30 |
| `train.lr` | float | Learning rate for optimizer | 0.001 |
| `train.beta` | float | KL divergence weight (1.0=VAE, >1=β-VAE) | 1.0 |
| `test.batch` | int | Batch size for validation/test | 128 |
| `dataset.dir` | str | Directory for downloaded MNIST data | outputs/data |

## Key Features

### Generative Modeling
- Learns a continuous latent space representation of images
- Can generate new images by sampling from the learned distribution
- Reconstructs images from noisy inputs through the encoder-decoder architecture

### Variational Inference
- Uses the reparameterization trick for differentiable sampling
- Balances reconstruction quality vs. latent space regularity through ELBO loss
- Beta parameter controls the trade-off (β>1 for more disentangled representations)

### Unsupervised Learning
- No labels required - learns from pixel values alone
- Encoder learns meaningful feature representations
- Decoder learns to reconstruct from compressed latent codes

### Visualization & Analysis
- Saves reconstruction grids comparing original and reconstructed images
- Generates new samples by sampling from the prior
- Plots ELBO loss curves over training epochs
- Saves model checkpoints for later use

## Understanding VAE Components

### Encoder
Compresses 28×28 images (784 pixels) into a 20-dimensional latent vector:
- Input: 784-dim flattened image
- Hidden: 400-dim with ReLU
- Output: Two 20-dim vectors (mu and logvar for Gaussian parameters)

### Latent Space
20-dimensional continuous representation with:
- Gaussian prior: p(z) = N(0, I)
- Learned posterior: q(z|x) = N(mu, exp(logvar))
- Reparameterization: z = mu + exp(0.5*logvar) * epsilon

### Decoder
Reconstructs images from latent vectors:
- Input: 20-dim latent vector z
- Hidden: 400-dim with ReLU
- Output: 784-dim logits (sigmoid applied for binary pixel values)

## Customization

### Adjusting Latent Space Dimension

```python
# 10-dim latent space for more compression
model = VAE(hidden_dim=400, z_dim=10)

# 50-dim latent space for richer representation
model = VAE(hidden_dim=400, z_dim=50)
```

### Using β-VAE for Disentanglement

```yaml
train:
  beta: 4.0  # β > 1 encourages disentangled representations
```

Higher beta values force the latent space to be more disentangled but may reduce reconstruction quality.

### Changing Network Capacity

```python
# Smaller network
model = VAE(hidden_dim=128, z_dim=20)

# Larger network for better reconstruction
model = VAE(hidden_dim=800, z_dim=32)
```

## Loss Components

The ELBO loss combines:

**Reconstruction Loss (Binary Cross-Entropy):**
- Measures how well the decoder reconstructs the original image
- Lower values mean better reconstruction
- Formula: -E[log p(x|z)]

**KL Divergence:**
- Measures how much the learned posterior q(z|x) diverges from the prior p(z)
- Acts as a regularizer preventing the latent space from collapsing
- Formula: KL(q(z|x) || p(z))

**Total ELBO:**
```
ELBO = Reconstruction Loss + β * KL Divergence
```

## Performance Monitoring

The trainer automatically logs during training:
- Training ELBO loss per epoch
- Training reconstruction loss
- Training KL divergence
- Validation ELBO, reconstruction, and KL divergence
- Test metrics on final evaluation

Example output:
```
Epoch 01/30 | train loss=269.5824 (recon=232.1234, kl=37.4590) | 
             val loss=268.9234 (recon=231.5678, kl=37.3556)
Epoch 02/30 | train loss=265.3421 (recon=228.1234, kl=37.2187) | 
             val loss=264.7890 (recon=227.5678, kl=37.2212)
...
TEST | loss=263.4567 (recon=226.1234, kl=37.3333)
```

## Output Artifacts

The template generates three key outputs:

1. **recon_grid.png**: Side-by-side comparison of original and reconstructed images (16 samples)
2. **samples_grid.png**: New images generated by sampling from the prior (64 samples)
3. **loss_curve.png**: Training and validation ELBO curves
4. **vae_mnist.pt**: Model checkpoint with weights and configuration

## Expected Results

On MNIST:
- Training ELBO: ~260-280
- Validation ELBO: ~265-285
- Test ELBO: ~265-285

Results depend on:
- Latent space dimension (z_dim)
- Hidden layer size (hidden_dim)
- Number of training epochs
- Beta parameter value
- Learning rate

## File Structure

```
vae-mnist/
├── main.py                       # Training script
├── modules/
│   ├── __init__.py              # Module imports
│   ├── model.py                 # VAE architecture
│   └── losses.py                # ELBO loss computation
├── fenn.yaml                    # Configuration file
├── requirements.txt             # Python dependencies
└── vae.md                       # This documentation
```

## Dependencies

Required packages:
- PyTorch
- torchvision (for MNIST dataset and utilities)
- matplotlib (for plotting)
- tqdm (for progress bars)
- FENN framework

Install with:
```bash
pip install torch torchvision matplotlib tqdm
```

## Running the Template

```bash
# Using FENN CLI
python main.py

# With custom config
FENN_CONFIG=custom.yaml python main.py

# Check available devices
python -c "import torch; print(torch.cuda.is_available())"
```

## Extending the Template

### Using Different Datasets

```python
# Using Fashion-MNIST instead
from torchvision import datasets

train_full = datasets.FashionMNIST(
    root=args["dataset"]["dir"], 
    train=True, 
    download=True, 
    transform=transform
)
```

### Larger Images

For 32×32 images like CIFAR-10, modify the VAE:

```python
class VAE32(nn.Module):
    def __init__(self, hidden_dim: int = 400, z_dim: int = 20):
        super().__init__()
        self.x_dim = 32 * 32  # Changed from 28*28
        # Rest of architecture...
```

### Convolutional VAE

For better reconstruction of complex images:

```python
class ConvVAE(nn.Module):
    def __init__(self, hidden_dim: int = 400, z_dim: int = 20):
        # Use Conv2d layers instead of Linear
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        # Decoder with ConvTranspose2d
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Blurry reconstructions | Increase hidden_dim or reduce beta |
| KL divergence dominates | Reduce beta parameter (try 0.1-0.5) |
| CUDA out of memory | Reduce batch size in fenn.yaml |
| Poor sample quality | Train for more epochs or increase z_dim |
| Slow training | Enable GPU: set `device: cuda` |
| Posterior collapse | Use KL annealing: gradually increase beta during training |

## References

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - Original VAE paper
- [β-VAE: Learning Basic Visual Concepts](https://openreview.net/forum?id=Sy2fzU9gl) - β-VAE paper
- [PyTorch VAE Tutorial](https://github.com/pytorch/examples/tree/main/vae)
- [Understanding ELBO Loss](https://towardsdatascience.com/elbo-explained-with-mnist-data-b62b6fbf8f8f)
- FENN Framework Documentation
