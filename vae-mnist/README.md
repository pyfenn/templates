# VAE on MNIST (template)

Beginner-friendly, minimal Variational Autoencoder (VAE) example trained on MNIST.
Everything is self-contained inside this template folder (model, loss, training, sampling, plots). No fenn core changes.

## What is a VAE?

VAE optimizes the **ELBO** objective:

**ELBO loss = reconstruction loss + β * KL divergence**

- **Reconstruction** term encourages the decoder to reconstruct the input.
- **KL** term regularizes `q(z|x)` towards the prior `p(z)=N(0, I)`.
- **β** (default `1.0`) controls the KL term weight.

## Expected input shape

MNIST images are tensors of shape `(B, 1, 28, 28)`.
Internally, the template flattens them to `(B, 784)` for an MLP VAE.

## Hyperparameters

All hyperparameters are configured in `fenn.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `model.hidden_dim` | 400 | Width of encoder/decoder hidden layers |
| `model.z_dim` | 20 | Latent space dimensionality |
| `train.beta` | 1.0 | KL divergence weight |
| `train.lr` | 1e-3 | Learning rate (Adam) |
| `train.batch` | 128 | Training batch size |
| `train.epochs` | 30 | Number of training epochs |
| `train.seed` | 42 | Random seed for reproducibility |

## How to run

```bash
cd vae-mnist
pip install -r requirements.txt
python main.py
```

## Outputs

After training, the `outputs/` directory will contain:

- `recon_grid.png` — original vs reconstructed images
- `samples_grid.png` — images generated from random latent vectors
- `loss_curve.png` — train/val loss over epochs
- `vae_mnist.pt` — saved model checkpoint
