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

- `--hidden-dim` (default: 400)
- `--z-dim` (default: 20)
- `--beta` (default: 1.0)
- `--lr` (default: 1e-3)
- `--batch-size` (default: 128)
- `--epochs` (default: 10)
- `--seed` (default: 42)

## How to run

```bash
cd vae-mnist
pip install -r requirements.txt
python train.py --epochs 5
