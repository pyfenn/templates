from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def reconstruction_bce_with_logits(x_hat_logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reconstruction loss for MNIST with Bernoulli likelihood. Returns sum over batch."""
    return F.binary_cross_entropy_with_logits(x_hat_logits, x, reduction="sum")


def kl_divergence_diag_gaussian(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x) || p(z)) for diagonal Gaussian. Returns sum over batch."""
    return -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - torch.exp(logvar))


@dataclass(frozen=True)
class ElboTerms:
    loss: torch.Tensor
    recon: torch.Tensor
    kl: torch.Tensor


def elbo_loss(
    x_hat_logits: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> ElboTerms:
    recon = reconstruction_bce_with_logits(x_hat_logits, x)
    kl = kl_divergence_diag_gaussian(mu, logvar)
    loss = recon + beta * kl
    return ElboTerms(loss=loss, recon=recon, kl=kl)



@dataclass(frozen=True)
class ElboTerms:
    loss: torch.Tensor
    recon: torch.Tensor
    kl: torch.Tensor


def elbo_loss(
    x_hat_logits: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> ElboTerms:
    recon = reconstruction_bce_with_logits(x_hat_logits, x)
    kl = kl_divergence_diag_gaussian(mu, logvar)
    loss = recon + beta * kl
    return ElboTerms(loss=loss, recon=recon, kl=kl)
