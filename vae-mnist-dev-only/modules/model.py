from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class VAE(nn.Module):
    """
    Minimal MLP VAE for MNIST.

    Expected input shape: (B, 1, 28, 28)
    Internally flattens to (B, 784).
    """

    def __init__(self, hidden_dim: int = 400, z_dim: int = 20) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.x_dim = 28 * 28

        # Encoder: x -> h -> (mu, logvar)
        self.encoder = nn.Sequential(
            nn.Linear(self.x_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

        # Decoder: z -> h -> x_hat_logits
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.x_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.view(x.size(0), -1)  # (B, 784)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x_hat_logits = self.decoder(z)  # (B, 784)
        return x_hat_logits.view(-1, 1, 28, 28)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat_logits = self.decode(z)
        return x_hat_logits, mu, logvar, z
