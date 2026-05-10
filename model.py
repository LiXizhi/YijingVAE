"""Beta-VAE with 6 latent variables (the six yao of a Yijing hexagram).

Encoder/decoder are simple convolutional networks suitable for 28x28
grayscale (MNIST/FashionMNIST) and 32x32 / 64x64 RGB images.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Six latent variables, one per yao (line) of a hexagram.
LATENT_DIM = 6


class BetaVAE(nn.Module):
    def __init__(self, in_channels: int = 1, image_size: int = 28,
                 latent_dim: int = LATENT_DIM, hidden: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.hidden = hidden

        # Encoder: three stride-2 convs reduce HxW by factor 8.
        # We pad/resize inputs to a multiple of 8 before feeding the network.
        self.enc_size = self._round_up(image_size, 8)
        feat = self.enc_size // 8

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 4, 2, 1),  nn.ReLU(inplace=True),
            nn.Conv2d(hidden,    hidden * 2, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden * 2, hidden * 4, 4, 2, 1), nn.ReLU(inplace=True),
        )
        self.flat_dim = hidden * 4 * feat * feat
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, self.flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden * 4, hidden * 2, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden * 2, hidden,     4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden,     in_channels, 4, 2, 1),
        )
        self._feat = feat

    @staticmethod
    def _round_up(x: int, m: int) -> int:
        return ((x + m - 1) // m) * m

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.enc_size or x.shape[-2] != self.enc_size:
            x = F.interpolate(x, size=(self.enc_size, self.enc_size),
                              mode="bilinear", align_corners=False)
        return x

    def encode(self, x: torch.Tensor):
        x = self._prep(x)
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(-1, self.hidden * 4, self._feat, self._feat)
        x = self.decoder(h)
        x = torch.sigmoid(x)
        if x.shape[-1] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size),
                              mode="bilinear", align_corners=False)
        return x

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


def beta_vae_loss(x_hat: torch.Tensor, x: torch.Tensor,
                  mu: torch.Tensor, logvar: torch.Tensor,
                  beta: float = 4.0):
    """Returns (total, recon, kld) per-batch averaged."""
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum") / x.size(0)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon + beta * kld, recon, kld
