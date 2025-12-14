import torch.nn.functional as F
import torch.nn as nn
import torch


# https://github.com/IParraMartin/Sparse-Autoencoder/blob/main/sae.py
# https://github.com/AntonP999/Sparse_autoencoder/blob/master/Sparse_autoencoder.ipynb


class Encoder(nn.Module):
    def __init__(self, io_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(io_dim, latent_dim), nn.Sigmoid())

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, io_dim: int, latent_dim: int):
        super().__init__()
        self.decoder = nn.Sequential(nn.Linear(latent_dim, io_dim), nn.Tanh())

    def forward(self, x):
        return self.decoder(x)


class SparseAutoEncoder(nn.Module):
    def __init__(self, io_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = Encoder(io_dim, latent_dim)
        self.decoder = Decoder(io_dim, latent_dim)
        self.encoded_mean = None

    def forward(self, x):
        encoded = self.encoder(x)
        self.encoded_mean = torch.mean(encoded, dim=0)
        return self.decoder(encoded)

    def sparsity_loss(self):
        eps = 1e-8
        assert self.encoded_mean is not None
        rho_hat = torch.clamp(self.encoded_mean, min=eps, max=1 - eps)
        rho = 0.05
        kl_div = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log(
            (1 - rho) / (1 - rho_hat)
        )
        return torch.sum(kl_div)

    def loss(self, x, target, **kwargs):
        mse_loss = F.mse_loss(x, target)
        sparsity_loss = self.sparsity_loss()
        return mse_loss + sparsity_loss * 1e-4
