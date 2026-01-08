import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import CHECKPOINT, LORA_OUTPUT_DIR, load_lora_model

# MNIST SAEs
# https://github.com/IParraMartin/Sparse-Autoencoder/blob/main/sae.py
# https://github.com/AntonP999/Sparse_autoencoder/blob/master/Sparse_autoencoder.ipynb
# LLM SAEs
# https://adamkarvonen.github.io/machine_learning/2024/06/11/sae-intuitions.html
# from what I can tell, this one doesn't use an activation function on the decoder
# https://www.goodfire.ai/blog/sae-open-source-announcement
#
# gated SAE
# https://arxiv.org/abs/2404.16014
#
# interpretability getting started thing
# https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J
# https://github.com/TransformerLensOrg/TransformerLens
# https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Main_Demo.ipynb#scrollTo=z_fpOjmtfdYx
# https://github.com/decoderesearch/SAELens/blob/main/tutorials/training_a_sparse_autoencoder.ipynb
#
# potential data to train an SAE
# * generate random characters
# * LMSYS Chat 1M but that requires signing something
# * openassitant oasst1 or oasst2
# * webtext + gpt2 output https://github.com/openai/gpt-2-output-dataset

# easiest thing is probably random characters


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


def train_sae(
    io_dim: int,
    latent_dim: int,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    n_epochs: int,
) -> nn.Module:
    model = SparseAutoEncoder(io_dim, latent_dim)
    opt = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    model.train()
    for epoch in tqdm(range(1, n_epochs + 1)):
        epoch_loss = 0
        for data in train_dataloader:
            opt.zero_grad()
            preds = model(data)
            loss = model.loss(preds, data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0, error_if_nonfinite=True
            )
            opt.step()
            epoch_loss += loss.item()
        if epoch % 20 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_data in val_dataloader:
                    preds = model(val_data)
                    loss = model.loss(preds, val_data)
                    val_loss += loss.item()
            model.train()

            print(
                f"Epoch {epoch} | Train Loss {epoch_loss:.6f} | Val Loss {val_loss:.6f}"
            )

        elif epoch == n_epochs:
            # do another eval at the end of training
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_data in val_dataloader:
                    preds = model(val_data)
                    loss = model.loss(preds, val_data)
                    val_loss += loss.item()
            model.train()

            print(
                f"Epoch {epoch} | Train Loss {epoch_loss:.6f} | Val Loss {val_loss:.6f}"
            )

    return model


if __name__ == "__main__":
    io_dim = 10
    latent_dim = 5

    data = DataLoader([torch.tensor([0.1] * io_dim)])
    model = train_sae(io_dim, latent_dim, data, data, 50)
