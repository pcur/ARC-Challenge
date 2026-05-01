"""
vae.py

Train BOTH:
1. Deep CNN Variational Autoencoder
2. Transformer Self-Attention Variational Autoencoder

On ARC-AGI grids.

Outputs saved separately for each model.
"""

import csv
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from util.data_utils import load_arc_tasks_batch


# ============================================================
# CONFIG
# ============================================================

@dataclass
class VAEConfig:
    data_dir: str = "data/training"
    output_dir: str = "results/vae"

    max_grid_size: int = 30
    num_colors: int = 10

    latent_dim: int = 256

    hidden_channels: tuple = (64, 128, 256, 256, 512)

    transformer_dim: int = 256
    transformer_heads: int = 8
    transformer_layers: int = 4

    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 3e-8

    cnn_epochs: int = 125
    transformer_epochs: int = 30

    beta_target: float = 0.2
    kl_warmup_epochs: int = 30

    train_on_outputs: bool = False

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0
    

# ============================================================
# DATASET
# ============================================================

def pad_grid(grid, max_size):
    h = len(grid)
    w = len(grid[0])

    padded = np.zeros((max_size, max_size), dtype=np.int64)
    padded[:h, :w] = np.array(grid)

    mask = np.zeros((max_size, max_size), dtype=np.float32)
    mask[:h, :w] = 1.0

    return padded, mask


def one_hot_grid(grid, num_colors):
    return F.one_hot(
        torch.tensor(grid, dtype=torch.long),
        num_classes=num_colors
    ).permute(2, 0, 1).float()


class ARCVAEDataset(Dataset):
    def __init__(self, tasks, max_grid_size=30, num_colors=10, use_outputs=False):
        self.samples = []

        for task in tasks.values():
            for pair in task.train_pairs:
                grid = pair["output"] if use_outputs else pair["input"]

                padded, mask = pad_grid(grid, max_grid_size)
                self.samples.append((padded, mask))

        self.num_colors = num_colors

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        grid, mask = self.samples[idx]

        return one_hot_grid(grid, self.num_colors), torch.tensor(mask)


# ============================================================
# CNN VAE
# ============================================================

class CNNEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        C = config.num_colors
        H = config.hidden_channels

        self.conv = nn.Sequential(
            nn.Conv2d(C, H[0], 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(H[0], H[1], 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv2d(H[1], H[2], 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(H[2], H[3], 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(H[3], H[4], 3, stride=2, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(
                1, C,
                config.max_grid_size,
                config.max_grid_size
            )
            out = self.conv(dummy)
            self.enc_shape = out.shape[1:]
            flat_dim = out.numel()

        self.fc_mu = nn.Linear(flat_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, config.latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.flatten(start_dim=1)

        return self.fc_mu(h), self.fc_logvar(h)


class CNNDecoder(nn.Module):
    def __init__(self, config, enc_shape):
        super().__init__()

        H = config.hidden_channels
        flat_dim = int(np.prod(enc_shape))

        self.enc_shape = enc_shape
        self.output_size = config.max_grid_size

        self.fc = nn.Linear(config.latent_dim, flat_dim)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(H[4], H[3], 4, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(H[3], H[2], 4, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(H[2], H[1], 4, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(H[1], H[0], 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(H[0], config.num_colors, 1),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, *self.enc_shape)

        x = self.deconv(h)

        return x[:, :, :self.output_size, :self.output_size]


# ============================================================
# TRANSFORMER VAE
# ============================================================

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.grid_size = config.max_grid_size
        self.num_tokens = self.grid_size * self.grid_size

        self.input_proj = nn.Linear(
            config.num_colors,
            config.transformer_dim
        )

        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_tokens, config.transformer_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.transformer_dim,
            nhead=config.transformer_heads,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer_layers,
        )

        flat_dim = self.num_tokens * config.transformer_dim

        self.fc_mu = nn.Linear(flat_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, config.latent_dim)

    def forward(self, x):
        B = x.size(0)

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(B, self.num_tokens, -1)

        x = self.input_proj(x)
        x = x + self.pos_embed

        x = self.transformer(x)
        x = x.flatten(start_dim=1)

        return self.fc_mu(x), self.fc_logvar(x)


class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_colors = config.num_colors
        self.grid_size = config.max_grid_size

        self.fc = nn.Sequential(
            nn.Linear(config.latent_dim, 1024),
            nn.ReLU(),

            nn.Linear(
                1024,
                self.num_colors * self.grid_size * self.grid_size
            )
        )

    def forward(self, z):
        x = self.fc(z)

        return x.view(
            -1,
            self.num_colors,
            self.grid_size,
            self.grid_size
        )


# ============================================================
# GENERIC VAE WRAPPER
# ============================================================

class ARCVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)

        z = self.reparameterize(mu, logvar)

        recon = self.decoder(z)

        return recon, mu, logvar


# ============================================================
# LOSS
# ============================================================

def vae_loss(recon_logits, target, mask, mu, logvar, beta):
    target_labels = target.argmax(dim=1)

    ce = F.cross_entropy(
        recon_logits,
        target_labels,
        reduction='none'
    )

    recon_loss = (ce * mask).sum() / mask.sum()

    kl = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    total = recon_loss + beta * kl

    return total, recon_loss, kl


def reconstruction_accuracy(recon_logits, target, mask):
    preds = recon_logits.argmax(dim=1)
    labels = target.argmax(dim=1)

    correct = ((preds == labels) * mask.bool()).sum().float()
    total = mask.sum()

    return correct / total


# ============================================================
# TRAINING
# ============================================================

def train_single_model(model, loader, config, num_epochs, model_name):
    outdir = Path(config.output_dir) / model_name
    outdir.mkdir(parents=True, exist_ok=True)

    model = model.to(config.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
    )

    history = {
        "epoch": [],
        "recon": [],
        "kl": [],
        "acc": [],
    }

    for epoch in range(num_epochs):
        model.train()

        beta = min(
            1.0,
            epoch / config.kl_warmup_epochs
        ) * config.beta_target

        total_recon = 0
        total_kl = 0
        total_acc = 0

        for batch, mask in loader:
            batch = batch.to(config.device)
            mask = mask.to(config.device)

            optimizer.zero_grad()

            recon, mu, logvar = model(batch)

            loss, recon_loss, kl = vae_loss(
                recon,
                batch,
                mask,
                mu,
                logvar,
                beta
            )

            acc = reconstruction_accuracy(
                recon,
                batch,
                mask
            )

            loss.backward()
            optimizer.step()

            total_recon += recon_loss.item()
            total_kl += kl.item()
            total_acc += acc.item()

        n = len(loader)

        avg_recon = total_recon / n
        avg_kl = total_kl / n
        avg_acc = total_acc / n

        scheduler.step(avg_recon)

        history["epoch"].append(epoch + 1)
        history["recon"].append(avg_recon)
        history["kl"].append(avg_kl)
        history["acc"].append(avg_acc)

        print(
            f"[{model_name}] "
            f"Epoch {epoch+1:03d}/{num_epochs} | "
            f"CE={avg_recon:.4f} | "
            f"KL={avg_kl:.4f} | "
            f"Acc={avg_acc:.4f}"
        )

    save_outputs(model, history, config, outdir)


# ============================================================
# SAVE / PLOT
# ============================================================

def save_outputs(model, history, config, outdir):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
        },
        outdir / "vae_model.pt",
    )

    with open(outdir / "training_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(history.keys())

        for row in zip(*history.values()):
            writer.writerow(row)

    plt.figure(figsize=(8, 5))

    plt.plot(history["epoch"], history["recon"], label="CE Loss")
    plt.plot(history["epoch"], history["kl"], label="KL")
    plt.plot(history["epoch"], history["acc"], label="Exact Match Accuracy")

    plt.legend()
    plt.grid()
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.tight_layout()

    plt.savefig(outdir / "training_plot.png", dpi=200)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    config = VAEConfig()
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version built with:", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())
    print(f"Using device: {config.device}")

    if config.device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    tasks = load_arc_tasks_batch(config.data_dir)

    if len(tasks) == 0:
        raise RuntimeError(f"No tasks found in {config.data_dir}")

    dataset = ARCVAEDataset(
        tasks,
        max_grid_size=config.max_grid_size,
        num_colors=config.num_colors,
        use_outputs=config.train_on_outputs,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    print("===================================")
    print("Training CNN VAE")
    print("===================================")

    cnn_encoder = CNNEncoder(config)
    cnn_decoder = CNNDecoder(config, cnn_encoder.enc_shape)

    cnn_vae = ARCVAE(cnn_encoder, cnn_decoder)

    train_single_model(
        cnn_vae,
        loader,
        config,
        config.cnn_epochs,
        "cnn_vae",
    )

    print("\n===================================")
    print("Training Transformer VAE")
    print("===================================")

    transformer_vae = ARCVAE(
        TransformerEncoder(config),
        TransformerDecoder(config)
    )

    train_single_model(
        transformer_vae,
        loader,
        config,
        config.transformer_epochs,
        "transformer_vae",
    )


if __name__ == "__main__":
    main()