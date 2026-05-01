"""
arcutils.py

Utilities for exploring trained ARC VAE latent spaces.

Supports:
- Load CNN / Transformer VAE checkpoints
- Encode ARC grid -> latent vector
- Decode latent vector -> ARC grid
- Visualize reconstruction quality
- Interpolate between latent vectors
- Shift latent dimensions interactively
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from vae import (
    VAEConfig,
    CNNEncoder,
    CNNDecoder,
    TransformerEncoder,
    TransformerDecoder,
    ARCVAE,
    pad_grid,
    one_hot_grid,
)


# ============================================================
# MODEL LOADING
# ============================================================

def load_vae_checkpoint(checkpoint_path, model_type, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = VAEConfig(**checkpoint["config"])

    if model_type == "cnn":
        encoder = CNNEncoder(config)
        decoder = CNNDecoder(config, encoder.enc_shape)

    elif model_type == "transformer":
        encoder = TransformerEncoder(config)
        decoder = TransformerDecoder(config)

    else:
        raise ValueError("model_type must be 'cnn' or 'transformer'")

    model = ARCVAE(encoder, decoder)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, config


# ============================================================
# ENCODE / DECODE
# ============================================================

def encode_grid(model, grid, config, device=None):
    device = device or next(model.parameters()).device

    padded, mask = pad_grid(grid, config.max_grid_size)

    x = one_hot_grid(padded, config.num_colors).unsqueeze(0).to(device)

    with torch.no_grad():
        mu, logvar = model.encoder(x)

    return mu.squeeze(0).cpu().numpy()


def decode_latent(model, latent_vector, config, device=None):
    device = device or next(model.parameters()).device

    z = torch.tensor(
        latent_vector,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    with torch.no_grad():
        logits = model.decoder(z)

    grid = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    return grid


# ============================================================
# VISUALIZATION
# ============================================================

ARC_CMAP = plt.cm.get_cmap("tab10", 10)


def plot_arc_grid(grid, title=None):
    plt.imshow(grid, cmap=ARC_CMAP, vmin=0, vmax=9)
    plt.xticks([])
    plt.yticks([])

    if title:
        plt.title(title)


def compare_reconstruction(model, grid, config):
    latent = encode_grid(model, grid, config)
    recon = decode_latent(model, latent, config)

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plot_arc_grid(np.array(grid), "Original")

    plt.subplot(1, 2, 2)
    plot_arc_grid(recon, "Reconstruction")

    plt.tight_layout()
    plt.show()

    exact = np.mean(
        recon[:len(grid), :len(grid[0])] == np.array(grid)
    )

    print(f"Reconstruction Accuracy: {exact:.4f}")


# ============================================================
# LATENT INTERPOLATION
# ============================================================

def interpolate_latents(model, z1, z2, config, steps=8):
    zs = []

    for alpha in np.linspace(0, 1, steps):
        z = (1 - alpha) * z1 + alpha * z2
        zs.append(z)

    plt.figure(figsize=(2 * steps, 2))

    for i, z in enumerate(zs):
        grid = decode_latent(model, z, config)

        plt.subplot(1, steps, i + 1)
        plot_arc_grid(grid, f"{i/(steps-1):.2f}")

    plt.tight_layout()
    plt.show()


# ============================================================
# LATENT SHIFTING
# ============================================================

def shift_latent_dimension(
    model,
    base_latent,
    dim,
    config,
    shift_range=(-3, 3),
    steps=7
):
    values = np.linspace(
        shift_range[0],
        shift_range[1],
        steps
    )

    plt.figure(figsize=(2 * steps, 2))

    for i, delta in enumerate(values):
        z = base_latent.copy()
        z[dim] += delta

        grid = decode_latent(model, z, config)

        plt.subplot(1, steps, i + 1)
        plot_arc_grid(grid, f"{delta:.1f}")

    plt.tight_layout()
    plt.show()


# ============================================================
# RANDOM LATENT SAMPLING
# ============================================================

def sample_random_latents(model, config, n=8):
    plt.figure(figsize=(2 * n, 2))

    for i in range(n):
        z = np.random.randn(config.latent_dim)

        grid = decode_latent(model, z, config)

        plt.subplot(1, n, i + 1)
        plot_arc_grid(grid)

    plt.tight_layout()
    plt.show()


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    MODEL_PATH = "results/vae/cnn_vae/vae_model.pt"

    model, config = load_vae_checkpoint(
        MODEL_PATH,
        model_type="cnn"
    )

    sample_random_latents(model, config)