"""
arcutils.py

Utilities for exploring trained ARC VAE latent spaces.

Supports:
- Load CNN / Transformer VAE checkpoints
- Encode ARC grid -> latent vector
- Decode latent vector -> ARC grid
- Visualize reconstruction quality
- Interpolate between latent vectors
- Shift latent dimensions
- Interactive latent exploration with Tkinter sliders
"""

import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
import torch
import tkinter as tk
from tkinter import Scale, HORIZONTAL
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
        mu, _ = model.encoder(x)

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

ARC_CMAP = plt.get_cmap("tab10", 10)


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
# INTERACTIVE LATENT EXPLORER (TKINTER)
# ============================================================

def interactive_latent_explorer(
    cnn_model,
    transformer_model,
    base_latent,
    config,
    dim=0,
    shift_range=(-5, 5)
):
    import json
    import tkinter as tk
    from tkinter import Scale, HORIZONTAL, filedialog
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    base_latent = np.array(base_latent).copy()

    root = tk.Tk()
    root.title("ARC Latent Explorer (Dual View)")

    current_model_name = tk.StringVar(value="cnn")

    def get_active_model():
        return cnn_model if current_model_name.get() == "cnn" else transformer_model

    # --------------------------
    # FIGURE: TWO PANELS
    # --------------------------
    fig = Figure(figsize=(8, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    original_grid = np.zeros((config.max_grid_size, config.max_grid_size))
    recon_grid = decode_latent(get_active_model(), base_latent, config)

    img1 = ax1.imshow(original_grid, cmap=ARC_CMAP, vmin=0, vmax=9)
    img2 = ax2.imshow(recon_grid, cmap=ARC_CMAP, vmin=0, vmax=9)

    ax1.set_title("Original")
    ax2.set_title("Reconstruction")

    for ax in (ax1, ax2):
        ax.set_xticks([])
        ax.set_yticks([])

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()

    current_dim = tk.IntVar(value=dim)
    loaded_grid = None

    # --------------------------
    # UPDATE FUNCTION
    # --------------------------
    def update_plot():
        model = get_active_model()

        z = base_latent.copy()
        z[current_dim.get()] += shift_slider.get()

        recon = decode_latent(model, z, config)

        img2.set_data(recon)

        title = (
            f"{current_model_name.get().upper()} | "
            f"Dim {current_dim.get()} | "
            f"Shift {shift_slider.get():.2f}"
        )
        ax2.set_title(f"Reconstruction\n{title}")

        canvas.draw()

    # --------------------------
    # LOAD JSON + DISPLAY ORIGINAL
    # --------------------------
    def load_arc_json():
        nonlocal loaded_grid, base_latent

        file_path = filedialog.askopenfilename(
            initialdir="data/evaluation",
            filetypes=[("JSON files", "*.json")]
        )

        if not file_path:
            return

        with open(file_path, "r") as f:
            data = json.load(f)

        try:
            loaded_grid = data["train"][0]["input"]
        except Exception as e:
            print("Invalid ARC format:", e)
            return

        model = get_active_model()

        base_latent = encode_grid(model, loaded_grid, config)

        # update left panel
        img1.set_data(np.array(loaded_grid))

        ax1.set_title("Original (Loaded ARC)")

        print(f"[INFO] Loaded: {file_path}")

        update_plot()

    # --------------------------
    # UI CONTROLS
    # --------------------------
    model_menu = tk.OptionMenu(
        root,
        current_model_name,
        "cnn",
        "transformer",
        command=lambda _: update_plot()
    )
    model_menu.pack(fill="x")

    tk.Button(root, text="Load ARC JSON", command=load_arc_json).pack(fill="x")

    dim_slider = Scale(
        root,
        from_=0,
        to=config.latent_dim - 1,
        orient=HORIZONTAL,
        label="Latent Dimension",
        variable=current_dim,
        command=lambda _: update_plot()
    )
    dim_slider.pack(fill="x")

    shift_slider = Scale(
        root,
        from_=shift_range[0],
        to=shift_range[1],
        resolution=0.1,
        orient=HORIZONTAL,
        label="Shift Amount",
        command=lambda _: update_plot()
    )
    shift_slider.pack(fill="x")

    update_plot()
    root.mainloop()


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
    import os

    CNN_PATH = "results/vae/cnn_vae/vae_model.pt"
    TRANSFORMER_PATH = "results/vae/transformer_vae/vae_model.pt"

    cnn_model = None
    transformer_model = None
    config = None

    # --------------------------
    # Load CNN if exists
    # --------------------------
    if os.path.exists(CNN_PATH):
        cnn_model, config = load_vae_checkpoint(
            CNN_PATH,
            model_type="cnn"
        )
        print("[INFO] Loaded CNN VAE")
    else:
        print("[WARN] CNN VAE not found")

    # --------------------------
    # Load Transformer if exists
    # --------------------------
    if os.path.exists(TRANSFORMER_PATH):
        transformer_model, _ = load_vae_checkpoint(
            TRANSFORMER_PATH,
            model_type="transformer"
        )
        print("[INFO] Loaded Transformer VAE")
    else:
        print("[WARN] Transformer VAE not found")

    # --------------------------
    # Ensure at least one model exists
    # --------------------------
    if cnn_model is None and transformer_model is None:
        raise RuntimeError("No VAE checkpoints found.")

    # --------------------------
    # Pick a valid config source
    # --------------------------
    active_config = config if config is not None else (
        load_vae_checkpoint(
            TRANSFORMER_PATH if transformer_model else CNN_PATH,
            model_type="transformer" if transformer_model else "cnn"
        )[1]
    )

    # --------------------------
    # Create latent seed
    # --------------------------
    z = np.random.randn(active_config.latent_dim)

    # --------------------------
    # Launch viewer (adaptive)
    # --------------------------
    if cnn_model is not None and transformer_model is not None:
        print("[INFO] Enabling CNN <-> Transformer switching UI")

        interactive_latent_explorer(
            cnn_model,
            transformer_model,
            z,
            active_config
        )

    elif cnn_model is not None:
        print("[INFO] Only CNN available (no switching)")

        interactive_latent_explorer(
            cnn_model,
            cnn_model,   # fallback duplicate
            z,
            active_config
        )

    else:
        print("[INFO] Only Transformer available (no switching)")

        interactive_latent_explorer(
            transformer_model,
            transformer_model,
            z,
            active_config
        )