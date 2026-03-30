"""
train.py
========
Training loop for the GAT-VAE on ARC graphs.

Features
--------
- CUDA GPU training with automatic fallback to CPU
- KL annealing   : kl_weight ramps from kl_start to kl_max over kl_anneal_epochs
- LR scheduler   : ReduceLROnPlateau — halves LR when val loss plateaus
- Checkpointing  : best model (lowest val loss) saved to disk

Usage
-----
    python train.py

Plug in your own dataset by replacing the placeholder `build_dataloaders()`
function with one that returns PyG DataLoader objects over your ARC graphs.

Each graph in the dataset should be a torch_geometric.data.Data with:
    x          : (N, 22)   node features  — one-hot color (10) + geometry (12)
    edge_index : (2, E)    fully connected COO edge index
    edge_attr  : (E, 12)   edge features

Dependencies:
    pip install torch torch_geometric
"""

import os
import torch
import torch.optim as optim

from gat_vae import GATVAE
from graph_decoder import gat_vae_loss
from dataset import build_dataloaders


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CFG = dict(
    # Data
    training_dir     = os.path.join("..", "builder", "data", "training"),
    max_nodes        = 50,       # graphs with more nodes are skipped
    val_split        = 0.1,      # 10% of training data held out for validation
    batch_size       = 32,

    # Model
    gat_hidden       = 128,
    gat_heads        = 4,
    gat_layers       = 3,
    gat_dropout      = 0.1,
    trunk_depth      = 4,
    trunk_dropout    = 0.1,
    latent_dim       = 256,
    dec_hidden       = 256,

    # Training
    epochs           = 1,
    lr               = 3e-4,

    # LR scheduler (ReduceLROnPlateau)
    lr_factor        = 0.5,      # multiply LR by this on plateau
    lr_patience      = 10,       # epochs with no improvement before reducing
    lr_min           = 1e-6,     # floor for LR

    # KL annealing — ramps linearly from kl_start to kl_max
    kl_start         = 0.0,
    kl_max           = 1e-3,
    kl_anneal_epochs = 50,       # reach kl_max after this many epochs

    # Loss weights (non-KL terms)
    color_weight     = 1.0,
    geom_weight      = 1.0,
    edge_cont_weight = 1.0,
    edge_bin_weight  = 1.0,

    # Checkpointing
    checkpoint_dir   = "checkpoints",
    checkpoint_name  = "gatvae_best.pt",
)


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
# build_dataloaders is imported from dataset.py — no placeholder needed here.


# ─────────────────────────────────────────────────────────────────────────────
# TARGET EXTRACTION
#   Pulls correctly typed targets from a PyG Batch for the loss function.
#   Targets are padded to (B, max_nodes, ...) to match decoder output shape.
# ─────────────────────────────────────────────────────────────────────────────

def extract_targets(batch, max_nodes: int, device: torch.device):
    """
    Convert a PyG Batch into the dense padded targets expected by gat_vae_loss.

    Returns
    -------
    target_color      : (B, max_nodes)              LongTensor  — color class indices
    target_node_geom  : (B, max_nodes, 12)          FloatTensor — continuous geometry
    target_edge_cont  : (B, max_nodes, max_nodes, 6) FloatTensor
    target_edge_bin   : (B, max_nodes, max_nodes, 6) FloatTensor
    """
    B           = batch.num_graphs
    node_counts = batch.ptr[1:] - batch.ptr[:-1]   # nodes per graph in batch

    # ── node targets ─────────────────────────────────────────────────────────
    color_idx  = batch.x[:, :10].argmax(dim=-1)    # (N_total,) long
    geom       = batch.x[:, 10:]                   # (N_total, 12)

    target_color     = torch.zeros(B, max_nodes, dtype=torch.long,  device=device)
    target_node_geom = torch.zeros(B, max_nodes, 12,                device=device)

    for i in range(B):
        start = batch.ptr[i].item()
        end   = batch.ptr[i + 1].item()
        n     = end - start
        target_color[i, :n]        = color_idx[start:end]
        target_node_geom[i, :n]    = geom[start:end]

    # ── edge targets ─────────────────────────────────────────────────────────
    # edge_attr layout: [dx,dy,manhattan,euclidean | binary×6 | ratio_ab,ratio_ba]
    target_edge_cont = torch.zeros(B, max_nodes, max_nodes, 6, device=device)
    target_edge_bin  = torch.zeros(B, max_nodes, max_nodes, 6, device=device)

    edge_ptr = 0
    for i in range(B):
        start = batch.ptr[i].item()
        n     = (batch.ptr[i + 1] - batch.ptr[i]).item()
        n_edges = n * (n - 1)                      # fully connected, no self-loops

        ea  = batch.edge_attr[edge_ptr : edge_ptr + n_edges]   # (E, 12)
        ei  = batch.edge_index[:, edge_ptr : edge_ptr + n_edges] - start  # local idx

        # scatter into dense (max_nodes, max_nodes) grid
        src, dst = ei[0], ei[1]
        target_edge_cont[i, src, dst] = torch.cat([ea[:, :4], ea[:, 10:]], dim=-1)
        target_edge_bin[i,  src, dst] = ea[:, 4:10]

        edge_ptr += n_edges

    return target_color, target_node_geom, target_edge_cont, target_edge_bin


# ─────────────────────────────────────────────────────────────────────────────
# KL ANNEALING
# ─────────────────────────────────────────────────────────────────────────────

def kl_weight_at(epoch: int, cfg: dict) -> float:
    """Linear ramp from kl_start to kl_max over kl_anneal_epochs."""
    if cfg["kl_anneal_epochs"] == 0:
        return cfg["kl_max"]
    progress = min(epoch / cfg["kl_anneal_epochs"], 1.0)
    return cfg["kl_start"] + progress * (cfg["kl_max"] - cfg["kl_start"])


# ─────────────────────────────────────────────────────────────────────────────
# ONE EPOCH
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, device, max_nodes, cfg, kl_weight, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss   = 0.0
    loss_buckets = dict(color_loss=0., geom_loss=0.,
                        edge_cont_loss=0., edge_bin_loss=0., kl_loss=0.)

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(device)

            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            target_color, target_node_geom, target_edge_cont, target_edge_bin = \
                extract_targets(batch, max_nodes, device)

            loss, breakdown = gat_vae_loss(
                out,
                target_color,
                target_node_geom,
                target_edge_cont,
                target_edge_bin,
                kl_weight        = kl_weight,
                color_weight     = cfg["color_weight"],
                geom_weight      = cfg["geom_weight"],
                edge_cont_weight = cfg["edge_cont_weight"],
                edge_bin_weight  = cfg["edge_bin_weight"],
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            for k, v in breakdown.items():
                loss_buckets[k] += v

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in loss_buckets.items()}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train():
    # ── device ───────────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not found. This training script requires a GPU.\n"
            "If you want to run on CPU, remove this check and set device='cpu'."
        )
    device = torch.device("cuda")
    print(f"Training on: {torch.cuda.get_device_name(device)}\n")

    # ── data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(
        training_dir = CFG["training_dir"],
        batch_size   = CFG["batch_size"],
        max_nodes    = CFG["max_nodes"],
        val_split    = CFG["val_split"],
    )

    # ── model ────────────────────────────────────────────────────────────────
    model = GATVAE(
        max_nodes    = CFG["max_nodes"],
        gat_hidden   = CFG["gat_hidden"],
        gat_heads    = CFG["gat_heads"],
        gat_layers   = CFG["gat_layers"],
        gat_dropout  = CFG["gat_dropout"],
        trunk_depth  = CFG["trunk_depth"],
        trunk_dropout= CFG["trunk_dropout"],
        latent_dim   = CFG["latent_dim"],
        dec_hidden   = CFG["dec_hidden"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}\n")

    # ── optimizer + scheduler ────────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode      = "min",
        factor    = CFG["lr_factor"],
        patience  = CFG["lr_patience"],
        min_lr    = CFG["lr_min"],
    )

    # ── checkpointing ────────────────────────────────────────────────────────
    os.makedirs(CFG["checkpoint_dir"], exist_ok=True)
    checkpoint_path = os.path.join(CFG["checkpoint_dir"], CFG["checkpoint_name"])
    best_val_loss   = float("inf")

    # ── training loop ────────────────────────────────────────────────────────
    print(f"{'Epoch':>6}  {'KL w':>7}  {'LR':>9}  "
          f"{'Train':>9}  {'Val':>9}  "
          f"{'color':>7}  {'geom':>7}  {'e_cont':>7}  {'e_bin':>7}  {'KL':>7}")
    print("─" * 95)

    for epoch in range(1, CFG["epochs"] + 1):
        kl_weight = kl_weight_at(epoch - 1, CFG)
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_bd = run_epoch(
            model, train_loader, device, CFG["max_nodes"], CFG, kl_weight,
            optimizer=optimizer,
        )
        val_loss, val_bd = run_epoch(
            model, val_loader, device, CFG["max_nodes"], CFG, kl_weight,
        )

        scheduler.step(val_loss)

        # ── checkpoint ───────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_loss"   : best_val_loss,
                "cfg"        : CFG,
            }, checkpoint_path)
            saved_marker = " ✓"
        else:
            saved_marker = ""

        print(
            f"{epoch:>6}  {kl_weight:>7.2e}  {current_lr:>9.2e}  "
            f"{train_loss:>9.4f}  {val_loss:>9.4f}  "
            f"{val_bd['color_loss']:>7.4f}  {val_bd['geom_loss']:>7.4f}  "
            f"{val_bd['edge_cont_loss']:>7.4f}  {val_bd['edge_bin_loss']:>7.4f}  "
            f"{val_bd['kl_loss']:>7.4f}"
            f"{saved_marker}"
        )

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    train()