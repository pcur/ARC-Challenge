"""
train_spatial_geom.py
=====================
Training loop for SpatialGeomAE — spatial object-level geometry autoencoder.

No global bottleneck — per-node embeddings throughout.
Each object node reconstructs its own geometry from local neighbourhood.

Feature extraction
------------------
Node input  : geometry dims [10:22] of the original 22-dim node features
              normalised to zero mean / unit variance
Edge input  : spatial dims [0:4, 7:10, 10:12] = 9 dims
              (excludes color-related edge features)

Self-loop removal
-----------------
Single-node graphs previously got a self-loop added to avoid empty edge_index.
We now let isolated nodes be isolated — GATv2 handles no-edge nodes gracefully
by simply propagating the node's own features without attention aggregation.

Dependencies:
    pip install torch torch_geometric
"""

import os
import torch
import torch.optim as optim

from spatial_geom_ae import SpatialGeomAE, spatial_geom_loss
from dual_dataset import build_dual_dataloaders
from training_utils import GPUMemoryGuard, EarlyStopping


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CFG = dict(
    # Data
    training_dir      = os.path.join("..", "builder", "data", "training"),
    max_object_nodes  = 50,
    max_pixel_nodes   = 400,
    val_split         = 0.2,
    batch_size        = 32,   # object graphs are small — larger batch is fine

    # Model
    hidden_dim        = 128,
    num_heads         = 4,
    num_layers        = 4,
    dropout           = 0.1,
    pos_enc_dim       = 16,   # sinusoidal encoding dims per coordinate (row + col)

    # Training
    epochs            = 2000,   # extended — model still improving at 500
    weight_decay      = 1e-3,
    lr_max            = 1e-4,
    lr_min            = 1e-6,
    lr_cycle_epochs   = 50,

    # Loss
    geom_weight         = 1.0,
    edge_spatial_weight = 1.0,

    # Early stopping + GPU
    es_patience       = 20,
    es_threshold      = 1.05,
    gpu_max_fraction  = 0.80,

    # Checkpointing
    checkpoint_dir    = "checkpoints",
    checkpoint_name   = "spatial_geom_ae_best.pt",
)

# Geometry feature names for reporting
GEOM_FEATURE_NAMES = [
    "area", "centroid_row", "centroid_col",
    "bbox_min_row", "bbox_min_col", "bbox_max_row", "bbox_max_col",
    "width", "height", "density", "aspect_ratio", "is_single_pixel",
]


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION  —  geometry-only slices from object graph
# ─────────────────────────────────────────────────────────────────────────────

def extract_geom_features(obj_batch, stats, device):
    """
    Extract and normalise geometry features from object graph batch.

    Node geom  : x[:, 10:]          normalised (N, 12)
    Edge geom  : edge_attr cols [0:4, 7:10, 10:12]  normalised (E, 9)
    """
    x  = obj_batch.x.clone()
    ea = obj_batch.edge_attr

    x[:, 10:] = (x[:, 10:] - stats["geom_mean"]) / stats["geom_std"]

    # guard against graphs with no edges (single-node graphs)
    if ea is None or ea.numel() == 0 or ea.dim() < 2:
        edge_geom           = torch.zeros(0, 9, device=device)
        edge_spatial_target = torch.zeros(0, 3, device=device)
        return x[:, 10:], edge_geom, edge_spatial_target

    ea = ea.clone()
    ea[:, :4]  = (ea[:, :4]  - stats["econt_mean"][:4])  / stats["econt_std"][:4]
    ea[:, 10:] = (ea[:, 10:] - stats["econt_mean"][4:])   / stats["econt_std"][4:]

    x_geom    = x[:, 10:]
    edge_geom = torch.cat([ea[:, :4], ea[:, 7:10], ea[:, 10:]], dim=-1)
    edge_spatial_target = ea[:, 7:10].clone()

    return x_geom, edge_geom, edge_spatial_target


def compute_norm_stats(train_loader, device):
    print("Computing geometry normalisation statistics...")
    geom_sum = torch.zeros(12, device=device)
    geom_sq  = torch.zeros(12, device=device)
    ec_sum   = torch.zeros(6,  device=device)
    ec_sq    = torch.zeros(6,  device=device)
    geom_n = ec_n = 0

    with torch.no_grad():
        for obj_batch, _, _ in train_loader:
            obj_batch = obj_batch.to(device)
            geom      = obj_batch.x[:, 10:]
            geom_sum += geom.sum(0);  geom_sq += geom.pow(2).sum(0)
            geom_n   += geom.size(0)
            ea        = obj_batch.edge_attr
            ec        = torch.cat([ea[:, :4], ea[:, 10:]], dim=-1)
            ec_sum   += ec.sum(0);    ec_sq   += ec.pow(2).sum(0)
            ec_n     += ec.size(0)

    gm  = geom_sum / geom_n
    gs  = (geom_sq / geom_n  - gm.pow(2)).clamp(min=1e-8).sqrt()
    ecm = ec_sum   / ec_n
    ecs = (ec_sq   / ec_n    - ecm.pow(2)).clamp(min=1e-8).sqrt()
    stats = dict(geom_mean=gm, geom_std=gs, econt_mean=ecm, econt_std=ecs)
    print(f"  geom mean [{gm.min():.3f}, {gm.max():.3f}]  "
          f"std [{gs.min():.3f}, {gs.max():.3f}]\n")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(node_geom_pred, edge_spatial_pred,
                    target_node_geom, target_edge_spatial, stats):
    """
    geom_mae      : mean absolute error on geometry in original scale
    edge_spat_acc : accuracy on same_row / same_col / same_area flags
    """
    gm = stats["geom_mean"]
    gs = stats["geom_std"]

    pred_orig = node_geom_pred * gs + gm
    true_orig = target_node_geom * gs + gm
    geom_mae  = (pred_orig - true_orig).abs().mean().item()

    pred_flags = edge_spatial_pred > 0.0
    true_flags = target_edge_spatial > 0.5
    edge_acc   = (pred_flags == true_flags).float().mean().item()

    return geom_mae, edge_acc


# ─────────────────────────────────────────────────────────────────────────────
# ONE EPOCH
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, device, cfg, stats,
              optimizer=None, scheduler=None, gpu_guard=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss  = 0.0
    buckets     = dict(geom_loss=0., spatial_loss=0.)
    mae_sum = eacc_sum = 0.0
    n_batches   = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for obj_batch, _, _ in loader:
            obj_batch = obj_batch.to(device)

            x_geom, edge_geom, target_edge_spatial = extract_geom_features(
                obj_batch, stats, device
            )
            target_node_geom = x_geom   # reconstruction target = normalised input

            out = model(x_geom, obj_batch.edge_index, edge_geom)

            loss, breakdown = spatial_geom_loss(
                out["node_geom"],
                out["edge_spatial"],
                target_node_geom,
                target_edge_spatial,
                geom_weight         = cfg["geom_weight"],
                edge_spatial_weight = cfg["edge_spatial_weight"] if target_edge_spatial.numel() > 0 else 0.0,
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                if gpu_guard is not None:
                    gpu_guard.check()

            total_loss += loss.item()
            for k, v in breakdown.items():
                buckets[k] += v

            gm, ea = compute_metrics(
                out["node_geom"], out["edge_spatial"],
                target_node_geom, target_edge_spatial, stats,
            )
            mae_sum  += gm
            eacc_sum += ea
            n_batches += 1

    n = len(loader)
    metrics = dict(geom_mae=mae_sum / n_batches, edge_spatial_acc=eacc_sum / n_batches)
    return total_loss / n, {k: v / n for k, v in buckets.items()}, metrics


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def train(resume: bool = False):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not found. Set device='cpu' to run on CPU.")
    device = torch.device("cuda")
    print(f"Training SpatialGeomAE on: {torch.cuda.get_device_name(device)}\n")

    train_loader, val_loader = build_dual_dataloaders(
        training_dir     = CFG["training_dir"],
        batch_size       = CFG["batch_size"],
        max_object_nodes = CFG["max_object_nodes"],
        max_pixel_nodes  = CFG["max_pixel_nodes"],
        val_split        = CFG["val_split"],
    )

    stats = compute_norm_stats(train_loader, device)

    model = SpatialGeomAE(
        hidden_dim  = CFG["hidden_dim"],
        num_heads   = CFG["num_heads"],
        num_layers  = CFG["num_layers"],
        dropout     = CFG["dropout"],
        pos_enc_dim = CFG["pos_enc_dim"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SpatialGeomAE parameters  : {n_params:,}")
    print(f"No global bottleneck      : per-node embeddings preserved")
    print(f"Self-loops removed        : isolated nodes handled natively by GATv2")
    print(f"Augmentation              : ON (rotation + color permutation)")
    print(f"Loss                      : geometry MSE + spatial edge BCE\n")

    optimizer = optim.AdamW(model.parameters(),
                            lr=CFG["lr_max"], weight_decay=CFG["weight_decay"])
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr      = CFG["lr_min"],
        max_lr       = CFG["lr_max"],
        step_size_up = CFG["lr_cycle_epochs"] * steps_per_epoch,
        mode         = "triangular2",
        cycle_momentum = False,
    )

    gpu_guard  = GPUMemoryGuard(device, max_fraction=CFG["gpu_max_fraction"])
    early_stop = EarlyStopping(patience=CFG["es_patience"],
                               divergence_threshold=CFG["es_threshold"])

    os.makedirs(CFG["checkpoint_dir"], exist_ok=True)
    ckpt_path     = os.path.join(CFG["checkpoint_dir"], CFG["checkpoint_name"])
    best_val_loss = float("inf")
    start_epoch   = 1

    # ── resume from checkpoint if requested ──────────────────────────────────
    if resume and os.path.exists(ckpt_path):
        print(f"Resuming from checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optim_state"])
        best_val_loss = state["val_loss"]
        start_epoch   = state["epoch"] + 1
        # restore norm stats from checkpoint so they're consistent
        stats = {k: v.to(device) for k, v in state["norm_stats"].items()}
        print(f"  Resumed from epoch {state['epoch']}  |  "
              f"Best val loss so far: {best_val_loss:.4f}\n")
    elif resume:
        print(f"No checkpoint found at {ckpt_path} — starting fresh\n")

    print(f"Starting training — epochs {start_epoch} to {CFG['epochs']}\n")

    for epoch in range(start_epoch, CFG["epochs"] + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, _, _ = run_epoch(
            model, train_loader, device, CFG, stats,
            optimizer=optimizer, scheduler=scheduler, gpu_guard=gpu_guard,
        )
        val_loss, val_bd, val_m = run_epoch(
            model, val_loader, device, CFG, stats,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_loss"   : best_val_loss,
                "cfg"        : CFG,
                "norm_stats" : {k: v.cpu() for k, v in stats.items()},
            }, ckpt_path)
            marker = " ✓ best"
        else:
            marker = ""

        es_status = f"{early_stop.counter}/{early_stop.patience}"

        print(f"\nEpoch {epoch}  |  LR: {current_lr:.2e}  |  VRAM: {gpu_guard.summary_str()}  |  ES: {es_status}{marker}")
        print(f"  Losses  —  Train: {train_loss:.4f}  |  Val: {val_loss:.4f}  |  "
              f"Geom MSE: {val_bd['geom_loss']:.4f}  |  Edge Spatial: {val_bd['spatial_loss']:.4f}")
        print(f"  Metrics —  Geom MAE: {val_m['geom_mae']:.4f}  |  "
              f"Edge Spatial Acc: {val_m['edge_spatial_acc']:.4f}")

        if early_stop.step(val_loss):
            print(f"\nEarly stopping at epoch {epoch} — "
                  f"val loss diverged for {early_stop.patience} epochs.")
            break

    print(f"\nSpatialGeomAE training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from existing checkpoint")
    args = parser.parse_args()
    train(resume=args.resume)