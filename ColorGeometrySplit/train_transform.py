"""
train_transform.py
==================
Training loop for TRMTransform.

Only the TRMTransform parameters are trained — both spatial encoders are
frozen throughout. Loss is deep supervision: weighted CE at every
refinement step, averaged across steps and batch.

Dataset
-------
TaskDataset with leave-one-out splits. Each batch is a list of task
examples (variable context length per example). Processing is per-example
since context sizes vary.

Dependencies:
    pip install torch torch_geometric
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Batch

from transform_model import TRMTransform, FrozenDualEncoder, mean_max_pool
from task_dataset import build_task_dataloaders
from training_utils import GPUMemoryGuard, EarlyStopping


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CFG = dict(
    # Data
    training_dir      = os.path.join("..", "builder", "data", "training"),
    max_object_nodes  = 50,
    max_pixels        = 400,
    val_split         = 0.2,
    batch_size        = 256,
    min_context       = 1,

    # Frozen encoder checkpoints
    geom_ckpt         = os.path.join("checkpoints", "spatial_geom_ae_best.pt"),
    color_ckpt        = os.path.join("checkpoints", "spatial_color_ae_best.pt"),

    # Transform model
    geom_hidden       = 128,
    color_hidden      = 128,
    fused_dim         = 256,
    context_dim       = 256,
    num_heads         = 4,
    dropout           = 0.1,
    num_colors        = 10,

    # TRM refinement steps
    num_steps         = 4,       # total refinement steps
    supervision_steps = None,    # which steps to compute loss on
                                 # None = all steps, [3] = last only,
                                 # [2,3] = last two, [0,2] = every other

    # Training
    epochs            = 40,
    weight_decay      = 1e-3,
    lr_max            = 1e-4,
    lr_min            = 1e-6,
    lr_cycle_epochs   = 50,

    # Loss
    bg_class_weight   = 0.1,   # background pixel weight in CE
    exact_bonus_weight = 0.5,  # weight on exact match bonus term
    exact_bonus_alpha  = 4,    # sharpness — higher = bonus spikes closer to perfect

    # Early stopping + GPU
    es_patience       = 20,
    es_threshold      = 1.05,
    gpu_max_fraction  = 0.80,

    # Checkpointing
    checkpoint_dir    = "checkpoints",
    checkpoint_name   = "trm_transform_best.pt",
)


# ─────────────────────────────────────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────────────────────────────────────

def deep_supervision_loss(all_logits, target_colors, bg_weight=0.1,
                          num_colors=10, device=None, supervision_steps=None,
                          exact_bonus_weight=0.5, exact_bonus_alpha=4):
    """
    CE loss over selected refinement steps + exact match bonus on final step.

    supervision_steps  : list of step indices to supervise, or None for all.
    exact_bonus_weight : weight on the exact match bonus term.
    exact_bonus_alpha  : sharpness of bonus — higher values make the bonus
                         spike sharply as grid accuracy approaches 1.0.
                         alpha=4 means a grid that is 90% correct gets
                         0.9^4 = 0.66 bonus, while 99% correct gets 0.96.
                         This specifically rewards near-perfect predictions.
    """
    if device is None:
        device = all_logits[0].device
    weights    = torch.ones(num_colors, device=device)
    weights[0] = bg_weight

    steps = supervision_steps if supervision_steps is not None \
            else list(range(len(all_logits)))

    # ── standard CE loss across supervised steps ──────────────────────────────
    ce_total = 0.0
    for i in steps:
        ce_total += F.cross_entropy(all_logits[i], target_colors, weight=weights)
    ce_loss = ce_total / max(len(steps), 1)

    # ── exact match bonus on final step only ──────────────────────────────────
    # per-pixel accuracy → raised to alpha → bonus grows sharply near 1.0
    # we negate it so it acts as a loss term (minimise → maximise accuracy)
    if exact_bonus_weight > 0:
        final_logits = all_logits[-1]
        pred         = final_logits.argmax(dim=-1)
        pixel_acc    = (pred == target_colors).float().mean()
        exact_bonus  = -(pixel_acc ** exact_bonus_alpha)   # negative = reward
    else:
        exact_bonus  = torch.tensor(0.0, device=device)

    return ce_loss + exact_bonus_weight * exact_bonus


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(logits, target_colors, rows, cols):
    """
    Compute both per-pixel accuracy and task-level exact match.

    Per-pixel FG accuracy : fraction of foreground pixels correct
    Exact match           : 1 if every pixel in the grid is correct, 0 otherwise

    rows, cols : output grid dimensions (needed to check exact match)
    """
    pred   = logits.argmax(dim=-1)   # (M_pix,)
    is_fg  = target_colors != 0

    fg_acc = ((pred == target_colors) & is_fg).sum().float() / \
              is_fg.sum().float().clamp(min=1)

    # exact match — every pixel must be correct
    exact  = (pred == target_colors).all().float()

    return fg_acc.item(), exact.item()


# ─────────────────────────────────────────────────────────────────────────────
# PROCESS ONE TASK EXAMPLE
# ─────────────────────────────────────────────────────────────────────────────

def process_example(example, encoder, model, device, num_steps):
    """
    Process a single task example through the full pipeline.

    Returns (all_logits, target_colors) for loss computation.
    """
    context_pairs = example["context_pairs"]
    test_in_obj   = example["test_in_obj"].to(device)
    test_in_pix   = example["test_in_pix"].to(device)
    test_out_pix  = example["test_out_pix"].to(device)

    # ── encode test input ─────────────────────────────────────────────────────
    test_in_obj_batch  = Batch.from_data_list([test_in_obj])
    test_in_pix_batch  = Batch.from_data_list([test_in_pix])
    h_geom_in          = encoder.encode_geom(test_in_obj_batch)
    h_color_in         = encoder.encode_color(test_in_pix_batch)

    # ── build task context from demonstration pairs ───────────────────────────
    pair_embeds = []
    for obj_in, pix_in, obj_out, pix_out in context_pairs:
        obj_in_b  = Batch.from_data_list([obj_in.to(device)])
        pix_in_b  = Batch.from_data_list([pix_in.to(device)])
        obj_out_b = Batch.from_data_list([obj_out.to(device)])
        pix_out_b = Batch.from_data_list([pix_out.to(device)])

        h_gi  = encoder.encode_geom(obj_in_b)
        h_ci  = encoder.encode_color(pix_in_b)
        h_go  = encoder.encode_geom(obj_out_b)
        h_co  = encoder.encode_color(pix_out_b)

        p_gi  = mean_max_pool(h_gi, obj_in_b.ptr)     # (1, 256)
        p_ci  = mean_max_pool(h_ci, pix_in_b.ptr)     # (1, 256)
        p_go  = mean_max_pool(h_go, obj_out_b.ptr)    # (1, 256)
        p_co  = mean_max_pool(h_co, pix_out_b.ptr)    # (1, 256)

        pair_embed = torch.cat([p_gi, p_ci, p_go, p_co], dim=-1)  # (1, 1024)
        pair_embeds.append(pair_embed)

    pair_embeds  = torch.cat(pair_embeds, dim=0)       # (K, 1024)
    task_context = model.task_ctx_encoder(pair_embeds) # (context_dim,)

    # ── forward through TRM ───────────────────────────────────────────────────
    all_logits = model(
        h_geom_in    = h_geom_in,
        h_color_in   = h_color_in,
        task_context = task_context,
        out_pix_graph = test_out_pix,
        num_steps    = num_steps,
    )

    # ── target colors from output pixel graph ─────────────────────────────────
    n_out         = test_out_pix.x.size(0)
    target_colors = test_out_pix.x[:n_out, :10].argmax(dim=-1)  # (M_pix,)
    rows          = test_out_pix.grid_rows
    cols          = test_out_pix.grid_cols

    return all_logits, target_colors, rows, cols


# ─────────────────────────────────────────────────────────────────────────────
# ONE EPOCH
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, encoder, loader, device, cfg,
              optimizer=None, scheduler=None, gpu_guard=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss  = 0.0
    fg_acc_sum  = exact_sum = 0.0
    n_examples  = 0

    ctx_mgr = torch.enable_grad() if is_train else torch.no_grad()
    with ctx_mgr:
        for batch in loader:
            batch_loss = 0.0

            for example in batch:
                try:
                    all_logits, target_colors, rows, cols = process_example(
                        example, encoder, model, device, cfg["num_steps"]
                    )
                except Exception as e:
                    continue

                loss = deep_supervision_loss(
                    all_logits, target_colors,
                    bg_weight          = cfg["bg_class_weight"],
                    num_colors         = cfg["num_colors"],
                    device             = device,
                    supervision_steps  = cfg["supervision_steps"],
                    exact_bonus_weight = cfg["exact_bonus_weight"],
                    exact_bonus_alpha  = cfg["exact_bonus_alpha"],
                )
                batch_loss += loss

                # metrics on final step only
                fg_acc, exact = compute_metrics(
                    all_logits[-1], target_colors, rows, cols
                )
                fg_acc_sum += fg_acc
                exact_sum  += exact
                n_examples += 1

            if n_examples == 0:
                continue

            if is_train and batch_loss.requires_grad:
                avg_loss = batch_loss / len(batch)
                optimizer.zero_grad()
                avg_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                if gpu_guard is not None:
                    gpu_guard.check()

            total_loss += batch_loss.item() if hasattr(batch_loss, 'item') \
                          else batch_loss

    n = max(n_examples, 1)
    return (total_loss / n,
            dict(fg_acc=fg_acc_sum/n, exact_match=exact_sum/n,
                 n_solved=exact_sum, n_total=n_examples))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def train():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not found.")
    device = torch.device("cuda")
    print(f"Training TRMTransform on: {torch.cuda.get_device_name(device)}\n")

    train_loader, val_loader = build_task_dataloaders(
        training_dir     = CFG["training_dir"],
        batch_size       = CFG["batch_size"],
        max_object_nodes = CFG["max_object_nodes"],
        max_pixels       = CFG["max_pixels"],
        val_split        = CFG["val_split"],
        min_context      = CFG["min_context"],
    )

    # frozen encoders
    encoder = FrozenDualEncoder(
        geom_ckpt  = CFG["geom_ckpt"],
        color_ckpt = CFG["color_ckpt"],
        device     = device,
    )

    # trainable transform model
    model = TRMTransform(
        geom_hidden  = CFG["geom_hidden"],
        color_hidden = CFG["color_hidden"],
        fused_dim    = CFG["fused_dim"],
        context_dim  = CFG["context_dim"],
        num_heads    = CFG["num_heads"],
        dropout      = CFG["dropout"],
        num_colors   = CFG["num_colors"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sup_steps = CFG["supervision_steps"] if CFG["supervision_steps"] is not None \
                else list(range(CFG["num_steps"]))
    print(f"\nTRMTransform parameters : {n_params:,}")
    print(f"Refinement steps        : {CFG['num_steps']}")
    print(f"Supervised steps        : {sup_steps}\n")

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

    print(f"Starting training — {CFG['epochs']} epochs max\n")

    for epoch in range(1, CFG["epochs"] + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, _        = run_epoch(
            model, encoder, train_loader, device, CFG,
            optimizer=optimizer, scheduler=scheduler, gpu_guard=gpu_guard,
        )
        val_loss, val_m      = run_epoch(
            model, encoder, val_loader, device, CFG,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_loss"   : best_val_loss,
                "cfg"        : CFG,
            }, ckpt_path)
            marker = " ✓ best"
        else:
            marker = ""

        es_status = f"{early_stop.counter}/{early_stop.patience}"
        n_solved  = int(val_m['n_solved'])
        n_total   = int(val_m['n_total'])
        print(f"\nEpoch {epoch}  |  LR: {current_lr:.2e}  |  "
              f"VRAM: {gpu_guard.summary_str()}  |  ES: {es_status}{marker}")
        print(f"  Losses  —  Train: {train_loss:.4f}  |  Val: {val_loss:.4f}")
        print(f"  Metrics —  FG Pixel Acc: {val_m['fg_acc']:.4f}  |  "
              f"Exact Match: {n_solved}/{n_total} "
              f"({val_m['exact_match']*100:.1f}%)")

        if early_stop.step(val_loss):
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    print(f"\nTRMTransform training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    train()