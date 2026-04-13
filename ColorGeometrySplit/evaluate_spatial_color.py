"""
evaluate_spatial_color.py
=========================
Evaluation script for SpatialColorAE.

Metrics
-------
  fg_color_acc  : accuracy on foreground pixels
  fg_bg_acc     : accuracy on fg/bg classification
  per_color P/R/F1 : per ARC color class (1-9)

Visualisation
-------------
  For N random val grids: original | reconstructed | error map PNGs

Usage
-----
    python evaluate_spatial_color.py \
        --checkpoint checkpoints/spatial_color_ae_best.pt \
        --n_samples 8 \
        --output_dir eval_outputs/spatial_color \
        --seed 42
"""

import os
import argparse
import random
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from sklearn.metrics import precision_score, recall_score, f1_score

from spatial_color_ae import SpatialColorAE
from dual_dataset import build_dual_dataloaders
from pixel_graph_builder import pixel_grid_to_graph


ARC_COLORS_HEX = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25",
]
ARC_COLOR_NAMES = [
    "black", "blue", "red", "green", "yellow",
    "grey", "magenta", "orange", "azure", "maroon",
]
ARC_CMAP = mcolors.ListedColormap(ARC_COLORS_HEX)
ARC_NORM  = mcolors.BoundaryNorm(boundaries=range(11), ncolors=10)


def load_model(checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device)
    cfg   = state["cfg"]
    model = SpatialColorAE(
        hidden_dim      = cfg["hidden_dim"],
        num_heads       = cfg["num_heads"],
        num_layers      = cfg["num_layers"],
        dropout         = cfg["dropout"],
        color_dropout_p = cfg["color_dropout_p"],
    ).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model, cfg


def compute_metrics(model, val_loader, device):
    all_pred, all_true = [], []
    fg_color_total = fg_color_correct = 0
    fg_bg_total = fg_bg_correct = 0

    with torch.no_grad():
        for _, pix_batch, _ in val_loader:
            pix_batch  = pix_batch.to(device)
            x          = pix_batch.x
            target     = x[:, :10].argmax(dim=-1)
            out        = model(x, pix_batch.edge_index, pix_batch.edge_attr)

            pred_color = out["color_logits"].argmax(dim=-1)
            is_fg      = target != 0

            fg_color_correct += ((pred_color == target) & is_fg).sum().item()
            fg_color_total   += is_fg.sum().item()

            pred_fg = out["fg_logits"].squeeze(-1) > 0.0
            fg_bg_correct += (pred_fg == is_fg).sum().item()
            fg_bg_total   += target.numel()

            all_pred.append(pred_color[is_fg].cpu().numpy())
            all_true.append(target[is_fg].cpu().numpy())

    fg_color_acc = fg_color_correct / max(fg_color_total, 1)
    fg_bg_acc    = fg_bg_correct    / max(fg_bg_total, 1)

    ap = np.concatenate(all_pred)
    at = np.concatenate(all_true)

    per_color = {}
    for c in range(1, 10):
        p = precision_score(at == c, ap == c, zero_division=0)
        r = recall_score(at == c, ap == c, zero_division=0)
        f = f1_score(at == c, ap == c, zero_division=0)
        per_color[ARC_COLOR_NAMES[c]] = {"precision": p, "recall": r, "f1": f}

    macro_p = precision_score(at, ap, average="macro", labels=list(range(1,10)), zero_division=0)
    macro_r = recall_score(at, ap, average="macro", labels=list(range(1,10)), zero_division=0)
    macro_f = f1_score(at, ap, average="macro", labels=list(range(1,10)), zero_division=0)

    return dict(
        fg_color_acc = fg_color_acc,
        fg_bg_acc    = fg_bg_acc,
        macro_p      = macro_p,
        macro_r      = macro_r,
        macro_f      = macro_f,
        per_color    = per_color,
    )


def print_metrics(results):
    print("\n" + "="*60)
    print("SPATIAL COLOR AE — EVALUATION METRICS")
    print("="*60)
    print(f"  FG Color Accuracy : {results['fg_color_acc']:.4f}")
    print(f"  FG/BG Accuracy    : {results['fg_bg_acc']:.4f}")
    print(f"  Macro Precision   : {results['macro_p']:.4f}")
    print(f"  Macro Recall      : {results['macro_r']:.4f}")
    print(f"  Macro F1          : {results['macro_f']:.4f}")
    print("\n  Per-color metrics:")
    for name, m in results["per_color"].items():
        print(f"    {name:10s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
    print("="*60 + "\n")


def reconstruct_grid(model, pix_graph, device):
    from torch_geometric.data import Batch
    batch      = Batch.from_data_list([pix_graph]).to(device)
    with torch.no_grad():
        out    = model(batch.x, batch.edge_index, batch.edge_attr)
    rows, cols = pix_graph.grid_rows, pix_graph.grid_cols
    n          = rows * cols
    pred       = out["color_logits"][:n].argmax(dim=-1).cpu().numpy().reshape(rows, cols)
    true       = batch.x[:n, :10].argmax(dim=-1).cpu().numpy().reshape(rows, cols)
    return true, pred


def draw_arc_grid(ax, grid, title):
    ax.imshow(grid, cmap=ARC_CMAP, norm=ARC_NORM, interpolation="nearest", aspect="equal")
    r, c = grid.shape
    for i in range(r+1): ax.axhline(i-0.5, color="white", linewidth=0.4)
    for i in range(c+1): ax.axvline(i-0.5, color="white", linewidth=0.4)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10)


def draw_error_map(ax, true_grid, pred_grid, title):
    r, c    = true_grid.shape
    err_img = np.zeros((r, c, 3))
    for i in range(r):
        for j in range(c):
            if true_grid[i,j] == 0:
                err_img[i,j] = [0.85, 0.85, 0.85]
            elif true_grid[i,j] == pred_grid[i,j]:
                err_img[i,j] = [0.18, 0.8, 0.18]
            else:
                err_img[i,j] = [0.9, 0.15, 0.15]
    ax.imshow(err_img, interpolation="nearest", aspect="equal")
    for i in range(r+1): ax.axhline(i-0.5, color="white", linewidth=0.4)
    for i in range(c+1): ax.axvline(i-0.5, color="white", linewidth=0.4)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10)
    ax.legend(handles=[
        Patch(facecolor=[0.18,0.8,0.18], label="correct"),
        Patch(facecolor=[0.9,0.15,0.15], label="wrong"),
        Patch(facecolor=[0.85,0.85,0.85], label="background"),
    ], loc="lower right", fontsize=7, framealpha=0.8)


def save_visualisations(model, val_pix_graphs, device, n_samples, output_dir, seed):
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    samples = random.sample(val_pix_graphs, min(n_samples, len(val_pix_graphs)))

    for idx, pix_graph in enumerate(samples):
        true_grid, pred_grid = reconstruct_grid(model, pix_graph, device)
        fg       = true_grid != 0
        n_fg     = fg.sum()
        n_ok     = ((true_grid == pred_grid) & fg).sum()
        acc      = n_ok / max(n_fg, 1)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(
            f"SpatialColorAE reconstruction — sample {idx+1}  "
            f"(fg acc={acc:.3f}, {n_ok}/{n_fg} fg pixels)", fontsize=11,
        )
        draw_arc_grid(axes[0], true_grid, "Original")
        draw_arc_grid(axes[1], pred_grid, "Reconstructed")
        draw_error_map(axes[2], true_grid, pred_grid, "Error map")
        plt.tight_layout()
        path = os.path.join(output_dir, f"spatial_sample_{idx+1:02d}.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   default="checkpoints/spatial_color_ae_best.pt")
    parser.add_argument("--training_dir", default=os.path.join("..", "builder", "data", "training"))
    parser.add_argument("--n_samples",    type=int, default=8)
    parser.add_argument("--output_dir",   default="eval_outputs/spatial_color")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--batch_size",   type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, cfg = load_model(args.checkpoint, device)
    print(f"Loaded SpatialColorAE from {args.checkpoint}\n")

    _, val_loader = build_dual_dataloaders(
        training_dir     = args.training_dir,
        batch_size       = args.batch_size,
        max_object_nodes = 50,
        max_pixel_nodes  = cfg.get("max_pixel_nodes", 400),
        val_split        = 0.2,
        seed             = args.seed,
    )

    results = compute_metrics(model, val_loader, device)
    print_metrics(results)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
        f.write("SPATIAL COLOR AE — EVALUATION METRICS\n")
        f.write(f"Checkpoint: {args.checkpoint}\n\n")
        f.write(f"FG Color Accuracy : {results['fg_color_acc']:.4f}\n")
        f.write(f"FG/BG Accuracy    : {results['fg_bg_acc']:.4f}\n")
        f.write(f"Macro Precision   : {results['macro_p']:.4f}\n")
        f.write(f"Macro Recall      : {results['macro_r']:.4f}\n")
        f.write(f"Macro F1          : {results['macro_f']:.4f}\n\n")
        f.write("Per-color metrics:\n")
        for name, m in results["per_color"].items():
            f.write(f"  {name:10s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}\n")

    val_pix_graphs = [item[1] for item in val_loader.dataset]
    print(f"Saving {args.n_samples} random visualisations to {args.output_dir}/")
    save_visualisations(model, val_pix_graphs, device,
                        args.n_samples, args.output_dir, args.seed)
    print("\nDone.")


if __name__ == "__main__":
    main()
