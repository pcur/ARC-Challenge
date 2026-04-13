"""
evaluate_spatial_geom.py
========================
Evaluation script for SpatialGeomAE.

Metrics
-------
  geom_mae          : mean absolute error on geometry (original scale)
  per_feature_mae   : MAE broken down per geometry feature
  edge_spatial_acc  : accuracy on same_row / same_col / same_area

Visualisation
-------------
  For N random val graphs: side-by-side scatter plot
    Left  : original — nodes at true centroid positions, sized by area
    Right : reconstructed — nodes at predicted positions, sized by predicted area
  Nodes coloured by their true ARC color for readability.

Usage
-----
    python evaluate_spatial_geom.py \
        --checkpoint checkpoints/spatial_geom_ae_best.pt \
        --n_samples 8 \
        --output_dir eval_outputs/spatial_geom \
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

from spatial_geom_ae import SpatialGeomAE
from train_spatial_geom import (
    compute_norm_stats, extract_geom_features,
    GEOM_FEATURE_NAMES,
)
from dual_dataset import build_dual_dataloaders


ARC_COLORS = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25",
]


def load_model(checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device)
    cfg   = state["cfg"]
    model = SpatialGeomAE(
        hidden_dim  = cfg["hidden_dim"],
        num_heads   = cfg["num_heads"],
        num_layers  = cfg["num_layers"],
        dropout     = cfg["dropout"],
        pos_enc_dim = cfg.get("pos_enc_dim", 16),
    ).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    norm_stats = {k: v.to(device) for k, v in state["norm_stats"].items()}
    return model, cfg, norm_stats


def compute_metrics(model, val_loader, stats, device):
    all_abs_err  = []
    all_edge_pred, all_edge_true = [], []

    with torch.no_grad():
        for obj_batch, _, _ in val_loader:
            obj_batch = obj_batch.to(device)
            x_geom, edge_geom, target_edge_spatial = extract_geom_features(
                obj_batch, stats, device
            )
            out = model(x_geom, obj_batch.edge_index, edge_geom)

            gm = stats["geom_mean"]
            gs = stats["geom_std"]
            pred_orig = out["node_geom"] * gs + gm
            true_orig = x_geom * gs + gm
            all_abs_err.append((pred_orig - true_orig).abs().cpu())

            pred_flags = (out["edge_spatial"] > 0.0).cpu().numpy()
            true_flags = (target_edge_spatial > 0.5).cpu().numpy()
            all_edge_pred.append(pred_flags)
            all_edge_true.append(true_flags)

    abs_err      = torch.cat(all_abs_err, dim=0)
    overall_mae  = abs_err.mean().item()
    per_feat_mae = abs_err.mean(dim=0).numpy()

    ep = np.concatenate(all_edge_pred)
    et = np.concatenate(all_edge_true)
    edge_acc = (ep == et).mean()

    return dict(
        overall_mae  = overall_mae,
        per_feat_mae = dict(zip(GEOM_FEATURE_NAMES, per_feat_mae.tolist())),
        edge_spatial_acc = float(edge_acc),
    )


def print_metrics(results):
    print("\n" + "="*60)
    print("SPATIAL GEOM AE — EVALUATION METRICS")
    print("="*60)
    print(f"  Overall Geom MAE     : {results['overall_mae']:.4f}")
    print(f"  Edge Spatial Acc     : {results['edge_spatial_acc']:.4f}")
    print("\n  Per-feature MAE (original scale):")
    for feat, val in results["per_feat_mae"].items():
        print(f"    {feat:20s}: {val:.4f}")
    print("="*60 + "\n")


def visualise_sample(model, obj_graph, stats, device, ax_orig, ax_recon):
    from torch_geometric.data import Batch
    batch  = Batch.from_data_list([obj_graph]).to(device)
    x_geom, edge_geom, _ = extract_geom_features(batch, stats, device)

    with torch.no_grad():
        out = model(x_geom, batch.edge_index, edge_geom)

    n  = batch.x.size(0)
    gm = stats["geom_mean"]
    gs = stats["geom_std"]

    orig_geom = (x_geom[:n] * gs + gm).cpu()
    pred_geom = (out["node_geom"][:n] * gs + gm).cpu()

    colors = [ARC_COLORS[int(batch.x[i, :10].argmax().item())] for i in range(n)]

    max_area = max(orig_geom[:, 0].max().item(),
                   pred_geom[:, 0].max().item(), 1.0)

    for ax, geom, title in [
        (ax_orig,  orig_geom, "Original geometry"),
        (ax_recon, pred_geom, "Reconstructed geometry"),
    ]:
        rows  = geom[:, 1].numpy()
        cols  = geom[:, 2].numpy()
        areas = geom[:, 0].numpy()
        sizes = [max(30, 800 * (a / max_area)) for a in areas]

        ax.scatter(cols, rows, s=sizes, c=colors, alpha=0.85,
                   edgecolors="white", linewidths=0.8, zorder=3)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_facecolor("#F5F5F5")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("col", fontsize=9)
        ax.set_ylabel("row", fontsize=9)


def save_visualisations(model, val_obj_graphs, stats, device,
                        n_samples, output_dir, seed):
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    samples = random.sample(val_obj_graphs, min(n_samples, len(val_obj_graphs)))

    for idx, obj_graph in enumerate(samples):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        fig.suptitle(f"SpatialGeomAE reconstruction — sample {idx+1}", fontsize=13)
        visualise_sample(model, obj_graph, stats, device, axes[0], axes[1])
        plt.tight_layout()
        path = os.path.join(output_dir, f"geom_sample_{idx+1:02d}.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   default="checkpoints/spatial_geom_ae_best.pt")
    parser.add_argument("--training_dir", default=os.path.join("..", "builder", "data", "training"))
    parser.add_argument("--n_samples",    type=int, default=8)
    parser.add_argument("--output_dir",   default="eval_outputs/spatial_geom")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--batch_size",   type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, cfg, norm_stats = load_model(args.checkpoint, device)
    print(f"Loaded SpatialGeomAE from {args.checkpoint}\n")

    _, val_loader = build_dual_dataloaders(
        training_dir     = args.training_dir,
        batch_size       = args.batch_size,
        max_object_nodes = cfg.get("max_object_nodes", 50),
        max_pixel_nodes  = cfg.get("max_pixel_nodes", 400),
        val_split        = 0.2,
        seed             = args.seed,
    )

    results = compute_metrics(model, val_loader, norm_stats, device)
    print_metrics(results)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
        f.write("SPATIAL GEOM AE — EVALUATION METRICS\n")
        f.write(f"Checkpoint: {args.checkpoint}\n\n")
        f.write(f"Overall Geom MAE     : {results['overall_mae']:.4f}\n")
        f.write(f"Edge Spatial Acc     : {results['edge_spatial_acc']:.4f}\n\n")
        f.write("Per-feature MAE:\n")
        for feat, val in results["per_feat_mae"].items():
            f.write(f"  {feat:20s}: {val:.4f}\n")

    val_obj_graphs = [item[0] for item in val_loader.dataset]
    print(f"Saving {args.n_samples} random visualisations to {args.output_dir}/")
    save_visualisations(model, val_obj_graphs, norm_stats, device,
                        args.n_samples, args.output_dir, args.seed)
    print("\nDone.")


if __name__ == "__main__":
    main()