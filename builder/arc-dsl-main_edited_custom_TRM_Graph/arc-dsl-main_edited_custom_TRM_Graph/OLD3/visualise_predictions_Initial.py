"""
visualise_predictions.py

Loads the trained VAE and transformer checkpoints, runs inference on a
selection of ARC tasks, and produces matplotlib figures showing:

    For each task:
        Row 1 — training demonstration pairs (input → output)
        Row 2 — test input | VAE reconstruction | transformer prediction | ground truth

    A summary page shows overall exact-match accuracy across all tasks.

Usage:
    python visualise_predictions.py

    Optionally edit the CONFIG section below to point at different checkpoints
    or to change how many tasks are visualised.
"""

import os
import json
import random
from typing import List, Dict, Optional

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from custom_object3 import grid_to_graph, graph_to_grid_from_predictions
from gat_vae3 import GATVAE
from arc_transformer import ARCTransformer, grid_to_z, z_to_grid


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  — edit these to match your setup
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_PATH       = r"C:\Users\tedhun\OneDrive - Wanzl GmbH & Co. KGaA\Microsoft Teams Chat Files\outputs\Documents\GS\ARC\Assignment10\Builder\data\training"
VAE_CKPT         = "gat_vae3_best.pt"
TRANSFORMER_CKPT = "arc_transformer_best.pt"
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

MAX_NODES          = 30
MAX_CELLS_PER_NODE = 40
LATENT_DIM         = 256

# How many tasks to visualise in detail (task-level plots)
N_TASKS_TO_PLOT = 10

# How many tasks to evaluate for the summary accuracy page
# Set to None to evaluate all available tasks (slow)
N_TASKS_TO_EVAL = None

# Random seed for reproducible task selection
SEED = 42

# Save figures to disk instead of showing interactively?
SAVE_FIGURES    = True
OUTPUT_DIR      = "visualisations"   # relative to script location

# ─────────────────────────────────────────────────────────────────────────────
# ARC COLOUR MAP
# 0=black 1=blue 2=red 3=green 4=yellow 5=grey
# 6=magenta 7=orange 8=azure 9=maroon
# ─────────────────────────────────────────────────────────────────────────────

ARC_COLORS = [
    "#000000",  # 0 black
    "#0074D9",  # 1 blue
    "#FF4136",  # 2 red
    "#2ECC40",  # 3 green
    "#FFDC00",  # 4 yellow
    "#AAAAAA",  # 5 grey
    "#F012BE",  # 6 magenta
    "#FF851B",  # 7 orange
    "#7FDBFF",  # 8 azure
    "#870C25",  # 9 maroon
]

ARC_CMAP  = ListedColormap(ARC_COLORS)
NORM      = matplotlib.colors.BoundaryNorm(boundaries=range(11), ncolors=10)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def grids_match(a, b) -> bool:
    """Exact pixel-level match between two grids."""
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if list(ra) != list(rb):
            return False
    return True


def draw_grid(ax, grid, title="", title_color="black"):
    """Draw a single ARC grid on a matplotlib axis."""
    arr = np.array(grid, dtype=int)
    ax.imshow(arr, cmap=ARC_CMAP, vmin=0, vmax=9, interpolation="nearest")

    h, w = arr.shape
    # Grid lines
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="both", bottom=False, left=False,
                   labelbottom=False, labelleft=False)

    ax.set_title(title, fontsize=8, color=title_color, pad=3)


def draw_blank(ax, msg="N/A"):
    """Draw a placeholder axis when a grid is unavailable."""
    ax.set_facecolor("#eeeeee")
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            fontsize=8, color="#888888", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("", fontsize=8)


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def encode_grid(raw_grid, vae, device):
    """Encode a raw grid → mu (deterministic latent). Returns None on failure."""
    grid  = tuple(tuple(row) for row in raw_grid)
    graph = grid_to_graph(grid, max_cells_per_node=MAX_CELLS_PER_NODE)
    n_nodes = len(graph["nodes"])
    n_edges = len(graph["edges"])
    if n_nodes < 2 or n_edges == 0 or n_nodes > MAX_NODES:
        return None
    x          = torch.tensor(graph["node_features"], dtype=torch.float32).to(device)
    edge_index = torch.tensor(graph["edge_index"],    dtype=torch.long).to(device)
    edge_attr  = torch.tensor(graph["edge_features"], dtype=torch.float32).to(device)
    batch      = torch.zeros(n_nodes, dtype=torch.long).to(device)
    with torch.no_grad():
        _, mu, _ = vae.encode(x, edge_index, edge_attr, batch)
    return mu.squeeze(0)


def vae_reconstruct(raw_grid, vae, device):
    """Encode then immediately decode — shows what the VAE alone can do."""
    z = encode_grid(raw_grid, vae, device)
    if z is None:
        return None
    h = len(raw_grid)
    w = len(raw_grid[0])
    return z_to_grid(z, vae, h, w, device)


def transformer_predict(task_sample, transformer, vae, device):
    """
    Run the full transformer pipeline on one task sample.
    Returns the predicted grid, or None if encoding fails.
    """
    transformer.eval()

    seq   = []
    roles = []
    ROLE_TRAIN_INPUT  = 0
    ROLE_TRAIN_OUTPUT = 1
    ROLE_TEST_INPUT   = 2

    for zi, zo in zip(task_sample["z_train_inputs"],
                      task_sample["z_train_outputs"]):
        seq.append(zi)
        seq.append(zo)
        roles.append(ROLE_TRAIN_INPUT)
        roles.append(ROLE_TRAIN_OUTPUT)

    seq.append(task_sample["z_test_input"])
    roles.append(ROLE_TEST_INPUT)

    z_seq  = torch.stack(seq).unsqueeze(0).to(device)
    role_t = torch.tensor(roles, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        z_pred = transformer(z_seq, role_t)

    h = task_sample["test_height"]
    w = task_sample["test_width"]
    return z_to_grid(z_pred.squeeze(0), vae, h, w, device)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET BUILDER  (lightweight — no full z encoding upfront)
# ─────────────────────────────────────────────────────────────────────────────

def build_samples(train_path, vae, device, max_files=None):
    """
    Build task samples for visualisation.
    Each sample has raw grids stored so we can display them,
    plus z encodings for the transformer.
    """
    files = sorted([f for f in os.listdir(train_path) if f.endswith(".json")])
    if max_files is not None:
        files = files[:max_files]

    samples  = []
    skipped  = 0
    vae.eval()

    for fname in files:
        fpath = os.path.join(train_path, fname)
        task  = load_json(fpath)

        train_pairs = task.get("train", [])
        test_pairs  = task.get("test",  [])
        if not train_pairs or not test_pairs:
            continue

        # Encode training pairs
        z_ins, z_outs = [], []
        train_inputs_raw, train_outputs_raw = [], []
        ok = True

        for pair in train_pairs:
            zi = encode_grid(pair["input"],  vae, device)
            zo = encode_grid(pair["output"], vae, device)
            if zi is None or zo is None:
                ok = False
                break
            z_ins.append(zi)
            z_outs.append(zo)
            train_inputs_raw.append(pair["input"])
            train_outputs_raw.append(pair["output"])

        if not ok:
            skipped += 1
            continue

        test_pair     = test_pairs[0]
        z_test_in     = encode_grid(test_pair["input"], vae, device)
        test_out_raw  = test_pair.get("output", None)
        z_test_out    = encode_grid(test_out_raw, vae, device) if test_out_raw else None

        if z_test_in is None:
            skipped += 1
            continue

        h = len(test_out_raw) if test_out_raw else len(test_pair["input"])
        w = len(test_out_raw[0]) if test_out_raw else len(test_pair["input"][0])

        samples.append({
            "task_id"           : fname.replace(".json", ""),
            "train_inputs_raw"  : train_inputs_raw,
            "train_outputs_raw" : train_outputs_raw,
            "test_input_raw"    : test_pair["input"],
            "test_output_raw"   : test_out_raw,
            "z_train_inputs"    : z_ins,
            "z_train_outputs"   : z_outs,
            "z_test_input"      : z_test_in,
            "z_test_output"     : z_test_out,
            "test_height"       : h,
            "test_width"        : w,
        })

    print(f"  Samples built : {len(samples)}")
    print(f"  Skipped       : {skipped}")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — PER-TASK DETAIL PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_task(sample, transformer, vae, device, save_dir=None):
    """
    One figure per task showing:
        Top rows    : training demonstration pairs
        Bottom row  : test input | VAE recon | transformer pred | ground truth
    """
    task_id       = sample["task_id"]
    n_train       = len(sample["train_inputs_raw"])
    test_in_raw   = sample["test_input_raw"]
    test_out_raw  = sample["test_output_raw"]

    # Run inference
    vae_recon  = vae_reconstruct(test_in_raw, vae, device)
    tr_pred    = transformer_predict(sample, transformer, vae, device)

    exact_vae  = grids_match(vae_recon,  test_out_raw) if vae_recon  and test_out_raw else False
    exact_tr   = grids_match(tr_pred,    test_out_raw) if tr_pred    and test_out_raw else False

    # Layout: n_train demo columns + 4 test columns
    n_cols   = max(n_train, 4)
    n_rows   = 2   # row 0 = demos, row 1 = test analysis

    fig_w    = max(n_cols * 2.2, 10)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, 5))

    # Make axes always 2D
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    # ── Row 0: training demonstrations ───────────────────────────────────
    for col in range(n_cols):
        if col < n_train:
            draw_grid(axes[0, col],
                      sample["train_inputs_raw"][col],
                      title=f"Train {col+1} Input")
        else:
            draw_blank(axes[0, col])

    # Second pass — outputs below inputs in same column
    # We'll use row 0 for inputs and squeeze outputs as subtitles
    # Actually use a second sub-row: rebuild with 3 rows
    plt.close(fig)

    n_rows  = 3
    fig_h   = n_rows * 2.4 + 0.8
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    # Row 0: train inputs
    for col in range(n_cols):
        if col < n_train:
            draw_grid(axes[0, col],
                      sample["train_inputs_raw"][col],
                      title=f"Demo {col+1}: Input")
        else:
            draw_blank(axes[0, col])

    # Row 1: train outputs
    for col in range(n_cols):
        if col < n_train:
            draw_grid(axes[1, col],
                      sample["train_outputs_raw"][col],
                      title=f"Demo {col+1}: Output")
        else:
            draw_blank(axes[1, col])

    # Row 2: test analysis — up to 4 panels
    panels = [
        (test_in_raw,  "Test Input",          "black"),
        (vae_recon,    f"VAE Recon {'✓' if exact_vae else '✗'}",
                       "green" if exact_vae else "red"),
        (tr_pred,      f"Transformer {'✓' if exact_tr else '✗'}",
                       "green" if exact_tr else "red"),
        (test_out_raw, "Ground Truth",         "black"),
    ]

    for col, (grid, title, tcol) in enumerate(panels):
        if col < n_cols:
            if grid is not None:
                draw_grid(axes[2, col], grid, title=title, title_color=tcol)
            else:
                draw_blank(axes[2, col], msg="Encoding failed")

    # Hide any unused columns in row 2
    for col in range(len(panels), n_cols):
        draw_blank(axes[2, col])

    fig.suptitle(
        f"Task: {task_id}   |   "
        f"VAE exact: {'YES' if exact_vae else 'NO'}   |   "
        f"Transformer exact: {'YES' if exact_tr else 'NO'}",
        fontsize=10, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    if save_dir:
        path = os.path.join(save_dir, f"{task_id}.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")
    else:
        plt.show()

    return exact_vae, exact_tr


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — SUMMARY ACCURACY PAGE
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary(results, save_dir=None):
    """
    Bar chart and breakdown table showing:
        - Overall exact-match accuracy (VAE recon vs transformer)
        - Per-task result grid (green=correct, red=wrong)
    """
    task_ids  = [r["task_id"]    for r in results]
    vae_exact = [r["exact_vae"]  for r in results]
    tr_exact  = [r["exact_tr"]   for r in results]

    n          = len(results)
    vae_acc    = 100.0 * sum(vae_exact) / max(n, 1)
    tr_acc     = 100.0 * sum(tr_exact)  / max(n, 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, n * 0.22 + 2)))

    # ── Left: accuracy bar chart ──────────────────────────────────────────
    ax = axes[0]
    bars = ax.bar(
        ["VAE\nReconstruction", "Transformer\nPrediction"],
        [vae_acc, tr_acc],
        color=["#4C9BE8", "#E8834C"],
        width=0.5, edgecolor="white",
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Exact Match Accuracy (%)", fontsize=11)
    ax.set_title(f"Accuracy over {n} tasks", fontsize=12, fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=0.5)

    for bar, val in zip(bars, [vae_acc, tr_acc]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1.5,
                f"{val:.1f}%",
                ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Right: per-task result grid ───────────────────────────────────────
    ax2 = axes[1]
    ax2.set_title("Per-task results  (green=correct, red=wrong)",
                  fontsize=10, fontweight="bold")

    for i, (vid, tid, task_id) in enumerate(
            zip(vae_exact, tr_exact, task_ids)):
        y = n - 1 - i

        # VAE column
        ax2.barh(y, 0.45, left=0,    height=0.8,
                 color="#2ECC40" if vid else "#FF4136", edgecolor="white")
        # Transformer column
        ax2.barh(y, 0.45, left=0.55, height=0.8,
                 color="#2ECC40" if tid else "#FF4136", edgecolor="white")

        ax2.text(-0.05, y, task_id[:18], va="center", ha="right", fontsize=5.5)

    ax2.set_xlim(-0.05, 1.1)
    ax2.set_ylim(-0.5, n + 0.5)
    ax2.set_xticks([0.225, 0.775])
    ax2.set_xticklabels(["VAE", "Transformer"], fontsize=9)
    ax2.set_yticks([])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    # Legend
    legend_elements = [
        mpatches.Patch(color="#2ECC40", label="Exact match ✓"),
        mpatches.Patch(color="#FF4136", label="Wrong ✗"),
    ]
    ax2.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()

    if save_dir:
        path = os.path.join(save_dir, "_summary.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"\nSummary saved: {path}")
    else:
        plt.show()

    # Print text summary too
    print(f"\n{'='*50}")
    print(f"RESULTS SUMMARY  ({n} tasks evaluated)")
    print(f"{'='*50}")
    print(f"  VAE reconstruction  exact match: {sum(vae_exact):3d}/{n}  ({vae_acc:.1f}%)")
    print(f"  Transformer predict exact match: {sum(tr_exact):3d}/{n}  ({tr_acc:.1f}%)")
    print(f"{'='*50}")

    correct_ids = [tid for tid, ok in zip(task_ids, tr_exact) if ok]
    if correct_ids:
        print(f"\n  Tasks solved by transformer:")
        for tid in correct_ids:
            print(f"    {tid}")
    else:
        print(f"\n  No tasks solved exactly by transformer yet.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if SAVE_FIGURES:
        out_dir = os.path.join(script_dir, OUTPUT_DIR)
        os.makedirs(out_dir, exist_ok=True)
        print(f"Figures will be saved to: {out_dir}\n")
    else:
        out_dir = None

    # ── load VAE ──────────────────────────────────────────────────────────
    print("Loading VAE...")
    vae_path = os.path.join(script_dir, VAE_CKPT)
    vae = GATVAE(
        max_nodes=MAX_NODES,
        max_cells_per_node=MAX_CELLS_PER_NODE,
        latent_dim=LATENT_DIM,
    ).to(DEVICE)
    vae_ckpt = torch.load(vae_path, map_location=DEVICE)
    vae.load_state_dict(vae_ckpt["model_state_dict"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    print(f"  VAE loaded  (best val loss: {vae_ckpt.get('best_val_loss','N/A'):.4f})")

    # ── load transformer ──────────────────────────────────────────────────
    print("Loading transformer...")
    tr_path = os.path.join(script_dir, TRANSFORMER_CKPT)
    tr_ckpt = torch.load(tr_path, map_location=DEVICE)
    transformer = ARCTransformer(
        latent_dim = tr_ckpt.get("latent_dim", LATENT_DIM),
        tr_hidden  = tr_ckpt.get("tr_hidden",  512),
        tr_heads   = tr_ckpt.get("tr_heads",   8),
        tr_layers  = tr_ckpt.get("tr_layers",  6),
    ).to(DEVICE)
    transformer.load_state_dict(tr_ckpt["model_state_dict"])
    transformer.eval()
    print(f"  Transformer loaded (best val loss: {tr_ckpt.get('best_val_loss','N/A'):.6f})")
    print(f"  Trained for {tr_ckpt.get('epoch','?')} epochs\n")

    # ── build samples ─────────────────────────────────────────────────────
    print("Building task samples...")
    all_samples = build_samples(TRAIN_PATH, vae, DEVICE)

    if not all_samples:
        print("No samples could be built. Check TRAIN_PATH and checkpoints.")
        return

    random.shuffle(all_samples)

    # ── detailed plots for N_TASKS_TO_PLOT tasks ──────────────────────────
    plot_samples = all_samples[:N_TASKS_TO_PLOT]
    eval_samples = all_samples if N_TASKS_TO_EVAL is None else all_samples[:N_TASKS_TO_EVAL]

    print(f"\nGenerating detail plots for {len(plot_samples)} tasks...")
    for i, sample in enumerate(plot_samples):
        print(f"  [{i+1}/{len(plot_samples)}] {sample['task_id']}")
        plot_task(sample, transformer, vae, DEVICE, save_dir=out_dir)

    # ── summary accuracy over all eval samples ────────────────────────────
    print(f"\nEvaluating {len(eval_samples)} tasks for summary...")
    results = []
    for i, sample in enumerate(eval_samples):
        vae_recon = vae_reconstruct(sample["test_input_raw"], vae, DEVICE)
        tr_pred   = transformer_predict(sample, transformer, vae, DEVICE)
        gt        = sample["test_output_raw"]

        exact_vae = grids_match(vae_recon, gt) if vae_recon and gt else False
        exact_tr  = grids_match(tr_pred,   gt) if tr_pred   and gt else False

        results.append({
            "task_id"   : sample["task_id"],
            "exact_vae" : exact_vae,
            "exact_tr"  : exact_tr,
        })

        if (i+1) % 50 == 0:
            print(f"  Evaluated {i+1}/{len(eval_samples)}...")

    plot_summary(results, save_dir=out_dir)

    if not SAVE_FIGURES:
        plt.show()


if __name__ == "__main__":
    main()