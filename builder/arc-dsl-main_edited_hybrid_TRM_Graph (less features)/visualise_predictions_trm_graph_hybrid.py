"""
visualise_predictions_trm_graph_hybrid.py

Visualisation script for arc_trm_graph_hybrid.py.

Pipeline:
    raw grid → grid_to_graph() (hybrid_object2) → token sequence → Graph TRM
    → predicted existence + color + shape mask + bbox
    → graph_to_grid_from_predictions() (hybrid_object2) → reconstructed grid

Layout per task figure:
    Row 0 — section label: Demonstrations Input
    Row 1 — demonstration input grids
    Row 2 — section label: Demonstrations Output
    Row 3 — demonstration output grids
    Row 4 — section label: Test Analysis
    Row 5 — test input | Graph TRM prediction | ground truth | diff map
    Row 6 — metrics panel (full width)

Usage:
    python visualise_predictions_trm_graph_hybrid.py
"""

import os
import json
import random
from typing import List, Dict, Optional
from collections import Counter

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap

from hybrid_object2 import grid_to_graph, graph_to_grid_from_predictions
from arc_trm_graph_hybrid import (
    GraphTinyRecursiveModel,
    graph_to_targets,
    collate_batch,
    preds_to_grid,
    encode_grid_to_graph,
    build_sequence_tensors,
    graph_to_token_features,
    HIDDEN_SIZE, N_HEADS, FF_DIM, DROPOUT,
    N_RECURSIONS, T_INNER, N_SUP,
    MAX_NODES, MAX_DEMOS, NUM_COLORS,
    EMA_DECAY,
)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_PATH   = r"C:\Users\tedhun\OneDrive - Wanzl GmbH & Co. KGaA\Microsoft Teams Chat Files\outputs\Documents\GS\ARC\Assignment10\Builder\data\training"
TRM_CKPT     = "arc_trm_graph_hybrid_best.pt"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

N_TASKS_TO_PLOT = 10
N_TASKS_TO_EVAL = None   # None = all available tasks

SEED         = 42
SAVE_FIGURES = True
OUTPUT_DIR   = "visualisations_trm_graph_hybrid"


# ─────────────────────────────────────────────────────────────────────────────
# ARC COLOUR MAP
# ─────────────────────────────────────────────────────────────────────────────

ARC_COLORS = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25",
]
ARC_CMAP = ListedColormap(ARC_COLORS)


# ─────────────────────────────────────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)


def grids_match(a, b):
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if list(ra) != list(rb):
            return False
    return True


def draw_grid(ax, grid, title="", title_color="white"):
    arr = np.array(grid, dtype=int)
    ax.imshow(arr, cmap=ARC_CMAP, vmin=0, vmax=9, interpolation="nearest")
    h, w = arr.shape
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="both", bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    ax.set_title(title, fontsize=8, color=title_color, pad=3)


def draw_diff(ax, pred, target, title="Diff Map"):
    """Green = correct foreground, dark = correct background, red = wrong."""
    if pred is None or target is None:
        draw_blank(ax, "No diff")
        ax.set_title(title, fontsize=8, color="white", pad=3)
        return

    h_t, w_t = len(target), len(target[0])
    h_p, w_p = len(pred),   len(pred[0])

    if h_t != h_p or w_t != w_p:
        draw_blank(ax, "Size\nmismatch")
        ax.set_title(title, fontsize=8, color="#FF4136", pad=3)
        return

    flat_t = [target[r][c] for r in range(h_t) for c in range(w_t)]
    bg     = Counter(flat_t).most_common(1)[0][0]

    diff = np.zeros((h_t, w_t, 3), dtype=np.float32)
    for r in range(h_t):
        for c in range(w_t):
            t = target[r][c]
            p = pred[r][c]
            if t == p and t == bg:
                diff[r, c] = [0.1, 0.1, 0.1]
            elif t == p:
                diff[r, c] = [0.18, 0.80, 0.25]
            else:
                diff[r, c] = [0.87, 0.13, 0.13]

    ax.imshow(diff, interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, w_t, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h_t, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.3)
    ax.tick_params(which="both", bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    ax.set_title(title, fontsize=8, color="white", pad=3)


def draw_blank(ax, msg="N/A"):
    ax.set_facecolor("#222222")
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            fontsize=8, color="#666666", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("", fontsize=8)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_samples(train_path: str, max_files: int = None) -> List[Dict]:
    """
    Build one sample per task (no augmentation).
    Each sample stores both the graph representation (for the TRM)
    and the raw grids (for display).
    """
    files = sorted([f for f in os.listdir(train_path) if f.endswith(".json")])
    if max_files is not None:
        files = files[:max_files]

    samples, skipped = [], 0
    n_files = len(files)

    for file_idx, fname in enumerate(files):
        if file_idx % 100 == 0:
            print(f"  Processing file {file_idx+1}/{n_files}  "
                  f"({len(samples)} samples so far)...")

        task    = load_json(os.path.join(train_path, fname))
        task_id = fname.replace(".json", "")

        train_pairs = task.get("train", [])
        test_pairs  = task.get("test",  [])

        if not train_pairs or not test_pairs:
            skipped += 1
            continue

        test_pair = test_pairs[0]
        if "output" not in test_pair:
            skipped += 1
            continue

        # Build graphs for all demo pairs
        demo_input_graphs, demo_output_graphs = [], []
        ok = True

        for pair in train_pairs[:MAX_DEMOS]:
            g_in  = encode_grid_to_graph(pair["input"])
            g_out = encode_grid_to_graph(pair["output"])
            if g_in is None or g_out is None:
                ok = False; break
            if len(g_in["nodes"]) > MAX_NODES or len(g_out["nodes"]) > MAX_NODES:
                ok = False; break
            demo_input_graphs.append(g_in)
            demo_output_graphs.append(g_out)

        if not ok:
            skipped += 1
            continue

        g_test_in  = encode_grid_to_graph(test_pair["input"])
        g_test_out = encode_grid_to_graph(test_pair["output"])

        if g_test_in is None or g_test_out is None:
            skipped += 1
            continue
        if len(g_test_in["nodes"]) > MAX_NODES or len(g_test_out["nodes"]) > MAX_NODES:
            skipped += 1
            continue

        samples.append({
            "task_id"           : task_id,
            # Graph representations for TRM
            "demo_input_graphs" : demo_input_graphs,
            "demo_output_graphs": demo_output_graphs,
            "test_input_graph"  : g_test_in,
            "test_output_graph" : g_test_out,
            "test_h"            : len(test_pair["output"]),
            "test_w"            : len(test_pair["output"][0]),
            "test_raw"          : test_pair["output"],
            # Raw grids for display
            "train_inputs_raw"  : [p["input"]  for p in train_pairs[:MAX_DEMOS]],
            "train_outputs_raw" : [p["output"] for p in train_pairs[:MAX_DEMOS]],
            "test_input_raw"    : test_pair["input"],
            "test_output_raw"   : test_pair["output"],
        })

    print(f"  Samples built : {len(samples)}  (skipped {skipped})")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def trm_predict(sample: Dict, model: GraphTinyRecursiveModel,
                device: str) -> Optional[List[List[int]]]:
    """Run Graph TRM inference on one task sample."""
    model.eval()
    seq_feats, seq_types, pad_mask, _, test_sizes, _ = \
        collate_batch([sample], device)

    with torch.no_grad():
        _, preds = model(seq_feats, seq_types, pad_mask, targets=None)

    h, w = test_sizes[0]
    return preds_to_grid(preds, h, w, sample_idx=0)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(pred, target):
    empty = {
        "exact_match": False, "pixel_acc": 0.0, "color_acc": 0.0,
        "size_match": False, "n_correct": 0, "n_total": 0,
        "n_wrong_color": 0, "n_wrong_pos": 0,
    }
    if pred is None or target is None:
        return empty

    h_t, w_t = len(target), len(target[0])
    h_p, w_p = len(pred),   len(pred[0])
    size_match = (h_t == h_p and w_t == w_p)

    if not size_match:
        return {**empty, "size_match": False, "n_total": h_t * w_t}

    flat_t = [target[r][c] for r in range(h_t) for c in range(w_t)]
    flat_p = [pred[r][c]   for r in range(h_t) for c in range(w_t)]
    n_total   = len(flat_t)
    n_correct = sum(a == b for a, b in zip(flat_t, flat_p))
    pixel_acc = n_correct / n_total

    bg = Counter(flat_t).most_common(1)[0][0]
    fg = [i for i, v in enumerate(flat_t) if v != bg]
    color_acc = sum(flat_t[i] == flat_p[i] for i in fg) / len(fg) if fg else 1.0

    n_wrong_color = sum(1 for a, b in zip(flat_t, flat_p) if a != b and b != bg)
    n_wrong_pos   = sum(1 for a, b in zip(flat_t, flat_p) if a != bg and b == bg)

    return {
        "exact_match"  : n_correct == n_total,
        "pixel_acc"    : pixel_acc,
        "color_acc"    : color_acc,
        "size_match"   : size_match,
        "n_correct"    : n_correct,
        "n_total"      : n_total,
        "n_wrong_color": n_wrong_color,
        "n_wrong_pos"  : n_wrong_pos,
    }


def draw_metrics_panel(ax, metrics, task_id):
    ax.set_facecolor("#1a1a2e")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for spine in ax.spines.values():
        spine.set_visible(False)

    def fmt_bool(v):
        return ("✓  YES", "#2ECC40") if v else ("✗  NO", "#FF4136")

    def fmt_pct(v):
        col = "#2ECC40" if v >= 0.9 else "#FFDC00" if v >= 0.5 else "#FF4136"
        return f"{v*100:.1f}%", col

    rows = [
        ("Exact match",    fmt_bool(metrics["exact_match"])),
        ("Pixel accuracy", fmt_pct(metrics["pixel_acc"])),
        ("Foreground acc", fmt_pct(metrics["color_acc"])),
        ("Correct pixels", (f"{metrics['n_correct']}/{metrics['n_total']}", "#AAAAAA")),
        ("Wrong color",    (str(metrics["n_wrong_color"]), "#FF851B")),
        ("Missing cells",  (str(metrics["n_wrong_pos"]),   "#7FDBFF")),
        ("Size match",     fmt_bool(metrics["size_match"])),
    ]

    top = 0.88; divider_y = 0.78; data_top = 0.72
    row_h = 0.68 / len(rows)

    ax.text(0.02, top, "Metric",           color="white",   fontsize=8,
            fontweight="bold", va="center", transform=ax.transAxes)
    ax.text(0.55, top, "Graph TRM Hybrid", color="#2ECC40", fontsize=8,
            fontweight="bold", va="center", transform=ax.transAxes)
    ax.axhline(y=divider_y, xmin=0.01, xmax=0.99, color="#555555", linewidth=0.8)

    for i, (label, (val_txt, val_col)) in enumerate(rows):
        y = data_top - i * row_h
        if i % 2 == 0:
            ax.axhspan(y - row_h*0.45, y + row_h*0.45,
                       xmin=0.0, xmax=1.0, color="#22223a", zorder=0)
        ax.text(0.02, y, label,   color="#cccccc", fontsize=7.5,
                va="center", transform=ax.transAxes)
        ax.text(0.55, y, val_txt, color=val_col,   fontsize=7.5,
                va="center", transform=ax.transAxes)


# ─────────────────────────────────────────────────────────────────────────────
# PER-TASK PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_task(sample: Dict, model: GraphTinyRecursiveModel,
              device: str, save_dir=None):
    task_id     = sample["task_id"]
    n_train     = len(sample["train_inputs_raw"])
    test_in_raw = sample["test_input_raw"]
    test_gt     = sample["test_output_raw"]

    trm_pred = trm_predict(sample, model, device)
    metrics  = compute_metrics(trm_pred, test_gt)
    exact    = metrics["exact_match"]

    n_cols = max(n_train, 4)
    fig_w  = max(n_cols * 2.2, 10)

    height_ratios = [0.22, 2.2, 0.22, 2.2, 0.22, 2.2, 1.8]
    fig = plt.figure(figsize=(fig_w, sum(height_ratios) + 1.0))
    fig.patch.set_facecolor("#111111")

    gs = gridspec.GridSpec(7, n_cols, figure=fig,
                           height_ratios=height_ratios, hspace=0.35, wspace=0.15)

    def add_row_label(row, text, color="#aaaaaa"):
        lax = fig.add_subplot(gs[row, :])
        lax.set_facecolor("#111111")
        lax.set_xticks([]); lax.set_yticks([])
        for sp in lax.spines.values():
            sp.set_visible(False)
        lax.text(0.01, 0.5, text, color=color, fontsize=8,
                 fontweight="bold", va="center", transform=lax.transAxes)

    add_row_label(0, f"  DEMONSTRATIONS — INPUT  ({n_train} pair{'s' if n_train!=1 else ''})")
    for col in range(n_cols):
        ax = fig.add_subplot(gs[1, col])
        if col < n_train:
            draw_grid(ax, sample["train_inputs_raw"][col], title=f"Demo {col+1}: Input")
        else:
            draw_blank(ax, "No Demo")
            ax.set_title("(unused)", fontsize=6, color="#444444", pad=3)

    add_row_label(2, f"  DEMONSTRATIONS — OUTPUT  ({n_train} pair{'s' if n_train!=1 else ''})")
    for col in range(n_cols):
        ax = fig.add_subplot(gs[3, col])
        if col < n_train:
            draw_grid(ax, sample["train_outputs_raw"][col], title=f"Demo {col+1}: Output")
        else:
            draw_blank(ax, "No Demo")
            ax.set_title("(unused)", fontsize=6, color="#444444", pad=3)

    add_row_label(4, "  TEST  —  Input  |  Graph TRM Hybrid  |  Ground Truth  |  Diff Map")

    panels = [
        ("Test Input",
         lambda ax: draw_grid(ax, test_in_raw, title="Test Input")),
        (f"Graph TRM  {'✓' if exact else '✗'}",
         lambda ax: draw_grid(ax, trm_pred,
                              title=f"Graph TRM  {'✓' if exact else '✗'}",
                              title_color="#2ECC40" if exact else "#FF4136")
         if trm_pred is not None
         else lambda ax: draw_blank(ax, "Prediction\nfailed")),
        ("Ground Truth",
         lambda ax: draw_grid(ax, test_gt, title="Ground Truth")),
        ("Diff Map",
         lambda ax: draw_diff(ax, trm_pred, test_gt, title="Diff Map")),
    ]

    for col in range(n_cols):
        ax = fig.add_subplot(gs[5, col])
        if col < len(panels):
            _, draw_fn = panels[col]
            draw_fn(ax)
        else:
            draw_blank(ax, "")
            ax.set_title("—", fontsize=7, color="#555555", pad=3)

    metrics_ax = fig.add_subplot(gs[6, :])
    draw_metrics_panel(metrics_ax, metrics, task_id)

    verdict = "✓ EXACT MATCH" if exact else "✗ NO MATCH"
    fig.suptitle(
        f"Task: {task_id}  [Graph TRM Hybrid]\n"
        f"{verdict}  —  {metrics['pixel_acc']*100:.1f}% pixel accuracy  |  "
        f"{metrics['n_correct']}/{metrics['n_total']} pixels correct",
        fontsize=9, fontweight="bold", color="white", y=1.02,
    )

    if save_dir:
        path = os.path.join(save_dir, f"{task_id}.png")
        plt.savefig(path, dpi=130, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"  Saved: {path}")
    else:
        plt.show()

    return exact


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary(results: List[Dict], save_dir=None):
    task_ids = [r["task_id"]   for r in results]
    exact    = [r["exact"]     for r in results]
    px_acc   = [r["pixel_acc"] for r in results]

    n       = len(results)
    acc     = 100.0 * sum(exact) / max(n, 1)
    mean_px = 100.0 * sum(px_acc) / max(n, 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, n * 0.22 + 2)))

    ax = axes[0]
    bars = ax.bar(
        ["Exact Match", "Mean Pixel Acc"],
        [acc, mean_px],
        color=["#2ECC40", "#4C9BE8"],
        width=0.5, edgecolor="white",
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title(f"Graph TRM Hybrid — {n} tasks", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, [acc, mean_px]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax2 = axes[1]
    ax2.set_title("Per-task  (green=exact, orange=partial >50%, red=wrong)",
                  fontsize=9, fontweight="bold")
    for i, (ex, px, tid) in enumerate(zip(exact, px_acc, task_ids)):
        y = n - 1 - i
        color = "#2ECC40" if ex else ("#FF851B" if px > 0.5 else "#FF4136")
        ax2.barh(y, 0.9, left=0, height=0.8, color=color, edgecolor="white")
        ax2.text(0.92, y, f"{px*100:.0f}%", va="center", ha="left",
                 fontsize=5.5, color="#cccccc")
        ax2.text(-0.02, y, tid[:18], va="center", ha="right", fontsize=5.5)

    ax2.set_xlim(-0.02, 1.15)
    ax2.set_ylim(-0.5, n + 0.5)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.legend(handles=[
        mpatches.Patch(color="#2ECC40", label="Exact match ✓"),
        mpatches.Patch(color="#FF851B", label="Partial (>50% pixels)"),
        mpatches.Patch(color="#FF4136", label="Wrong ✗"),
    ], loc="lower right", fontsize=7)

    plt.tight_layout()

    if save_dir:
        path = os.path.join(save_dir, "_summary_trm_graph_hybrid.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"\nSummary saved: {path}")
    else:
        plt.show()

    print(f"\n{'='*55}")
    print(f"GRAPH TRM HYBRID RESULTS  ({n} tasks evaluated)")
    print(f"{'='*55}")
    print(f"  Exact match accuracy : {sum(exact):3d}/{n}  ({acc:.1f}%)")
    print(f"  Mean pixel accuracy  : {mean_px:.1f}%")
    print(f"{'='*55}")

    correct = [tid for tid, ok in zip(task_ids, exact) if ok]
    if correct:
        print(f"\n  Tasks solved exactly:")
        for tid in correct:
            print(f"    {tid}")
    else:
        print(f"\n  No tasks solved exactly yet.")

    partial = [(tid, px) for tid, ok, px in zip(task_ids, exact, px_acc)
               if not ok and px > 0.5]
    if partial:
        print(f"\n  Tasks with >50% pixel accuracy (partial progress):")
        for tid, px in sorted(partial, key=lambda x: -x[1])[:10]:
            print(f"    {tid}  ({px*100:.1f}%)")


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

    # ── load checkpoint ───────────────────────────────────────────────────
    print("Loading Graph TRM Hybrid checkpoint...")
    ckpt_path = os.path.join(script_dir, TRM_CKPT)
    ckpt      = torch.load(ckpt_path, map_location=DEVICE)

    model = GraphTinyRecursiveModel(
        hidden_size  = ckpt.get("hidden_size",  HIDDEN_SIZE),
        n_heads      = N_HEADS,
        ff_dim       = FF_DIM,
        dropout      = DROPOUT,
        n_recursions = ckpt.get("n_recursions", N_RECURSIONS),
        t_inner      = T_INNER,
        n_sup        = N_SUP,
        max_nodes    = MAX_NODES,
        num_colors   = NUM_COLORS,
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"])

    # Apply EMA weights if available
    if ckpt.get("ema_shadow") is not None:
        print("  Applying EMA weights...")
        for name, param in model.named_parameters():
            if name in ckpt["ema_shadow"]:
                param.data.copy_(ckpt["ema_shadow"][name])

    model.eval()
    print(f"  Loaded  (epoch {ckpt.get('epoch','?')}, "
          f"best val loss: {ckpt.get('best_val_loss','N/A')})\n")

    # ── build samples ─────────────────────────────────────────────────────
    print("Building task samples...")
    all_samples = build_samples(TRAIN_PATH)

    if not all_samples:
        print("No samples built. Check TRAIN_PATH.")
        return

    random.shuffle(all_samples)
    plot_samples = all_samples[:N_TASKS_TO_PLOT]
    eval_samples = all_samples if N_TASKS_TO_EVAL is None else all_samples[:N_TASKS_TO_EVAL]

    # ── per-task plots ────────────────────────────────────────────────────
    print(f"\nGenerating detail plots for {len(plot_samples)} tasks...")
    for i, sample in enumerate(plot_samples):
        print(f"  [{i+1}/{len(plot_samples)}] {sample['task_id']}")
        plot_task(sample, model, DEVICE, save_dir=out_dir)

    # ── summary ───────────────────────────────────────────────────────────
    print(f"\nEvaluating {len(eval_samples)} tasks for summary...")
    results = []
    for i, sample in enumerate(eval_samples):
        pred    = trm_predict(sample, model, DEVICE)
        gt      = sample["test_output_raw"]
        metrics = compute_metrics(pred, gt)
        results.append({
            "task_id"   : sample["task_id"],
            "exact"     : metrics["exact_match"],
            "pixel_acc" : metrics["pixel_acc"],
        })
        if (i + 1) % 50 == 0:
            print(f"  Evaluated {i+1}/{len(eval_samples)}...")

    plot_summary(results, save_dir=out_dir)

    if not SAVE_FIGURES:
        plt.show()


if __name__ == "__main__":
    main()