import os
import json
import math
import random
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data, Batch

from OLD2.hybrid_object import grid_to_graph
from OLD2.gat_vae_hybrid import GATVAE
from OLD2.graph_decoder_hybrid import gat_vae_loss


# ============================================================================
# CONFIG
# ============================================================================

TRAIN_PATH = r"C:\Users\tedhun\OneDrive - Wanzl GmbH & Co. KGaA\Microsoft Teams Chat Files\outputs\Documents\GS\ARC\Assignment10\Builder\data\training"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42

BATCH_SIZE = 2
EPOCHS = 20
LR = 1e-3

# If True, skip examples where either input or output graph is too tiny.
SKIP_TINY_GRAPHS = True

# Optional cap for quick testing. Set to None to use all files.
MAX_FILES = 10

# Optional node cap for stability / speed.
MAX_NODE_CAP = 100

# Number of sample ARC pairs to plot before training begins
NUM_PREVIEW_SAMPLES = 3


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# JSON / GRID HELPERS
# ============================================================================

def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def to_dsl_grid(grid: List[List[int]]) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(row) for row in grid)


# ============================================================================
# VISUALIZATION
# ============================================================================

def show_grid(grid, title="Grid"):
    plt.figure(figsize=(4, 4))
    plt.imshow(grid, cmap="tab10", vmin=0, vmax=9)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()


def show_input_output_pair(input_grid, output_grid, file_name="", pair_idx=0):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(input_grid, cmap="tab10", vmin=0, vmax=9)
    axes[0].set_title(f"INPUT\n{file_name} pair {pair_idx}")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].imshow(output_grid, cmap="tab10", vmin=0, vmax=9)
    axes[1].set_title("TARGET OUTPUT")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    plt.tight_layout()
    plt.show()


def preview_samples(samples: List[Dict], num_to_show: int = 3) -> None:
    print("\nPreviewing sample grids...")
    for s in samples[:num_to_show]:
        show_input_output_pair(
            s["input_grid"],
            s["output_grid"],
            file_name=s["file_name"],
            pair_idx=s["train_idx"]
        )
        print(
            f"{s['file_name']} | pair {s['train_idx']} | "
            f"in_nodes={s['input_num_nodes']} in_edges={s['input_num_edges']} | "
            f"out_nodes={s['output_num_nodes']} out_edges={s['output_num_edges']}"
        )


# ============================================================================
# GRAPH -> PYG
# ============================================================================

def graph_to_pyg_data(graph: dict) -> Data:
    """
    Convert graph_builder_minimal_hybrid.py output into a PyTorch Geometric
    Data object.

    Expected graph keys:
      - node_features : [N, 110]
      - edge_features : [E, 5]
      - edge_index    : [2, E]

    HYBRID NODE FEATURE LAYOUT
    --------------------------
      first 10 dims   = object color multi-hot
      next 100 dims   = padded flattened 10x10 shape mask

    HYBRID EDGE FEATURE LAYOUT
    --------------------------
      [0] dx
      [1] dy
      [2] touching
      [3] same_row
      [4] same_col
    """
    x = torch.tensor(graph["node_features"], dtype=torch.float32)
    edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(graph["edge_features"], dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ============================================================================
# TARGET BUILDING
# ============================================================================

def build_targets(
    target_graph: dict,
    max_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build padded HYBRID targets for gat_vae_loss(...).

    IMPORTANT:
    ----------
    These targets come from the OUTPUT graph, not the INPUT graph.

    Output shapes:
      target_color      : [max_nodes]                long
      target_node_shape : [max_nodes, 100]           float
      target_edge_cont  : [max_nodes, max_nodes, 2]  float
      target_edge_bin   : [max_nodes, max_nodes, 3]  float

    HYBRID NODE FEATURES
    --------------------
      first 10 dims  = color multi-hot
      next 100 dims  = padded flattened shape mask

    HYBRID EDGE FEATURES
    --------------------
      [0] dx
      [1] dy
      [2] touching
      [3] same_row
      [4] same_col
    """
    node_feats = torch.tensor(target_graph["node_features"], dtype=torch.float32)
    edge_feats = torch.tensor(target_graph["edge_features"], dtype=torch.float32)
    edge_index = torch.tensor(target_graph["edge_index"], dtype=torch.long)

    n_nodes = node_feats.size(0)

    # Color target
    target_color = torch.zeros(max_nodes, dtype=torch.long)
    target_color[:n_nodes] = node_feats[:, :10].argmax(dim=1)

    # Shape target
    target_node_shape = torch.zeros(max_nodes, 100, dtype=torch.float32)
    target_node_shape[:n_nodes] = node_feats[:, 10:]

    # Dense edge targets
    target_edge_cont = torch.zeros(max_nodes, max_nodes, 2, dtype=torch.float32)
    target_edge_bin = torch.zeros(max_nodes, max_nodes, 3, dtype=torch.float32)

    for e in range(edge_feats.size(0)):
        s = int(edge_index[0, e].item())
        d = int(edge_index[1, e].item())
        ef = edge_feats[e]

        target_edge_cont[s, d] = torch.tensor([ef[0], ef[1]], dtype=torch.float32)
        target_edge_bin[s, d] = torch.tensor([ef[2], ef[3], ef[4]], dtype=torch.float32)

    return target_color, target_node_shape, target_edge_cont, target_edge_bin


# ============================================================================
# DATASET BUILD
# ============================================================================

def collect_graph_samples(train_path: str, max_files: int = None) -> List[Dict]:
    """
    Build one sample per ARC train pair:
        input_grid  -> input_graph
        output_grid -> output_graph

    This is the ARC change:
    the model now learns INPUT GRAPH -> OUTPUT GRAPH
    instead of INPUT GRAPH -> INPUT GRAPH.
    """
    samples = []
    files = sorted([f for f in os.listdir(train_path) if f.endswith(".json")])

    if max_files is not None:
        files = files[:max_files]

    for fname in files:
        fpath = os.path.join(train_path, fname)
        task = load_json(fpath)

        for i, pair in enumerate(task.get("train", [])):
            raw_input_grid = pair["input"]
            raw_output_grid = pair["output"]

            input_grid = to_dsl_grid(raw_input_grid)
            output_grid = to_dsl_grid(raw_output_grid)

            input_graph = grid_to_graph(input_grid)
            output_graph = grid_to_graph(output_grid)

            in_nodes = len(input_graph["nodes"])
            in_edges = len(input_graph["edges"])
            out_nodes = len(output_graph["nodes"])
            out_edges = len(output_graph["edges"])

            if SKIP_TINY_GRAPHS:
                if in_nodes < 2 or in_edges == 0:
                    continue
                if out_nodes < 1:
                    continue

            if MAX_NODE_CAP is not None:
                if in_nodes > MAX_NODE_CAP or out_nodes > MAX_NODE_CAP:
                    continue

            samples.append({
                "file_name": fname,
                "train_idx": i,

                "input_grid": raw_input_grid,
                "output_grid": raw_output_grid,

                "input_graph": input_graph,
                "output_graph": output_graph,

                "input_num_nodes": in_nodes,
                "input_num_edges": in_edges,
                "output_num_nodes": out_nodes,
                "output_num_edges": out_edges,
            })

    return samples


def compute_max_nodes(samples: List[Dict]) -> int:
    """
    We care most about OUTPUT graph size, since that is what the model decodes.
    """
    if not samples:
        raise ValueError("No valid graph samples were found.")
    return max(s["output_num_nodes"] for s in samples)


def make_batch(samples: List[Dict], max_nodes: int, device: str):
    """
    Convert a list of samples into:
      - PyG Batch built from INPUT graphs
      - stacked target tensors built from OUTPUT graphs
    """
    data_list = []
    color_targets = []
    shape_targets = []
    edge_cont_targets = []
    edge_bin_targets = []

    for s in samples:
        input_graph = s["input_graph"]
        output_graph = s["output_graph"]

        # Model input = INPUT graph
        data_list.append(graph_to_pyg_data(input_graph))

        # Training target = OUTPUT graph
        target_color, target_node_shape, target_edge_cont, target_edge_bin = build_targets(
            output_graph, max_nodes
        )

        color_targets.append(target_color)
        shape_targets.append(target_node_shape)
        edge_cont_targets.append(target_edge_cont)
        edge_bin_targets.append(target_edge_bin)

    batch = Batch.from_data_list(data_list).to(device)

    target_color = torch.stack(color_targets).to(device)
    target_node_shape = torch.stack(shape_targets).to(device)
    target_edge_cont = torch.stack(edge_cont_targets).to(device)
    target_edge_bin = torch.stack(edge_bin_targets).to(device)

    return batch, target_color, target_node_shape, target_edge_cont, target_edge_bin


def chunk_list(items: List, batch_size: int) -> List[List]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


# ============================================================================
# TRAINING
# ============================================================================

def train_one_epoch(model, optimizer, samples, max_nodes, batch_size, device):
    model.train()
    random.shuffle(samples)

    total_loss = 0.0
    total_batches = 0

    for mini in chunk_list(samples, batch_size):
        batch, target_color, target_node_shape, target_edge_cont, target_edge_bin = make_batch(
            mini, max_nodes=max_nodes, device=device
        )

        optimizer.zero_grad()

        # Input to model = INPUT graph batch
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        # Loss target = OUTPUT graph tensors
        loss, breakdown = gat_vae_loss(
            out,
            target_color,
            target_node_shape,
            target_edge_cont,
            target_edge_bin,
        )

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

        if total_batches % 50 == 0:
            print(f"Batch {total_batches} | loss={loss.item():.4f}")

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate(model, samples, max_nodes, batch_size, device):
    model.eval()

    total_loss = 0.0
    total_batches = 0

    for mini in chunk_list(samples, batch_size):
        batch, target_color, target_node_shape, target_edge_cont, target_edge_bin = make_batch(
            mini, max_nodes=max_nodes, device=device
        )

        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        loss, breakdown = gat_vae_loss(
            out,
            target_color,
            target_node_shape,
            target_edge_cont,
            target_edge_bin,
        )

        total_loss += float(loss.item())
        total_batches += 1

    return total_loss / max(total_batches, 1)


# ============================================================================
# OPTIONAL DEBUG PRINTS
# ============================================================================

def print_sample_summary(samples: List[Dict], num_to_show: int = 5) -> None:
    print("\nSample summary:")
    for s in samples[:num_to_show]:
        print(
            f"  {s['file_name']} | pair {s['train_idx']} | "
            f"in_nodes={s['input_num_nodes']} in_edges={s['input_num_edges']} | "
            f"out_nodes={s['output_num_nodes']} out_edges={s['output_num_edges']}"
        )


# ============================================================================
# MAIN
# ============================================================================

def main():
    set_seed(SEED)

    print(f"Using device: {DEVICE}")
    print(f"Loading samples from: {TRAIN_PATH}")

    samples = collect_graph_samples(TRAIN_PATH, max_files=MAX_FILES)

    if not samples:
        raise RuntimeError("No usable samples were built. Check TRAIN_PATH or graph construction.")

    print(f"Total graph samples: {len(samples)}")
    print_sample_summary(samples, num_to_show=5)
    preview_samples(samples, num_to_show=NUM_PREVIEW_SAMPLES)

    max_nodes = compute_max_nodes(samples)
    print(f"Max OUTPUT nodes across dataset: {max_nodes}")

    # Simple split
    random.shuffle(samples)
    split_idx = max(1, int(0.9 * len(samples)))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:] if split_idx < len(samples) else samples[:1]

    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples:   {len(val_samples)}")

    # ------------------------------------------------------------------------
    # Model input dimensions match HYBRID builder:
    #   node_in_dim    = 110
    #   edge_in_dim    = 5
    #   node_shape_dim = 100
    #
    # INPUT  = input_graph
    # TARGET = output_graph
    # ------------------------------------------------------------------------
    model = GATVAE(
        max_nodes=max_nodes,
        node_in_dim=110,
        edge_in_dim=5,
        latent_dim=256,
        dec_hidden=256,
        node_shape_dim=100,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = math.inf
    save_path = "gat_vae_hybrid_arc_solver_best.pt"

    print("\nStarting ARC-style graph transformation training...\n")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model, optimizer, train_samples, max_nodes, BATCH_SIZE, DEVICE
        )
        val_loss = evaluate(
            model, val_samples, max_nodes, BATCH_SIZE, DEVICE
        )

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "max_nodes": max_nodes,
                    "best_val_loss": best_val,
                },
                save_path,
            )
            print(f"  Saved best model -> {save_path}")

    print("\nTraining complete.")
    print(f"Best val loss: {best_val:.6f}")


if __name__ == "__main__":
    main()