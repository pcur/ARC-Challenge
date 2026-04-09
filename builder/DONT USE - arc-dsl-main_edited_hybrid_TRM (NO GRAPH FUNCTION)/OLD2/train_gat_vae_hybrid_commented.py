import os
import json
import math
import random
from typing import List, Tuple, Dict

import torch
from torch_geometric.data import Data, Batch

# Converts an ARC grid into your custom object-graph representation.
# This is the "builder" step: grid -> nodes/edges/features.
from OLD2.hybrid_object import grid_to_graph

# Your hybrid graph model:
# - GAT encoder
# - VAE latent bottleneck
# - decoder that reconstructs node/edge information
from OLD2.gat_vae_hybrid import GATVAE

# Loss function for the hybrid decoder output
# (reconstruction + likely KL / latent regularization inside)
from OLD2.graph_decoder_hybrid import gat_vae_loss


# ============================================================================
# CONFIG
# ============================================================================

# Folder containing ARC training JSON task files.
# Each file usually contains:
#   {
#     "train": [...],
#     "test": [...]
#   }
TRAIN_PATH = r"C:\Users\tedhun\OneDrive - Wanzl GmbH & Co. KGaA\Microsoft Teams Chat Files\outputs\Documents\GS\ARC\Assignment10\Builder\data\training"

# Use GPU if available, otherwise fall back to CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Random seed for reproducibility.
SEED = 42

# Small batch size is safer here because:
# 1) graphs can vary in size
# 2) your decoder becomes dense in max_nodes x max_nodes
# 3) memory usage can rise quickly
BATCH_SIZE = 2

# Number of training epochs
EPOCHS = 20

# Learning rate for Adam optimizer
LR = 1e-3

# If True, skip graphs with fewer than 2 nodes or zero edges.
# Why?
# Because the encoder/graph reasoning is not very meaningful for tiny graphs,
# and some hybrid graph models behave awkwardly with degenerate edge structure.
SKIP_TINY_GRAPHS = True

# Optional quick-debug limit on number of JSON files to use.
# Set to None if you want the full dataset.
MAX_FILES = 10

# Optional safety cap on graph size.
# Since the decoder predicts dense structures over max_nodes x max_nodes,
# huge graphs can become slow and memory-heavy.
MAX_NODE_CAP = 100


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for repeatable runs.

    This affects:
    - Python random shuffling
    - Torch weight initialization / randomness
    - CUDA randomness (if running on GPU)

    Important note:
    This improves reproducibility, though some GPU ops can still have small
    nondeterministic differences depending on backend details.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# JSON / GRID HELPERS
# ============================================================================

def load_json(path: str) -> dict:
    """
    Load one ARC task JSON file from disk.

    Example structure:
    {
      "train": [
          {"input": [[...]], "output": [[...]]},
          ...
      ],
      "test": [...]
    }
    """
    with open(path, "r") as f:
        return json.load(f)


def to_dsl_grid(grid: List[List[int]]) -> Tuple[Tuple[int, ...], ...]:
    """
    Convert a list-of-lists grid into a tuple-of-tuples grid.

    Why do this?
    - Some ARC / DSL-style utilities prefer immutable structures
    - Tuple-based grids are hashable and safer to pass around when you want
      stable, read-only behavior
    """
    return tuple(tuple(row) for row in grid)


# ============================================================================
# GRAPH -> PYG
# ============================================================================

def graph_to_pyg_data(graph: dict) -> Data:
    """
    Convert your custom graph dictionary into a PyTorch Geometric Data object.

    Expected graph dictionary fields:
      - node_features : shape [N, 110]
      - edge_features : shape [E, 5]
      - edge_index    : shape [2, E]

    Node feature layout:
      first 10 dims   = object color multi-hot
      next 100 dims   = padded flattened 10x10 shape mask

    Edge feature layout:
      [0] dx
      [1] dy
      [2] touching
      [3] same_row
      [4] same_col

    PyG expects:
      x          = node feature matrix
      edge_index = connectivity in COO format
      edge_attr  = per-edge feature matrix
    """

    # Convert node feature list/array into float tensor.
    # Shape: [num_nodes, 110]
    x = torch.tensor(graph["node_features"], dtype=torch.float32)

    # Convert edge index into integer tensor.
    # Shape: [2, num_edges]
    # Row 0 = source nodes, Row 1 = destination nodes
    edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)

    # Convert edge features into float tensor.
    # Shape: [num_edges, 5]
    edge_attr = torch.tensor(graph["edge_features"], dtype=torch.float32)

    # Return PyG Data object that your GAT encoder can consume.
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def build_targets(
    graph: dict,
    max_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build padded training targets from one graph.

    This is needed because your model appears to decode into fixed-size tensors
    based on max_nodes across the dataset, even though each graph itself has a
    variable number of nodes.

    Output shapes:
      target_color      : [max_nodes]                long
      target_node_shape : [max_nodes, 100]           float
      target_edge_cont  : [max_nodes, max_nodes, 2]  float
      target_edge_bin   : [max_nodes, max_nodes, 3]  float

    Why padding is needed:
    - each ARC graph has a different number of objects
    - batching variable-sized decoder outputs is awkward
    - so everything is padded to the same max_nodes size

    HYBRID NODE FEATURE FORMAT:
      [0:10]   color multi-hot
      [10:110] flattened 10x10 shape mask

    HYBRID EDGE FEATURE FORMAT:
      [0] dx
      [1] dy
      [2] touching
      [3] same_row
      [4] same_col
    """

    # Convert graph contents into tensors for easy slicing/manipulation.
    node_feats = torch.tensor(graph["node_features"], dtype=torch.float32)   # [N, 110]
    edge_feats = torch.tensor(graph["edge_features"], dtype=torch.float32)   # [E, 5]
    edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)         # [2, E]

    # Number of actual nodes in this graph (before padding)
    n_nodes = node_feats.size(0)

    # ------------------------------------------------------------------------
    # NODE COLOR TARGET
    # ------------------------------------------------------------------------
    # We allocate a full-length target vector of size max_nodes.
    # Any positions beyond the real graph size remain zero-padded.
    target_color = torch.zeros(max_nodes, dtype=torch.long)

    # First 10 dims are the color multi-hot encoding.
    # Since ARC objects are typically single-color, argmax gives the class index.
    # Example:
    #   [0,0,1,0,0,0,0,0,0,0] -> class 2
    target_color[:n_nodes] = node_feats[:, :10].argmax(dim=1)

    # ------------------------------------------------------------------------
    # NODE SHAPE TARGET
    # ------------------------------------------------------------------------
    # Shape target stores the flattened 10x10 mask for each node/object.
    # Again, padded up to max_nodes.
    target_node_shape = torch.zeros(max_nodes, 100, dtype=torch.float32)
    target_node_shape[:n_nodes] = node_feats[:, 10:]

    # ------------------------------------------------------------------------
    # EDGE TARGETS
    # ------------------------------------------------------------------------
    # The decoder appears to predict dense pairwise edge information,
    # meaning it effectively reasons over all possible node pairs
    # in a [max_nodes, max_nodes] grid.
    #
    # So we build dense targets rather than sparse edge lists.
    #
    # Continuous edge channels:
    #   [dx, dy]
    #
    # Binary edge channels:
    #   [touching, same_row, same_col]
    target_edge_cont = torch.zeros(max_nodes, max_nodes, 2, dtype=torch.float32)
    target_edge_bin = torch.zeros(max_nodes, max_nodes, 3, dtype=torch.float32)

    # Fill only the actual edges present in the graph.
    # All other node pairs remain zero by default.
    for e in range(edge_feats.size(0)):
        # Source node index for edge e
        s = int(edge_index[0, e].item())

        # Destination node index for edge e
        d = int(edge_index[1, e].item())

        # Edge feature vector [dx, dy, touching, same_row, same_col]
        ef = edge_feats[e]

        # Store continuous part [dx, dy]
        target_edge_cont[s, d] = torch.tensor(
            [ef[0], ef[1]],
            dtype=torch.float32
        )

        # Store binary relational part [touching, same_row, same_col]
        target_edge_bin[s, d] = torch.tensor(
            [ef[2], ef[3], ef[4]],
            dtype=torch.float32
        )

    return target_color, target_node_shape, target_edge_cont, target_edge_bin


# ============================================================================
# DATASET BUILD
# ============================================================================

def collect_graph_samples(train_path: str, max_files: int = None) -> List[Dict]:
    """
    Build one graph sample for every train input grid in every ARC JSON file.

    Important:
    - This uses ONLY the train input grids, not outputs
    - Each train pair's input becomes one graph sample
    - Metadata is kept so you can later trace which file / train index it came from

    Returned sample format:
      {
        "file_name": ...,
        "train_idx": ...,
        "grid": ...,
        "graph": ...,
        "num_nodes": ...,
        "num_edges": ...
      }
    """
    samples = []

    # Gather only JSON files from the training folder.
    files = sorted([f for f in os.listdir(train_path) if f.endswith(".json")])

    # Optional debug truncation so you do not load everything.
    if max_files is not None:
        files = files[:max_files]

    # Loop through ARC task files
    for fname in files:
        fpath = os.path.join(train_path, fname)

        # Load one ARC task JSON
        task = load_json(fpath)

        # For each training example in the ARC task...
        for i, pair in enumerate(task.get("train", [])):
            # Raw list-of-lists input grid
            raw_grid = pair["input"]

            # Convert to immutable tuple-of-tuples version
            grid = to_dsl_grid(raw_grid)

            # Build graph from grid using your custom object builder
            graph = grid_to_graph(grid)

            # Number of nodes/edges in the constructed graph
            n_nodes = len(graph["nodes"])
            n_edges = len(graph["edges"])

            # If tiny-graph skipping is enabled, reject graphs that have too little
            # structure to be meaningful for graph learning.
            if SKIP_TINY_GRAPHS and (n_nodes < 2 or n_edges == 0):
                continue

            # Skip graphs that exceed size cap, if cap is active.
            # This avoids very large dense decoder targets.
            if MAX_NODE_CAP is not None and n_nodes > MAX_NODE_CAP:
                continue

            # Store sample with useful metadata
            samples.append({
                "file_name": fname,
                "train_idx": i,
                "grid": raw_grid,
                "graph": graph,
                "num_nodes": n_nodes,
                "num_edges": n_edges,
            })

    return samples


def compute_max_nodes(samples: List[Dict]) -> int:
    """
    Compute the maximum number of nodes over all collected samples.

    This value is very important because:
    - it sets the padded target size
    - it sets the decoder's effective output size
    - it strongly affects memory usage
    """
    if not samples:
        raise ValueError("No valid graph samples were found.")

    return max(s["num_nodes"] for s in samples)


def make_batch(samples: List[Dict], max_nodes: int, device: str):
    """
    Convert a list of sample dictionaries into:
      1) a PyG Batch for the encoder
      2) stacked padded target tensors for the decoder/loss

    Output:
      batch              : PyG Batch object
      target_color       : [B, max_nodes]
      target_node_shape  : [B, max_nodes, 100]
      target_edge_cont   : [B, max_nodes, max_nodes, 2]
      target_edge_bin    : [B, max_nodes, max_nodes, 3]

    Why not just batch raw graphs directly?
    Because the model needs BOTH:
    - sparse graph form for the encoder
    - dense padded targets for the decoder loss
    """
    data_list = []
    color_targets = []
    shape_targets = []
    edge_cont_targets = []
    edge_bin_targets = []

    # Process each graph sample individually, then stack them
    for s in samples:
        graph = s["graph"]

        # Build sparse PyG Data object for the encoder
        data_list.append(graph_to_pyg_data(graph))

        # Build fixed-size padded targets for the decoder/loss
        target_color, target_node_shape, target_edge_cont, target_edge_bin = build_targets(
            graph, max_nodes
        )

        color_targets.append(target_color)
        shape_targets.append(target_node_shape)
        edge_cont_targets.append(target_edge_cont)
        edge_bin_targets.append(target_edge_bin)

    # Combine variable-sized sparse graphs into one PyG batch.
    # batch.batch will track which node belongs to which graph.
    batch = Batch.from_data_list(data_list).to(device)

    # Stack fixed-size targets into normal batch tensors
    target_color = torch.stack(color_targets).to(device)
    target_node_shape = torch.stack(shape_targets).to(device)
    target_edge_cont = torch.stack(edge_cont_targets).to(device)
    target_edge_bin = torch.stack(edge_bin_targets).to(device)

    return batch, target_color, target_node_shape, target_edge_cont, target_edge_bin


def chunk_list(items: List, batch_size: int) -> List[List]:
    """
    Split a list into consecutive chunks of size batch_size.

    Example:
      [a,b,c,d,e], batch_size=2 -> [[a,b], [c,d], [e]]
    """
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


# ============================================================================
# TRAINING
# ============================================================================

def train_one_epoch(model, optimizer, samples, max_nodes, batch_size, device):
    """
    Run one full training epoch.

    Steps per mini-batch:
      1) build sparse graph batch + dense targets
      2) forward pass through model
      3) compute loss
      4) backpropagate gradients
      5) optimizer step

    Returns:
      average loss across all batches in the epoch
    """
    model.train()

    # Shuffle training samples each epoch for better stochastic training
    random.shuffle(samples)

    total_loss = 0.0
    total_batches = 0

    # Loop through mini-batches
    for mini in chunk_list(samples, batch_size):
        # Build encoder inputs + decoder targets
        batch, target_color, target_node_shape, target_edge_cont, target_edge_bin = make_batch(
            mini, max_nodes=max_nodes, device=device
        )

        # Clear old gradients before current backward pass
        optimizer.zero_grad()

        # Forward pass through model
        #
        # batch.x         = all node features in the merged PyG batch
        # batch.edge_index= all edges in merged graph form
        # batch.edge_attr = per-edge features
        # batch.batch     = graph-membership index for each node
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        # Compute total training loss
        # breakdown may contain components like:
        # - node color reconstruction loss
        # - shape reconstruction loss
        # - edge reconstruction loss
        # - KL divergence
        loss, breakdown = gat_vae_loss(
            out,
            target_color,
            target_node_shape,
            target_edge_cont,
            target_edge_bin,
        )

        # Backpropagate gradients
        loss.backward()

        # Update model weights
        optimizer.step()

        # Track total loss statistics
        total_loss += float(loss.item())
        total_batches += 1

        # Periodic progress print
        if total_batches % 50 == 0:
            print(f"Batch {total_batches} | loss={loss.item():.4f}")

        # If you want every batch printed, uncomment below:
        # print(f"Batch {total_batches} | loss={loss.item():.4f}")

    # Return average loss for this epoch
    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate(model, samples, max_nodes, batch_size, device):
    """
    Run evaluation/validation without gradient tracking.

    @torch.no_grad() disables autograd, which:
    - saves memory
    - speeds up validation
    - prevents accidental gradient updates
    """
    model.eval()

    total_loss = 0.0
    total_batches = 0

    for mini in chunk_list(samples, batch_size):
        # Build batch the same way as in training
        batch, target_color, target_node_shape, target_edge_cont, target_edge_bin = make_batch(
            mini, max_nodes=max_nodes, device=device
        )

        # Forward pass
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        # Compute validation loss
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
# MAIN
# ============================================================================

def main():
    """
    Full training entry point.

    High-level flow:
      1) set random seed
      2) collect graph samples from ARC JSON files
      3) compute max_nodes for padding/decoder sizing
      4) split into train/validation
      5) construct GATVAE model
      6) train for EPOCHS
      7) save best checkpoint based on validation loss
    """
    set_seed(SEED)

    print(f"Using device: {DEVICE}")
    print(f"Loading samples from: {TRAIN_PATH}")

    # Build graph dataset from ARC train inputs
    samples = collect_graph_samples(TRAIN_PATH, max_files=MAX_FILES)

    # Fail early if nothing usable was found
    if not samples:
        raise RuntimeError("No usable samples were built. Check TRAIN_PATH or graph construction.")

    print(f"Total graph samples: {len(samples)}")

    # Find largest graph size in dataset
    max_nodes = compute_max_nodes(samples)
    print(f"Max nodes across dataset: {max_nodes}")

    # Simple train/val split
    #
    # 90% train
    # 10% validation
    #
    # max(1, ...) ensures at least one sample ends up in training
    random.shuffle(samples)
    split_idx = max(1, int(0.9 * len(samples)))

    train_samples = samples[:split_idx]

    # If split somehow leaves validation empty, fall back to one sample
    val_samples = samples[split_idx:] if split_idx < len(samples) else samples[:1]

    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples:   {len(val_samples)}")

    # ------------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------------
    # Must match your builder feature dimensions:
    #
    # node_in_dim = 110
    #   - 10 color dims
    #   - 100 shape mask dims
    #
    # edge_in_dim = 5
    #   - dx, dy, touching, same_row, same_col
    #
    # node_shape_dim = 100
    #   - decoder reconstructs flattened 10x10 mask
    model = GATVAE(
        max_nodes=max_nodes,
        node_in_dim=110,
        edge_in_dim=5,
        latent_dim=256,
        dec_hidden=256,
        node_shape_dim=100,
    ).to(DEVICE)

    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Track best validation loss seen so far
    best_val = math.inf

    # Checkpoint output path
    save_path = "gat_vae_hybrid_best.pt"

    print("\nStarting training...\n")

    # Main epoch loop
    for epoch in range(1, EPOCHS + 1):
        # Train one epoch
        train_loss = train_one_epoch(
            model, optimizer, train_samples, max_nodes, BATCH_SIZE, DEVICE
        )

        # Evaluate on validation set
        val_loss = evaluate(
            model, val_samples, max_nodes, BATCH_SIZE, DEVICE
        )

        # Print summary for this epoch
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        # Save checkpoint whenever validation improves
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


# Standard Python entrypoint guard:
# This ensures main() runs only when this file is executed directly,
# not when it is imported as a module somewhere else.
if __name__ == "__main__":
    main()