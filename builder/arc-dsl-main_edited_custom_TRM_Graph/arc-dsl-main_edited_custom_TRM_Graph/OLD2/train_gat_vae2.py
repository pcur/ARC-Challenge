import os
import json
import math
import random
from typing import List, Tuple, Dict

import torch
from torch_geometric.data import Data, Batch

from OLD2.custom_object2 import grid_to_graph      # FIX v2: deterministic ordering
from OLD2.gat_vae2 import GATVAE                   # FIX v2: existence logits wired through
from OLD2.graph_decoder2 import gat_vae_loss       # FIX v2: masked loss + existence loss


# ============================================================================
# CONFIG
# ============================================================================

TRAIN_PATH = r"C:\Users\tedhun\OneDrive - Wanzl GmbH & Co. KGaA\Microsoft Teams Chat Files\outputs\Documents\GS\ARC\Assignment10\Builder\data\training"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
BATCH_SIZE = 2
EPOCHS = 20
LR = 1e-3

# If True, skips graphs with fewer than 2 nodes, because the encoder expects
# edge information and tiny graphs can be awkward for this model setup.
SKIP_TINY_GRAPHS = True

# Optional cap for quick testing. Set to None to use all files.
MAX_FILES = 10


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
# GRAPH -> PYG
# ============================================================================

def graph_to_pyg_data(graph: dict) -> Data:
    """
    Convert custom_object2.py output into a PyTorch Geometric Data object.

    Expected graph keys:
      - node_features : [N, 22]
      - edge_features : [E, 12]
      - edge_index    : [2, E]
    """
    x = torch.tensor(graph["node_features"], dtype=torch.float32)
    edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(graph["edge_features"], dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def build_targets(
    graph: dict, max_nodes: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build padded targets for gat_vae_loss(...).

    Output shapes:
      target_color     : [max_nodes]                long
      target_node_geom : [max_nodes, 12]            float
      target_edge_cont : [max_nodes, max_nodes, 6]  float
      target_edge_bin  : [max_nodes, max_nodes, 6]  float
      target_existence : [max_nodes]                float  ← NEW v2

    FIX v2: target_existence is a binary mask where:
        1.0 = this node slot corresponds to a real object in the graph
        0.0 = this slot is padding (beyond the actual number of nodes)

    This mask is used in gat_vae_loss to ensure padded slots do not
    contribute to the color, geometry, or edge reconstruction losses.

    custom_object2.py node features:
      first 10 dims  = colors (one-hot)
      last 12 dims   = geometry

    custom_object2.py edge features:
      [0]  dx
      [1]  dy
      [2]  manhattan
      [3]  dist
      [4]  same_color
      [5]  touching
      [6]  bbox_overlap
      [7]  same_row
      [8]  same_col
      [9]  same_area
      [10] area_ratio_ab
      [11] area_ratio_ba
    """
    node_feats = torch.tensor(graph["node_features"], dtype=torch.float32)   # [N, 22]
    edge_feats = torch.tensor(graph["edge_features"], dtype=torch.float32)   # [E, 12]
    edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)         # [2, E]

    n_nodes = node_feats.size(0)

    # ── FIX v2: existence mask ───────────────────────────────────────────────
    # Real nodes get 1.0, padded slots stay 0.0
    target_existence = torch.zeros(max_nodes, dtype=torch.float32)
    target_existence[:n_nodes] = 1.0

    # ── Color target: class index from first 10 dims ─────────────────────────
    target_color = torch.zeros(max_nodes, dtype=torch.long)
    target_color[:n_nodes] = node_feats[:, :10].argmax(dim=1)

    # ── Node geometry target: last 12 dims ───────────────────────────────────
    target_node_geom = torch.zeros(max_nodes, 12, dtype=torch.float32)
    target_node_geom[:n_nodes] = node_feats[:, 10:]

    # ── Dense edge targets ───────────────────────────────────────────────────
    # Only real-to-real edges are filled; padding rows/cols stay zero.
    target_edge_cont = torch.zeros(max_nodes, max_nodes, 6, dtype=torch.float32)
    target_edge_bin  = torch.zeros(max_nodes, max_nodes, 6, dtype=torch.float32)

    # Continuous = [dx, dy, manhattan, euclidean, ratio_ab, ratio_ba]
    # Binary     = [same_color, touching, bbox_overlap, same_row, same_col, same_area]
    for e in range(edge_feats.size(0)):
        s = int(edge_index[0, e].item())
        d = int(edge_index[1, e].item())
        ef = edge_feats[e]

        target_edge_cont[s, d] = torch.tensor(
            [ef[0], ef[1], ef[2], ef[3], ef[10], ef[11]],
            dtype=torch.float32,
        )
        target_edge_bin[s, d] = torch.tensor(
            [ef[4], ef[5], ef[6], ef[7], ef[8], ef[9]],
            dtype=torch.float32,
        )

    return target_color, target_node_geom, target_edge_cont, target_edge_bin, target_existence


# ============================================================================
# DATASET BUILD
# ============================================================================

def collect_graph_samples(train_path: str, max_files: int = None) -> List[Dict]:
    """
    Build one graph sample for every train input grid in every ARC JSON file.
    """
    samples = []
    files = sorted([f for f in os.listdir(train_path) if f.endswith(".json")])

    if max_files is not None:
        files = files[:max_files]

    for fname in files:
        fpath = os.path.join(train_path, fname)
        task = load_json(fpath)

        for i, pair in enumerate(task.get("train", [])):
            raw_grid = pair["input"]
            grid = to_dsl_grid(raw_grid)

            graph = grid_to_graph(grid)

            n_nodes = len(graph["nodes"])
            n_edges = len(graph["edges"])

            if SKIP_TINY_GRAPHS and (n_nodes < 2 or n_edges == 0):
                continue

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
    if not samples:
        raise ValueError("No valid graph samples were found.")
    return max(s["num_nodes"] for s in samples)


def make_batch(samples: List[Dict], max_nodes: int, device: str):
    """
    Convert a list of samples into:
      - PyG Batch
      - stacked target tensors

    FIX v2: now also returns stacked target_existence tensor.
    """
    data_list        = []
    color_targets    = []
    geom_targets     = []
    edge_cont_targets = []
    edge_bin_targets  = []
    existence_targets = []          # FIX v2

    for s in samples:
        graph = s["graph"]

        data_list.append(graph_to_pyg_data(graph))

        # FIX v2: build_targets now returns 5 tensors (added existence)
        (target_color,
         target_node_geom,
         target_edge_cont,
         target_edge_bin,
         target_existence) = build_targets(graph, max_nodes)

        color_targets.append(target_color)
        geom_targets.append(target_node_geom)
        edge_cont_targets.append(target_edge_cont)
        edge_bin_targets.append(target_edge_bin)
        existence_targets.append(target_existence)      # FIX v2

    batch = Batch.from_data_list(data_list).to(device)

    target_color      = torch.stack(color_targets).to(device)
    target_node_geom  = torch.stack(geom_targets).to(device)
    target_edge_cont  = torch.stack(edge_cont_targets).to(device)
    target_edge_bin   = torch.stack(edge_bin_targets).to(device)
    target_existence  = torch.stack(existence_targets).to(device)   # FIX v2

    return batch, target_color, target_node_geom, target_edge_cont, target_edge_bin, target_existence


def chunk_list(items: List, batch_size: int) -> List[List]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


# ============================================================================
# TRAINING
# ============================================================================

def train_one_epoch(model, optimizer, samples, max_nodes, batch_size, device):
    model.train()
    random.shuffle(samples)

    total_loss    = 0.0
    total_batches = 0

    for mini in chunk_list(samples, batch_size):
        # FIX v2: unpack 6 values (added target_existence)
        (batch,
         target_color,
         target_node_geom,
         target_edge_cont,
         target_edge_bin,
         target_existence) = make_batch(mini, max_nodes=max_nodes, device=device)

        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        # FIX v2: pass target_existence to loss
        loss, breakdown = gat_vae_loss(
            out,
            target_color,
            target_node_geom,
            target_edge_cont,
            target_edge_bin,
            target_existence,       # FIX v2
        )

        loss.backward()
        optimizer.step()

        total_loss    += float(loss.item())
        total_batches += 1

        if total_batches % 50 == 0:
            print(f"  Batch {total_batches} | loss={loss.item():.4f}")

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate(model, samples, max_nodes, batch_size, device):
    model.eval()

    total_loss    = 0.0
    total_batches = 0

    for mini in chunk_list(samples, batch_size):
        # FIX v2: unpack 6 values
        (batch,
         target_color,
         target_node_geom,
         target_edge_cont,
         target_edge_bin,
         target_existence) = make_batch(mini, max_nodes=max_nodes, device=device)

        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        # FIX v2: pass target_existence to loss
        loss, breakdown = gat_vae_loss(
            out,
            target_color,
            target_node_geom,
            target_edge_cont,
            target_edge_bin,
            target_existence,       # FIX v2
        )

        total_loss    += float(loss.item())
        total_batches += 1

    return total_loss / max(total_batches, 1)


# ============================================================================
# MAIN
# ============================================================================

def main():
    set_seed(SEED)

    print(f"Using device: {DEVICE}")
    print(f"Loading samples from: {TRAIN_PATH}")

    samples = collect_graph_samples(TRAIN_PATH, max_files=MAX_FILES)

    if not samples:
        raise RuntimeError(
            "No usable samples were built. Check TRAIN_PATH or graph construction."
        )

    print(f"Total graph samples: {len(samples)}")

    max_nodes = compute_max_nodes(samples)
    print(f"Max nodes across dataset: {max_nodes}")

    # Simple train/val split
    random.shuffle(samples)
    split_idx    = max(1, int(0.9 * len(samples)))
    train_samples = samples[:split_idx]
    val_samples   = samples[split_idx:] if split_idx < len(samples) else samples[:1]

    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples:   {len(val_samples)}")

    model = GATVAE(
        max_nodes=max_nodes,
        node_in_dim=22,
        edge_in_dim=12,
        latent_dim=256,
        dec_hidden=256,
        node_geom_dim=12,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val  = math.inf
    save_path = "gat_vae_best.pt"

    print("\nStarting training...\n")

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
                    "model_state_dict"      : model.state_dict(),
                    "optimizer_state_dict"  : optimizer.state_dict(),
                    "max_nodes"             : max_nodes,
                    "best_val_loss"         : best_val,
                },
                save_path,
            )
            print(f"  Saved best model → {save_path}")

    print("\nTraining complete.")
    print(f"Best val loss: {best_val:.6f}")


if __name__ == "__main__":
    main()
