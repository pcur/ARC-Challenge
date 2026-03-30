"""
dataset.py
==========
ARCGraphDataset — PyTorch Geometric dataset that loads raw ARC task JSON
files, converts each grid to a graph via the builder, and returns PyG
Data objects ready for the GAT-VAE training loop.

Each ARC task JSON contains multiple train pairs. Every *input* grid in
every train pair becomes one graph in the dataset. (Output grids are not
used here — they are the target for the transform model later.)

Directory layout assumed
------------------------
/Project/
    GAT_VAE/
        dataset.py      ← this file
        train.py
        gat_vae.py
        gat_encoder.py
        graph_decoder.py
    builder/
        graph_builder.py
        data/
            training/   ← JSON files used here
            evaluation/ ← not used during VAE training

Dependencies:
    pip install torch torch_geometric
"""

import os
import sys
import json
import torch
from torch_geometric.data import Data, Dataset


# ── import graph_builder from the sibling builder directory ─────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
_BUILDER = os.path.join(_HERE, "..", "builder", "arc-dsl-main_edited")
if _BUILDER not in sys.path:
    sys.path.insert(0, _BUILDER)

from graph_builder import grid_to_graph          # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class ARCGraphDataset(Dataset):
    """
    In-memory PyG dataset built from ARC task JSON files.

    Each input grid in each task's train pairs becomes one Data object:
        x          : (N, 22)  node features  — one-hot color (10) + geometry (12)
        edge_index : (2, E)   COO edge index (fully connected, no self-loops)
        edge_attr  : (E, 12)  edge features

    Parameters
    ----------
    data_dir  : path to the folder of ARC JSON files (e.g. .../data/training)
    max_nodes : graphs with more nodes than this are skipped (keeps decoder
                output size bounded — set to match GATVAE max_nodes)
    """

    def __init__(self, data_dir: str, max_nodes: int = 50):
        super().__init__()
        self.max_nodes = max_nodes
        self.graphs    = []
        self._load(data_dir)

    # ── loading ──────────────────────────────────────────────────────────────

    def _load(self, data_dir: str):
        json_files = sorted(
            f for f in os.listdir(data_dir) if f.endswith(".json")
        )
        skipped_size  = 0
        skipped_empty = 0

        for fname in json_files:
            path = os.path.join(data_dir, fname)
            try:
                with open(path, "r") as f:
                    task = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"  [warn] could not load {fname}: {e}")
                continue

            for pair in task.get("train", []):
                raw_grid = pair.get("input")
                if raw_grid is None:
                    continue

                grid = tuple(tuple(row) for row in raw_grid)

                try:
                    graph = grid_to_graph(grid)
                except Exception as e:
                    print(f"  [warn] grid_to_graph failed in {fname}: {e}")
                    continue

                if len(graph["node_features"]) == 0:
                    skipped_empty += 1
                    continue

                n_nodes = len(graph["nodes"])
                if n_nodes > self.max_nodes:
                    skipped_size += 1
                    continue

                data = self._to_pyg(graph)
                if data is not None:
                    self.graphs.append(data)

        print(f"Loaded {len(self.graphs)} graphs from {len(json_files)} tasks "
              f"({skipped_empty} empty grids skipped, "
              f"{skipped_size} skipped — exceeded max_nodes={self.max_nodes})")

    def _to_pyg(self, graph: dict):
        """Convert a graph_builder output dict to a PyG Data object."""
        try:
            # ── skip empty graphs (no objects detected in grid) ───────────
            if len(graph["node_features"]) == 0:
                return None

            # ── node features: (N, 22) ────────────────────────────────────
            x = torch.tensor(graph["node_features"], dtype=torch.float)  # (N, 22)

            # ── edge index: (2, E) ────────────────────────────────────────
            src = torch.tensor(graph["edge_index"][0], dtype=torch.long)
            dst = torch.tensor(graph["edge_index"][1], dtype=torch.long)
            edge_index = torch.stack([src, dst], dim=0)                   # (2, E)

            # ── edge features: (E, 12) ────────────────────────────────────
            edge_attr = torch.tensor(graph["edge_features"], dtype=torch.float)

            # ── self-loop for single-node graphs ──────────────────────────
            # A single node has no neighbours so GATv2Conv has nothing to
            # aggregate over. Adding a self-loop lets it at least compute an
            # attention-weighted self-representation through each layer.
            # Edge feature layout:
            #   [dx, dy, manhattan, euclidean,          ← 0.0 (no displacement)
            #    same_color, touching, bbox_overlap,
            #    same_row, same_col, same_area,          ← 1.0 (node is identical to itself)
            #    area_ratio_ab, area_ratio_ba]           ← 1.0 (same area)
            if x.size(0) == 1:
                edge_index = torch.zeros(2, 1, dtype=torch.long)          # (2, 1)
                edge_attr  = torch.tensor(
                    [[0.0, 0.0, 0.0, 0.0,   # dx, dy, manhattan, euclidean
                      1.0, 1.0, 1.0,         # same_color, touching, bbox_overlap
                      1.0, 1.0, 1.0,         # same_row, same_col, same_area
                      1.0, 1.0]],            # area_ratio_ab, area_ratio_ba
                    dtype=torch.float,
                )                                                          # (1, 12)

            # Basic shape validation
            assert x.dim() == 2 and x.size(1) == 22, \
                f"Expected node features (N, 22), got {tuple(x.shape)}"
            assert edge_attr.dim() == 2 and edge_attr.size(1) == 12, \
                f"Expected edge features (E, 12), got {tuple(edge_attr.shape)}"
            assert edge_index.size(1) == edge_attr.size(0), \
                "edge_index and edge_attr have mismatched edge counts"

            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        except Exception as e:
            print(f"  [warn] _to_pyg failed: {e}")
            return None

    # ── PyG Dataset interface ─────────────────────────────────────────────────

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    training_dir: str,
    batch_size: int,
    max_nodes: int,
    val_split: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
):
    """
    Load all training graphs, split into train/val, return DataLoaders.

    Parameters
    ----------
    training_dir : path to .../builder/data/training
    batch_size   : graphs per batch
    max_nodes    : graphs exceeding this are dropped
    val_split    : fraction of data held out for validation (default 10%)
    num_workers  : DataLoader workers (0 = main process, safe on Windows)
    seed         : random seed for the split
    """
    from torch_geometric.loader import DataLoader
    from torch.utils.data import random_split

    full_dataset = ARCGraphDataset(training_dir, max_nodes=max_nodes)

    n_total = len(full_dataset)
    n_val   = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val],
                                    generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    print(f"Train: {len(train_ds)} graphs  |  Val: {len(val_ds)} graphs\n")
    return train_loader, val_loader