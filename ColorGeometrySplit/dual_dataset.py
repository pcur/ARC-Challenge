"""
dual_dataset.py
===============
DualARCDataset — loads ARC task JSON files and builds paired graph
representations for each input grid on-the-fly with augmentation.

On-the-fly augmentation
------------------------
Raw grids are stored in memory. Each call to __getitem__ applies a
random augmentation (geometric transform + color permutation) before
building both graphs. This means every epoch sees a different augmented
version of each grid — effectively infinite variety from 2,464 base grids.

Augmentation is disabled for the validation split so metrics are
comparable across runs.

Graph types
-----------
  object_graph : object-level PyG Data (nodes = connected same-color regions)
  pixel_graph  : pixel-level PyG Data  (nodes = individual grid cells)

Both are built from the same augmented grid so they're always consistent.

Dependencies:
    pip install torch torch_geometric
"""

import os
import sys
import json
import random
import torch
from typing import Optional
from torch_geometric.data import Data, Dataset
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from pixel_graph_builder import pixel_grid_to_graph
from augmentation import augment_grid

# ── import object graph builder ──────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
_BUILDER = os.path.join(_HERE, "..", "builder", "arc-dsl-main_edited")
if _BUILDER not in sys.path:
    sys.path.insert(0, _BUILDER)
from graph_builder import grid_to_graph


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class DualARCDataset(Dataset):
    """
    Dataset that stores raw grids and builds dual graphs on-the-fly.

    During training (augment=True), each __getitem__ call applies a fresh
    random augmentation before building both graphs. During validation
    (augment=False), graphs are built from the original grid every time.

    Parameters
    ----------
    data_dir         : path to folder of ARC JSON files
    max_object_nodes : skip grids whose object graph exceeds this
    max_pixel_nodes  : skip grids with more pixels than this
    augment          : whether to apply augmentation in __getitem__
    """

    def __init__(
        self,
        data_dir: str,
        max_object_nodes: int = 50,
        max_pixel_nodes: int  = 400,
        augment: bool         = False,
    ):
        super().__init__()
        self.max_object_nodes = max_object_nodes
        self.max_pixel_nodes  = max_pixel_nodes
        self.augment          = augment
        self.raw_grids        = []   # list of (raw_grid, task_id)
        self._load(data_dir)

    def _load(self, data_dir: str):
        json_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".json"))
        skipped_empty = skipped_obj = skipped_pix = skipped_err = 0

        for fname in json_files:
            path = os.path.join(data_dir, fname)
            try:
                task = json.load(open(path))
            except Exception:
                continue

            task_id = fname.replace(".json", "")

            for pair in task.get("train", []):
                raw_grid = pair.get("input")
                if raw_grid is None:
                    continue

                rows  = len(raw_grid)
                cols  = len(raw_grid[0]) if rows > 0 else 0
                n_pix = rows * cols

                if n_pix > self.max_pixel_nodes:
                    skipped_pix += 1
                    continue

                # validate the original grid builds valid graphs
                grid = tuple(tuple(r) for r in raw_grid)
                try:
                    obj_dict = grid_to_graph(grid)
                except Exception:
                    skipped_err += 1
                    continue

                if len(obj_dict.get("node_features", [])) == 0:
                    skipped_empty += 1
                    continue

                if len(obj_dict["nodes"]) > self.max_object_nodes:
                    skipped_obj += 1
                    continue

                # store raw grid — graphs built on-the-fly
                self.raw_grids.append((raw_grid, task_id))

        print(
            f"Loaded {len(self.raw_grids)} grids from {len(json_files)} tasks  "
            f"({skipped_empty} empty, {skipped_obj} obj-too-large, "
            f"{skipped_pix} pix-too-large, {skipped_err} errors)  "
            f"augment={'ON' if self.augment else 'OFF'}"
        )

    def _build_graphs(self, raw_grid, task_id):
        """Build dual graphs from a (possibly augmented) raw grid."""
        # apply augmentation if enabled
        if self.augment:
            grid_list, _ = augment_grid(raw_grid, geometric=True, color_perm=True)
        else:
            grid_list = [list(row) for row in raw_grid]

        grid = tuple(tuple(r) for r in grid_list)

        try:
            obj_dict = grid_to_graph(grid)
            if len(obj_dict.get("node_features", [])) == 0:
                return None

            x          = torch.tensor(obj_dict["node_features"], dtype=torch.float)
            src        = torch.tensor(obj_dict["edge_index"][0], dtype=torch.long)
            dst        = torch.tensor(obj_dict["edge_index"][1], dtype=torch.long)
            edge_index = torch.stack([src, dst])
            edge_attr  = torch.tensor(obj_dict["edge_features"], dtype=torch.float)
            obj_graph  = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

            pix_graph = pixel_grid_to_graph(grid)

        except Exception:
            return None

        return obj_graph, pix_graph, task_id

    # ── PyG Dataset interface ─────────────────────────────────────────────────

    def len(self):
        return len(self.raw_grids)

    def get(self, idx):
        raw_grid, task_id = self.raw_grids[idx]
        result = self._build_graphs(raw_grid, task_id)
        if result is None:
            # fallback: return unaugmented version
            grid   = tuple(tuple(r) for r in raw_grid)
            obj_d  = grid_to_graph(grid)
            x      = torch.tensor(obj_d["node_features"], dtype=torch.float)
            ei     = torch.stack([
                torch.tensor(obj_d["edge_index"][0], dtype=torch.long),
                torch.tensor(obj_d["edge_index"][1], dtype=torch.long),
            ])
            ea     = torch.tensor(obj_d["edge_features"], dtype=torch.float)
            obj_g  = Data(x=x, edge_index=ei, edge_attr=ea)
            pix_g  = pixel_grid_to_graph(grid)
            return obj_g, pix_g, task_id
        return result


# ─────────────────────────────────────────────────────────────────────────────
# COLLATE
# ─────────────────────────────────────────────────────────────────────────────

def dual_collate(batch):
    from torch_geometric.data import Batch
    obj_graphs = [b[0] for b in batch]
    pix_graphs = [b[1] for b in batch]
    task_ids   = [b[2] for b in batch]
    return Batch.from_data_list(obj_graphs), Batch.from_data_list(pix_graphs), task_ids


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_dual_dataloaders(
    training_dir: str,
    batch_size: int       = 32,
    max_object_nodes: int = 50,
    max_pixel_nodes: int  = 400,
    val_split: float      = 0.2,
    num_workers: int      = 0,
    seed: int             = 42,
):
    """
    Build train and val dataloaders with on-the-fly augmentation.

    Training loader has augment=True — fresh random augmentation each epoch.
    Validation loader has augment=False — original grids for consistent metrics.
    """
    # load full dataset without augmentation to get stable indices
    full = DualARCDataset(
        training_dir,
        max_object_nodes = max_object_nodes,
        max_pixel_nodes  = max_pixel_nodes,
        augment          = False,
    )

    n_val   = max(1, int(len(full) * val_split))
    n_train = len(full) - n_val
    gen     = torch.Generator().manual_seed(seed)

    # split indices
    from torch.utils.data import Subset
    indices   = torch.randperm(len(full), generator=gen).tolist()
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:]

    # training dataset with augmentation enabled
    train_ds_aug = DualARCDataset(
        training_dir,
        max_object_nodes = max_object_nodes,
        max_pixel_nodes  = max_pixel_nodes,
        augment          = True,
    )
    train_ds = Subset(train_ds_aug, train_idx)
    val_ds   = Subset(full,         val_idx)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=dual_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=dual_collate,
    )

    print(f"Train: {len(train_ds)} grids (augmented)  |  Val: {len(val_ds)} grids\n")
    return train_loader, val_loader
