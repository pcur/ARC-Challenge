"""
task_dataset.py
===============
TaskDataset — loads complete ARC tasks and produces leave-one-out
training examples.

Each ARC task has K demonstration pairs (typically 3-5). For training,
we use leave-one-out: hold out one demo pair as the (input, output) to
predict, and use the remaining K-1 pairs as task context. A task with 3
pairs produces 3 training examples.

Each training example contains:
  context_pairs : list of (in_obj, in_pix, out_obj, out_pix) — K-1 pairs
  test_input    : (in_obj, in_pix) — the held-out input
  test_output   : (out_obj, out_pix) — the held-out output (supervision)
  task_id       : str

At inference time, all K demonstration pairs form the context and the
actual test input from the task JSON is the query.

Graph types
-----------
  obj_graph : object-level PyG Data (SpatialGeomAE input)
  pix_graph : pixel-level PyG Data  (SpatialColorAE input)

Augmentation
------------
Applied consistently to ALL grids in a task (context + test) using the
same geometric transform, so spatial relationships between pairs are
preserved. Color permutation is also applied consistently so the model
learns color relationships rather than absolute color indices.
"""

import os
import sys
import json
import random
import torch
from typing import List, Tuple, Optional
from torch_geometric.data import Data
from torch.utils.data import Dataset, random_split, DataLoader

from pixel_graph_builder import pixel_grid_to_graph
from augmentation import augment_grid

# ── import object graph builder ──────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
_BUILDER = os.path.join(_HERE, "..", "builder", "arc-dsl-main_edited")
if _BUILDER not in sys.path:
    sys.path.insert(0, _BUILDER)
from graph_builder import grid_to_graph


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH BUILDING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_obj_graph(grid) -> Optional[Data]:
    """Build object-level PyG Data from raw grid. Returns None if invalid."""
    try:
        obj_dict = grid_to_graph(tuple(tuple(r) for r in grid))
        if not obj_dict.get("node_features"):
            return None
        x          = torch.tensor(obj_dict["node_features"], dtype=torch.float)
        src        = torch.tensor(obj_dict["edge_index"][0], dtype=torch.long)
        dst        = torch.tensor(obj_dict["edge_index"][1], dtype=torch.long)
        edge_index = torch.stack([src, dst])
        edge_attr  = torch.tensor(obj_dict["edge_features"], dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    except Exception:
        return None


def _build_pix_graph(grid, max_pixels: int) -> Optional[Data]:
    """Build pixel-level PyG Data from raw grid. Returns None if too large."""
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    if rows * cols > max_pixels:
        return None
    try:
        return pixel_grid_to_graph(grid)
    except Exception:
        return None


def _build_dual(grid, max_object_nodes: int, max_pixels: int):
    """Build (obj_graph, pix_graph) pair. Returns None if either fails."""
    obj = _build_obj_graph(grid)
    if obj is None or obj.x.size(0) > max_object_nodes:
        return None
    pix = _build_pix_graph(grid, max_pixels)
    if pix is None:
        return None
    return obj, pix


# ─────────────────────────────────────────────────────────────────────────────
# TASK DATASET
# ─────────────────────────────────────────────────────────────────────────────

class TaskDataset(Dataset):
    """
    Leave-one-out task dataset for transform model training.

    Each item is a dict:
    {
        "context_pairs" : list of (obj_in, pix_in, obj_out, pix_out)
                          K-1 demonstration pairs used as task context
        "test_in_obj"   : Data  — held-out input object graph
        "test_in_pix"   : Data  — held-out input pixel graph
        "test_out_obj"  : Data  — held-out output object graph
        "test_out_pix"  : Data  — held-out output pixel graph
        "task_id"       : str
    }

    Parameters
    ----------
    data_dir         : path to ARC task JSON files
    max_object_nodes : skip tasks where any grid exceeds this object count
    max_pixels       : skip tasks where any grid exceeds this pixel count
    augment          : apply random augmentation (training only)
    min_context      : minimum context pairs required (skip if fewer available)
    """

    def __init__(
        self,
        data_dir: str,
        max_object_nodes: int = 50,
        max_pixels: int       = 400,
        augment: bool         = False,
        min_context: int      = 1,
        registry_path: str    = None,  # path to save/load augmented task registry
    ):
        self.max_object_nodes = max_object_nodes
        self.max_pixels       = max_pixels
        self.augment          = augment
        self.min_context      = min_context

        # default registry path sits alongside the training data
        if registry_path is None:
            suffix = "augmented" if augment else "original"
            registry_path = os.path.join(data_dir, f"task_registry_{suffix}.json")
        self.registry_path = registry_path

        self.examples = []   # list of (task_id, raw_pairs, held_out_idx)
        self._load(data_dir)

    def _load(self, data_dir: str):
        # ── load from registry if it exists ──────────────────────────────────
        if os.path.exists(self.registry_path):
            print(f"Loading task registry from {self.registry_path} ...")
            with open(self.registry_path, "r") as f:
                registry = json.load(f)
            skipped = 0
            for variant_id, raw_pairs in registry.items():
                # raw_pairs stored as list of [in_grid, out_grid]
                pairs = [(entry[0], entry[1]) for entry in raw_pairs]
                if len(pairs) < self.min_context + 1:
                    skipped += 1
                    continue
                for held_out_idx in range(len(pairs)):
                    self.examples.append((variant_id, pairs, held_out_idx))
            n_tasks = len({vid.split("_g")[0] for vid in registry.keys()})
            print(f"TaskDataset: {len(self.examples)} leave-one-out examples "
                  f"from {n_tasks} base tasks, "
                  f"{len(registry)} variants "
                  f"({skipped} skipped)\n")
            return

        # ── build registry from scratch ───────────────────────────────────────
        print(f"Building task registry (will save to {self.registry_path}) ...")
        from augmentation import apply_geometric, random_color_permutation
        import random as _random

        json_files = sorted(f for f in os.listdir(data_dir)
                            if f.endswith(".json"))
        skipped  = 0
        registry = {}   # variant_id → list of [in_grid, out_grid]

        if self.augment:
            aug_variants = [(0, None)]
            for geo_idx in range(1, 8):
                aug_variants.append((geo_idx, None))
                aug_variants.append((geo_idx, geo_idx * 17))
            for cs in [101, 202, 303]:
                aug_variants.append((0, cs))
        else:
            aug_variants = [(0, None)]

        for fname in json_files:
            path = os.path.join(data_dir, fname)
            try:
                task = json.load(open(path))
            except Exception:
                continue

            base_task_id = fname.replace(".json", "")
            pairs        = task.get("train", [])

            raw_pairs = []
            valid     = True
            for pair in pairs:
                in_grid  = pair.get("input")
                out_grid = pair.get("output")
                if in_grid is None or out_grid is None:
                    valid = False
                    break
                in_pix  = len(in_grid) * (len(in_grid[0]) if in_grid else 0)
                out_pix = len(out_grid) * (len(out_grid[0]) if out_grid else 0)
                if in_pix > self.max_pixels or out_pix > self.max_pixels:
                    valid = False
                    break
                raw_pairs.append(([list(r) for r in in_grid],
                                  [list(r) for r in out_grid]))

            if not valid or len(raw_pairs) < self.min_context + 1:
                skipped += 1
                continue

            for geo_idx, color_seed in aug_variants:
                if geo_idx == 0 and color_seed is None:
                    aug_pairs  = raw_pairs
                    variant_id = base_task_id
                else:
                    aug_pairs  = []
                    valid_aug  = True
                    for in_grid, out_grid in raw_pairs:
                        try:
                            aug_in  = apply_geometric(
                                [list(r) for r in in_grid], geo_idx)
                            aug_out = apply_geometric(
                                [list(r) for r in out_grid], geo_idx)
                            if color_seed is not None:
                                rng = _random.Random(color_seed)
                                aug_in,  _ = random_color_permutation(aug_in,  rng)
                                rng = _random.Random(color_seed)
                                aug_out, _ = random_color_permutation(aug_out, rng)
                            aug_pairs.append((aug_in, aug_out))
                        except Exception:
                            valid_aug = False
                            break
                    if not valid_aug:
                        continue

                    suffix     = f"_g{geo_idx}"
                    if color_seed is not None:
                        suffix += f"_c{color_seed}"
                    variant_id = f"{base_task_id}{suffix}"

                # store in registry as list of [in_grid, out_grid]
                registry[variant_id] = [[list(ig), list(og)]
                                        for ig, og in aug_pairs]

                for held_out_idx in range(len(aug_pairs)):
                    self.examples.append((variant_id, aug_pairs, held_out_idx))

        # ── save registry to disk ─────────────────────────────────────────────
        with open(self.registry_path, "w") as f:
            json.dump(registry, f)
        print(f"Saved {len(registry)} task variants to {self.registry_path}")

        n_tasks = len({vid.split("_g")[0] for vid in registry.keys()})
        print(f"TaskDataset: {len(self.examples)} leave-one-out examples "
              f"from {n_tasks} base tasks, "
              f"{len(aug_variants)} augmentation variant(s) "
              f"({skipped} tasks skipped)\n")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        task_id, raw_pairs, held_out_idx = self.examples[idx]

        # augmentation already applied at load time — raw_pairs are pre-augmented
        built = []
        for in_grid, out_grid in raw_pairs:
            in_dual  = _build_dual(in_grid,  self.max_object_nodes, self.max_pixels)
            out_dual = _build_dual(out_grid, self.max_object_nodes, self.max_pixels)
            if in_dual is None or out_dual is None:
                return self.__getitem__((idx + 1) % len(self))
            built.append((in_dual[0], in_dual[1], out_dual[0], out_dual[1]))

        context_pairs = [built[i] for i in range(len(built)) if i != held_out_idx]
        test_pair     = built[held_out_idx]

        return {
            "context_pairs" : context_pairs,
            "test_in_obj"   : test_pair[0],
            "test_in_pix"   : test_pair[1],
            "test_out_obj"  : test_pair[2],
            "test_out_pix"  : test_pair[3],
            "task_id"       : task_id,
        }


# ─────────────────────────────────────────────────────────────────────────────
# COLLATE
# ─────────────────────────────────────────────────────────────────────────────

def task_collate(batch):
    """
    Collate a list of task examples into batched form.

    Because context length varies per task, we keep examples separate
    rather than trying to batch across them. Returns a list of dicts,
    one per example in the batch.

    The transform model processes each example independently — batching
    happens at the loss level by averaging across the batch.
    """
    return batch


def build_task_dataloaders(
    training_dir: str,
    batch_size: int       = 8,
    max_object_nodes: int = 50,
    max_pixels: int       = 400,
    val_split: float      = 0.2,
    min_context: int      = 1,
    seed: int             = 42,
    registry_dir: str     = None,  # where to save/load registries (default: training_dir)
):
    """
    Build train and val task dataloaders.
    Train uses augmented variants with stable IDs, val uses original grids.
    Split is at the task level — no task appears in both train and val.
    Registries are saved to disk so augmented variants are stable across runs.
    """
    if registry_dir is None:
        registry_dir = training_dir

    aug_registry_path  = os.path.join(registry_dir, "task_registry_augmented.json")
    orig_registry_path = os.path.join(registry_dir, "task_registry_original.json")

    # load full dataset without augmentation to get stable task list
    full = TaskDataset(
        training_dir,
        max_object_nodes = max_object_nodes,
        max_pixels       = max_pixels,
        augment          = False,
        min_context      = min_context,
        registry_path    = orig_registry_path,
    )

    # split by unique BASE task IDs (strip variant suffix) to prevent leakage
    base_ids   = list({ex[0].split("_g")[0] for ex in full.examples})
    rng        = random.Random(seed)
    rng.shuffle(base_ids)
    n_val      = max(1, int(len(base_ids) * val_split))
    val_base   = set(base_ids[:n_val])
    train_base = set(base_ids[n_val:])

    # val examples: original grids only, base IDs in val split
    val_examples = [(tid, pairs, idx) for tid, pairs, idx in full.examples
                    if tid.split("_g")[0] in val_base]

    # train examples: load augmented registry, filter to train base IDs
    aug_full = TaskDataset(
        training_dir,
        max_object_nodes = max_object_nodes,
        max_pixels       = max_pixels,
        augment          = True,
        min_context      = min_context,
        registry_path    = aug_registry_path,
    )
    train_examples = [(tid, pairs, idx) for tid, pairs, idx in aug_full.examples
                      if tid.split("_g")[0] in train_base]

    train_ds = TaskDataset.__new__(TaskDataset)
    train_ds.max_object_nodes = max_object_nodes
    train_ds.max_pixels       = max_pixels
    train_ds.augment          = True
    train_ds.min_context      = min_context
    train_ds.examples         = train_examples

    val_ds = TaskDataset.__new__(TaskDataset)
    val_ds.max_object_nodes   = max_object_nodes
    val_ds.max_pixels         = max_pixels
    val_ds.augment            = False
    val_ds.min_context        = min_context
    val_ds.examples           = val_examples

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  collate_fn=task_collate)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, collate_fn=task_collate)

    print(f"Transform train: {len(train_ds)} examples from "
          f"{len(train_base)} tasks")
    print(f"Transform val  : {len(val_ds)} examples from "
          f"{len(val_base)} tasks\n")

    return train_loader, val_loader