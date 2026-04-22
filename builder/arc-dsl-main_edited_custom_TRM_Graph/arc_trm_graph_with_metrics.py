"""
arc_trm_graph.py

Graph-based Tiny Recursive Model for ARC.

How this differs from arc_trm.py
──────────────────────────────────
arc_trm.py:
    - Flattens raw pixel grid → 900 integer tokens
    - No structural understanding — learns everything implicitly
    - Token embedding table maps color integers → hidden vectors

arc_trm_graph.py (this file):
    - Calls grid_to_graph() from custom_object3.py to extract objects
    - Each object (node) becomes ONE token — its 22-dim feature vector
      is projected to HIDDEN_SIZE via a learned linear layer
    - Each relationship (edge) also becomes a token — its 12-dim feature
      vector is projected to HIDDEN_SIZE and interleaved between nodes
    - Sequence is dramatically shorter: ~5-30 node+edge tokens vs 900 pixels
    - Output is predicted per-node: existence, color, cell coords
    - Grid is reconstructed via graph_to_grid_from_predictions()

This makes the TRM's recursive computation operate directly over your
hand-engineered structural representation, combining:
    - Your object extraction work (custom_object3.py)
    - TRM's recursive depth and dynamic halting
    - No VAE bottleneck — end-to-end from graph to grid

Sequence layout per task
─────────────────────────
[puzzle_emb (PUZZLE_EMB_LEN tokens)]
[node_0_proj] [edge_0→1_proj] [node_1_proj] [edge_1→2_proj] ... (demo 1 input)
[SEP]
[node_0_proj] ... (demo 1 output)
[SEP]
...
[node_0_proj] ... (test input)
→ predict: per-node existence, color, cell coords for test output

Comparison value for your project
───────────────────────────────────
You now have four architectures to compare:
    1. Custom VAE + Transformer  (arc_transformer.py)
    2. Hybrid VAE + Transformer  (arc_transformer_hybrid.py)
    3. Raw-pixel TRM             (arc_trm.py)
    4. Graph TRM                 (arc_trm_graph.py)   ← this file

Comparing (1) vs (4) isolates the effect of TRM recursion vs standard
transformer depth, holding the graph representation constant.
Comparing (3) vs (4) isolates the effect of graph structure vs raw pixels,
holding the TRM architecture constant.
"""

import os
import json
import math
import random
from typing import List, Dict, Optional, Tuple
import gc
import pathlib
from datetime import datetime


import torch
import torch.nn as nn
import torch.nn.functional as F

from custom_object3 import (
    grid_to_graph,
    graph_to_grid_from_predictions,
)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

#TRAIN_PATH = r"C:\Users\tedhun\OneDrive - Wanzl GmbH & Co. KGaA\Microsoft Teams Chat Files\outputs\Documents\GS\ARC\Assignment10\Builder\data\training"

TRAIN_PATH = fr"{pathlib.Path.cwd()}" + r"\builder\data\training"

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path  = os.path.join(script_dir, f"test_model.pt")

SEED       = 42
#EPOCHS     = 60
EPOCHS     = 10
LR         = 1e-3
EMBED_LR   = 1e-2
BATCH_SIZE = 50
MAX_FILES  = None
PRINT_EVERY = 1

# Graph settings — must match what custom_object3 produces
NODE_FEAT_DIM      = 22    # node_to_feature_vector output dim
EDGE_FEAT_DIM      = 12    # edge_to_feature_vector output dim
MAX_NODES          = 30    # cap graphs at this many nodes
MAX_CELLS_PER_NODE = 40    # for cell coord decoder targets

# Sequence layout
PUZZLE_EMB_LEN = 16        # learned task-identity tokens prepended to sequence
MAX_EDGES_PER_NODE = 4     # only emit this many outgoing edges per node (sorted by distance)
                           # full graph = n^2 tokens which exceeds pos_emb size
# Special separator token index (used as a learned vector between demo pairs)
# We reserve index 0 for the SEP token in the puzzle embedding table

# TRM architecture
HIDDEN_SIZE  = 256
N_HEADS      = 4
FF_DIM       = 1024
DROPOUT      = 0.1
N_RECURSIONS = 6
T_INNER      = 4
N_SUP        = 16

# Augmentation
N_AUGMENTATIONS = 10
MAX_DEMOS       = 3

# EMA
EMA_DECAY = 0.999
USE_EMA   = True

# Output heads — predict per-node outputs (same targets as the VAE decoder)
NUM_COLORS = 10

# Debug / diagnostics toggles — set to False to preserve the original run style as much as possible
DEBUG_METRICS = {
    "tiny_subset_overfit": True,
    "gradient_norms": True,
    "activation_norms": True,
    "per_recursion_step_accuracy": True,
    "output_validity": True,
    "train_val_gap": True, #True by default
    "confidence_calibration": True,
}

# Debug settings
TINY_SUBSET_SIZE = 16
TINY_SUBSET_BATCH_SIZE = 8
CALIBRATION_BINS = 10
VALIDITY_EVAL_TASKS = 25
REPORT_PER_RECURSION_DETAIL = True


# ─────────────────────────────────────────────────────────────────────────────
# 1.  GRAPH → TOKEN SEQUENCE
# ─────────────────────────────────────────────────────────────────────────────

def graph_to_token_features(graph: dict, max_nodes: int = MAX_NODES):
    """
    Convert a graph into a flat list of feature vectors for the TRM.

    Sequence layout:
        [node_0_feats (22-dim)] [edge_0→1_feats (12-dim)] [node_1_feats] ...

    We interleave ALL edges after each node in a fixed canonical order:
        for each node n:
            emit node_n features
            emit all outgoing edges from node_n (sorted by dst)

    This gives the TRM access to both node properties AND pairwise
    relationships in a single flat sequence, without needing a separate
    graph attention mechanism.

    Returns:
        node_feats  : (n_nodes, NODE_FEAT_DIM)         float tensor
        edge_feats  : (n_nodes, n_nodes, EDGE_FEAT_DIM) float tensor (dense)
        n_nodes     : int
        positions   : list of (type, index) where type is 'node' or 'edge'
    """
    nodes = graph["nodes"]
    n_nodes = min(len(nodes), max_nodes)

    if n_nodes == 0:
        return None

    # Node features — (n_nodes, 22)
    node_feats = torch.tensor(
        graph["node_features"][:n_nodes], dtype=torch.float32
    )

    # Edge features — build dense (n_nodes, n_nodes, 12) matrix
    edge_feats = torch.zeros(n_nodes, n_nodes, EDGE_FEAT_DIM, dtype=torch.float32)
    edge_index = graph["edge_index"]
    ef_list    = graph["edge_features"]

    for e_idx in range(len(ef_list)):
        s = edge_index[0][e_idx]
        d = edge_index[1][e_idx]
        if s < n_nodes and d < n_nodes:
            edge_feats[s, d] = torch.tensor(ef_list[e_idx], dtype=torch.float32)

    return node_feats, edge_feats, n_nodes


def build_sequence_tensors(
    node_feats: torch.Tensor,   # (n_nodes, 22)
    edge_feats: torch.Tensor,   # (n_nodes, n_nodes, 12)
    n_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build the interleaved node+edge feature sequence for one graph.

    Layout: [node_0, edge_0→1, edge_0→2, ..., node_1, edge_1→0, ...]

    Returns:
        seq_feats : (seq_len, max(NODE_FEAT_DIM, EDGE_FEAT_DIM)) padded
        seq_types : (seq_len,) int — 0=node, 1=edge
    """
    tokens = []
    types  = []

    for n in range(n_nodes):
        # Node token
        nf = node_feats[n]                          # (22,)
        tokens.append(nf)
        types.append(0)   # node

        # Outgoing edge tokens — sorted by L2 distance, capped at MAX_EDGES_PER_NODE
        # This prevents sequence length exploding to n^2 for large graphs
        dists = []
        for m in range(n_nodes):
            if m == n:
                continue
            ef = edge_feats[n, m]
            # dx=ef[0], dy=ef[1] — use these for sorting by proximity
            dist = (ef[0]**2 + ef[1]**2).item()
            dists.append((dist, m))
        dists.sort()   # closest edges first

        for _, m in dists[:MAX_EDGES_PER_NODE]:
            ef = edge_feats[n, m]                   # (12,)
            tokens.append(ef)
            types.append(1)   # edge

    if not tokens:
        return None, None

    # Pad all tokens to NODE_FEAT_DIM (22) so they can be stacked
    padded = []
    for t, tp in zip(tokens, types):
        if tp == 0:
            padded.append(t)                        # already 22-dim
        else:
            # Pad 12-dim edge features to 22-dim with zeros
            padded.append(F.pad(t, (0, NODE_FEAT_DIM - EDGE_FEAT_DIM)))

    seq_feats = torch.stack(padded)                 # (seq_len, 22)
    seq_types = torch.tensor(types, dtype=torch.long)

    return seq_feats, seq_types


# ─────────────────────────────────────────────────────────────────────────────
# 2.  AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def augment_pair(input_grid, output_grid):
    """Dihedral group transform applied identically to input and output."""
    k    = random.randint(0, 3)
    flip = random.random() > 0.5

    def transform(grid):
        for _ in range(k):
            grid = [list(row) for row in zip(*grid[::-1])]
        if flip:
            grid = [row[::-1] for row in grid]
        return grid

    return transform(input_grid), transform(output_grid)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DATASET
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)


def encode_grid_to_graph(raw_grid, max_cells_per_node=MAX_CELLS_PER_NODE):
    """Convert raw grid → graph dict via custom_object3."""
    grid  = tuple(tuple(row) for row in raw_grid)
    graph = grid_to_graph(grid, max_cells_per_node=max_cells_per_node)
    return graph if len(graph["nodes"]) > 0 else None


def build_task_samples(
    train_path: str,
    max_files: int = None,
    n_augmentations: int = N_AUGMENTATIONS,
    max_demos: int = MAX_DEMOS,
) -> List[Dict]:
    """
    Build augmented samples. Each sample stores:
        demo_input_graphs  : list of graph dicts (up to max_demos)
        demo_output_graphs : list of graph dicts
        test_input_graph   : graph dict
        test_output_graph  : graph dict  (supervision target)
        test_h, test_w     : output grid dimensions
        test_raw           : raw test output grid (for exact-match eval)
        task_id            : filename without .json
    """
    files = sorted([f for f in os.listdir(train_path) if f.endswith(".json")]) #os.listdir: returns list of the names of the entries in a directory
    if max_files is not None:
        files = files[:max_files] #only take max_files amount of samples if we set a max_files quantity

    samples, skipped = [], 0
    n_files = len(files)

    for file_idx, fname in enumerate(files): # enumerate turns each element in files into an (idx, value) object
        if file_idx % 50 == 0:
            print(f"  Processing file {file_idx+1}/{n_files}  "
                  f"({len(samples)} samples so far)...")

        task    = load_json(os.path.join(train_path, fname))
        task_id = fname.replace(".json", "")

        train_pairs = task.get("train", [])
        test_pairs  = task.get("test",  [])

        if not train_pairs or not test_pairs: # skip tasks if they are missing train_pairs or test_pairs
            skipped += 1
            continue #skip remainder of current iteration 

        test_pair = test_pairs[0]
        if "output" not in test_pair: #if test pair only has an input grid but no output grid skip it, this has a bit of a hole though because some tasks have more than one test pair
            skipped += 1
            continue

        test_h = len(test_pair["output"])
        test_w = len(test_pair["output"][0])

        for aug_idx in range(n_augmentations):
            demo_input_graphs  = []
            demo_output_graphs = []
            ok = True

            for pair in train_pairs[:max_demos]:
                if aug_idx == 0:
                    in_grid  = pair["input"]
                    out_grid = pair["output"]
                else:
                    in_grid, out_grid = augment_pair(pair["input"], pair["output"])

                g_in  = encode_grid_to_graph(in_grid)
                g_out = encode_grid_to_graph(out_grid)

                if g_in is None or g_out is None:
                    ok = False
                    break

                # Cap to MAX_NODES
                if (len(g_in["nodes"])  > MAX_NODES or
                    len(g_out["nodes"]) > MAX_NODES):
                    ok = False
                    break

                demo_input_graphs.append(g_in)
                demo_output_graphs.append(g_out)

            if not ok:
                skipped += 1
                continue

            # Test pair
            if aug_idx == 0:
                test_in_raw  = test_pair["input"]
                test_out_raw = test_pair["output"]
            else:
                test_in_raw, test_out_raw = augment_pair(
                    test_pair["input"], test_pair["output"]
                )

            g_test_in  = encode_grid_to_graph(test_in_raw)
            g_test_out = encode_grid_to_graph(test_out_raw)

            if g_test_in is None or g_test_out is None:
                skipped += 1
                continue

            if (len(g_test_in["nodes"])  > MAX_NODES or
                len(g_test_out["nodes"]) > MAX_NODES):
                skipped += 1
                continue

            samples.append({
                "task_id"           : task_id,
                "demo_input_graphs" : demo_input_graphs,
                "demo_output_graphs": demo_output_graphs,
                "test_input_graph"  : g_test_in,
                "test_output_graph" : g_test_out,
                "test_h"            : test_h,
                "test_w"            : test_w,
                "test_raw"          : test_pair["output"],
            })

    print(f"  Samples built : {len(samples)}  (skipped {skipped})")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 4.  COLLATION
# ─────────────────────────────────────────────────────────────────────────────

def graph_to_targets(graph: dict, max_nodes: int, max_cells: int):
    """
    Build padded supervision targets from one output graph.
    Matches the target format used in train_gat_vae3.py.

    Returns:
        target_existence  : (max_nodes,)              float
        target_color      : (max_nodes,)              long
        target_cell_coords: (max_nodes, max_cells, 2) float  normalised /30
        target_cell_colors: (max_nodes, max_cells)    long
        target_cell_mask  : (max_nodes, max_cells)    float
    """
    nodes   = graph["nodes"]
    n_nodes = min(len(nodes), max_nodes)

    target_existence   = torch.zeros(max_nodes,                       dtype=torch.float32)
    target_color       = torch.zeros(max_nodes,                       dtype=torch.long)
    target_cell_coords = torch.zeros(max_nodes, max_cells, 2,         dtype=torch.float32)
    target_cell_colors = torch.zeros(max_nodes, max_cells,            dtype=torch.long)
    target_cell_mask   = torch.zeros(max_nodes, max_cells,            dtype=torch.float32)

    for ni in range(n_nodes):
        node = nodes[ni]
        target_existence[ni] = 1.0
        # Color = first (dominant) color of the object
        if node["colors"]:
            target_color[ni] = max(0, min(NUM_COLORS - 1, node["colors"][0]))

        # Cell coordinate targets (normalised by 30)
        cc = torch.tensor(node["cell_coords"],      dtype=torch.float32)  # (max_cells, 2)
        cl = torch.tensor(node["cell_colors_list"], dtype=torch.long).clamp(0, NUM_COLORS - 1)
        cm = torch.tensor(node["cell_mask"],        dtype=torch.float32)

        target_cell_coords[ni] = cc / 30.0
        target_cell_colors[ni] = cl
        target_cell_mask[ni]   = cm

    return (target_existence, target_color,
            target_cell_coords, target_cell_colors, target_cell_mask)


def collate_batch(samples: List[Dict], device: str):
    """
    Build batch tensors from a list of task samples.

    Context sequence layout per sample:
        [puzzle_emb_tokens (PUZZLE_EMB_LEN)]
        [demo1_in_seq] [SEP] [demo1_out_seq] [SEP]
        ...
        [test_in_seq]

    Because graph sequences vary in length (different numbers of nodes),
    we pad all sequences to the maximum length in the batch.

    Returns:
        seq_feats    : (B, max_seq_len, NODE_FEAT_DIM)  float
        seq_types    : (B, max_seq_len)                 long  0=node,1=edge,2=puzzle,3=SEP
        pad_mask     : (B, max_seq_len)                 bool  True=padding
        targets      : dict of (B, max_nodes, ...) tensors
        test_sizes   : list of (h, w)
        test_raws    : list of raw grids
        n_test_nodes : (B,) int — number of real nodes in test output
    """
    B = len(samples)

    # Pre-compute sequences for all graphs in all samples
    all_seqs   = []   # list of lists of (feat_tensor, type_int) pairs
    all_targets = []

    for s in samples:
        sample_seq = []

        # Puzzle embedding placeholders — filled by the model's emb table
        # We use a special marker tensor of zeros with type=2
        for pe in range(PUZZLE_EMB_LEN):
            sample_seq.append((
                torch.zeros(NODE_FEAT_DIM, dtype=torch.float32),
                2,   # type: puzzle
            ))

        # Demo pairs
        for g_in, g_out in zip(s["demo_input_graphs"], s["demo_output_graphs"]):
            result_in = graph_to_token_features(g_in)
            if result_in is not None:
                nf, ef, nn = result_in
                sf, st = build_sequence_tensors(nf, ef, nn)
                if sf is not None:
                    for f, t in zip(sf, st):
                        sample_seq.append((f, t.item()))

            # SEP token between input and output
            sample_seq.append((torch.zeros(NODE_FEAT_DIM), 3))

            result_out = graph_to_token_features(g_out)
            if result_out is not None:
                nf, ef, nn = result_out
                sf, st = build_sequence_tensors(nf, ef, nn)
                if sf is not None:
                    for f, t in zip(sf, st):
                        sample_seq.append((f, t.item()))

            # SEP between demo pairs
            sample_seq.append((torch.zeros(NODE_FEAT_DIM), 3))

        # Test input
        result_test = graph_to_token_features(s["test_input_graph"])
        if result_test is not None:
            nf, ef, nn = result_test
            sf, st = build_sequence_tensors(nf, ef, nn)
            if sf is not None:
                for f, t in zip(sf, st):
                    sample_seq.append((f, t.item()))

        all_seqs.append(sample_seq)

        # Targets from test output graph
        tgts = graph_to_targets(
            s["test_output_graph"], MAX_NODES, MAX_CELLS_PER_NODE
        )
        all_targets.append(tgts)

    # Pad all sequences to max length
    max_len = max(len(s) for s in all_seqs)

    seq_feats = torch.zeros(B, max_len, NODE_FEAT_DIM, dtype=torch.float32)
    seq_types = torch.zeros(B, max_len,                dtype=torch.long)
    pad_mask  = torch.ones( B, max_len,                dtype=torch.bool)   # True=pad

    for i, seq in enumerate(all_seqs):
        L = len(seq)
        for j, (feat, typ) in enumerate(seq):
            seq_feats[i, j] = feat
            seq_types[i, j] = typ
        pad_mask[i, :L] = False

    # Stack targets
    (tex, tcol, tcc, tcl, tcm) = zip(*all_targets)
    targets = {
        "existence"   : torch.stack(tex).to(device),   # (B, N)
        "color"       : torch.stack(tcol).to(device),  # (B, N)
        "cell_coords" : torch.stack(tcc).to(device),   # (B, N, MC, 2)
        "cell_colors" : torch.stack(tcl).to(device),   # (B, N, MC)
        "cell_mask"   : torch.stack(tcm).to(device),   # (B, N, MC)
    }

    test_sizes = [(s["test_h"], s["test_w"]) for s in samples]
    test_raws  = [s["test_raw"] for s in samples]

    return (
        seq_feats.to(device),
        seq_types.to(device),
        pad_mask.to(device),
        targets,
        test_sizes,
        test_raws,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5.  GRAPH TRM MODEL
# ─────────────────────────────────────────────────────────────────────────────

class GraphTRMBlock(nn.Module):
    """
    Shared recursive transformer block — identical structure to arc_trm.py
    but operates on projected graph token sequences instead of pixel tokens.
    """

    def __init__(self, hidden_size=HIDDEN_SIZE, n_heads=N_HEADS,
                 ff_dim=FF_DIM, dropout=DROPOUT, t_inner=T_INNER):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=t_inner, norm=nn.LayerNorm(hidden_size),
        )

    def forward(self, z_h, src_key_padding_mask=None):
        return self.transformer(z_h, src_key_padding_mask=src_key_padding_mask)


class GraphTinyRecursiveModel(nn.Module):
    """
    Graph-based Tiny Recursive Model.

    Input: interleaved node+edge feature sequence from custom_object3.py
    Output: per-node predictions (existence, color, cell coords)

    Key differences from TinyRecursiveModel in arc_trm.py:
        1. Input projection: linear layer maps raw float features → hidden
           (instead of an embedding table for integer tokens)
        2. Type embedding: 4 types (node=0, edge=1, puzzle=2, SEP=3)
        3. Puzzle embedding: PUZZLE_EMB_LEN learned vectors prepended
        4. Output heads: predict per-node existence, color, cell coords
           (same target format as the VAE decoder)
        5. Halt head: same ACT-style mechanism as arc_trm.py
    """

    def __init__(
        self,
        hidden_size: int  = HIDDEN_SIZE,
        n_heads: int      = N_HEADS,
        ff_dim: int       = FF_DIM,
        dropout: float    = DROPOUT,
        n_recursions: int = N_RECURSIONS,
        t_inner: int      = T_INNER,
        n_sup: int        = N_SUP,
        max_seq_len: int  = 4096,   # large enough for capped graph sequences
        max_nodes: int    = MAX_NODES,
        max_cells: int    = MAX_CELLS_PER_NODE,
        num_colors: int   = NUM_COLORS,
    ):
        super().__init__()

        self.hidden_size  = hidden_size
        self.n_recursions = n_recursions
        self.n_sup        = n_sup
        self.max_nodes    = max_nodes
        self.max_cells    = max_cells
        self.num_colors   = num_colors

        # ── input side ────────────────────────────────────────────────────

        # Project raw float features (22-dim) → hidden
        self.feat_proj = nn.Linear(NODE_FEAT_DIM, hidden_size)

        # Type embedding: node / edge / puzzle / SEP
        self.type_emb = nn.Embedding(4, hidden_size)

        # Puzzle token embeddings (PUZZLE_EMB_LEN separate learned vectors)
        # These give the model a way to identify which task it's solving
        self.puzzle_emb = nn.Embedding(PUZZLE_EMB_LEN, hidden_size)

        # Positional embedding (learned)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)

        # ── shared recursive block ─────────────────────────────────────────
        self.block = GraphTRMBlock(hidden_size, n_heads, ff_dim, dropout, t_inner)

        # ── halt head ─────────────────────────────────────────────────────
        self.q_head = nn.Linear(hidden_size, 2, bias=True)

        # ── output heads (per-node predictions) ───────────────────────────
        # These operate on the first MAX_NODES positions of z_h that
        # correspond to node tokens in the context sequence.
        # We use a dedicated "node readout" MLP that maps from z_h[:, 0:1]
        # (the first hidden state, same as the halt head) to per-node outputs.
        # More precisely: we pool over node-type positions, then decode.

        self.node_existence_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, max_nodes),
        )
        self.node_color_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, max_nodes * num_colors),
        )
        self.node_cell_coord_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2), nn.SiLU(),
            nn.Linear(hidden_size * 2, max_nodes * max_cells * 2),
        )
        self.node_cell_color_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, max_nodes * max_cells * num_colors),
        )
        self.node_cell_mask_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, max_nodes * max_cells),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.type_emb.weight,   std=0.02)
        nn.init.normal_(self.puzzle_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight,    std=0.02)
        nn.init.zeros_(self.q_head.bias)

    def _embed_input(self, seq_feats, seq_types):
        """
        seq_feats : (B, L, NODE_FEAT_DIM)
        seq_types : (B, L)  0=node, 1=edge, 2=puzzle, 3=SEP
        """
        B, L, _ = seq_feats.shape
        device   = seq_feats.device

        # Project continuous features
        x = self.feat_proj(seq_feats)   # (B, L, H)

        # Replace puzzle positions with learned puzzle embeddings
        puzzle_mask = (seq_types == 2)  # (B, L)
        if puzzle_mask.any():
            # Assign puzzle position indices 0..PUZZLE_EMB_LEN-1
            puzzle_ids = torch.zeros_like(seq_types)
            for b in range(B):
                cnt = 0
                for l in range(L):
                    if seq_types[b, l] == 2:
                        puzzle_ids[b, l] = cnt
                        cnt = min(cnt + 1, PUZZLE_EMB_LEN - 1)
            x = x + puzzle_mask.unsqueeze(-1).float() * self.puzzle_emb(puzzle_ids)

        # Add type embedding
        x = x + self.type_emb(seq_types)

        # Add positional embedding
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(pos_ids)

        return x   # (B, L, H)

    def _decode_from_hidden(self, z_h):
        """
        Decode per-node predictions from the hidden state.

        We use the mean of all node-type positions (type=0) across the
        last demo + test input portion of the sequence. In practice we
        simply pool z_h across the sequence dimension and decode from
        that summary, which is simple and effective.

        z_h : (B, L, H)
        Returns dict of prediction tensors.
        """
        # Global mean pool over sequence → (B, H)
        h = z_h.mean(dim=1)

        B = h.shape[0]
        N = self.max_nodes
        MC = self.max_cells
        C  = self.num_colors

        existence_logits  = self.node_existence_head(h)           # (B, N)
        color_logits      = self.node_color_head(h).view(B, N, C) # (B, N, C)
        cell_coord_logits = self.node_cell_coord_head(h).view(B, N, MC, 2)
        cell_color_logits = self.node_cell_color_head(h).view(B, N, MC, C)
        cell_mask_logits  = self.node_cell_mask_head(h).view(B, N, MC)

        return {
            "existence"        : existence_logits,
            "color"            : color_logits,
            "cell_coords"      : cell_coord_logits,
            "cell_color_logits": cell_color_logits,
            "cell_mask"        : cell_mask_logits,
        }

    def forward(self, seq_feats, seq_types, pad_mask, targets=None, n_sup=None, return_debug=False):
        """
        seq_feats : (B, L, NODE_FEAT_DIM)
        seq_types : (B, L)
        pad_mask  : (B, L)  True=padding position
        targets   : dict of target tensors (None at inference)
        n_sup     : supervision steps (defaults to self.n_sup)

        Returns:
            loss    : scalar
            preds   : dict of prediction tensors (from final step)
        """
        if n_sup is None:
            n_sup = self.n_sup

        z_h = self._embed_input(seq_feats, seq_types)   # (B, L, H)

        debug_info = {
            "embed_norm": tensor_l2_norm(z_h),
            "step_activation_norms": [],
            "step_metrics": [],
            "final_activation_norm": 0.0,
            "halt_prob_mean": 0.0,
        }

        # ACT accumulation
        cumulative_preds    = None
        halt_prob_remaining = torch.ones(z_h.shape[0], 1, device=z_h.device)
        step_losses         = []
        preds               = None

        for step in range(self.n_recursions):
            z_h = self.block(z_h, src_key_padding_mask=pad_mask)
            debug_info["step_activation_norms"].append(tensor_l2_norm(z_h))

            # Halt decision
            halt_logits = self.q_head(z_h[:, 0, :])              # (B, 2)
            p_halt = torch.sigmoid(
                halt_logits[:, 0] - halt_logits[:, 1]
            ).unsqueeze(-1)                                        # (B, 1)
            debug_info["halt_prob_mean"] += float(p_halt.mean().item())

            # Decode predictions at this step
            preds = self._decode_from_hidden(z_h)
            if return_debug and targets is not None:
                debug_info["step_metrics"].append(compute_prediction_metrics(preds, targets))

            # Accumulate weighted predictions
            # w shape: (B, 1) — broadcast over all prediction dimensions
            w = halt_prob_remaining * p_halt   # (B, 1)

            if cumulative_preds is None:
                cumulative_preds = {}
                for k, v in preds.items():
                    weight = w.view(w.shape[0], *([1] * (v.dim() - 1)))
                    cumulative_preds[k] = weight * v
            else:
                for k in preds:
                    v      = preds[k]
                    weight = w.view(w.shape[0], *([1] * (v.dim() - 1)))
                    cumulative_preds[k] = cumulative_preds[k] + weight * v

            halt_prob_remaining = halt_prob_remaining * (1.0 - p_halt)

            # Supervision schedule loss
            if targets is not None and step < n_sup:
                step_loss = self._compute_loss(preds, targets)
                weight    = 1.0 / (step + 1)
                step_losses.append(weight * step_loss)

        # Flush remainder to final prediction
        for k in preds:
            v      = preds[k]
            weight = halt_prob_remaining.view(halt_prob_remaining.shape[0], *([1] * (v.dim() - 1)))
            cumulative_preds[k] = cumulative_preds[k] + weight * v

        debug_info["final_activation_norm"] = tensor_l2_norm(z_h)
        debug_info["halt_prob_mean"] = debug_info["halt_prob_mean"] / max(self.n_recursions, 1)

        # Final loss
        if targets is not None:
            main_loss  = self._compute_loss(cumulative_preds, targets)
            total_loss = main_loss
            if step_losses:
                total_loss = total_loss + sum(step_losses) / len(step_losses)
        else:
            total_loss = torch.tensor(0.0, device=z_h.device)

        if return_debug:
            return total_loss, cumulative_preds, debug_info
        return total_loss, cumulative_preds

    def _compute_loss(self, preds, targets):
        """
        Multi-head loss matching the target format from graph_to_targets().

        Losses:
            existence BCE  — all node slots
            color CE       — real nodes only (masked)
            cell coord MSE — real cells only (masked)
            cell color CE  — real cells only (masked)
            cell mask BCE  — all cell slots
        """
        B, N = targets["existence"].shape
        MC   = targets["cell_mask"].shape[-1]
        C    = self.num_colors

        exist_tgt = targets["existence"]   # (B, N)
        node_mask = exist_tgt.bool()       # (B, N)

        # Existence BCE
        loss_exist = F.binary_cross_entropy_with_logits(
            preds["existence"], exist_tgt,
        )

        # Color CE (real nodes only)
        color_logits_flat = preds["color"].view(B*N, C)
        color_tgt_flat    = targets["color"].view(B*N)
        node_mask_flat    = node_mask.view(B*N)
        if node_mask_flat.any():
            loss_color = F.cross_entropy(
                color_logits_flat[node_mask_flat],
                color_tgt_flat[node_mask_flat],
            )
        else:
            loss_color = preds["color"].sum() * 0.0

        # Cell mask BCE
        loss_cell_mask = F.binary_cross_entropy_with_logits(
            preds["cell_mask"], targets["cell_mask"],
        )

        # Cell mask for real cells
        cell_mask_bool = targets["cell_mask"].bool()   # (B, N, MC)

        # Cell coord MSE (real cells of real nodes)
        real_cell_mask = node_mask.unsqueeze(-1) & cell_mask_bool   # (B, N, MC)
        if real_cell_mask.any():
            loss_coords = F.mse_loss(
                preds["cell_coords"][real_cell_mask],
                targets["cell_coords"][real_cell_mask],
            )
        else:
            loss_coords = preds["cell_coords"].sum() * 0.0

        # Cell color CE (real cells of real nodes)
        cell_color_logits_flat = preds["cell_color_logits"].view(B*N*MC, C)
        cell_color_tgt_flat    = targets["cell_colors"].view(B*N*MC)
        real_cell_flat         = real_cell_mask.view(B*N*MC)
        if real_cell_flat.any():
            loss_cell_color = F.cross_entropy(
                cell_color_logits_flat[real_cell_flat],
                cell_color_tgt_flat[real_cell_flat],
            )
        else:
            loss_cell_color = preds["cell_color_logits"].sum() * 0.0

        total = (
            1.0 * loss_exist
          + 1.0 * loss_color
          + 1.0 * loss_cell_mask
          + 0.5 * loss_coords
          + 1.0 * loss_cell_color
        )
        return total




# ─────────────────────────────────────────────────────────────────────────────
# 5B.  DEBUG / DIAGNOSTIC HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def tensor_l2_norm(x: Optional[torch.Tensor]) -> float:
    if x is None:
        return 0.0
    return float(x.detach().float().norm().item())


def safe_mean(values):
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


def compute_global_grad_norm(model: nn.Module) -> float:
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach().float()
            total_sq += float(torch.sum(g * g).item())
    return total_sq ** 0.5


def compute_headline_grad_norms(model: nn.Module) -> Dict[str, float]:
    groups = {
        "feat_proj": [model.feat_proj.weight, model.feat_proj.bias],
        "block": list(model.block.parameters()),
        "halt_head": list(model.q_head.parameters()),
        "output_heads": (
            list(model.node_existence_head.parameters())
            + list(model.node_color_head.parameters())
            + list(model.node_cell_coord_head.parameters())
            + list(model.node_cell_color_head.parameters())
            + list(model.node_cell_mask_head.parameters())
        ),
    }
    out = {}
    for name, params in groups.items():
        total_sq = 0.0
        for p in params:
            if p is not None and getattr(p, "grad", None) is not None:
                g = p.grad.detach().float()
                total_sq += float(torch.sum(g * g).item())
        out[name] = total_sq ** 0.5
    return out


def compute_prediction_metrics(preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
    metrics = {}
    exist_prob = torch.sigmoid(preds["existence"])
    exist_pred = (exist_prob >= 0.5).float()
    exist_tgt  = targets["existence"].float()
    metrics["existence_acc"] = float((exist_pred == exist_tgt).float().mean().item())

    node_mask = targets["existence"].bool()
    if node_mask.any():
        color_pred = preds["color"].argmax(dim=-1)
        color_tgt  = targets["color"]
        metrics["node_color_acc"] = float((color_pred[node_mask] == color_tgt[node_mask]).float().mean().item())
    else:
        metrics["node_color_acc"] = 0.0

    cell_mask_tgt = targets["cell_mask"].bool()
    cell_mask_prob = torch.sigmoid(preds["cell_mask"])
    cell_mask_pred = cell_mask_prob >= 0.5
    metrics["cell_mask_acc"] = float((cell_mask_pred == cell_mask_tgt).float().mean().item())

    real_cell_mask = node_mask.unsqueeze(-1) & cell_mask_tgt
    if real_cell_mask.any():
        cell_color_pred = preds["cell_color_logits"].argmax(dim=-1)
        cell_color_tgt  = targets["cell_colors"]
        metrics["cell_color_acc"] = float((cell_color_pred[real_cell_mask] == cell_color_tgt[real_cell_mask]).float().mean().item())
        coord_mae = (preds["cell_coords"][real_cell_mask] - targets["cell_coords"][real_cell_mask]).abs().mean()
        metrics["cell_coord_mae"] = float(coord_mae.item())
    else:
        metrics["cell_color_acc"] = 0.0
        metrics["cell_coord_mae"] = 0.0

    return metrics


def compute_ece(confidences: torch.Tensor, correctness: torch.Tensor, n_bins: int = CALIBRATION_BINS) -> float:
    if confidences.numel() == 0:
        return 0.0
    confidences = confidences.detach().float().flatten().clamp(0.0, 1.0)
    correctness = correctness.detach().float().flatten().clamp(0.0, 1.0)
    bin_edges = torch.linspace(0, 1, n_bins + 1, device=confidences.device)
    ece = torch.tensor(0.0, device=confidences.device)
    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if mask.any():
            bin_conf = confidences[mask].mean()
            bin_acc  = correctness[mask].mean()
            ece += mask.float().mean() * torch.abs(bin_conf - bin_acc)
    return float(ece.item())


def compute_existence_calibration(preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], n_bins: int = CALIBRATION_BINS) -> Dict[str, float]:
    probs = torch.sigmoid(preds["existence"])
    labels = targets["existence"].float()
    pred_labels = (probs >= 0.5).float()
    correctness = (pred_labels == labels).float()
    confidence = torch.where(pred_labels > 0.5, probs, 1.0 - probs)
    return {
        "existence_ece": compute_ece(confidence, correctness, n_bins=n_bins),
        "existence_confidence_mean": float(confidence.mean().item()),
        "existence_accuracy_mean": float(correctness.mean().item()),
    }


def compute_output_validity_stats(preds: Dict[str, torch.Tensor], test_sizes) -> Dict[str, float]:
    total = len(test_sizes)
    if total == 0:
        return {
            "valid_grid_rate": 0.0,
            "coord_in_bounds_rate": 0.0,
            "predicted_active_nodes_mean": 0.0,
        }

    valid_grids = 0
    total_coords = 0
    in_bounds = 0
    active_nodes = []

    existence_prob = torch.sigmoid(preds["existence"])
    cell_mask_prob = torch.sigmoid(preds["cell_mask"])
    coords = preds["cell_coords"] * 30.0

    for i, (h, w) in enumerate(test_sizes):
        try:
            grid = preds_to_grid(preds, h, w, sample_idx=i)
            ok_shape = len(grid) == h and all(len(row) == w for row in grid)
            ok_values = ok_shape and all(0 <= int(v) < NUM_COLORS for row in grid for v in row)
            if ok_shape and ok_values:
                valid_grids += 1
        except Exception:
            pass

        node_on = existence_prob[i] >= 0.5
        active_nodes.append(float(node_on.float().sum().item()))
        coord_i = coords[i]
        mask_i = cell_mask_prob[i] >= 0.5
        if mask_i.any():
            xy = coord_i[mask_i]
            total_coords += int(xy.shape[0])
            x_ok = (xy[:, 0] >= 0) & (xy[:, 0] < h)
            y_ok = (xy[:, 1] >= 0) & (xy[:, 1] < w)
            in_bounds += int((x_ok & y_ok).sum().item())

    return {
        "valid_grid_rate": valid_grids / total,
        "coord_in_bounds_rate": (in_bounds / total_coords) if total_coords > 0 else 1.0,
        "predicted_active_nodes_mean": safe_mean(active_nodes),
    }


def format_metric_dict(d: Dict[str, float], prefix: str = "") -> str:
    pieces = []
    for k, v in d.items():
        if isinstance(v, float):
            pieces.append(f"{prefix}{k}={v:.4f}")
        else:
            pieces.append(f"{prefix}{k}={v}")
    return " | ".join(pieces)

# ─────────────────────────────────────────────────────────────────────────────
# 6.  EMA
# ─────────────────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay  = decay
        self.shadow = {n: p.data.clone()
                       for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n] = self.decay * self.shadow[n] + (1-self.decay) * p.data

    def apply_shadow(self, model):
        self._backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self._backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if n in self._backup:
                p.data.copy_(self._backup[n])
        self._backup = {}


# ─────────────────────────────────────────────────────────────────────────────
# 7.  GRID RECONSTRUCTION FROM PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────

def preds_to_grid(preds: dict, height: int, width: int, sample_idx: int = 0):
    """
    Convert model predictions → reconstructed ARC grid.
    Uses graph_to_grid_from_predictions from custom_object3.py.
    """
    existence_prob  = torch.sigmoid(preds["existence"][sample_idx]).cpu().tolist()
    cell_mask_prob  = torch.sigmoid(preds["cell_mask"][sample_idx]).cpu().tolist()
    cell_color_pred = preds["cell_color_logits"][sample_idx].argmax(dim=-1).cpu().tolist()
    # Denormalise coords (* 30)
    cell_coords_dn  = (preds["cell_coords"][sample_idx] * 30.0).cpu().tolist()

    return graph_to_grid_from_predictions(
        pred_cell_coords = cell_coords_dn,
        pred_cell_colors = cell_color_pred,
        pred_cell_mask   = cell_mask_prob,
        pred_existence   = existence_prob,
        height=height,
        width=width,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 8.  TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, optimizer, samples, batch_size, device,
              ema=None, train=True, metrics_config=None):
    metrics_config = metrics_config or DEBUG_METRICS
    model.train() if train else model.eval()
    
    if train:
        random.shuffle(samples)  #note we are shuffling the samples, 

    total_loss    = 0.0
    total_batches = 0
    chunks = [samples[i:i+batch_size] for i in range(0, len(samples), batch_size)]
    n_chunks = len(chunks)

    epoch_metrics = {
        "mean_loss": 0.0,
        "gradient_norm": [],
        "gradient_norm_by_group": {},
        "activation_embed_norm": [],
        "activation_final_norm": [],
        "activation_step_norms": [],
        "per_step_metrics": {},
        "calibration": [],
    }

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for chunk_idx, chunk in enumerate(chunks):
            seq_feats, seq_types, pad_mask, targets, _, _ = collate_batch(chunk, device)

            need_debug = any([
                metrics_config.get("activation_norms", False),
                metrics_config.get("per_recursion_step_accuracy", False),
                metrics_config.get("confidence_calibration", False),
            ])

            if need_debug:
                loss, preds, debug_info = model(seq_feats, seq_types, pad_mask, targets, return_debug=True)
            else:
                loss, preds = model(seq_feats, seq_types, pad_mask, targets)
                debug_info = None

            if train:
                if not torch.isfinite(loss):
                    print(f"  WARNING: non-finite loss at batch {chunk_idx+1}, skipping")
                    optimizer.zero_grad()
                    continue

                optimizer.zero_grad()
                loss.backward()

                if metrics_config.get("gradient_norms", False):
                    epoch_metrics["gradient_norm"].append(compute_global_grad_norm(model))
                    grouped = compute_headline_grad_norms(model)
                    for k, v in grouped.items():
                        epoch_metrics["gradient_norm_by_group"].setdefault(k, []).append(v)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                if ema is not None:
                    ema.update(model)

                if (chunk_idx + 1) % PRINT_EVERY == 0 or chunk_idx == 0:
                    print(f"  Batch {chunk_idx+1:4d}/{n_chunks} | loss={loss.item():.4f}")

            if debug_info is not None:
                if metrics_config.get("activation_norms", False):
                    epoch_metrics["activation_embed_norm"].append(debug_info["embed_norm"])
                    epoch_metrics["activation_final_norm"].append(debug_info["final_activation_norm"])
                    epoch_metrics["activation_step_norms"].append(debug_info["step_activation_norms"])
                if metrics_config.get("per_recursion_step_accuracy", False):
                    for step_idx, step_dict in enumerate(debug_info["step_metrics"]):
                        step_store = epoch_metrics["per_step_metrics"].setdefault(step_idx, {})
                        for k, v in step_dict.items():
                            step_store.setdefault(k, []).append(v)
                if metrics_config.get("confidence_calibration", False):
                    epoch_metrics["calibration"].append(compute_existence_calibration(preds, targets))

            if torch.isfinite(loss):
                total_loss    += float(loss.item())
                total_batches += 1

    summary = {"mean_loss": total_loss / max(total_batches, 1)}

    if metrics_config.get("gradient_norms", False):
        summary["gradient_norm"] = safe_mean(epoch_metrics["gradient_norm"])
        summary["gradient_norm_by_group"] = {
            k: safe_mean(v) for k, v in epoch_metrics["gradient_norm_by_group"].items()
        }

    if metrics_config.get("activation_norms", False):
        summary["activation_embed_norm"] = safe_mean(epoch_metrics["activation_embed_norm"])
        summary["activation_final_norm"] = safe_mean(epoch_metrics["activation_final_norm"])
        step_lists = epoch_metrics["activation_step_norms"]
        if step_lists:
            max_steps = max(len(x) for x in step_lists)
            step_means = []
            for i in range(max_steps):
                vals = [x[i] for x in step_lists if i < len(x)]
                step_means.append(safe_mean(vals))
            summary["activation_step_norms"] = step_means
        else:
            summary["activation_step_norms"] = []

    if metrics_config.get("per_recursion_step_accuracy", False):
        summary["per_step_metrics"] = {}
        for step_idx, metric_dict in epoch_metrics["per_step_metrics"].items():
            summary["per_step_metrics"][step_idx] = {k: safe_mean(v) for k, v in metric_dict.items()}

    if metrics_config.get("confidence_calibration", False):
        calib = {}
        for entry in epoch_metrics["calibration"]:
            for k, v in entry.items():
                calib.setdefault(k, []).append(v)
        summary["calibration"] = {k: safe_mean(v) for k, v in calib.items()}

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 9.  EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def grids_match(a, b):
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if list(ra) != list(rb):
            return False
    return True


def evaluate_task(sample, model, device):
    """Run inference on one task. Returns True if exact match."""
    model.eval()
    seq_feats, seq_types, pad_mask, _, test_sizes, test_raws = \
        collate_batch([sample], device)

    with torch.no_grad():
        _, preds = model(seq_feats, seq_types, pad_mask, targets=None)

    h, w = test_sizes[0]
    predicted = preds_to_grid(preds, h, w, sample_idx=0)
    return grids_match(predicted, test_raws[0])




def run_tiny_subset_diagnostics(model, samples, device, metrics_config):
    if not samples:
        return {}
    subset = samples[:min(TINY_SUBSET_SIZE, len(samples))]
    summary = run_epoch(
        model, None, subset, TINY_SUBSET_BATCH_SIZE, device,
        ema=None, train=False, metrics_config=metrics_config,
    )
    out = {
        "tiny_subset_loss": summary["mean_loss"],
    }
    unique_subset = {s["task_id"]: s for s in subset}
    eval_tasks = list(unique_subset.values())
    if eval_tasks:
        n_correct = sum(evaluate_task(s, model, device) for s in eval_tasks)
        out["tiny_subset_exact_match"] = n_correct / len(eval_tasks)
    return out


def summarize_epoch_diagnostics(label: str, summary: Dict, metrics_config: Dict):
    parts = [f"{label}_loss={summary['mean_loss']:.4f}"]

    if metrics_config.get("gradient_norms", False) and "gradient_norm" in summary:
        parts.append(f"{label}_grad_norm={summary['gradient_norm']:.4f}")
        for k, v in summary.get("gradient_norm_by_group", {}).items():
            parts.append(f"{label}_{k}_grad={v:.4f}")

    if metrics_config.get("activation_norms", False):
        parts.append(f"{label}_embed_norm={summary.get('activation_embed_norm', 0.0):.4f}")
        parts.append(f"{label}_final_act_norm={summary.get('activation_final_norm', 0.0):.4f}")
        step_norms = summary.get("activation_step_norms", [])
        if step_norms:
            parts.append(f"{label}_step_act_norms=[" + ", ".join(f"{v:.3f}" for v in step_norms) + "]")

    if metrics_config.get("confidence_calibration", False) and "calibration" in summary:
        calib = summary["calibration"]
        parts.append(f"{label}_ece={calib.get('existence_ece', 0.0):.4f}")
        parts.append(f"{label}_conf={calib.get('existence_confidence_mean', 0.0):.4f}")
        parts.append(f"{label}_conf_acc={calib.get('existence_accuracy_mean', 0.0):.4f}")

    print(" | ".join(parts))

    if metrics_config.get("per_recursion_step_accuracy", False):
        per_step = summary.get("per_step_metrics", {})
        if per_step:
            print(f"  {label} per-step metrics:")
            for step_idx in sorted(per_step.keys()):
                metric_dict = per_step[step_idx]
                print(f"    step {step_idx+1}: {format_metric_dict(metric_dict)}")

# ─────────────────────────────────────────────────────────────────────────────
# 10.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    try:
           
        set_seed(SEED)


        print(f"Device          : {DEVICE}")
        print(f"Hidden size     : {HIDDEN_SIZE}")
        print(f"Recursions      : {N_RECURSIONS}")
        print(f"Inner layers    : {T_INNER}")
        print(f"Node feat dim   : {NODE_FEAT_DIM}")
        print(f"Edge feat dim   : {EDGE_FEAT_DIM}")
        print(f"Max nodes       : {MAX_NODES}")
        print(f"Augmentations   : {N_AUGMENTATIONS}")
        print(f"Batch size      : {BATCH_SIZE}")
        print(f"Debug metrics   : {DEBUG_METRICS}\n")

        # ── build dataset ─────────────────────────────────────────────────────
        print("Building graph task samples...")
        all_samples = build_task_samples(
            TRAIN_PATH,
            max_files=MAX_FILES,
            n_augmentations=N_AUGMENTATIONS,
            max_demos=MAX_DEMOS,
        )

        if not all_samples:
            raise RuntimeError("No samples built. Check TRAIN_PATH.")

        random.shuffle(all_samples)
        split_idx     = max(1, int(0.9 * len(all_samples)))
        
        train_samples = all_samples[:split_idx]
        val_samples   = all_samples[split_idx:] if split_idx < len(all_samples) else all_samples[:1]
        if DEBUG_METRICS.get("gradient_norms", False):
            train_samples = all_samples[:TINY_SUBSET_SIZE]
            val_samples = all_samples[:TINY_SUBSET_SIZE]

        print(f"Train samples   : {len(train_samples)}")
        print(f"Val samples     : {len(val_samples)}\n")

        # ── build model ───────────────────────────────────────────────────────
        model = GraphTinyRecursiveModel(
            hidden_size  = HIDDEN_SIZE,
            n_heads      = N_HEADS,
            ff_dim       = FF_DIM,
            dropout      = DROPOUT,
            n_recursions = N_RECURSIONS,
            t_inner      = T_INNER,
            n_sup        = N_SUP,
            max_nodes    = MAX_NODES,
            max_cells    = MAX_CELLS_PER_NODE,
            num_colors   = NUM_COLORS,
        ).to(DEVICE)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model params    : {n_params:,}\n")

        # ── optimizer with separate embedding LR ──────────────────────────────
        emb_params   = (list(model.puzzle_emb.parameters()) +
                        list(model.type_emb.parameters()) +
                        list(model.pos_emb.parameters()))
        other_params = [p for p in model.parameters()
                        if p.requires_grad and
                        not any(p is ep for ep in emb_params)]

        optimizer = torch.optim.AdamW([
            {"params": emb_params,   "lr": EMBED_LR},
            {"params": other_params, "lr": LR},
        ], betas=(0.9, 0.95), weight_decay=0.1)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=LR/10,
        )

        ema      = EMA(model, decay=EMA_DECAY) if USE_EMA else None
        best_val = math.inf

        print("Starting Graph TRM training...\n")
        start = datetime.now() 
        for epoch in range(1, EPOCHS + 1):
            train_summary = run_epoch(model, optimizer, train_samples,
                                BATCH_SIZE, DEVICE, ema=ema, train=True, metrics_config=DEBUG_METRICS)
            train_loss = train_summary["mean_loss"]

            if ema is not None:
                ema.apply_shadow(model)

            val_summary = run_epoch(model, None, val_samples,
                                BATCH_SIZE, DEVICE, ema=None, train=False, metrics_config=DEBUG_METRICS)
            val_loss = val_summary["mean_loss"]
            print(datetime.now() - start)
            gap = train_loss - val_loss
            summarize_epoch_diagnostics("train", train_summary, DEBUG_METRICS)
            summarize_epoch_diagnostics("val", val_summary, DEBUG_METRICS)

            epoch_parts = [
                f"Epoch {epoch:03d}",
                f"train={train_loss:.4f}",
                f"val={val_loss:.4f}",
            ]
            if DEBUG_METRICS.get("train_val_gap", False):
                epoch_parts.append(f"gap={gap:.4f}")

            if epoch % 10 == 0:
                unique_val = {s["task_id"]: s for s in val_samples}
                eval_tasks = list(unique_val.values())[:50]
                n_correct  = sum(evaluate_task(s, model, DEVICE) for s in eval_tasks)
                acc = 100.0 * n_correct / max(len(eval_tasks), 1)
                epoch_parts.append(f"exact_match={n_correct}/{len(eval_tasks)} ({acc:.1f}%)")

            print(" | ".join(epoch_parts))

            if DEBUG_METRICS.get("tiny_subset_overfit", False):
                tiny = run_tiny_subset_diagnostics(model, train_samples, DEVICE, DEBUG_METRICS)
                print(f"  tiny_subset_loss={tiny.get('tiny_subset_loss', 0.0):.4f} | tiny_subset_exact_match={100.0 * tiny.get('tiny_subset_exact_match', 0.0):.1f}%")

            if DEBUG_METRICS.get("output_validity", False):
                validity_subset = val_samples[:min(VALIDITY_EVAL_TASKS, len(val_samples))]
                if validity_subset:
                    seq_feats, seq_types, pad_mask, _, test_sizes, _ = collate_batch(validity_subset, DEVICE)
                    with torch.no_grad():
                        _, preds = model(seq_feats, seq_types, pad_mask, targets=None)
                    validity = compute_output_validity_stats(preds, test_sizes)
                    print(f"  validity: {format_metric_dict(validity)}")

            if val_loss < best_val:
                best_val = val_loss
                total_time = datetime.now() - start
                now = datetime.now()
                formatted_now = now.strftime("%m_%d_%Y___%H_%M_%S")
                print(formatted_now)
                save_path  = fr"{pathlib.Path(__file__).parent.resolve()}" + fr"\models_folder\trm_{formatted_now}__{round(train_loss,2)}_{round(val_loss,2)}.pt"
                print(save_path)
                torch.save({
                    "model_state_dict"    : model.state_dict(),
                    "ema_shadow"          : ema.shadow if ema else None,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "hidden_size"         : HIDDEN_SIZE,
                    "n_recursions"        : N_RECURSIONS,
                    "t_inner"             : T_INNER,
                    "best_val_loss"       : best_val,
                    "epoch"               : epoch,
                    "description"         : (
                        f"total time = {total_time}; "  
                        f"optimizer = {optimizer if optimizer else 'NA'}; " 
                        f"scheduler = {scheduler if scheduler else 'NA'}; "
                        f"Loss = {val_loss}; "
                        f"EPOCHS = {EPOCHS if EPOCHS else 'NA'}; "
                        f"N_RECURSIONS = {N_RECURSIONS if N_RECURSIONS else 'NA'}; "
                        f"T_INNER = {T_INNER if T_INNER else 'NA'}; "
                        f" N_SUP = {N_SUP if N_SUP else 'NA'}; "
                        f"Stop Epoch = {epoch if epoch else 'NA'}; " 
                        f"LR = {LR}; EMBED_LR = {EMBED_LR if EMBED_LR else 'NA'}; " 
                        f"HIDDEN_SIZE = {HIDDEN_SIZE if HIDDEN_SIZE else 'NA'}; " 
                        f"N_HEADS = {N_HEADS if N_HEADS else 'NA'}; " 
                        f"FF_DIM = {FF_DIM if FF_DIM else 'NA'}; " 
                        f"DROPOUT = {DROPOUT if DROPOUT else 'NA'}; "
                        f"DEBUG_METRICS = {DEBUG_METRICS}; "
                    ),
                }, save_path)
                print(f"  → Saved best Graph TRM (val={best_val:.4f})")

            if ema is not None:
                ema.restore(model)

            scheduler.step()

        print(f"\nTraining complete. Best val loss: {best_val:.4f}")
    except KeyboardInterrupt:
        print("Interrupted by Ctrl + C")
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()