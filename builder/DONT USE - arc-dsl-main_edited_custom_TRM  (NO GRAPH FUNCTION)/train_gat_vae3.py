import os
import json
import math
import random
from typing import List, Tuple, Dict

import torch
from torch_geometric.data import Data, Batch

from custom_object3 import grid_to_graph
from gat_vae3 import GATVAE
from graph_decoder3 import gat_vae_loss


# ============================================================================
# CONFIG
# ============================================================================

TRAIN_PATH = r"C:\Users\tedhun\OneDrive - Wanzl GmbH & Co. KGaA\Microsoft Teams Chat Files\outputs\Documents\GS\ARC\Assignment10\Builder\data\training"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

SEED       = 42
BATCH_SIZE = 8
#EPOCHS     = 40
EPOCHS     = 5
LR         = 1e-3

# v3: fixed capacities from scan_object_sizes.py results
MAX_NODES          = 30    # covers ~90% of graphs; larger graphs are skipped
MAX_CELLS_PER_NODE = 40    # covers 99% of objects; larger objects are truncated

# v3: also train on output grids so the encoder learns a latent space that
# is meaningful for both inputs and outputs
INCLUDE_OUTPUT_GRIDS = True

# Skip graphs that are too small (encoder needs edges) or too large
SKIP_TINY_GRAPHS  = True
SKIP_LARGE_GRAPHS = True   # v3: skip graphs with more than MAX_NODES nodes

# Set to None to use all 1000 training files
MAX_FILES = None


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
    x          = torch.tensor(graph["node_features"], dtype=torch.float32)
    edge_index = torch.tensor(graph["edge_index"],    dtype=torch.long)
    edge_attr  = torch.tensor(graph["edge_features"], dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def build_targets(graph: dict, max_nodes: int, max_cells: int):
    """
    Build padded targets for gat_vae_loss.

    v3: now also builds cell-level targets:
        target_cell_coords  : (max_nodes, max_cells, 2)  float
        target_cell_colors  : (max_nodes, max_cells)     long
        target_cell_mask    : (max_nodes, max_cells)     float

    Cell coords are normalised to [0, 1] by dividing by 30 (max ARC grid dim).
    This keeps the MSE loss on a sensible scale regardless of grid size.
    """
    node_feats = torch.tensor(graph["node_features"], dtype=torch.float32)
    edge_feats = torch.tensor(graph["edge_features"], dtype=torch.float32)
    edge_index = torch.tensor(graph["edge_index"],    dtype=torch.long)
    n_nodes    = node_feats.size(0)

    # ── existence ────────────────────────────────────────────────────────
    target_existence = torch.zeros(max_nodes, dtype=torch.float32)
    target_existence[:n_nodes] = 1.0

    # ── node color (argmax over one-hot) ─────────────────────────────────
    target_color = torch.zeros(max_nodes, dtype=torch.long)
    target_color[:n_nodes] = node_feats[:, :10].argmax(dim=1)

    # ── node geometry ─────────────────────────────────────────────────────
    target_node_geom = torch.zeros(max_nodes, 12, dtype=torch.float32)
    target_node_geom[:n_nodes] = node_feats[:, 10:]

    # ── edge targets ──────────────────────────────────────────────────────
    target_edge_cont   = torch.zeros(max_nodes, max_nodes, 6,  dtype=torch.float32)
    target_edge_binary = torch.zeros(max_nodes, max_nodes, 6,  dtype=torch.float32)

    for e in range(edge_feats.size(0)):
        s  = int(edge_index[0, e].item())
        d  = int(edge_index[1, e].item())
        ef = edge_feats[e]
        target_edge_cont[s, d]   = torch.tensor([ef[0],ef[1],ef[2],ef[3],ef[10],ef[11]])
        target_edge_binary[s, d] = torch.tensor([ef[4],ef[5],ef[6],ef[7],ef[8], ef[9]])

    # ── v3: cell-level targets ────────────────────────────────────────────
    target_cell_coords = torch.zeros(max_nodes, max_cells, 2,  dtype=torch.float32)
    target_cell_colors = torch.zeros(max_nodes, max_cells,     dtype=torch.long)
    target_cell_mask   = torch.zeros(max_nodes, max_cells,     dtype=torch.float32)

    for ni, node in enumerate(graph["nodes"]):
        if ni >= max_nodes:
            break
        coords = torch.tensor(node["cell_coords"],       dtype=torch.float32)  # (max_cells, 2)
        colors = torch.tensor(node["cell_colors_list"],  dtype=torch.long)     # (max_cells,)
        mask   = torch.tensor(node["cell_mask"],         dtype=torch.float32)  # (max_cells,)

        # Normalise coordinates to [0, 1] — makes MSE loss scale-independent
        coords = coords / 30.0

        target_cell_coords[ni] = coords
        target_cell_colors[ni] = colors
        target_cell_mask[ni]   = mask

    return (
        target_color,
        target_node_geom,
        target_edge_cont,
        target_edge_binary,
        target_existence,
        target_cell_coords,   # v3
        target_cell_colors,   # v3
        target_cell_mask,     # v3
    )


# ============================================================================
# DATASET BUILD
# ============================================================================

def collect_graph_samples(train_path: str, max_files: int = None) -> List[Dict]:
    """
    v3 changes:
        1. Also collects output grids when INCLUDE_OUTPUT_GRIDS=True.
           This gives the VAE a consistent latent space for both grid types.
        2. Skips graphs with more than MAX_NODES nodes (SKIP_LARGE_GRAPHS).
           Extreme outlier graphs (up to 626 nodes) would bloat the model.
    """
    samples = []
    files   = sorted([f for f in os.listdir(train_path) if f.endswith(".json")])

    if max_files is not None:
        files = files[:max_files]

    skipped_large = 0
    skipped_tiny  = 0

    for fname in files:
        fpath = os.path.join(train_path, fname)
        task  = load_json(fpath)

        for split in ["train", "test"]:
            for i, pair in enumerate(task.get(split, [])):

                grids_to_process = [("input", pair["input"])]

                # v3: also process output grids
                if INCLUDE_OUTPUT_GRIDS and "output" in pair:
                    grids_to_process.append(("output", pair["output"]))

                for grid_type, raw_grid in grids_to_process:
                    grid  = to_dsl_grid(raw_grid)
                    graph = grid_to_graph(
                        grid,
                        max_cells_per_node=MAX_CELLS_PER_NODE,
                    )

                    n_nodes = len(graph["nodes"])
                    n_edges = len(graph["edges"])

                    if SKIP_TINY_GRAPHS and (n_nodes < 2 or n_edges == 0):
                        skipped_tiny += 1
                        continue

                    # v3: skip graphs that exceed MAX_NODES
                    if SKIP_LARGE_GRAPHS and n_nodes > MAX_NODES:
                        skipped_large += 1
                        continue

                    samples.append({
                        "file_name" : fname,
                        "split"     : split,
                        "pair_idx"  : i,
                        "grid_type" : grid_type,
                        "grid"      : raw_grid,
                        "graph"     : graph,
                        "num_nodes" : n_nodes,
                        "num_edges" : n_edges,
                    })

    print(f"  Skipped (tiny)  : {skipped_tiny}")
    print(f"  Skipped (large) : {skipped_large}")
    return samples


# ============================================================================
# BATCHING
# ============================================================================

def make_batch(samples: List[Dict], max_nodes: int, max_cells: int, device: str):
    data_list          = []
    color_tgts         = []
    geom_tgts          = []
    edge_cont_tgts     = []
    edge_bin_tgts      = []
    existence_tgts     = []
    cell_coord_tgts    = []
    cell_color_tgts    = []
    cell_mask_tgts     = []

    for s in samples:
        data_list.append(graph_to_pyg_data(s["graph"]))
        (tc, tg, tec, teb, te,
         tcc, tcol, tcm) = build_targets(s["graph"], max_nodes, max_cells)
        color_tgts.append(tc)
        geom_tgts.append(tg)
        edge_cont_tgts.append(tec)
        edge_bin_tgts.append(teb)
        existence_tgts.append(te)
        cell_coord_tgts.append(tcc)
        cell_color_tgts.append(tcol)
        cell_mask_tgts.append(tcm)

    batch            = Batch.from_data_list(data_list).to(device)
    target_color     = torch.stack(color_tgts).to(device)
    target_node_geom = torch.stack(geom_tgts).to(device)
    target_edge_cont = torch.stack(edge_cont_tgts).to(device)
    target_edge_bin  = torch.stack(edge_bin_tgts).to(device)
    target_existence = torch.stack(existence_tgts).to(device)
    target_cell_coords = torch.stack(cell_coord_tgts).to(device)
    target_cell_colors = torch.stack(cell_color_tgts).to(device)
    target_cell_mask   = torch.stack(cell_mask_tgts).to(device)

    return (batch, target_color, target_node_geom, target_edge_cont,
            target_edge_bin, target_existence,
            target_cell_coords, target_cell_colors, target_cell_mask)


def chunk_list(items: List, batch_size: int) -> List[List]:
    return [items[i:i+batch_size] for i in range(0, len(items), batch_size)]


# ============================================================================
# TRAINING
# ============================================================================

def run_epoch(model, optimizer, samples, max_nodes, max_cells,
              batch_size, device, train=True):
    model.train() if train else model.eval()
    if train:
        random.shuffle(samples)

    total_loss    = 0.0
    total_batches = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for mini in chunk_list(samples, batch_size):
            (batch, tc, tg, tec, teb, te,
             tcc, tcol, tcm) = make_batch(mini, max_nodes, max_cells, device)

            if train:
                optimizer.zero_grad()

            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            loss, breakdown = gat_vae_loss(
                out,
                tc, tg, tec, teb, te,
                tcc, tcol, tcm,
                # FIX v3b: reduce edge_cont weight — it was dominating the
                # total loss and driving NaN explosions on large grids
                edge_cont_weight=0.1,
            )

            if train:
                # FIX v3b: NaN guard — skip this batch if loss is invalid
                # rather than letting corrupt gradients poison the weights
                if not torch.isfinite(loss):
                    print(f"  WARNING: non-finite loss at batch {total_batches+1}, skipping")
                    optimizer.zero_grad()
                    continue

                loss.backward()

                # FIX v3b: gradient clipping — caps gradient norm so no
                # single bad batch can cause a weight explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

            if not torch.isfinite(loss):
                continue   # don't accumulate NaN into total_loss

            total_loss    += float(loss.item())
            total_batches += 1

            if train and total_batches % 100 == 0:
                print(f"  Batch {total_batches:4d} | loss={loss.item():.4f} | "
                      + " | ".join(f"{k}={v:.3f}" for k,v in breakdown.items()))

    return total_loss / max(total_batches, 1)


# ============================================================================
# MAIN
# ============================================================================

def main():
    set_seed(SEED)
    print(f"Device         : {DEVICE}")
    print(f"MAX_NODES      : {MAX_NODES}")
    print(f"MAX_CELLS/NODE : {MAX_CELLS_PER_NODE}")
    print(f"Include outputs: {INCLUDE_OUTPUT_GRIDS}")
    print(f"\nLoading samples from: {TRAIN_PATH}\n")

    samples = collect_graph_samples(TRAIN_PATH, max_files=MAX_FILES)

    if not samples:
        raise RuntimeError("No samples found. Check TRAIN_PATH.")

    print(f"Total samples  : {len(samples)}")

    random.shuffle(samples)
    split_idx     = max(1, int(0.9 * len(samples)))
    train_samples = samples[:split_idx]
    val_samples   = samples[split_idx:] if split_idx < len(samples) else samples[:1]

    print(f"Train samples  : {len(train_samples)}")
    print(f"Val samples    : {len(val_samples)}\n")

    model = GATVAE(
        max_nodes=MAX_NODES,
        max_cells_per_node=MAX_CELLS_PER_NODE,
        node_in_dim=22,
        edge_in_dim=12,
        latent_dim=256,
        dec_hidden=256,
        node_geom_dim=12,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params   : {n_params:,}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR/10
    )

    best_val  = math.inf
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path  = os.path.join(script_dir, "gat_vae3_best.pt")

    print("Starting training...\n")

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(
            model, optimizer, train_samples,
            MAX_NODES, MAX_CELLS_PER_NODE, BATCH_SIZE, DEVICE, train=True,
        )
        val_loss = run_epoch(
            model, None, val_samples,
            MAX_NODES, MAX_CELLS_PER_NODE, BATCH_SIZE, DEVICE, train=False,
        )
        scheduler.step()

        print(f"Epoch {epoch:03d} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict"     : model.state_dict(),
                "optimizer_state_dict" : optimizer.state_dict(),
                "max_nodes"            : MAX_NODES,
                "max_cells_per_node"   : MAX_CELLS_PER_NODE,
                "latent_dim"           : 256,
                "best_val_loss"        : best_val,
                "epoch"                : epoch,
            }, save_path)
            print(f"  → Saved best model (val={best_val:.4f})")

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
