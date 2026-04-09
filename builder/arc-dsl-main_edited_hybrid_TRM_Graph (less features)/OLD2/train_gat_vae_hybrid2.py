import os
import json
import math
import random
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data, Batch

from hybrid_object2 import grid_to_graph
from gat_vae_hybrid2 import GATVAE
from graph_decoder_hybrid2 import gat_vae_loss


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

SKIP_TINY_GRAPHS  = True
SKIP_LARGE_GRAPHS = True
MAX_NODES         = 30
MAX_NODE_CAP      = MAX_NODES

INCLUDE_OUTPUT_GRIDS = True
NUM_PREVIEW_SAMPLES  = 3
MAX_FILES            = None


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
# VISUALISATION
# ============================================================================

def show_input_output_pair(input_grid, output_grid, file_name="", pair_idx=0):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(input_grid, cmap="tab10", vmin=0, vmax=9)
    axes[0].set_title(f"INPUT\n{file_name} pair {pair_idx}")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[1].imshow(output_grid, cmap="tab10", vmin=0, vmax=9)
    axes[1].set_title("TARGET OUTPUT")
    axes[1].set_xticks([]); axes[1].set_yticks([])
    plt.tight_layout()
    plt.show()


def preview_samples(samples: List[Dict], num_to_show: int = 3) -> None:
    print("\nPreviewing sample grids...")
    for s in samples[:num_to_show]:
        show_input_output_pair(
            s["input_grid"], s["output_grid"],
            file_name=s["file_name"], pair_idx=s["train_idx"],
        )
        print(
            f"{s['file_name']} | pair {s['train_idx']} | "
            f"in_nodes={s['input_num_nodes']} | out_nodes={s['output_num_nodes']}"
        )


# ============================================================================
# GRAPH -> PYG
# ============================================================================

def graph_to_pyg_data(graph: dict) -> Data:
    x          = torch.tensor(graph["node_features"], dtype=torch.float32)
    edge_index = torch.tensor(graph["edge_index"],    dtype=torch.long)
    edge_attr  = torch.tensor(graph["edge_features"], dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ============================================================================
# TARGET BUILDING
# ============================================================================

def build_targets(target_graph: dict, max_nodes: int, grid_norm: float = 30.0):
    node_feats = torch.tensor(target_graph["node_features"], dtype=torch.float32)
    edge_feats = torch.tensor(target_graph["edge_features"], dtype=torch.float32)
    edge_index = torch.tensor(target_graph["edge_index"],    dtype=torch.long)
    n_nodes    = node_feats.size(0)

    target_existence = torch.zeros(max_nodes, dtype=torch.float32)
    target_existence[:n_nodes] = 1.0

    target_color = torch.zeros(max_nodes, dtype=torch.long)
    target_color[:n_nodes] = node_feats[:, :10].argmax(dim=1)

    target_node_shape = torch.zeros(max_nodes, 100, dtype=torch.float32)
    target_node_shape[:n_nodes] = node_feats[:, 10:]

    target_bbox = torch.zeros(max_nodes, 4, dtype=torch.float32)
    for ni, node in enumerate(target_graph["nodes"]):
        if ni >= max_nodes:
            break
        min_r, min_c, max_r, max_c = node["bbox"]
        target_bbox[ni] = torch.tensor(
            [min_r / grid_norm, min_c / grid_norm,
             max_r / grid_norm, max_c / grid_norm],
            dtype=torch.float32,
        )

    target_edge_cont = torch.zeros(max_nodes, max_nodes, 2,  dtype=torch.float32)
    target_edge_bin  = torch.zeros(max_nodes, max_nodes, 3,  dtype=torch.float32)

    for e in range(edge_feats.size(0)):
        s  = int(edge_index[0, e].item())
        d  = int(edge_index[1, e].item())
        ef = edge_feats[e]
        target_edge_cont[s, d] = torch.tensor([ef[0], ef[1]])
        target_edge_bin[s, d]  = torch.tensor([ef[2], ef[3], ef[4]])

    return (target_color, target_node_shape, target_edge_cont,
            target_edge_bin, target_existence, target_bbox)


# ============================================================================
# DATASET BUILD
# ============================================================================

def collect_graph_samples(train_path: str, max_files: int = None) -> List[Dict]:
    samples       = []
    files         = sorted([f for f in os.listdir(train_path) if f.endswith(".json")])
    if max_files is not None:
        files = files[:max_files]

    skipped_tiny  = 0
    skipped_large = 0

    for fname in files:
        fpath = os.path.join(train_path, fname)
        task  = load_json(fpath)

        for split in ["train", "test"]:
            for i, pair in enumerate(task.get(split, [])):
                raw_input  = pair["input"]
                raw_output = pair.get("output", None)
                if raw_output is None:
                    continue

                input_graph  = grid_to_graph(to_dsl_grid(raw_input))
                output_graph = grid_to_graph(to_dsl_grid(raw_output))

                in_nodes  = len(input_graph["nodes"])
                in_edges  = len(input_graph["edges"])
                out_nodes = len(output_graph["nodes"])
                out_edges = len(output_graph["edges"])

                if SKIP_TINY_GRAPHS:
                    if in_nodes < 2 or in_edges == 0:
                        skipped_tiny += 1; continue
                    if out_nodes < 1:
                        skipped_tiny += 1; continue

                if SKIP_LARGE_GRAPHS:
                    if in_nodes > MAX_NODES or out_nodes > MAX_NODES:
                        skipped_large += 1; continue

                samples.append({
                    "file_name"       : fname,
                    "split"           : split,
                    "train_idx"       : i,
                    "input_grid"      : raw_input,
                    "output_grid"     : raw_output,
                    "input_graph"     : input_graph,
                    "output_graph"    : output_graph,
                    "input_num_nodes" : in_nodes,
                    "input_num_edges" : in_edges,
                    "output_num_nodes": out_nodes,
                    "output_num_edges": out_edges,
                })

    print(f"  Skipped (tiny)  : {skipped_tiny}")
    print(f"  Skipped (large) : {skipped_large}")
    return samples


def print_sample_summary(samples: List[Dict], num_to_show: int = 5) -> None:
    print("\nSample summary:")
    for s in samples[:num_to_show]:
        print(
            f"  {s['file_name']} | pair {s['train_idx']} | "
            f"in_nodes={s['input_num_nodes']} | out_nodes={s['output_num_nodes']}"
        )


# ============================================================================
# BATCHING
# ============================================================================

def make_batch(samples: List[Dict], max_nodes: int, device: str):
    data_list      = []
    color_tgts     = []
    shape_tgts     = []
    edge_cont_tgts = []
    edge_bin_tgts  = []
    existence_tgts = []
    bbox_tgts      = []

    for s in samples:
        data_list.append(graph_to_pyg_data(s["input_graph"]))
        (tc, ts, tec, teb, te, tb) = build_targets(s["output_graph"], max_nodes)
        color_tgts.append(tc)
        shape_tgts.append(ts)
        edge_cont_tgts.append(tec)
        edge_bin_tgts.append(teb)
        existence_tgts.append(te)
        bbox_tgts.append(tb)

    batch            = Batch.from_data_list(data_list).to(device)
    target_color     = torch.stack(color_tgts).to(device)
    target_shape     = torch.stack(shape_tgts).to(device)
    target_edge_cont = torch.stack(edge_cont_tgts).to(device)
    target_edge_bin  = torch.stack(edge_bin_tgts).to(device)
    target_existence = torch.stack(existence_tgts).to(device)
    target_bbox      = torch.stack(bbox_tgts).to(device)

    return (batch, target_color, target_shape, target_edge_cont,
            target_edge_bin, target_existence, target_bbox)


def chunk_list(items: List, batch_size: int) -> List[List]:
    return [items[i:i+batch_size] for i in range(0, len(items), batch_size)]


# ============================================================================
# TRAINING
# ============================================================================

def run_epoch(model, optimizer, samples, max_nodes, batch_size, device, train=True):
    model.train() if train else model.eval()
    if train:
        random.shuffle(samples)

    total_loss    = 0.0
    total_batches = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for mini in chunk_list(samples, batch_size):
            (batch, tc, ts, tec, teb, te, tb) = make_batch(mini, max_nodes, device)

            if train:
                optimizer.zero_grad()

            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            loss, breakdown = gat_vae_loss(
                out, tc, ts, tec, teb, te, tb,
            )

            if train:
                if not torch.isfinite(loss):
                    print(f"  WARNING: non-finite loss at batch {total_batches+1}, skipping")
                    optimizer.zero_grad()
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            if not torch.isfinite(loss):
                continue

            total_loss    += float(loss.item())
            total_batches += 1

            if train and total_batches % 100 == 0:
                print(f"  Batch {total_batches:4d} | loss={loss.item():.4f} | "
                      + " | ".join(f"{k}={v:.3f}" for k, v in breakdown.items()))

    return total_loss / max(total_batches, 1)


# ============================================================================
# MAIN
# ============================================================================

def main():
    set_seed(SEED)
    print(f"Device         : {DEVICE}")
    print(f"MAX_NODES      : {MAX_NODES}")
    print(f"Include outputs: {INCLUDE_OUTPUT_GRIDS}")
    print(f"\nLoading samples from: {TRAIN_PATH}\n")

    samples = collect_graph_samples(TRAIN_PATH, max_files=MAX_FILES)

    if not samples:
        raise RuntimeError("No samples found. Check TRAIN_PATH.")

    print(f"Total samples  : {len(samples)}")
    print_sample_summary(samples, num_to_show=5)
    preview_samples(samples, num_to_show=NUM_PREVIEW_SAMPLES)

    random.shuffle(samples)
    split_idx     = max(1, int(0.9 * len(samples)))
    train_samples = samples[:split_idx]
    val_samples   = samples[split_idx:] if split_idx < len(samples) else samples[:1]

    print(f"Train samples  : {len(train_samples)}")
    print(f"Val samples    : {len(val_samples)}\n")

    model = GATVAE(
        max_nodes=MAX_NODES,
        node_in_dim=110,
        edge_in_dim=5,
        latent_dim=256,
        dec_hidden=256,
        node_shape_dim=100,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params   : {n_params:,}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR/10,
    )

    best_val   = math.inf
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path  = os.path.join(script_dir, "gat_vae_hybrid2_best.pt")

    print("Starting training...\n")

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(
            model, optimizer, train_samples,
            MAX_NODES, BATCH_SIZE, DEVICE, train=True,
        )
        val_loss = run_epoch(
            model, None, val_samples,
            MAX_NODES, BATCH_SIZE, DEVICE, train=False,
        )
        scheduler.step()

        print(f"Epoch {epoch:03d} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict"    : model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "max_nodes"           : MAX_NODES,
                "best_val_loss"       : best_val,
                "epoch"               : epoch,
            }, save_path)
            print(f"  → Saved best model (val={best_val:.4f})")

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()