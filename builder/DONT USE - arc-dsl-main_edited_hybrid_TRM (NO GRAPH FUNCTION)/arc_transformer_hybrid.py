"""
arc_transformer_hybrid.py

Hybrid-pipeline equivalent of arc_transformer.py.

The architecture is identical to the custom transformer — an in-context
learning transformer that takes a sequence of (z_input, z_output) demo
pairs followed by z_test_input and predicts z_test_output.

The only differences from arc_transformer.py are:
    1. Uses gat_vae_hybrid2.GATVAE instead of gat_vae3.GATVAE
    2. Uses hybrid_object2.grid_to_graph (110-dim nodes, 5-dim edges)
    3. z_to_grid uses the shape mask + bbox decoder path instead of
       cell coordinates

Pipeline
────────
1. Frozen hybrid VAE encodes every grid → z ∈ R^latent_dim
2. Transformer sees:
       [z_in_1, z_out_1, z_in_2, z_out_2, ..., z_test_in]
   and predicts z_test_out
3. Frozen hybrid VAE decoder converts z_test_out → grid via
   shape mask + bbox reconstruction

Checkpoint dependency
─────────────────────
Requires gat_vae_hybrid2_best.pt produced by train_gat_vae_hybrid2.py.
"""

import os
import json
import math
import random
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

from hybrid_object2 import grid_to_graph, graph_to_grid_from_predictions
from gat_vae_hybrid2 import GATVAE


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_PATH    = r"C:\Users\tedhun\OneDrive - Wanzl GmbH & Co. KGaA\Microsoft Teams Chat Files\outputs\Documents\GS\ARC\Assignment10\Builder\data\training"

VAE_CKPT      = "gat_vae_hybrid2_best.pt"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

SEED          = 42
BATCH_SIZE    = 4
EPOCHS        = 60
LR            = 1e-4

MAX_NODES     = 30
LATENT_DIM    = 256

# Transformer hyperparameters
TR_HIDDEN  = 512
TR_HEADS   = 8
TR_LAYERS  = 6
TR_FF_DIM  = 2048
TR_DROPOUT = 0.1

MAX_FILES  = None   # set to e.g. 50 for quick testing


# ─────────────────────────────────────────────────────────────────────────────
# ROLE TOKENS
# ─────────────────────────────────────────────────────────────────────────────

ROLE_TRAIN_INPUT  = 0
ROLE_TRAIN_OUTPUT = 1
ROLE_TEST_INPUT   = 2


# ─────────────────────────────────────────────────────────────────────────────
# 1.  ARC TRANSFORMER  (identical architecture to arc_transformer.py)
# ─────────────────────────────────────────────────────────────────────────────

class ARCTransformer(nn.Module):
    """
    Sequence-to-one transformer that predicts z_test_out from a sequence
    of (z_in, z_out) demonstration pairs followed by z_test_in.

    Role embeddings:
        0 = training input
        1 = training output
        2 = test input  (always the last token)
    """

    def __init__(
        self,
        latent_dim: int   = 256,
        tr_hidden: int    = 512,
        tr_heads: int     = 8,
        tr_layers: int    = 6,
        tr_ff_dim: int    = 2048,
        tr_dropout: float = 0.1,
        max_seq_len: int  = 64,
        num_roles: int    = 3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.tr_hidden  = tr_hidden

        self.input_proj = nn.Linear(latent_dim, tr_hidden)
        self.role_emb   = nn.Embedding(num_roles, tr_hidden)
        self.register_buffer(
            "pos_enc",
            self._make_sinusoidal(max_seq_len, tr_hidden),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=tr_hidden,
            nhead=tr_heads,
            dim_feedforward=tr_ff_dim,
            dropout=tr_dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=tr_layers,
            norm=nn.LayerNorm(tr_hidden),
        )
        self.output_head = nn.Sequential(
            nn.Linear(tr_hidden, tr_hidden),
            nn.SiLU(),
            nn.Linear(tr_hidden, latent_dim),
        )

    @staticmethod
    def _make_sinusoidal(max_len: int, d_model: int) -> torch.Tensor:
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    def forward(self, z_sequence, roles):
        """
        z_sequence : (B, seq_len, latent_dim)
        roles      : (B, seq_len)  int
        Returns    : (B, latent_dim)  predicted z_test_out
        """
        B, S, _ = z_sequence.shape
        x  = self.input_proj(z_sequence)
        x  = x + self.role_emb(roles)
        x  = x + self.pos_enc[:, :S, :]
        x  = self.transformer(x)
        return self.output_head(x[:, -1, :])


# ─────────────────────────────────────────────────────────────────────────────
# 2.  GRID → z  (frozen hybrid VAE encoder)
# ─────────────────────────────────────────────────────────────────────────────

def grid_to_z(raw_grid, vae: GATVAE, device: str) -> Optional[torch.Tensor]:
    """
    Encode one raw ARC grid → latent mu vector.
    Returns None if the grid produces a graph that is too small or too large.
    """
    grid  = tuple(tuple(row) for row in raw_grid)
    graph = grid_to_graph(grid)

    n_nodes = len(graph["nodes"])
    n_edges = len(graph["edges"])

    if n_nodes < 2 or n_edges == 0 or n_nodes > MAX_NODES:
        return None

    x          = torch.tensor(graph["node_features"], dtype=torch.float32).to(device)
    edge_index = torch.tensor(graph["edge_index"],    dtype=torch.long).to(device)
    edge_attr  = torch.tensor(graph["edge_features"], dtype=torch.float32).to(device)
    batch      = torch.zeros(n_nodes, dtype=torch.long).to(device)

    with torch.no_grad():
        _, mu, _ = vae.encode(x, edge_index, edge_attr, batch)

    return mu.squeeze(0)   # (latent_dim,)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  z → GRID  (frozen hybrid VAE decoder)
# ─────────────────────────────────────────────────────────────────────────────

def z_to_grid(
    z: torch.Tensor,
    vae: GATVAE,
    height: int,
    width: int,
    device: str,
) -> List[List[int]]:
    """
    Decode a latent vector z → reconstructed ARC grid.

    Uses the hybrid decoder path:
        existence_logits → which node slots are real
        color_logits     → color per node
        node_shape       → 10×10 shape mask per node
        bbox             → bounding box per node (normalised)

    Then calls graph_to_grid_from_predictions from hybrid_object2.py
    to paint each predicted object back onto a blank canvas.
    """
    z = z.unsqueeze(0).to(device)   # (1, latent_dim)

    with torch.no_grad():
        (color_logits,
         node_shape,
         _edge_feats,
         _edge_binary,
         existence_logits,
         bbox) = vae.decode(z)

    # Convert to numpy-friendly lists
    existence_prob = torch.sigmoid(existence_logits[0]).cpu().tolist()  # [N]
    color_pred     = color_logits[0].argmax(dim=-1).cpu().tolist()      # [N]
    shape_pred     = node_shape[0].cpu().tolist()                       # [N, 100]
    bbox_pred      = bbox[0].cpu().tolist()                             # [N, 4]

    return graph_to_grid_from_predictions(
        pred_shape_masks = shape_pred,
        pred_colors      = color_pred,
        pred_existence   = existence_prob,
        pred_bboxes      = bbox_pred,
        height           = height,
        width            = width,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4.  DATASET  (one sample per ARC task)
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def build_task_samples(
    train_path: str,
    vae: GATVAE,
    device: str,
    max_files: int = None,
) -> List[Dict]:
    """
    One sample per ARC task.

    Each sample stores pre-encoded z vectors for all demo pairs and the
    test pair, plus the raw test output grid for evaluation.
    """
    files   = sorted([f for f in os.listdir(train_path) if f.endswith(".json")])
    if max_files is not None:
        files = files[:max_files]

    samples  = []
    skipped  = 0
    vae.eval()

    for fname in files:
        fpath = os.path.join(train_path, fname)
        task  = load_json(fpath)

        train_pairs = task.get("train", [])
        test_pairs  = task.get("test",  [])

        if not train_pairs or not test_pairs:
            continue

        # Encode all training pairs
        z_ins, z_outs = [], []
        ok = True

        for pair in train_pairs:
            zi = grid_to_z(pair["input"],  vae, device)
            zo = grid_to_z(pair["output"], vae, device)
            if zi is None or zo is None:
                ok = False
                break
            z_ins.append(zi)
            z_outs.append(zo)

        if not ok:
            skipped += 1
            continue

        # Encode test pair
        test_pair  = test_pairs[0]
        z_test_in  = grid_to_z(test_pair["input"], vae, device)
        test_out   = test_pair.get("output", test_pair["input"])
        z_test_out = grid_to_z(test_out, vae, device)

        if z_test_in is None or z_test_out is None:
            skipped += 1
            continue

        h = len(test_out)
        w = len(test_out[0])

        samples.append({
            "task_id"        : fname.replace(".json", ""),
            "z_train_inputs" : z_ins,
            "z_train_outputs": z_outs,
            "z_test_input"   : z_test_in,
            "z_test_output"  : z_test_out,
            "test_height"    : h,
            "test_width"     : w,
            "test_output_raw": test_out,
        })

    print(f"  Task samples built : {len(samples)}")
    print(f"  Tasks skipped      : {skipped}")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 5.  BATCHING
# ─────────────────────────────────────────────────────────────────────────────

def collate_task_batch(
    samples: List[Dict],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build padded sequence batch from a list of task samples.

    Sequence layout per task:
        [z_in_1, z_out_1, z_in_2, z_out_2, ..., z_test_in]

    Left-padded so the test token is always the last position.
    """
    sequences, role_seqs, targets = [], [], []

    for s in samples:
        seq, roles = [], []
        for zi, zo in zip(s["z_train_inputs"], s["z_train_outputs"]):
            seq.append(zi);  roles.append(ROLE_TRAIN_INPUT)
            seq.append(zo);  roles.append(ROLE_TRAIN_OUTPUT)
        seq.append(s["z_test_input"]);  roles.append(ROLE_TEST_INPUT)
        sequences.append(seq)
        role_seqs.append(roles)
        targets.append(s["z_test_output"])

    max_len = max(len(s) for s in sequences)
    latent  = targets[0].shape[0]
    B       = len(samples)

    z_seq    = torch.zeros(B, max_len, latent, dtype=torch.float32)
    role_t   = torch.zeros(B, max_len,         dtype=torch.long)
    pad_mask = torch.ones( B, max_len,         dtype=torch.bool)

    for i, (seq, roles) in enumerate(zip(sequences, role_seqs)):
        L     = len(seq)
        start = max_len - L
        for j, (z, r) in enumerate(zip(seq, roles)):
            z_seq[i, start+j]  = z
            role_t[i, start+j] = r
        pad_mask[i, start:] = False

    z_target = torch.stack(targets)

    return (
        z_seq.to(device),
        role_t.to(device),
        z_target.to(device),
        pad_mask.to(device),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6.  TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(transformer, optimizer, samples, batch_size, device, train=True):
    transformer.train() if train else transformer.eval()
    if train:
        random.shuffle(samples)

    total_loss    = 0.0
    total_batches = 0
    chunks = [samples[i:i+batch_size] for i in range(0, len(samples), batch_size)]

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for chunk in chunks:
            z_seq, roles, z_target, _ = collate_task_batch(chunk, device)
            z_pred = transformer(z_seq, roles)
            loss   = F.mse_loss(z_pred, z_target)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
                optimizer.step()

            total_loss    += float(loss.item())
            total_batches += 1

    return total_loss / max(total_batches, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def grids_match(a, b) -> bool:
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if list(ra) != list(rb):
            return False
    return True


def evaluate_task(sample: Dict, transformer, vae: GATVAE, device: str) -> bool:
    """Run full pipeline on one task. Returns True if grid exactly matches."""
    transformer.eval()

    seq, roles = [], []
    for zi, zo in zip(sample["z_train_inputs"], sample["z_train_outputs"]):
        seq.append(zi);  roles.append(ROLE_TRAIN_INPUT)
        seq.append(zo);  roles.append(ROLE_TRAIN_OUTPUT)
    seq.append(sample["z_test_input"]);  roles.append(ROLE_TEST_INPUT)

    z_seq  = torch.stack(seq).unsqueeze(0).to(device)
    role_t = torch.tensor(roles, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        z_pred = transformer(z_seq, role_t)

    predicted = z_to_grid(
        z_pred.squeeze(0), vae,
        sample["test_height"], sample["test_width"], device,
    )

    return grids_match(predicted, sample["test_output_raw"])


# ─────────────────────────────────────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(SEED)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path  = os.path.join(script_dir, "arc_transformer_hybrid_best.pt")

    print(f"Device     : {DEVICE}")
    print(f"VAE ckpt   : {VAE_CKPT}\n")

    # ── load frozen hybrid VAE ────────────────────────────────────────────
    vae_path = os.path.join(script_dir, VAE_CKPT)
    vae = GATVAE(
        max_nodes=MAX_NODES,
        node_in_dim=110,
        edge_in_dim=5,
        node_shape_dim=100,
        latent_dim=LATENT_DIM,
    ).to(DEVICE)

    ckpt = torch.load(vae_path, map_location=DEVICE)
    vae.load_state_dict(ckpt["model_state_dict"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    print(f"Hybrid VAE loaded and frozen.")
    print(f"  Best VAE val loss : {ckpt.get('best_val_loss', 'N/A')}\n")

    # ── build task dataset ────────────────────────────────────────────────
    print("Building task samples (encoding all grids with hybrid VAE)...")
    all_samples = build_task_samples(TRAIN_PATH, vae, DEVICE, max_files=MAX_FILES)

    if not all_samples:
        raise RuntimeError("No task samples built. Check TRAIN_PATH and VAE checkpoint.")

    random.shuffle(all_samples)
    split_idx     = max(1, int(0.9 * len(all_samples)))
    train_samples = all_samples[:split_idx]
    val_samples   = all_samples[split_idx:] if split_idx < len(all_samples) else all_samples[:1]

    print(f"Train tasks : {len(train_samples)}")
    print(f"Val tasks   : {len(val_samples)}\n")

    # ── build transformer ─────────────────────────────────────────────────
    transformer = ARCTransformer(
        latent_dim=LATENT_DIM,
        tr_hidden=TR_HIDDEN,
        tr_heads=TR_HEADS,
        tr_layers=TR_LAYERS,
        tr_ff_dim=TR_FF_DIM,
        tr_dropout=TR_DROPOUT,
        max_seq_len=64,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print(f"Transformer params : {n_params:,}\n")

    optimizer = torch.optim.AdamW(transformer.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR/10,
    )

    best_val = math.inf
    print("Starting transformer training...\n")

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(transformer, optimizer, train_samples,
                               BATCH_SIZE, DEVICE, train=True)
        val_loss   = run_epoch(transformer, None, val_samples,
                               BATCH_SIZE, DEVICE, train=False)
        scheduler.step()

        # Exact-match check every 10 epochs
        if epoch % 10 == 0:
            n_correct = sum(
                evaluate_task(s, transformer, vae, DEVICE)
                for s in val_samples
            )
            acc = 100.0 * n_correct / max(len(val_samples), 1)
            print(f"Epoch {epoch:03d} | train={train_loss:.6f} | "
                  f"val={val_loss:.6f} | exact_match={n_correct}/{len(val_samples)} "
                  f"({acc:.1f}%)")
        else:
            print(f"Epoch {epoch:03d} | train={train_loss:.6f} | val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict"    : transformer.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "latent_dim"          : LATENT_DIM,
                "tr_hidden"           : TR_HIDDEN,
                "tr_heads"            : TR_HEADS,
                "tr_layers"           : TR_LAYERS,
                "best_val_loss"       : best_val,
                "epoch"               : epoch,
            }, save_path)
            print(f"  → Saved best transformer (val={best_val:.6f})")

    print(f"\nTraining complete. Best val loss: {best_val:.6f}")


if __name__ == "__main__":
    main()