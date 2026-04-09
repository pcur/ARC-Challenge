"""
arc_transformer.py

Transformer that solves ARC tasks by in-context learning over latent vectors.

Pipeline overview
─────────────────
Each ARC task has K training pairs (input_grid, output_grid) plus one test input.

1. The frozen VAE encoder converts every grid into a latent vector z ∈ R^latent_dim.

2. The transformer receives a sequence:
       [z_in_1, z_out_1, z_in_2, z_out_2, ..., z_in_K, z_out_K, z_test_in]
   and predicts z_test_out.

3. The frozen VAE decoder converts z_test_out back into a grid via the
   cell coordinate / cell color / cell mask heads.

Training
────────
During training we have ground truth (z_test_out) from the output grid of
the held-out pair. The transformer is trained with MSE loss between its
predicted z and the true z.

The VAE is FROZEN throughout — only the transformer weights are updated.

Sequence encoding
─────────────────
Each position in the sequence is tagged with a role embedding:
    0 = training input
    1 = training output
    2 = test input  (always the last token)

This lets the transformer distinguish "this is an example input I should
learn from" vs "this is the query I must answer".
"""

import os
import json
import math
import random
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

from custom_object3 import grid_to_graph, graph_to_grid_from_predictions
from gat_vae3 import GATVAE


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_PATH    = r"C:\Users\tedhun\OneDrive - Wanzl GmbH & Co. KGaA\Microsoft Teams Chat Files\outputs\Documents\GS\ARC\Assignment10\Builder\data\training"
VAE_CKPT      = "gat_vae3_best.pt"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

SEED          = 42
BATCH_SIZE    = 4       # number of ARC tasks per batch
#EPOCHS        = 60
EPOCHS        = 5
LR            = 1e-4

MAX_NODES          = 30
MAX_CELLS_PER_NODE = 40
LATENT_DIM         = 256

# Transformer hyperparameters
TR_HIDDEN    = 512     # transformer model dimension
TR_HEADS     = 8       # attention heads
TR_LAYERS    = 6       # transformer encoder layers
TR_DROPOUT   = 0.1
TR_FF_DIM    = 2048    # feedforward dimension inside transformer

MAX_FILES    = None    # set to e.g. 50 for quick testing


# ─────────────────────────────────────────────────────────────────────────────
# ROLE TOKENS
# ─────────────────────────────────────────────────────────────────────────────

ROLE_TRAIN_INPUT  = 0
ROLE_TRAIN_OUTPUT = 1
ROLE_TEST_INPUT   = 2


# ─────────────────────────────────────────────────────────────────────────────
# 1.  ARC TRANSFORMER
# ─────────────────────────────────────────────────────────────────────────────

class ARCTransformer(nn.Module):
    """
    Sequence-to-one transformer that predicts z_test_out from a sequence of
    (z_in, z_out) demonstration pairs followed by z_test_in.

    Architecture
    ────────────
    - Input projection: latent_dim → tr_hidden
    - Role embedding:   3 roles    → tr_hidden  (added to projected z)
    - Positional encoding: sinusoidal, up to max_seq_len positions
    - Transformer encoder: tr_layers × (self-attention + FFN)
    - Output head: takes the representation of the LAST token (z_test_in)
                   and projects it to latent_dim → predicted z_test_out
    """

    def __init__(
        self,
        latent_dim: int  = 256,
        tr_hidden: int   = 512,
        tr_heads: int    = 8,
        tr_layers: int   = 6,
        tr_ff_dim: int   = 2048,
        tr_dropout: float = 0.1,
        max_seq_len: int  = 64,   # max number of tokens (pairs*2 + 1)
        num_roles: int    = 3,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.tr_hidden  = tr_hidden

        # Project z vectors into transformer space
        self.input_proj = nn.Linear(latent_dim, tr_hidden)

        # Role embedding (train_input / train_output / test_input)
        self.role_emb = nn.Embedding(num_roles, tr_hidden)

        # Sinusoidal positional encoding (not learned — generalises to variable
        # sequence lengths without needing to see all lengths during training)
        self.register_buffer(
            "pos_enc",
            self._make_sinusoidal(max_seq_len, tr_hidden),
        )

        # Transformer encoder (we use encoder-only; the last token attends to
        # all prior demonstration tokens via full self-attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=tr_hidden,
            nhead=tr_heads,
            dim_feedforward=tr_ff_dim,
            dropout=tr_dropout,
            batch_first=True,    # (B, seq, hidden)
            norm_first=True,     # pre-norm — more stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=tr_layers,
            norm=nn.LayerNorm(tr_hidden),
        )

        # Output head: transformer hidden → predicted z
        self.output_head = nn.Sequential(
            nn.Linear(tr_hidden, tr_hidden),
            nn.SiLU(),
            nn.Linear(tr_hidden, latent_dim),
        )

    @staticmethod
    def _make_sinusoidal(max_len: int, d_model: int) -> torch.Tensor:
        """Standard sinusoidal positional encoding."""
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)   # (1, max_len, d_model)

    def forward(
        self,
        z_sequence: torch.Tensor,   # (B, seq_len, latent_dim)
        roles: torch.Tensor,        # (B, seq_len)  int — role ids
    ) -> torch.Tensor:
        """
        Returns predicted z_test_out: (B, latent_dim)

        The last token in the sequence is always z_test_in.
        Its output representation is used to predict z_test_out.
        """
        B, S, _ = z_sequence.shape

        # Project + add role embeddings + add positional encoding
        x  = self.input_proj(z_sequence)               # (B, S, tr_hidden)
        x  = x + self.role_emb(roles)                  # (B, S, tr_hidden)
        x  = x + self.pos_enc[:, :S, :]                # (B, S, tr_hidden)

        # Transformer encoder — full self-attention (all tokens see all others)
        x  = self.transformer(x)                        # (B, S, tr_hidden)

        # Take representation of last token (z_test_in position)
        last = x[:, -1, :]                              # (B, tr_hidden)

        # Project to latent space
        z_pred = self.output_head(last)                 # (B, latent_dim)
        return z_pred


# ─────────────────────────────────────────────────────────────────────────────
# 2.  GRID → z  HELPER  (uses frozen VAE encoder)
# ─────────────────────────────────────────────────────────────────────────────

def grid_to_z(raw_grid, vae: GATVAE, device: str) -> Optional[torch.Tensor]:
    """
    Convert one raw ARC grid into a latent vector z.

    Returns None if the grid has no extractable objects (all background).
    """
    grid  = tuple(tuple(row) for row in raw_grid)
    graph = grid_to_graph(grid, max_cells_per_node=MAX_CELLS_PER_NODE)

    n_nodes = len(graph["nodes"])
    n_edges = len(graph["edges"])

    if n_nodes < 2 or n_edges == 0 or n_nodes > MAX_NODES:
        return None

    x          = torch.tensor(graph["node_features"], dtype=torch.float32).to(device)
    edge_index = torch.tensor(graph["edge_index"],    dtype=torch.long).to(device)
    edge_attr  = torch.tensor(graph["edge_features"], dtype=torch.float32).to(device)
    batch      = torch.zeros(n_nodes, dtype=torch.long).to(device)

    with torch.no_grad():
        z, mu, _ = vae.encode(x, edge_index, edge_attr, batch)

    # At inference time the VAE bottleneck returns mu (deterministic).
    # We use mu rather than z for stability.
    return mu.squeeze(0)   # (latent_dim,)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  z → GRID  HELPER  (uses frozen VAE decoder)
# ─────────────────────────────────────────────────────────────────────────────

def z_to_grid(z: torch.Tensor, vae: GATVAE,
              height: int, width: int, device: str) -> List[List[int]]:
    """
    Convert a latent vector z into a reconstructed ARC grid.

    Uses the cell coordinate / cell mask / cell color heads from the v3 decoder.
    """
    z = z.unsqueeze(0).to(device)   # (1, latent_dim)

    with torch.no_grad():
        (_, _, _, _,
         existence_logits,
         cell_coords,
         cell_color_logits,
         cell_mask_logits) = vae.decode(z)

    # Convert logits to predictions
    existence_prob  = torch.sigmoid(existence_logits[0]).cpu().tolist()
    cell_mask_prob  = torch.sigmoid(cell_mask_logits[0]).cpu().tolist()
    # Argmax over color dimension
    cell_color_pred = cell_color_logits[0].argmax(dim=-1).cpu().tolist()
    # Denormalise coordinates (we normalised by /30 during training)
    cell_coords_dn  = (cell_coords[0] * 30.0).cpu().tolist()

    return graph_to_grid_from_predictions(
        pred_cell_coords  = cell_coords_dn,
        pred_cell_colors  = cell_color_pred,
        pred_cell_mask    = cell_mask_prob,
        pred_existence    = existence_prob,
        height=height,
        width=width,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4.  DATASET
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def build_task_samples(train_path: str, vae: GATVAE,
                       device: str, max_files: int = None) -> List[Dict]:
    """
    Build one sample per ARC task.

    Each sample contains:
        z_train_inputs  : list of z tensors for training input grids
        z_train_outputs : list of z tensors for training output grids
        z_test_input    : z tensor for the test input grid
        z_test_output   : z tensor for the test output grid (supervision target)
        test_height     : height of test output grid (for reconstruction)
        test_width      : width  of test output grid
        test_output_raw : the actual test output grid (for evaluation)

    Tasks are skipped if any grid fails to encode (too few/many nodes).
    """
    files   = sorted([f for f in os.listdir(train_path) if f.endswith(".json")])
    if max_files is not None:
        files = files[:max_files]

    samples      = []
    skipped      = 0
    vae.eval()

    for fname in files:
        fpath = os.path.join(train_path, fname)
        task  = load_json(fpath)

        train_pairs = task.get("train", [])
        test_pairs  = task.get("test",  [])

        if not train_pairs or not test_pairs:
            continue

        # Encode all training pairs
        z_ins  = []
        z_outs = []
        ok     = True

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

        # Use the first test pair as the prediction target
        test_pair = test_pairs[0]
        z_test_in  = grid_to_z(test_pair["input"],  vae, device)
        z_test_out = grid_to_z(test_pair.get("output", test_pair["input"]),
                               vae, device)

        if z_test_in is None or z_test_out is None:
            skipped += 1
            continue

        test_out_raw = test_pair.get("output", test_pair["input"])
        h = len(test_out_raw)
        w = len(test_out_raw[0])

        samples.append({
            "task_id"        : fname.rstrip(".json"),
            "z_train_inputs" : z_ins,
            "z_train_outputs": z_outs,
            "z_test_input"   : z_test_in,
            "z_test_output"  : z_test_out,     # supervision target
            "test_height"    : h,
            "test_width"     : w,
            "test_output_raw": test_out_raw,
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate a list of task samples into padded batch tensors.

    Sequence layout per task:
        [z_in_1, z_out_1, z_in_2, z_out_2, ..., z_in_K, z_out_K, z_test_in]

    Tasks can have different numbers of training pairs (K varies).
    We pad shorter sequences with zeros on the LEFT so the test token
    is always the LAST position. A key_padding_mask is returned to tell
    the transformer which positions are padding.

    Returns:
        z_seq   : (B, max_seq_len, latent_dim)
        roles   : (B, max_seq_len)  int
        z_target: (B, latent_dim)   — ground truth z_test_out
        pad_mask: (B, max_seq_len)  bool — True means IGNORE this position
    """
    # Build sequences for each sample
    sequences = []
    role_seqs = []
    targets   = []

    for s in samples:
        seq   = []
        roles = []
        for zi, zo in zip(s["z_train_inputs"], s["z_train_outputs"]):
            seq.append(zi)
            seq.append(zo)
            roles.append(ROLE_TRAIN_INPUT)
            roles.append(ROLE_TRAIN_OUTPUT)
        seq.append(s["z_test_input"])
        roles.append(ROLE_TEST_INPUT)
        sequences.append(seq)
        role_seqs.append(roles)
        targets.append(s["z_test_output"])

    max_len = max(len(s) for s in sequences)
    latent  = targets[0].shape[0]
    B       = len(samples)

    z_seq    = torch.zeros(B, max_len, latent,   dtype=torch.float32)
    role_t   = torch.zeros(B, max_len,           dtype=torch.long)
    pad_mask = torch.ones( B, max_len,           dtype=torch.bool)   # True=ignore

    for i, (seq, roles) in enumerate(zip(sequences, role_seqs)):
        L    = len(seq)
        start = max_len - L   # left-pad
        for j, (z, r) in enumerate(zip(seq, roles)):
            z_seq[i, start+j] = z
            role_t[i, start+j] = r
        pad_mask[i, start:] = False   # real positions

    z_target = torch.stack(targets)   # (B, latent_dim)

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
            z_seq, roles, z_target, pad_mask = collate_task_batch(chunk, device)

            z_pred = transformer(z_seq, roles)   # (B, latent_dim)

            # MSE loss between predicted and true z_test_out
            loss = F.mse_loss(z_pred, z_target)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
                optimizer.step()

            total_loss    += float(loss.item())
            total_batches += 1

    return total_loss / max(total_batches, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  EVALUATION: predict and reconstruct grids
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_task(sample: Dict, transformer: ARCTransformer,
                  vae: GATVAE, device: str) -> bool:
    """
    Run the full pipeline on one task and check if output matches ground truth.

    Returns True if the predicted grid exactly matches the target.
    """
    transformer.eval()

    # Build sequence
    seq   = []
    roles = []
    for zi, zo in zip(sample["z_train_inputs"], sample["z_train_outputs"]):
        seq.append(zi)
        seq.append(zo)
        roles.append(ROLE_TRAIN_INPUT)
        roles.append(ROLE_TRAIN_OUTPUT)
    seq.append(sample["z_test_input"])
    roles.append(ROLE_TEST_INPUT)

    z_seq  = torch.stack(seq).unsqueeze(0).to(device)       # (1, S, latent)
    role_t = torch.tensor(roles, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        z_pred = transformer(z_seq, role_t)   # (1, latent_dim)

    predicted_grid = z_to_grid(
        z_pred.squeeze(0), vae,
        sample["test_height"], sample["test_width"], device,
    )

    return predicted_grid == sample["test_output_raw"]


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
    print(f"Device     : {DEVICE}")
    print(f"VAE ckpt   : {VAE_CKPT}\n")

    # ── load frozen VAE ───────────────────────────────────────────────────
    vae = GATVAE(
        max_nodes=MAX_NODES,
        max_cells_per_node=MAX_CELLS_PER_NODE,
        latent_dim=LATENT_DIM,
    ).to(DEVICE)

    ckpt = torch.load(VAE_CKPT, map_location=DEVICE)
    vae.load_state_dict(ckpt["model_state_dict"])
    vae.eval()

    # Freeze all VAE parameters
    for p in vae.parameters():
        p.requires_grad = False

    print(f"VAE loaded and frozen.")
    print(f"  Best VAE val loss: {ckpt.get('best_val_loss', 'N/A')}\n")

    # ── build task dataset ────────────────────────────────────────────────
    print("Building task samples (encoding all grids with VAE)...")
    all_samples = build_task_samples(TRAIN_PATH, vae, DEVICE, max_files=MAX_FILES)

    if not all_samples:
        raise RuntimeError("No task samples built. Check TRAIN_PATH and VAE checkpoint.")

    random.shuffle(all_samples)
    split_idx     = max(1, int(0.9 * len(all_samples)))
    train_samples = all_samples[:split_idx]
    val_samples   = all_samples[split_idx:] if split_idx < len(all_samples) else all_samples[:1]

    print(f"Train tasks: {len(train_samples)}")
    print(f"Val tasks  : {len(val_samples)}\n")

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
    print(f"Transformer params: {n_params:,}\n")

    optimizer = torch.optim.AdamW(transformer.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR/10
    )

    best_val  = math.inf
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path  = os.path.join(script_dir, "arc_transformer_best.pt")

    print("Starting transformer training...\n")

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(transformer, optimizer, train_samples,
                               BATCH_SIZE, DEVICE, train=True)
        val_loss   = run_epoch(transformer, None, val_samples,
                               BATCH_SIZE, DEVICE, train=False)
        scheduler.step()

        # Every 10 epochs check exact-match accuracy on val set
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
                "model_state_dict"     : transformer.state_dict(),
                "optimizer_state_dict" : optimizer.state_dict(),
                "latent_dim"           : LATENT_DIM,
                "tr_hidden"            : TR_HIDDEN,
                "tr_heads"             : TR_HEADS,
                "tr_layers"            : TR_LAYERS,
                "best_val_loss"        : best_val,
                "epoch"                : epoch,
            }, save_path)
            print(f"  → Saved best transformer (val={best_val:.6f})")

    print(f"\nTraining complete. Best val loss: {best_val:.6f}")


if __name__ == "__main__":
    main()
