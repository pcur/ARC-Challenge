"""
arc_trm.py

Tiny Recursive Model (TRM) for ARC grid solving.

Based on the TRM paper as presented in Tiny_Recursive_Models.pptx.

How TRM differs from arc_transformer.py
────────────────────────────────────────
arc_transformer.py:
    - Fixed-depth transformer (N separate layers, each run once)
    - Operates on z latent vectors produced by the VAE
    - Predicts z_test_out from a sequence of z demo pairs

arc_trm.py (this file):
    - ONE shared transformer block run recursively N times
    - A halt head (q_head) decides after each recursion whether to stop
    - Operates directly on flattened grid tokens (900 + 16 puzzle embedding)
    - Predicts the output grid token-by-token via cross-entropy
    - Does NOT use the VAE — it works end-to-end on raw grids

Key TRM ideas implemented here
───────────────────────────────
1. Single shared block: the same transformer block is applied at every
   recursion step, keeping parameter count tiny (~7M in the paper, scaled
   down here for your hardware).

2. Dynamic halting: a 2-output linear head (q_head) reads the first
   hidden state z_h[:,0] and produces halt/continue logits. An ACT-style
   (Adaptive Computation Time) weighted sum accumulates predictions across
   steps. Training uses a supervision schedule: early steps are supervised
   more heavily, encouraging the model to arrive at a good answer quickly.

3. Grid tokenisation: ARC grids are flattened to 900 tokens (30×30),
   padded with zeros if smaller. Colors are shifted: 0=padding, 1=EOS,
   2-11=colors 0-9. Each demo pair adds input+output grids. A 16-token
   puzzle embedding is prepended to identify the task.

4. In-context learning: all demo pairs are concatenated as context,
   then the test input is appended. The model must predict the test output.

Token layout per task
─────────────────────
[puzzle_emb (16)] [demo1_in (900)] [demo1_out (900)] ... [test_in (900)]
→ predict: [test_out (900)]

Total sequence length with 3 demos: 16 + 3*2*900 + 900 = 6316 tokens
(This is large — reduce N_DEMOS or GRID_FLAT_LEN for memory constraints)

Hardware-scaled defaults
─────────────────────────
Paper used: hidden=512, 4×H100 80GB, 3 days, batch=768
These defaults: hidden=256, single GPU, batch=2, reduced sequence length
"""

import os
import json
import math
import random
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_PATH = r"C:\Users\tedhun\OneDrive - Wanzl GmbH & Co. KGaA\Microsoft Teams Chat Files\outputs\Documents\GS\ARC\Assignment10\Builder\data\training"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

SEED       = 42
EPOCHS     = 60
#EPOCHS     = 2
LR         = 1e-4
EMBED_LR   = 1e-2       # paper uses separate higher LR for embeddings
BATCH_SIZE = 2          # reduce if OOM — paper uses 768 on 4×H100
MAX_FILES  = None       # None = all 1000 tasks
PRINT_EVERY = 10        # print loss every N batches so you can see progress

# Grid tokenisation
# KEY MEMORY SAVING: cap grids at 10×10 instead of 30×30.
# Most ARC grids are small. Context length scales as O(seq^2) in attention,
# so halving grid size cuts attention cost by 4×.
# Set GRID_H/GRID_W back to 30 if you have enough VRAM.
GRID_H          = 10    # paper: 30 — reduce for memory
GRID_W          = 10    # paper: 30 — reduce for memory
GRID_FLAT_LEN   = GRID_H * GRID_W   # 100 (vs 900 in paper)
PUZZLE_EMB_LEN  = 16    # extra learnable tokens to identify the task
SEQ_LEN         = GRID_FLAT_LEN     # sequence length for one grid

# Token vocabulary: 0=pad, 1=EOS, 2-11=colors 0-9
PAD_TOKEN = 0
EOS_TOKEN = 1
COLOR_OFFSET = 2        # color c → token c+2

# TRM architecture (scaled down from paper for single GPU)
HIDDEN_SIZE   = 256     # paper: 512
N_HEADS       = 4       # paper: 8
FF_DIM        = 1024    # paper: 2048
DROPOUT       = 0.1
N_RECURSIONS  = 6       # paper: 6  — number of recursive steps (max)
T_INNER       = 3       # paper: 3  — inner transformer layers per block
N_SUP         = 8       # paper: 16 — supervision steps (ACT schedule)

# Number of demo pairs to include in context (reduce for memory)
MAX_DEMOS = 3

# Augmentations per task (paper: 1000, reduce heavily for limited compute)
N_AUGMENTATIONS = 10

# EMA for model averaging (paper: 0.999)
EMA_DECAY = 0.999
USE_EMA   = True


# ─────────────────────────────────────────────────────────────────────────────
# 1.  GRID TOKENISATION
# ─────────────────────────────────────────────────────────────────────────────

def grid_to_tokens(raw_grid: List[List[int]]) -> torch.Tensor:
    """
    Flatten an ARC grid to a 1D token sequence of length GRID_FLAT_LEN.

    Steps:
        1. Convert each color c → token c + COLOR_OFFSET (so 0→2, 1→3, etc.)
        2. Pad to 30×30 with PAD_TOKEN (0)
        3. Flatten row-major to length 900
    """
    h = len(raw_grid)
    w = len(raw_grid[0]) if h > 0 else 0

    tokens = torch.zeros(GRID_FLAT_LEN, dtype=torch.long)
    for r in range(min(h, GRID_H)):
        for c in range(min(w, GRID_W)):
            tokens[r * GRID_W + c] = raw_grid[r][c] + COLOR_OFFSET

    return tokens   # (900,)


def tokens_to_grid(tokens: torch.Tensor, height: int, width: int) -> List[List[int]]:
    """
    Convert a 1D token sequence back to a 2D ARC grid.

    Reverses the color shift: token t → color t - COLOR_OFFSET.
    Tokens that are PAD or EOS → color 0 (background).
    """
    grid = [[0] * width for _ in range(height)]
    for r in range(height):
        for c in range(width):
            idx = r * GRID_W + c
            if idx >= GRID_FLAT_LEN:
                break
            t = tokens[idx].item()
            if t >= COLOR_OFFSET:
                grid[r][c] = t - COLOR_OFFSET
    return grid


# ─────────────────────────────────────────────────────────────────────────────
# 2.  AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def augment_pair(
    input_grid: List[List[int]],
    output_grid: List[List[int]],
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Apply a random dihedral group transformation (rotation + flip) to a
    matched input/output grid pair.

    The paper uses:
        - Color permutation
        - Dihedral group (8 symmetries: 4 rotations × 2 flips)
        - Translation

    We implement dihedral transforms here. Color permutation is skipped
    because it requires consistent mapping across all grids in a task.
    """
    # Choose one of 8 dihedral transforms: k rotations × flip
    k    = random.randint(0, 3)   # number of 90° CCW rotations
    flip = random.random() > 0.5

    def transform(grid):
        # Rotate k times CCW
        for _ in range(k):
            grid = [list(row) for row in zip(*grid[::-1])]
        # Flip horizontally
        if flip:
            grid = [row[::-1] for row in grid]
        return grid

    return transform(input_grid), transform(output_grid)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DATASET
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_task_samples(
    train_path: str,
    max_files: int = None,
    n_augmentations: int = N_AUGMENTATIONS,
    max_demos: int = MAX_DEMOS,
) -> List[Dict]:
    """
    Build augmented samples from ARC training tasks.

    Each sample represents one (augmented) version of one task and contains:
        demo_inputs   : list of tokenised input grids  (up to max_demos)
        demo_outputs  : list of tokenised output grids (up to max_demos)
        test_input    : tokenised test input grid
        test_output   : tokenised test output grid (supervision target)
        test_h, test_w: original test output dimensions (for reconstruction)
        test_raw      : raw test output grid (for exact-match evaluation)
        task_id       : filename without .json

    The paper augments 1000× per task. We default to N_AUGMENTATIONS.
    """
    files = sorted([f for f in os.listdir(train_path) if f.endswith(".json")])
    if max_files is not None:
        files = files[:max_files]

    samples  = []
    skipped  = 0
    n_files  = len(files)

    for file_idx, fname in enumerate(files):
        # Progress every 50 files so you can see it's working
        if file_idx % 50 == 0:
            print(f"  Processing file {file_idx+1}/{n_files}  "
                  f"({len(samples)} samples so far)...")

        task = load_json(os.path.join(train_path, fname))
        task_id     = fname.replace(".json", "")
        train_pairs = task.get("train", [])
        test_pairs  = task.get("test",  [])

        if not train_pairs or not test_pairs:
            skipped += 1
            continue

        test_pair = test_pairs[0]
        if "output" not in test_pair:
            skipped += 1
            continue

        test_h = len(test_pair["output"])
        test_w = len(test_pair["output"][0])

        for aug_idx in range(n_augmentations):
            # Augment each demo pair consistently (same transform applied
            # to both input and output of each pair)
            demo_inputs  = []
            demo_outputs = []

            pairs_to_use = train_pairs[:max_demos]

            for pair in pairs_to_use:
                if aug_idx == 0:
                    # First augmentation is always identity (no transform)
                    aug_in  = pair["input"]
                    aug_out = pair["output"]
                else:
                    aug_in, aug_out = augment_pair(pair["input"], pair["output"])

                demo_inputs.append(grid_to_tokens(aug_in))
                demo_outputs.append(grid_to_tokens(aug_out))

            # Augment test pair with same strategy
            if aug_idx == 0:
                test_in_raw  = test_pair["input"]
                test_out_raw = test_pair["output"]
            else:
                test_in_raw, test_out_raw = augment_pair(
                    test_pair["input"], test_pair["output"]
                )

            samples.append({
                "task_id"     : task_id,
                "demo_inputs" : demo_inputs,   # list of (900,) tensors
                "demo_outputs": demo_outputs,  # list of (900,) tensors
                "test_input"  : grid_to_tokens(test_in_raw),   # (900,)
                "test_output" : grid_to_tokens(test_out_raw),  # (900,)
                "test_h"      : test_h,
                "test_w"      : test_w,
                "test_raw"    : test_pair["output"],  # always unaugmented
            })

    print(f"  Samples built : {len(samples)}  (skipped {skipped} tasks)")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 4.  COLLATION
# ─────────────────────────────────────────────────────────────────────────────

def collate_batch(samples: List[Dict], device: str):
    """
    Build batch tensors from a list of task samples.

    Sequence layout (tokens):
        [puzzle_emb_ids] [demo1_in] [demo1_out] ... [demoK_in] [demoK_out] [test_in]

    The puzzle embedding positions use special token IDs in range
    [12, 12+PUZZLE_EMB_LEN) — outside the color range — so the embedding
    table learns task-specific representations at those positions.

    Returns:
        input_ids   : (B, seq_len)          long  — full context sequence
        target_ids  : (B, GRID_FLAT_LEN)    long  — test output tokens
        test_sizes  : list of (h, w) tuples
        test_raws   : list of raw output grids (for evaluation)
    """
    B = len(samples)

    # Determine max number of demos in this batch
    max_demos = max(len(s["demo_inputs"]) for s in samples)

    # Sequence: puzzle_emb + demos*2 + test_in
    context_len = PUZZLE_EMB_LEN + max_demos * 2 * GRID_FLAT_LEN + GRID_FLAT_LEN

    input_ids  = torch.full((B, context_len), PAD_TOKEN, dtype=torch.long)
    target_ids = torch.zeros(B, GRID_FLAT_LEN, dtype=torch.long)
    test_sizes = []
    test_raws  = []

    for i, s in enumerate(samples):
        pos = 0

        # Puzzle embedding: tokens 12, 13, ..., 12+PUZZLE_EMB_LEN-1
        for pe in range(PUZZLE_EMB_LEN):
            input_ids[i, pos] = 12 + pe
            pos += 1

        # Demo pairs
        n_demos = len(s["demo_inputs"])
        for d in range(max_demos):
            if d < n_demos:
                din  = s["demo_inputs"][d]
                dout = s["demo_outputs"][d]
            else:
                # Pad missing demos with all-padding tokens
                din  = torch.zeros(GRID_FLAT_LEN, dtype=torch.long)
                dout = torch.zeros(GRID_FLAT_LEN, dtype=torch.long)

            input_ids[i, pos:pos + GRID_FLAT_LEN] = din;  pos += GRID_FLAT_LEN
            input_ids[i, pos:pos + GRID_FLAT_LEN] = dout; pos += GRID_FLAT_LEN

        # Test input
        input_ids[i, pos:pos + GRID_FLAT_LEN] = s["test_input"]

        # Target
        target_ids[i] = s["test_output"]

        test_sizes.append((s["test_h"], s["test_w"]))
        test_raws.append(s["test_raw"])

    return (
        input_ids.to(device),
        target_ids.to(device),
        test_sizes,
        test_raws,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5.  TRM MODEL
# ─────────────────────────────────────────────────────────────────────────────

class TRMBlock(nn.Module):
    """
    The single shared transformer block that is applied recursively.

    This is the core of TRM: instead of N separate layers (like a standard
    transformer), we have ONE block with T_INNER layers that is called
    N_RECURSIONS times on the same hidden state.

    This dramatically reduces parameter count while allowing deep computation
    through repeated application of the same learned transformation.
    """

    def __init__(
        self,
        hidden_size: int  = HIDDEN_SIZE,
        n_heads: int      = N_HEADS,
        ff_dim: int       = FF_DIM,
        dropout: float    = DROPOUT,
        t_inner: int      = T_INNER,
    ):
        super().__init__()

        # T_INNER transformer layers — all share the same block instance
        # but are distinct layers within the block (not weight-shared within
        # the block itself, only across recursion calls)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,    # pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=t_inner,
            norm=nn.LayerNorm(hidden_size),
        )

    def forward(self, z_h: torch.Tensor) -> torch.Tensor:
        """
        z_h : (B, seq_len + puzzle_emb_len, hidden_size)
        Returns updated z_h of same shape.
        """
        return self.transformer(z_h)


class TinyRecursiveModel(nn.Module):
    """
    Full TRM for ARC.

    Architecture
    ────────────
    1. Token embedding: vocab_size → hidden_size
       Vocab: 0=pad, 1=EOS, 2-11=colors, 12-27=puzzle embedding tokens
    2. Positional embedding: learned, for context_len positions
    3. TRMBlock: one shared block, called N_RECURSIONS times
    4. Halt head (q_head): reads z_h[:,0] → 2 logits (halt / continue)
       At each step, produces a weighted prediction via ACT-style accumulation
    5. Output head: z_h[:, -GRID_FLAT_LEN:] → (GRID_FLAT_LEN, vocab_size)
       Predicts the test output grid tokens

    Forward pass (recursive loop)
    ──────────────────────────────
    z_h = embed(input_ids) + pos_emb
    cumulative_pred = 0
    halt_prob_remaining = 1.0

    for step in range(N_RECURSIONS):
        z_h = TRMBlock(z_h)

        halt_logits, continue_logits = q_head(z_h[:,0]).chunk(2, dim=-1)
        p_halt = sigmoid(halt_logits - continue_logits)  # prob of halting NOW

        step_pred = output_head(z_h[:, -GRID_FLAT_LEN:])
        cumulative_pred += halt_prob_remaining * p_halt * step_pred

        halt_prob_remaining *= (1 - p_halt)

    cumulative_pred += halt_prob_remaining * step_pred  # flush remainder

    Loss = cross_entropy(cumulative_pred, target_ids)
         + supervision_schedule_weight * intermediate_step_losses
    """

    def __init__(
        self,
        context_len: int,
        hidden_size: int  = HIDDEN_SIZE,
        n_heads: int      = N_HEADS,
        ff_dim: int       = FF_DIM,
        dropout: float    = DROPOUT,
        n_recursions: int = N_RECURSIONS,
        t_inner: int      = T_INNER,
        n_sup: int        = N_SUP,
    ):
        super().__init__()

        self.hidden_size  = hidden_size
        self.n_recursions = n_recursions
        self.n_sup        = n_sup
        self.context_len  = context_len

        # Vocabulary: pad(0), EOS(1), colors(2-11), puzzle_emb(12-27)
        vocab_size = 12 + PUZZLE_EMB_LEN  # = 28

        # Token embedding — separate learning rate in optimizer
        self.token_emb = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_TOKEN)

        # Learned positional embedding (paper uses learned, not sinusoidal)
        self.pos_emb = nn.Embedding(context_len, hidden_size)

        # THE shared recursive block
        self.block = TRMBlock(hidden_size, n_heads, ff_dim, dropout, t_inner)

        # Halt head: reads first hidden state → halt/continue logits
        # q_head(y) in the paper
        self.q_head = nn.Linear(hidden_size, 2, bias=True)

        # Output head: last GRID_FLAT_LEN positions → token logits
        self.output_head = nn.Linear(hidden_size, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight,   std=0.02)
        nn.init.zeros_(self.q_head.bias)
        nn.init.normal_(self.q_head.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,   # (B, context_len)
        target_ids: torch.Tensor,  # (B, GRID_FLAT_LEN)  — None at inference
        n_sup: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            loss        : scalar training loss (0 at inference if target=None)
            pred_tokens : (B, GRID_FLAT_LEN) predicted token indices

        ACT-style accumulation:
            At each recursion step the model produces a partial prediction
            weighted by its halt probability. The final prediction is the
            probability-weighted sum across all steps.

        Supervision schedule (n_sup):
            The first n_sup steps also contribute individual cross-entropy
            losses. This encourages the model to produce good predictions
            early, not just at the final step.
        """
        B, L = input_ids.shape
        device = input_ids.device

        if n_sup is None:
            n_sup = self.n_sup

        # ── embed ──────────────────────────────────────────────────────────
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        z_h = self.token_emb(input_ids) + self.pos_emb(pos_ids)  # (B, L, H)

        # ── recursive loop ─────────────────────────────────────────────────
        cumulative_logits    = torch.zeros(B, GRID_FLAT_LEN,
                                           self.output_head.out_features,
                                           device=device)
        halt_prob_remaining  = torch.ones(B, 1, 1, device=device)

        step_losses = []

        for step in range(self.n_recursions):
            z_h = self.block(z_h)   # (B, L, H)

            # Halt decision from first position
            halt_logits = self.q_head(z_h[:, 0, :])            # (B, 2)
            p_halt = torch.sigmoid(
                halt_logits[:, 0] - halt_logits[:, 1]
            ).unsqueeze(-1).unsqueeze(-1)                       # (B, 1, 1)

            # Prediction at this step from last GRID_FLAT_LEN positions
            step_logits = self.output_head(
                z_h[:, -GRID_FLAT_LEN:, :]
            )                                                   # (B, 900, vocab)

            # ACT accumulation
            cumulative_logits += halt_prob_remaining * p_halt * step_logits
            halt_prob_remaining = halt_prob_remaining * (1.0 - p_halt)

            # Supervision schedule: add individual step loss for early steps
            if target_ids is not None and step < n_sup:
                step_loss = F.cross_entropy(
                    step_logits.reshape(-1, step_logits.size(-1)),
                    target_ids.reshape(-1),
                    ignore_index=PAD_TOKEN,
                )
                # Weight decays with step: step 0 gets weight 1.0,
                # step n_sup-1 gets weight 1/n_sup
                weight = 1.0 / (step + 1)
                step_losses.append(weight * step_loss)

        # Flush remaining probability mass to final step prediction
        cumulative_logits += halt_prob_remaining * step_logits

        # ── main loss ──────────────────────────────────────────────────────
        if target_ids is not None:
            main_loss = F.cross_entropy(
                cumulative_logits.reshape(-1, cumulative_logits.size(-1)),
                target_ids.reshape(-1),
                ignore_index=PAD_TOKEN,
            )
            total_loss = main_loss
            if step_losses:
                total_loss = total_loss + sum(step_losses) / len(step_losses)
        else:
            total_loss = torch.tensor(0.0, device=device)

        # ── decode prediction ──────────────────────────────────────────────
        pred_tokens = cumulative_logits.argmax(dim=-1)   # (B, GRID_FLAT_LEN)

        return total_loss, pred_tokens


# ─────────────────────────────────────────────────────────────────────────────
# 6.  EMA (EXPONENTIAL MOVING AVERAGE) FOR MODEL WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

class EMA:
    """
    Maintains a shadow copy of model weights updated as:
        shadow = decay * shadow + (1 - decay) * param

    The paper uses EMA=0.999 and reports it helps avoid overfitting.
    At evaluation/inference, the EMA weights are used instead of the
    live weights.
    """

    def __init__(self, model: nn.Module, decay: float = EMA_DECAY):
        self.decay  = decay
        self.shadow = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name]
                    + (1.0 - self.decay) * param.data
                )

    def apply_shadow(self, model: nn.Module):
        """Replace model weights with EMA weights (for evaluation)."""
        self._backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original weights after evaluation."""
        for name, param in model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup = {}


# ─────────────────────────────────────────────────────────────────────────────
# 7.  TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model: TinyRecursiveModel,
    optimizer: torch.optim.Optimizer,
    samples: List[Dict],
    batch_size: int,
    device: str,
    ema: Optional[EMA] = None,
    train: bool = True,
) -> float:
    model.train() if train else model.eval()
    if train:
        random.shuffle(samples)

    total_loss    = 0.0
    total_batches = 0
    chunks = [samples[i:i+batch_size] for i in range(0, len(samples), batch_size)]
    n_chunks = len(chunks)

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for chunk_idx, chunk in enumerate(chunks):
            input_ids, target_ids, _, _ = collate_batch(chunk, device)

            loss, _ = model(input_ids, target_ids)

            if train:
                if not torch.isfinite(loss):
                    print(f"  WARNING: non-finite loss at batch {chunk_idx+1}, skipping")
                    optimizer.zero_grad()
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                if ema is not None:
                    ema.update(model)

                # Progress update so you can see training is alive
                if (chunk_idx + 1) % PRINT_EVERY == 0 or chunk_idx == 0:
                    print(f"  Batch {chunk_idx+1:4d}/{n_chunks} | loss={loss.item():.4f}")

            if torch.isfinite(loss):
                total_loss    += float(loss.item())
                total_batches += 1

    return total_loss / max(total_batches, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_task(
    sample: Dict,
    model: TinyRecursiveModel,
    device: str,
) -> bool:
    """
    Run inference on one task and check exact grid match.
    Returns True if the predicted grid exactly matches the target.
    """
    model.eval()

    # Build single-sample batch
    input_ids, _, test_sizes, test_raws = collate_batch([sample], device)

    with torch.no_grad():
        _, pred_tokens = model(input_ids, target_ids=None)

    h, w = test_sizes[0]
    predicted = tokens_to_grid(pred_tokens[0].cpu(), h, w)

    return predicted == test_raws[0]


# ─────────────────────────────────────────────────────────────────────────────
# 9.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(SEED)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path  = os.path.join(script_dir, "arc_trm_hybrid_best.pt")

    print(f"Device          : {DEVICE}")
    print(f"Hidden size     : {HIDDEN_SIZE}")
    print(f"Recursions      : {N_RECURSIONS}")
    print(f"Inner layers    : {T_INNER}")
    print(f"Supervision N   : {N_SUP}")
    print(f"Augmentations   : {N_AUGMENTATIONS}")
    print(f"Batch size      : {BATCH_SIZE}")
    print(f"Epochs          : {EPOCHS}\n")

    # ── build dataset ─────────────────────────────────────────────────────
    print("Building augmented task samples...")
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

    print(f"Train samples   : {len(train_samples)}")
    print(f"Val samples     : {len(val_samples)}\n")

    # ── compute context length ────────────────────────────────────────────
    # Must be consistent across all batches
    max_demos_in_data = max(len(s["demo_inputs"]) for s in all_samples)
    context_len = (PUZZLE_EMB_LEN
                   + max_demos_in_data * 2 * GRID_FLAT_LEN
                   + GRID_FLAT_LEN)

    print(f"Context length  : {context_len} tokens")
    print(f"  = {PUZZLE_EMB_LEN} puzzle + "
          f"{max_demos_in_data}×2×{GRID_FLAT_LEN} demos + "
          f"{GRID_FLAT_LEN} test_in")

    # Warn if context is very large — attention is O(seq^2)
    if context_len > 2000:
        print(f"\n  ⚠  Context length {context_len} is large.")
        print(f"     Attention cost ∝ {context_len}² = {context_len**2:,} ops per head.")
        print(f"     Consider reducing GRID_H/GRID_W or MAX_DEMOS if training is slow.\n")
    else:
        print(f"  ✓ Context length is manageable.\n")

    # ── build model ───────────────────────────────────────────────────────
    model = TinyRecursiveModel(
        context_len=context_len,
        hidden_size=HIDDEN_SIZE,
        n_heads=N_HEADS,
        ff_dim=FF_DIM,
        dropout=DROPOUT,
        n_recursions=N_RECURSIONS,
        t_inner=T_INNER,
        n_sup=N_SUP,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params    : {n_params:,}\n")

    # ── optimizer with separate embedding LR (as per paper) ───────────────
    embedding_params = list(model.token_emb.parameters()) + \
                       list(model.pos_emb.parameters())
    other_params     = [p for p in model.parameters()
                        if p.requires_grad and
                        not any(p is ep for ep in embedding_params)]

    optimizer = torch.optim.AdamW([
        {"params": embedding_params, "lr": EMBED_LR},
        {"params": other_params,     "lr": LR},
    ], betas=(0.9, 0.95), weight_decay=0.1)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR / 10,
    )

    ema = EMA(model, decay=EMA_DECAY) if USE_EMA else None

    best_val  = math.inf
    print("Starting TRM training...\n")

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(
            model, optimizer, train_samples,
            BATCH_SIZE, DEVICE, ema=ema, train=True,
        )

        # Evaluate with EMA weights if available
        if ema is not None:
            ema.apply_shadow(model)

        val_loss = run_epoch(
            model, None, val_samples,
            BATCH_SIZE, DEVICE, ema=None, train=False,
        )

        # Exact-match accuracy every 10 epochs
        if epoch % 10 == 0:
            # Use a small subset of unique tasks for speed
            unique_val = {s["task_id"]: s for s in val_samples}
            eval_tasks = list(unique_val.values())[:50]
            n_correct  = sum(
                evaluate_task(s, model, DEVICE) for s in eval_tasks
            )
            acc = 100.0 * n_correct / max(len(eval_tasks), 1)
            print(f"Epoch {epoch:03d} | train={train_loss:.4f} | "
                  f"val={val_loss:.4f} | "
                  f"exact_match={n_correct}/{len(eval_tasks)} ({acc:.1f}%)")
        else:
            print(f"Epoch {epoch:03d} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict"    : model.state_dict(),
                "ema_shadow"          : ema.shadow if ema else None,
                "optimizer_state_dict": optimizer.state_dict(),
                "context_len"         : context_len,
                "hidden_size"         : HIDDEN_SIZE,
                "n_recursions"        : N_RECURSIONS,
                "t_inner"             : T_INNER,
                "best_val_loss"       : best_val,
                "epoch"               : epoch,
            }, save_path)
            print(f"  → Saved best TRM (val={best_val:.4f})")

        if ema is not None:
            ema.restore(model)

        scheduler.step()

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()