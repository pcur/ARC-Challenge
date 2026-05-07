"""
arc_coord_color_trm.py

ARC-AGI-1 coordinate-color recursive model.

Expected folder layout:
    your_project/
    ├── arc_coord_color_trm.py
    └── data/
        ├── training/*.json
        └── evaluation/*.json

Core representation:
    Each real grid cell has:
        row id
        column id
        color id
        one-hot color vector

Color one-hot convention:
    ARC color 0 -> one_hot[0] = 1
    ARC color 1 -> one_hot[1] = 1
    ...
    ARC color 9 -> one_hot[9] = 1

Coordinate-color token convention:
    token_id = row * 30 * 10 + col * 10 + color

Example:
    row=5, col=12, color=3
    token_id = 5*30*10 + 12*10 + 3 = 1623

The model predicts:
    1. output grid colors over a padded 30x30 canvas
    2. output height
    3. output width

After training finishes, the script saves evaluation plots to ./eval_plots/ by default.
Each plot shows:
    input grid | model output grid | actual output grid

Plot filename format:
    taskid_sampleN_correctX-900.png

This is not the official TRM implementation.
It is a TRM-inspired recursive model using coordinate-color embeddings.
"""

import argparse
import json
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    data_dir: str = "data"

    max_rows: int = 30
    max_cols: int = 30
    num_colors: int = 10

    hidden_size: int = 512
    n_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.1

    n_recursions: int = 4

    batch_size: int = 16
    epochs: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # Loss weights
    color_loss_weight: float = 1.0
    shape_loss_weight: float = 0.25

    # Use evaluation/train pairs for validation.
    # Official evaluation tasks still contain train/demo pairs.
    eval_split: str = "evaluation"

    # Padding / ignore behavior
    pad_token_id: int = 9000
    ignore_index: int = -100

    seed: int = 1234
    num_workers: int = 0

    save_path: str = "coord_color_arc_model.pt"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# ARC plot colors
# ============================================================

ARC_COLORS = [
    "#000000",  # 0 black
    "#0074D9",  # 1 blue
    "#FF4136",  # 2 red
    "#2ECC40",  # 3 green
    "#FFDC00",  # 4 yellow
    "#AAAAAA",  # 5 gray
    "#F012BE",  # 6 magenta
    "#FF851B",  # 7 orange
    "#7FDBFF",  # 8 cyan
    "#870C25",  # 9 brown/maroon
]

ARC_CMAP = ListedColormap(ARC_COLORS)
ARC_NORM = BoundaryNorm(boundaries=list(range(11)), ncolors=10)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# ARC loading
# ============================================================

def load_arc_task(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def grid_to_tensor(grid: List[List[int]]) -> torch.Tensor:
    """
    Converts a JSON ARC grid into a LongTensor [H, W].
    """
    return torch.tensor(grid, dtype=torch.long)


def validate_grid(grid: torch.Tensor, name: str):
    if grid.dim() != 2:
        raise ValueError(f"{name} must be 2D, got shape {tuple(grid.shape)}")

    H, W = grid.shape

    if not (1 <= H <= 30 and 1 <= W <= 30):
        raise ValueError(f"{name} has invalid ARC shape {H}x{W}")

    if grid.min().item() < 0 or grid.max().item() > 9:
        raise ValueError(
            f"{name} has color outside ARC range 0-9: "
            f"min={grid.min().item()}, max={grid.max().item()}"
        )


def pad_input_grid(grid: torch.Tensor, cfg: Config) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads input grid to [30, 30].

    For input colors:
        padded cells are filled with 0, but input_mask marks them as invalid.

    The model uses input_mask and one-hot zeros for invalid cells, so padded 0s
    are not treated as real ARC color 0.
    """
    validate_grid(grid, "input_grid")

    H, W = grid.shape

    padded = torch.zeros((cfg.max_rows, cfg.max_cols), dtype=torch.long)
    mask = torch.zeros((cfg.max_rows, cfg.max_cols), dtype=torch.bool)

    padded[:H, :W] = grid
    mask[:H, :W] = True

    return padded, mask


def pad_target_grid(grid: torch.Tensor, cfg: Config) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads output grid to [30, 30].

    For target:
        real cells are 0-9
        padded cells are ignore_index

    This lets cross entropy ignore cells outside the real output shape.
    """
    validate_grid(grid, "target_grid")

    H, W = grid.shape

    padded = torch.full(
        (cfg.max_rows, cfg.max_cols),
        fill_value=cfg.ignore_index,
        dtype=torch.long,
    )
    mask = torch.zeros((cfg.max_rows, cfg.max_cols), dtype=torch.bool)

    padded[:H, :W] = grid
    mask[:H, :W] = True

    return padded, mask


class ARCPairDataset(Dataset):
    """
    Turns official ARC task JSON files into supervised input-output pairs.

    For each file:
        task["train"] contains demonstration pairs with input and output.
        task["test"] may or may not contain output depending on the dataset copy.

    This dataset uses:
        - all train/demo pairs
        - optionally test pairs if they include output
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        cfg: Config,
        include_test_with_outputs: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.cfg = cfg
        self.include_test_with_outputs = include_test_with_outputs

        self.split_dir = self.root_dir / split

        if not self.split_dir.exists():
            raise FileNotFoundError(
                f"Could not find split directory: {self.split_dir}\n"
                f"Expected something like data/training/*.json"
            )

        self.samples = []

        json_paths = sorted(self.split_dir.glob("*.json"))

        if len(json_paths) == 0:
            raise FileNotFoundError(f"No JSON files found in {self.split_dir}")

        for task_path in json_paths:
            task_id = task_path.stem
            task = load_arc_task(task_path)

            for pair_idx, pair in enumerate(task.get("train", [])):
                if "input" not in pair or "output" not in pair:
                    continue

                self.samples.append(
                    {
                        "task_id": task_id,
                        "source": "train",
                        "pair_idx": pair_idx,
                        "input": pair["input"],
                        "output": pair["output"],
                    }
                )

            if include_test_with_outputs:
                for pair_idx, pair in enumerate(task.get("test", [])):
                    if "input" in pair and "output" in pair:
                        self.samples.append(
                            {
                                "task_id": task_id,
                                "source": "test",
                                "pair_idx": pair_idx,
                                "input": pair["input"],
                                "output": pair["output"],
                            }
                        )

        if len(self.samples) == 0:
            raise RuntimeError(f"No usable supervised pairs found in {self.split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        input_grid = grid_to_tensor(item["input"])
        output_grid = grid_to_tensor(item["output"])

        input_padded, input_mask = pad_input_grid(input_grid, self.cfg)
        target_padded, target_mask = pad_target_grid(output_grid, self.cfg)

        out_h, out_w = output_grid.shape

        return {
            "task_id": item["task_id"],
            "source": item["source"],
            "pair_idx": item["pair_idx"],

            "input_grid": input_padded,       # [30, 30], values 0-9
            "input_mask": input_mask,         # [30, 30], bool

            "target_grid": target_padded,     # [30, 30], values 0-9 or ignore_index
            "target_mask": target_mask,       # [30, 30], bool

            # Classes 0-29 represent sizes 1-30.
            "target_h_class": torch.tensor(out_h - 1, dtype=torch.long),
            "target_w_class": torch.tensor(out_w - 1, dtype=torch.long),

            "input_h": torch.tensor(input_grid.shape[0], dtype=torch.long),
            "input_w": torch.tensor(input_grid.shape[1], dtype=torch.long),
            "output_h": torch.tensor(out_h, dtype=torch.long),
            "output_w": torch.tensor(out_w, dtype=torch.long),
        }


# ============================================================
# Coordinate-color IDs and one-hot encoding
# ============================================================

def make_coord_color_ids(
    row_ids: torch.Tensor,
    col_ids: torch.Tensor,
    color_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    cfg: Config,
) -> torch.Tensor:
    """
    Converts [row, col, color] into one token ID.

    For real cells:
        token_id = row * 30 * 10 + col * 10 + color

    For padded cells:
        token_id = cfg.pad_token_id

    With max_rows=30, max_cols=30, num_colors=10:
        real IDs are 0..8999
        pad ID is 9000
    """
    token_ids = (
        row_ids * cfg.max_cols * cfg.num_colors
        + col_ids * cfg.num_colors
        + color_ids
    )

    token_ids = torch.where(
        valid_mask,
        token_ids,
        torch.full_like(token_ids, cfg.pad_token_id),
    )

    return token_ids


def one_hot_arc_colors(
    color_grid: torch.Tensor,
    valid_mask: torch.Tensor,
    num_colors: int = 10,
) -> torch.Tensor:
    """
    One-hot encodes ARC colors.

    Matching convention:
        color 0 -> one_hot[..., 0]
        color 1 -> one_hot[..., 1]
        ...
        color 9 -> one_hot[..., 9]

    Padded cells receive the all-zero vector.
    """
    one_hot = F.one_hot(color_grid.clamp(0, num_colors - 1), num_classes=num_colors)
    one_hot = one_hot.float()
    one_hot = one_hot * valid_mask.unsqueeze(-1).float()
    return one_hot


# ============================================================
# Embedding module
# ============================================================

class ARCCoordColorEmbedding(nn.Module):
    """
    Input representation:

        x_cell =
            coord_color_emb[row, col, color]
          + color_onehot_projection(one_hot_color)
          + row_emb[row]
          + col_emb[col]
          + valid_mask_projection(valid)

    The one-hot color vector follows direct ARC color identity:
        color k is represented by element k.
    """

    def __init__(self, cfg: Config):
        super().__init__()

        self.cfg = cfg

        vocab_size = cfg.max_rows * cfg.max_cols * cfg.num_colors + 1

        self.coord_color_emb = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=cfg.hidden_size,
            padding_idx=cfg.pad_token_id,
        )

        self.row_emb = nn.Embedding(cfg.max_rows, cfg.hidden_size)
        self.col_emb = nn.Embedding(cfg.max_cols, cfg.hidden_size)

        self.color_onehot_proj = nn.Linear(cfg.num_colors, cfg.hidden_size)
        self.valid_proj = nn.Linear(1, cfg.hidden_size)

        self.norm = nn.LayerNorm(cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, color_grid: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        color_grid:
            [B, 30, 30], values 0-9

        valid_mask:
            [B, 30, 30], bool

        returns:
            [B, 900, hidden_size]
        """
        B, H, W = color_grid.shape
        device = color_grid.device

        row_ids = torch.arange(H, device=device).view(1, H, 1).expand(B, H, W)
        col_ids = torch.arange(W, device=device).view(1, 1, W).expand(B, H, W)

        token_ids = make_coord_color_ids(
            row_ids=row_ids,
            col_ids=col_ids,
            color_ids=color_grid,
            valid_mask=valid_mask,
            cfg=self.cfg,
        )

        color_oh = one_hot_arc_colors(
            color_grid=color_grid,
            valid_mask=valid_mask,
            num_colors=self.cfg.num_colors,
        )

        valid_float = valid_mask.unsqueeze(-1).float()

        x = (
            self.coord_color_emb(token_ids)
            + self.row_emb(row_ids)
            + self.col_emb(col_ids)
            + self.color_onehot_proj(color_oh)
            + self.valid_proj(valid_float)
        )

        x = self.norm(x)
        x = self.dropout(x)

        return x.view(B, H * W, self.cfg.hidden_size)


# ============================================================
# Recursive block
# ============================================================

class RecursiveBlock(nn.Module):
    """
    Compact transformer-style recursive block.

    TRM-inspired behavior:
        same block is applied repeatedly to refine the representation.
    """

    def __init__(self, cfg: Config):
        super().__init__()

        self.norm1 = nn.LayerNorm(cfg.hidden_size)

        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.hidden_size,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(cfg.hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size * cfg.mlp_ratio),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_size * cfg.mlp_ratio, cfg.hidden_size),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        """
        x:
            [B, L, D]

        key_padding_mask:
            [B, L], True means masked/ignored for attention.
        """
        h = self.norm1(x)

        attn_out, _ = self.attn(
            h,
            h,
            h,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        x = x + attn_out

        h = self.norm2(x)
        x = x + self.mlp(h)

        return x


# ============================================================
# Main model
# ============================================================

class CoordColorARCModel(nn.Module):
    """
    Predicts:
        color_logits: [B, 30, 30, 10]
        height_logits: [B, 30]
        width_logits: [B, 30]

    Height/width classes:
        class 0 -> size 1
        class 1 -> size 2
        ...
        class 29 -> size 30
    """

    def __init__(self, cfg: Config):
        super().__init__()

        self.cfg = cfg

        self.embedding = ARCCoordColorEmbedding(cfg)
        self.recursive_block = RecursiveBlock(cfg)

        self.final_norm = nn.LayerNorm(cfg.hidden_size)
        self.color_head = nn.Linear(cfg.hidden_size, cfg.num_colors)

        self.shape_pool_norm = nn.LayerNorm(cfg.hidden_size)
        self.height_head = nn.Linear(cfg.hidden_size, cfg.max_rows)
        self.width_head = nn.Linear(cfg.hidden_size, cfg.max_cols)

    def forward(
        self,
        input_grid: torch.Tensor,
        input_mask: torch.Tensor,
        return_all_steps: bool = False,
    ):
        """
        input_grid:
            [B, 30, 30]

        input_mask:
            [B, 30, 30], True for real cells

        returns:
            dict with color_logits, height_logits, width_logits
        """
        B = input_grid.shape[0]

        x = self.embedding(input_grid, input_mask)

        # MultiheadAttention key_padding_mask uses True for tokens to ignore.
        key_padding_mask = ~input_mask.view(B, self.cfg.max_rows * self.cfg.max_cols)

        all_step_logits = []

        for _ in range(self.cfg.n_recursions):
            x = self.recursive_block(x, key_padding_mask=key_padding_mask)

            if return_all_steps:
                step_h = self.final_norm(x)
                step_color_logits = self.color_head(step_h)
                step_color_logits = step_color_logits.view(
                    B,
                    self.cfg.max_rows,
                    self.cfg.max_cols,
                    self.cfg.num_colors,
                )
                all_step_logits.append(step_color_logits)

        h = self.final_norm(x)

        color_logits = self.color_head(h)
        color_logits = color_logits.view(
            B,
            self.cfg.max_rows,
            self.cfg.max_cols,
            self.cfg.num_colors,
        )

        # Masked mean pool over valid input cells.
        valid = input_mask.view(B, -1).float().unsqueeze(-1)
        pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        pooled = self.shape_pool_norm(pooled)

        height_logits = self.height_head(pooled)
        width_logits = self.width_head(pooled)

        out = {
            "color_logits": color_logits,
            "height_logits": height_logits,
            "width_logits": width_logits,
        }

        if return_all_steps:
            out["all_step_color_logits"] = all_step_logits

        return out


# ============================================================
# Loss and metrics
# ============================================================

def compute_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], cfg: Config):
    color_logits = outputs["color_logits"]
    height_logits = outputs["height_logits"]
    width_logits = outputs["width_logits"]

    target_grid = batch["target_grid"]
    target_h_class = batch["target_h_class"]
    target_w_class = batch["target_w_class"]

    color_loss = F.cross_entropy(
        color_logits.reshape(-1, cfg.num_colors),
        target_grid.reshape(-1),
        ignore_index=cfg.ignore_index,
    )

    height_loss = F.cross_entropy(height_logits, target_h_class)
    width_loss = F.cross_entropy(width_logits, target_w_class)
    shape_loss = height_loss + width_loss

    total_loss = (
        cfg.color_loss_weight * color_loss
        + cfg.shape_loss_weight * shape_loss
    )

    return {
        "loss": total_loss,
        "color_loss": color_loss.detach(),
        "shape_loss": shape_loss.detach(),
        "height_loss": height_loss.detach(),
        "width_loss": width_loss.detach(),
    }


@torch.no_grad()
def compute_metrics(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], cfg: Config):
    color_logits = outputs["color_logits"]
    height_logits = outputs["height_logits"]
    width_logits = outputs["width_logits"]

    target_grid = batch["target_grid"]
    target_mask = batch["target_mask"]

    pred_grid = color_logits.argmax(dim=-1)

    valid = target_mask
    correct_cells = ((pred_grid == target_grid) & valid).sum().item()
    total_real_cells = valid.sum().item()
    cell_acc = correct_cells / max(total_real_cells, 1)

    B = pred_grid.shape[0]

    per_sample_exact = []
    per_sample_correct_900 = []

    for b in range(B):
        mask_b = target_mask[b]
        exact = torch.equal(pred_grid[b][mask_b], target_grid[b][mask_b])
        per_sample_exact.append(float(exact))

        correct_900 = ((pred_grid[b] == target_grid[b]) & target_mask[b]).sum().item()
        per_sample_correct_900.append(correct_900)

    exact_match = sum(per_sample_exact) / max(len(per_sample_exact), 1)
    correct_over_900_avg = sum(per_sample_correct_900) / max(len(per_sample_correct_900), 1)

    pred_h_class = height_logits.argmax(dim=-1)
    pred_w_class = width_logits.argmax(dim=-1)

    height_acc = (pred_h_class == batch["target_h_class"]).float().mean().item()
    width_acc = (pred_w_class == batch["target_w_class"]).float().mean().item()
    shape_acc = (
        (pred_h_class == batch["target_h_class"])
        & (pred_w_class == batch["target_w_class"])
    ).float().mean().item()

    return {
        "cell_acc": cell_acc,
        "exact_match": exact_match,
        "height_acc": height_acc,
        "width_acc": width_acc,
        "shape_acc": shape_acc,
        "correct_over_900_avg": correct_over_900_avg,
    }


# ============================================================
# Train / eval loops
# ============================================================

def move_batch_to_device(batch: Dict, device: str):
    moved = {}

    for k, v in batch.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device)
        else:
            moved[k] = v

    return moved


def train_one_epoch(model, loader, optimizer, cfg: Config, epoch: int):
    model.train()
    epoch_start = time.perf_counter()

    running = {
        "loss": 0.0,
        "color_loss": 0.0,
        "shape_loss": 0.0,
        "cell_acc": 0.0,
        "exact_match": 0.0,
        "shape_acc": 0.0,
        "correct_over_900_avg": 0.0,
    }

    n = 0

    for batch in loader:
        batch = move_batch_to_device(batch, cfg.device)

        outputs = model(
            input_grid=batch["input_grid"],
            input_mask=batch["input_mask"],
        )

        losses = compute_loss(outputs, batch, cfg)

        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()

        if cfg.grad_clip is not None and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()

        metrics = compute_metrics(outputs, batch, cfg)

        running["loss"] += losses["loss"].item()
        running["color_loss"] += losses["color_loss"].item()
        running["shape_loss"] += losses["shape_loss"].item()
        running["cell_acc"] += metrics["cell_acc"]
        running["exact_match"] += metrics["exact_match"]
        running["shape_acc"] += metrics["shape_acc"]
        running["correct_over_900_avg"] += metrics["correct_over_900_avg"]

        n += 1

    elapsed = time.perf_counter() - epoch_start
    out = {k: v / max(n, 1) for k, v in running.items()}
    out["epoch_seconds"] = elapsed

    print(
        f"epoch {epoch:03d} train | "
        f"time {elapsed:.2f}s | "
        f"loss {out['loss']:.4f} | "
        f"color {out['color_loss']:.4f} | "
        f"shape {out['shape_loss']:.4f} | "
        f"cell_acc {out['cell_acc']:.4f} | "
        f"exact {out['exact_match']:.4f} | "
        f"shape_acc {out['shape_acc']:.4f} | "
        f"correct/900 avg {out['correct_over_900_avg']:.2f}"
    )

    return out


@torch.no_grad()
def evaluate(model, loader, cfg: Config, name: str = "eval"):
    model.eval()
    eval_start = time.perf_counter()

    running = {
        "loss": 0.0,
        "color_loss": 0.0,
        "shape_loss": 0.0,
        "cell_acc": 0.0,
        "exact_match": 0.0,
        "shape_acc": 0.0,
        "correct_over_900_avg": 0.0,
    }

    n = 0

    for batch in loader:
        batch = move_batch_to_device(batch, cfg.device)

        outputs = model(
            input_grid=batch["input_grid"],
            input_mask=batch["input_mask"],
        )

        losses = compute_loss(outputs, batch, cfg)
        metrics = compute_metrics(outputs, batch, cfg)

        running["loss"] += losses["loss"].item()
        running["color_loss"] += losses["color_loss"].item()
        running["shape_loss"] += losses["shape_loss"].item()
        running["cell_acc"] += metrics["cell_acc"]
        running["exact_match"] += metrics["exact_match"]
        running["shape_acc"] += metrics["shape_acc"]
        running["correct_over_900_avg"] += metrics["correct_over_900_avg"]

        n += 1

    elapsed = time.perf_counter() - eval_start
    out = {k: v / max(n, 1) for k, v in running.items()}
    out["eval_seconds"] = elapsed

    print(
        f"{name} | "
        f"time {elapsed:.2f}s | "
        f"loss {out['loss']:.4f} | "
        f"color {out['color_loss']:.4f} | "
        f"shape {out['shape_loss']:.4f} | "
        f"cell_acc {out['cell_acc']:.4f} | "
        f"exact {out['exact_match']:.4f} | "
        f"shape_acc {out['shape_acc']:.4f} | "
        f"correct/900 avg {out['correct_over_900_avg']:.2f}"
    )

    return out


# ============================================================
# Plotting after training
# ============================================================

@torch.no_grad()
def save_eval_plots_after_training(
    model: CoordColorARCModel,
    dataset: ARCPairDataset,
    cfg: Config,
    output_dir: str = "eval_plots",
    max_plots: Optional[int] = None,
):
    """
    Saves matplotlib plots after training.

    Each plot contains:
        1. input grid
        2. model output grid
        3. actual output grid

    Filename format:
        taskid_sampleN_correctX-900.png

    correctX-900 means:
        number of correct predicted cells inside the actual target area,
        using 900 as the denominator because ARC max canvas is 30x30.
    """
    model.eval()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total = len(dataset) if max_plots is None else min(len(dataset), max_plots)

    print()
    print("=" * 80)
    print(f"Saving evaluation plots to: {output_path.resolve()}")
    print(f"Number of plots: {total}")
    print("=" * 80)

    for idx in range(total):
        item = dataset[idx]

        task_id = item["task_id"]
        sample_num = int(item["pair_idx"])

        input_grid = item["input_grid"].unsqueeze(0).to(cfg.device)
        input_mask = item["input_mask"].unsqueeze(0).to(cfg.device)

        target_grid = item["target_grid"]
        target_mask = item["target_mask"]

        input_h = int(item["input_h"])
        input_w = int(item["input_w"])
        output_h = int(item["output_h"])
        output_w = int(item["output_w"])

        outputs = model(input_grid, input_mask)

        pred_canvas = outputs["color_logits"].argmax(dim=-1)[0].cpu()  # [30, 30]

        pred_h = int(outputs["height_logits"].argmax(dim=-1).item()) + 1
        pred_w = int(outputs["width_logits"].argmax(dim=-1).item()) + 1

        pred_h = max(1, min(pred_h, cfg.max_rows))
        pred_w = max(1, min(pred_w, cfg.max_cols))

        input_display = item["input_grid"][:input_h, :input_w].cpu()
        pred_display = pred_canvas[:pred_h, :pred_w].cpu()
        target_display = target_grid[:output_h, :output_w].cpu()

        correct_cells = ((pred_canvas == target_grid) & target_mask).sum().item()

        filename = f"{task_id}_sample{sample_num}_correct{correct_cells}-900.png"
        save_file = output_path / filename

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        panels = [
            ("Input Grid", input_display),
            (f"Model Output\npred shape {pred_h}x{pred_w}", pred_display),
            (f"Actual Grid\ntrue shape {output_h}x{output_w}", target_display),
        ]

        for ax, (title, grid) in zip(axes, panels):
            ax.imshow(grid, cmap=ARC_CMAP, norm=ARC_NORM)
            ax.set_title(title)
            ax.set_xticks(range(grid.shape[1]))
            ax.set_yticks(range(grid.shape[0]))
            ax.grid(True, linewidth=0.5)
            ax.tick_params(labelbottom=False, labelleft=False, length=0)

        fig.suptitle(
            f"task={task_id} | sample={sample_num} | correct={correct_cells}/900",
            fontsize=12,
        )

        plt.tight_layout()
        plt.savefig(save_file, dpi=150)
        plt.close(fig)

        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            print(f"Saved {idx + 1}/{total} plots")

    print("Finished saving evaluation plots.")
    print()


# ============================================================
# Inference helper
# ============================================================

@torch.no_grad()
def predict_single_grid(model: CoordColorARCModel, raw_grid: List[List[int]], cfg: Config):
    """
    Predicts output for one raw ARC input grid.

    Returns:
        predicted_grid:
            List[List[int]] cropped to predicted output H,W
    """
    model.eval()

    grid = grid_to_tensor(raw_grid)
    input_padded, input_mask = pad_input_grid(grid, cfg)

    batch = {
        "input_grid": input_padded.unsqueeze(0).to(cfg.device),
        "input_mask": input_mask.unsqueeze(0).to(cfg.device),
    }

    outputs = model(batch["input_grid"], batch["input_mask"])

    pred_canvas = outputs["color_logits"].argmax(dim=-1)[0]

    pred_h = outputs["height_logits"].argmax(dim=-1).item() + 1
    pred_w = outputs["width_logits"].argmax(dim=-1).item() + 1

    pred_grid = pred_canvas[:pred_h, :pred_w].cpu().tolist()
    return pred_grid


# ============================================================
# CLI / main
# ============================================================

def build_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_recursions", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save_path", type=str, default="coord_color_arc_model.pt")
    parser.add_argument("--include_test_with_outputs", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--plot_dir", type=str, default="eval_plots")
    parser.add_argument("--max_eval_plots", type=int, default=None)
    parser.add_argument("--skip_plots", action="store_true")

    return parser


def main():
    args = build_argparser().parse_args()

    cfg = Config(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        n_heads=args.n_heads,
        n_recursions=args.n_recursions,
        learning_rate=args.lr,
        save_path=args.save_path,
        seed=args.seed,
    )

    set_seed(cfg.seed)

    print("=" * 80)
    print("ARC coordinate-color recursive model")
    print("=" * 80)
    print(asdict(cfg))
    print()

    train_dataset = ARCPairDataset(
        root_dir=cfg.data_dir,
        split="training",
        cfg=cfg,
        include_test_with_outputs=args.include_test_with_outputs,
    )

    eval_dataset = ARCPairDataset(
        root_dir=cfg.data_dir,
        split=cfg.eval_split,
        cfg=cfg,
        include_test_with_outputs=args.include_test_with_outputs,
    )

    print(f"Training supervised pairs: {len(train_dataset)}")
    print(f"Evaluation supervised pairs: {len(eval_dataset)}")
    print()

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
    )

    model = CoordColorARCModel(cfg).to(cfg.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 80)
    print("Model")
    print("=" * 80)
    print(f"Device: {cfg.device}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    best_eval_exact = -1.0

    total_start = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, cfg, epoch)
        eval_metrics = evaluate(model, eval_loader, cfg, name=f"epoch {epoch:03d} evaluation")

        if eval_metrics["exact_match"] > best_eval_exact:
            best_eval_exact = eval_metrics["exact_match"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "epoch": epoch,
                    "best_eval_exact": best_eval_exact,
                    "train_metrics": train_metrics,
                    "eval_metrics": eval_metrics,
                },
                cfg.save_path,
            )

            print(f"Saved best checkpoint to {cfg.save_path}")

        print()

    total_elapsed = time.perf_counter() - total_start

    print("Training complete.")
    print(f"Total training+eval time: {total_elapsed:.2f}s")
    print(f"Best eval exact match: {best_eval_exact:.4f}")

    if not args.skip_plots:
        checkpoint = torch.load(cfg.save_path, map_location=cfg.device)
        model.load_state_dict(checkpoint["model_state_dict"])

        save_eval_plots_after_training(
            model=model,
            dataset=eval_dataset,
            cfg=cfg,
            output_dir=args.plot_dir,
            max_plots=args.max_eval_plots,
        )
    else:
        print("Skipping evaluation plots because --skip_plots was used.")


if __name__ == "__main__":
    main()









'''
#!/bin/bash

#SBATCH --job-name=qwen3-sft
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=02:00:00

# Change 'rbunescu' to your <username>
cd /users/mmajeske/hw06/hw06/sft

# Make sure the right Python environment is activated before running this.
python sft_qa.py -train -model_load /projects/class/itcs6101_091/hw06/models/Qwen3-0.6B-

'''