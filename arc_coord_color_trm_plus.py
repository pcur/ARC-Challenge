"""
arc_coord_color_trm_plus.py

A TRM-inspired ARC-AGI-1 experiment script that extends the earlier
coordinate-color model with several missing TRM-like behaviors.

Expected folder layout:
    your_project/
    ├── arc_coord_color_trm_plus.py
    └── data/
        ├── training/*.json
        └── evaluation/*.json

Adds these behaviors compared with the earlier script:
    1. Task-level context:
       - Uses demo input/output pairs as context.
       - Uses one pair as the query target.

    2. Puzzle/task embeddings:
       - Each JSON file gets a learned puzzle embedding.
       - Augmented JSON files get their own puzzle IDs because their stems differ.

    3. Role embeddings:
       - demo input, demo output, query input, answer state.

    4. Pair-index embeddings:
       - Tokens know which demo pair they came from.

    5. Coordinate-color embeddings:
       - Context cell token uses [row, column, color] -> learned vector.

    6. Answer-state feedback:
       - The model maintains a 30x30 answer canvas.
       - Each recursive step predicts colors.
       - The next step embeds the previous predicted answer.

    7. Deep supervision:
       - Every recursive step receives color loss.

    8. Q/halting head:
       - Each recursion step predicts whether it should halt.
       - During training, the best step in the current forward pass is treated as the halt target.
       - During evaluation/plots, the selected step can be chosen by the highest halt logit.

    9. EMA:
       - Maintains exponential moving average weights and uses EMA for eval/plots/checkpointing.

    10. Plots after training:
       - Saves support/query/model-output/actual-output visualizations.

Important:
    This is NOT the official TRM implementation.
    It is a practical TRM-inspired research script built on our coordinate-color embedding idea.

Run examples:
    python arc_coord_color_trm_plus.py --epochs 2 --batch_size 2 --hidden_size 256 --n_heads 4 --n_recursions 3 --max_eval_plots 20

    python arc_coord_color_trm_plus.py --epochs 20 --batch_size 4 --hidden_size 512 --n_heads 8 --n_recursions 6
"""

import argparse
import json
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

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
    train_split: str = "training"
    eval_split: str = "evaluation"

    max_rows: int = 30
    max_cols: int = 30
    num_colors: int = 10
    answer_unknown_color: int = 10

    max_demo_pairs: int = 8
    max_seq_len: int = 4096

    hidden_size: int = 512
    n_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.1
    n_recursions: int = 6

    batch_size: int = 4
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    color_loss_weight: float = 1.0
    shape_loss_weight: float = 0.25
    halt_loss_weight: float = 0.10
    deep_supervision_decay: float = 0.85

    ignore_index: int = -100
    context_pad_color_id: int = 9000

    # Role IDs.
    role_demo_input: int = 0
    role_demo_output: int = 1
    role_query_input: int = 2
    role_answer_state: int = 3
    num_roles: int = 4

    seed: int = 1234
    num_workers: int = 0

    use_ema: bool = True
    ema_decay: float = 0.995

    save_path: str = "checkpoints/coord_color_trm_plus.pt"
    plot_dir: str = "eval_plots_plus"
    max_eval_plots: Optional[int] = 100

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


ARC_COLORS = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25",
]
ARC_CMAP = ListedColormap(ARC_COLORS)
ARC_NORM = BoundaryNorm(boundaries=list(range(11)), ncolors=10)


# ============================================================
# Helpers
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def grid_to_tensor(grid: List[List[int]]) -> torch.Tensor:
    return torch.tensor(grid, dtype=torch.long)


def validate_grid(grid: torch.Tensor, name: str):
    if grid.dim() != 2:
        raise ValueError(f"{name} must be 2D, got {tuple(grid.shape)}")
    h, w = grid.shape
    if not (1 <= h <= 30 and 1 <= w <= 30):
        raise ValueError(f"{name} has invalid ARC shape {h}x{w}")
    if grid.min().item() < 0 or grid.max().item() > 9:
        raise ValueError(f"{name} has colors outside 0-9")


def pad_input_grid(grid: torch.Tensor, cfg: Config) -> Tuple[torch.Tensor, torch.Tensor]:
    validate_grid(grid, "input_grid")
    h, w = grid.shape
    padded = torch.zeros((cfg.max_rows, cfg.max_cols), dtype=torch.long)
    mask = torch.zeros((cfg.max_rows, cfg.max_cols), dtype=torch.bool)
    padded[:h, :w] = grid
    mask[:h, :w] = True
    return padded, mask


def pad_target_grid(grid: torch.Tensor, cfg: Config) -> Tuple[torch.Tensor, torch.Tensor]:
    validate_grid(grid, "target_grid")
    h, w = grid.shape
    padded = torch.full((cfg.max_rows, cfg.max_cols), cfg.ignore_index, dtype=torch.long)
    mask = torch.zeros((cfg.max_rows, cfg.max_cols), dtype=torch.bool)
    padded[:h, :w] = grid
    mask[:h, :w] = True
    return padded, mask


def coord_color_id(row: int, col: int, color: int, cfg: Config) -> int:
    return row * cfg.max_cols * cfg.num_colors + col * cfg.num_colors + color


def make_grid_tokens(
    grid: torch.Tensor,
    role_id: int,
    pair_idx: int,
    cfg: Config,
) -> Dict[str, torch.Tensor]:
    """
    Converts a variable-sized grid into a list of real-cell tokens.

    Each cell token gets:
        coord_color_id, row, col, color, role, pair_idx
    """
    validate_grid(grid, "token_grid")
    h, w = grid.shape

    coord_ids = []
    rows = []
    cols = []
    colors = []
    roles = []
    pair_ids = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c].item())
            coord_ids.append(coord_color_id(r, c, color, cfg))
            rows.append(r)
            cols.append(c)
            colors.append(color)
            roles.append(role_id)
            pair_ids.append(min(pair_idx, cfg.max_demo_pairs))

    return {
        "coord_ids": torch.tensor(coord_ids, dtype=torch.long),
        "rows": torch.tensor(rows, dtype=torch.long),
        "cols": torch.tensor(cols, dtype=torch.long),
        "colors": torch.tensor(colors, dtype=torch.long),
        "roles": torch.tensor(roles, dtype=torch.long),
        "pair_ids": torch.tensor(pair_ids, dtype=torch.long),
    }


def concat_token_dicts(dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = ["coord_ids", "rows", "cols", "colors", "roles", "pair_ids"]
    return {k: torch.cat([d[k] for d in dicts], dim=0) for k in keys}


# ============================================================
# Dataset
# ============================================================

class ARCTaskContextDataset(Dataset):
    """
    Each sample is:
        support demos + query input -> query output

    For training/evaluation split files:
        each train pair becomes a query once;
        all other train pairs become support context.

    If include_test_with_outputs is used, test pairs that contain outputs also become queries,
    using all train pairs as support.
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        cfg: Config,
        puzzle_id_map: Optional[Dict[str, int]] = None,
        include_test_with_outputs: bool = False,
        max_tasks: Optional[int] = None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.cfg = cfg
        self.split_dir = self.root_dir / split

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {self.split_dir}")

        paths = sorted(self.split_dir.glob("*.json"))
        if max_tasks is not None:
            paths = paths[:max_tasks]
        if not paths:
            raise FileNotFoundError(f"No JSON files found in {self.split_dir}")

        if puzzle_id_map is None:
            self.puzzle_id_map = {p.stem: i for i, p in enumerate(paths)}
        else:
            self.puzzle_id_map = puzzle_id_map

        self.samples = []

        for path in paths:
            task_id = path.stem
            if task_id not in self.puzzle_id_map:
                # Evaluation may contain task IDs unseen in training map.
                self.puzzle_id_map[task_id] = len(self.puzzle_id_map)

            task = load_json(path)
            train_pairs = task.get("train", [])
            test_pairs = task.get("test", [])

            # Every train pair becomes query; remaining train pairs are support.
            for q_idx, q_pair in enumerate(train_pairs):
                if "input" not in q_pair or "output" not in q_pair:
                    continue
                support = [p for i, p in enumerate(train_pairs) if i != q_idx and "input" in p and "output" in p]
                self.samples.append({
                    "task_id": task_id,
                    "puzzle_id": self.puzzle_id_map[task_id],
                    "query_source": "train",
                    "query_idx": q_idx,
                    "support": support[:cfg.max_demo_pairs],
                    "query_input": q_pair["input"],
                    "query_output": q_pair["output"],
                })

            if include_test_with_outputs:
                support = [p for p in train_pairs if "input" in p and "output" in p]
                for q_idx, q_pair in enumerate(test_pairs):
                    if "input" in q_pair and "output" in q_pair:
                        self.samples.append({
                            "task_id": task_id,
                            "puzzle_id": self.puzzle_id_map[task_id],
                            "query_source": "test",
                            "query_idx": q_idx,
                            "support": support[:cfg.max_demo_pairs],
                            "query_input": q_pair["input"],
                            "query_output": q_pair["output"],
                        })

        if not self.samples:
            raise RuntimeError(f"No usable samples found in {self.split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        cfg = self.cfg

        token_parts = []

        for pair_i, pair in enumerate(s["support"]):
            x = grid_to_tensor(pair["input"])
            y = grid_to_tensor(pair["output"])
            token_parts.append(make_grid_tokens(x, cfg.role_demo_input, pair_i, cfg))
            token_parts.append(make_grid_tokens(y, cfg.role_demo_output, pair_i, cfg))

        query_x = grid_to_tensor(s["query_input"])
        query_y = grid_to_tensor(s["query_output"])

        token_parts.append(make_grid_tokens(query_x, cfg.role_query_input, cfg.max_demo_pairs, cfg))
        context = concat_token_dicts(token_parts)

        # Hard truncate from the left if context gets too long.
        # Keeps the query input at the end.
        L = context["coord_ids"].shape[0]
        if L > cfg.max_seq_len:
            start = L - cfg.max_seq_len
            context = {k: v[start:] for k, v in context.items()}

        input_padded, input_mask = pad_input_grid(query_x, cfg)
        target_padded, target_mask = pad_target_grid(query_y, cfg)
        out_h, out_w = query_y.shape

        context["context_mask"] = torch.ones(context["coord_ids"].shape[0], dtype=torch.bool)

        return {
            "task_id": s["task_id"],
            "query_source": s["query_source"],
            "query_idx": s["query_idx"],
            "puzzle_id": torch.tensor(s["puzzle_id"], dtype=torch.long),
            "support": s["support"],

            "coord_ids": context["coord_ids"],
            "rows": context["rows"],
            "cols": context["cols"],
            "colors": context["colors"],
            "roles": context["roles"],
            "pair_ids": context["pair_ids"],
            "context_mask": context["context_mask"],

            "query_input_grid": input_padded,
            "query_input_mask": input_mask,
            "target_grid": target_padded,
            "target_mask": target_mask,
            "target_h_class": torch.tensor(out_h - 1, dtype=torch.long),
            "target_w_class": torch.tensor(out_w - 1, dtype=torch.long),
            "input_h": torch.tensor(query_x.shape[0], dtype=torch.long),
            "input_w": torch.tensor(query_x.shape[1], dtype=torch.long),
            "output_h": torch.tensor(out_h, dtype=torch.long),
            "output_w": torch.tensor(out_w, dtype=torch.long),
        }


def collate_task_context(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_len = max(item["coord_ids"].shape[0] for item in batch)
    B = len(batch)

    def pad_1d(name: str, pad_value: int = 0):
        out = torch.full((B, max_len), pad_value, dtype=torch.long)
        for i, item in enumerate(batch):
            v = item[name]
            out[i, :v.shape[0]] = v
        return out

    coord_pad = 9000
    out = {
        "task_id": [item["task_id"] for item in batch],
        "query_source": [item["query_source"] for item in batch],
        "query_idx": torch.tensor([item["query_idx"] for item in batch], dtype=torch.long),
        "support": [item["support"] for item in batch],

        "puzzle_id": torch.stack([item["puzzle_id"] for item in batch]),
        "coord_ids": pad_1d("coord_ids", coord_pad),
        "rows": pad_1d("rows", 0),
        "cols": pad_1d("cols", 0),
        "colors": pad_1d("colors", 0),
        "roles": pad_1d("roles", 0),
        "pair_ids": pad_1d("pair_ids", 0),

        "context_mask": torch.zeros((B, max_len), dtype=torch.bool),
        "query_input_grid": torch.stack([item["query_input_grid"] for item in batch]),
        "query_input_mask": torch.stack([item["query_input_mask"] for item in batch]),
        "target_grid": torch.stack([item["target_grid"] for item in batch]),
        "target_mask": torch.stack([item["target_mask"] for item in batch]),
        "target_h_class": torch.stack([item["target_h_class"] for item in batch]),
        "target_w_class": torch.stack([item["target_w_class"] for item in batch]),
        "input_h": torch.stack([item["input_h"] for item in batch]),
        "input_w": torch.stack([item["input_w"] for item in batch]),
        "output_h": torch.stack([item["output_h"] for item in batch]),
        "output_w": torch.stack([item["output_w"] for item in batch]),
    }

    for i, item in enumerate(batch):
        L = item["coord_ids"].shape[0]
        out["context_mask"][i, :L] = True

    return out


# ============================================================
# EMA
# ============================================================

class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.detach().clone()
                param.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.copy_(self.backup[name])
        self.backup = {}


# ============================================================
# Model
# ============================================================

class ContextEmbedding(nn.Module):
    def __init__(self, cfg: Config, num_puzzles: int):
        super().__init__()
        self.cfg = cfg
        self.coord_color_emb = nn.Embedding(cfg.max_rows * cfg.max_cols * cfg.num_colors + 1, cfg.hidden_size, padding_idx=cfg.context_pad_color_id)
        self.row_emb = nn.Embedding(cfg.max_rows, cfg.hidden_size)
        self.col_emb = nn.Embedding(cfg.max_cols, cfg.hidden_size)
        self.color_emb = nn.Embedding(cfg.num_colors, cfg.hidden_size)
        self.role_emb = nn.Embedding(cfg.num_roles, cfg.hidden_size)
        self.pair_emb = nn.Embedding(cfg.max_demo_pairs + 1, cfg.hidden_size)
        self.puzzle_emb = nn.Embedding(max(num_puzzles, 1), cfg.hidden_size)
        self.norm = nn.LayerNorm(cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        B, L = batch["coord_ids"].shape
        puzzle = self.puzzle_emb(batch["puzzle_id"]).unsqueeze(1)
        x = (
            self.coord_color_emb(batch["coord_ids"])
            + self.row_emb(batch["rows"])
            + self.col_emb(batch["cols"])
            + self.color_emb(batch["colors"])
            + self.role_emb(batch["roles"])
            + self.pair_emb(batch["pair_ids"])
            + puzzle
        )
        return self.dropout(self.norm(x))


class AnswerEmbedding(nn.Module):
    def __init__(self, cfg: Config, context_embedding: ContextEmbedding):
        super().__init__()
        self.cfg = cfg
        self.answer_color_emb = nn.Embedding(cfg.num_colors + 1, cfg.hidden_size)
        self.row_emb = context_embedding.row_emb
        self.col_emb = context_embedding.col_emb
        self.role_emb = context_embedding.role_emb
        self.puzzle_emb = context_embedding.puzzle_emb
        self.norm = nn.LayerNorm(cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, answer_colors: torch.Tensor, puzzle_id: torch.Tensor) -> torch.Tensor:
        """
        answer_colors: [B, 30, 30], values 0-10 where 10 is unknown.
        """
        B, H, W = answer_colors.shape
        device = answer_colors.device
        rows = torch.arange(H, device=device).view(1, H, 1).expand(B, H, W)
        cols = torch.arange(W, device=device).view(1, 1, W).expand(B, H, W)
        role = torch.full((B, H, W), self.cfg.role_answer_state, dtype=torch.long, device=device)
        puzzle = self.puzzle_emb(puzzle_id).view(B, 1, 1, -1)

        x = (
            self.answer_color_emb(answer_colors)
            + self.row_emb(rows)
            + self.col_emb(cols)
            + self.role_emb(role)
            + puzzle
        )
        x = self.dropout(self.norm(x))
        return x.view(B, H * W, self.cfg.hidden_size)


class RecursiveBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_size)
        self.attn = nn.MultiheadAttention(cfg.hidden_size, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(cfg.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size * cfg.mlp_ratio),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_size * cfg.mlp_ratio, cfg.hidden_size),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class TRMPlusARCModel(nn.Module):
    def __init__(self, cfg: Config, num_puzzles: int):
        super().__init__()
        self.cfg = cfg
        self.context_embedding = ContextEmbedding(cfg, num_puzzles)
        self.answer_embedding = AnswerEmbedding(cfg, self.context_embedding)
        self.block = RecursiveBlock(cfg)
        self.final_norm = nn.LayerNorm(cfg.hidden_size)
        self.color_head = nn.Linear(cfg.hidden_size, cfg.num_colors)
        self.height_head = nn.Linear(cfg.hidden_size, cfg.max_rows)
        self.width_head = nn.Linear(cfg.hidden_size, cfg.max_cols)
        self.q_head = nn.Linear(cfg.hidden_size, 1)

    def forward(self, batch: Dict[str, torch.Tensor], teacher_forcing_answer: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        cfg = self.cfg
        B = batch["puzzle_id"].shape[0]
        device = batch["puzzle_id"].device

        context_x = self.context_embedding(batch)
        context_mask = batch["context_mask"]

        answer_colors = torch.full((B, cfg.max_rows, cfg.max_cols), cfg.answer_unknown_color, dtype=torch.long, device=device)

        step_logits = []
        step_height_logits = []
        step_width_logits = []
        step_q_logits = []

        for step in range(cfg.n_recursions):
            ans_x = self.answer_embedding(answer_colors, batch["puzzle_id"])
            x = torch.cat([context_x, ans_x], dim=1)

            answer_mask = torch.ones((B, cfg.max_rows * cfg.max_cols), dtype=torch.bool, device=device)
            full_mask = torch.cat([context_mask, answer_mask], dim=1)
            key_padding_mask = ~full_mask

            x = self.block(x, key_padding_mask=key_padding_mask)
            ans_h = x[:, -cfg.max_rows * cfg.max_cols:, :]
            ans_h = self.final_norm(ans_h)

            color_logits = self.color_head(ans_h).view(B, cfg.max_rows, cfg.max_cols, cfg.num_colors)
            pooled = ans_h.mean(dim=1)
            height_logits = self.height_head(pooled)
            width_logits = self.width_head(pooled)
            q_logits = self.q_head(pooled).squeeze(-1)

            step_logits.append(color_logits)
            step_height_logits.append(height_logits)
            step_width_logits.append(width_logits)
            step_q_logits.append(q_logits)

            # Answer-state feedback: feed previous prediction into next step.
            # Optional teacher forcing can be enabled if desired by passing target canvas.
            if teacher_forcing_answer is not None and self.training:
                pred = color_logits.argmax(dim=-1)
                mask = batch["target_mask"]
                answer_colors = torch.where(mask, teacher_forcing_answer.clamp(0, 9), pred)
            else:
                answer_colors = color_logits.argmax(dim=-1)

        return {
            "step_color_logits": step_logits,
            "step_height_logits": step_height_logits,
            "step_width_logits": step_width_logits,
            "step_q_logits": torch.stack(step_q_logits, dim=1),  # [B, T]
            "color_logits": step_logits[-1],
            "height_logits": step_height_logits[-1],
            "width_logits": step_width_logits[-1],
        }


# ============================================================
# Loss and metrics
# ============================================================

def per_sample_correct(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((pred == target) & mask).view(pred.shape[0], -1).sum(dim=1)


def compute_loss(outputs: Dict[str, Any], batch: Dict[str, torch.Tensor], cfg: Config) -> Dict[str, torch.Tensor]:
    target = batch["target_grid"]
    target_mask = batch["target_mask"]
    B = target.shape[0]
    T = len(outputs["step_color_logits"])

    color_loss_total = 0.0
    weight_total = 0.0
    step_corrects = []

    for t, logits in enumerate(outputs["step_color_logits"]):
        # Later steps get slightly larger weight.
        w = cfg.deep_supervision_decay ** (T - 1 - t)
        color_loss = F.cross_entropy(
            logits.reshape(-1, cfg.num_colors),
            target.reshape(-1),
            ignore_index=cfg.ignore_index,
        )
        color_loss_total = color_loss_total + w * color_loss
        weight_total += w

        pred = logits.argmax(dim=-1)
        step_corrects.append(per_sample_correct(pred, target, target_mask))

    color_loss_total = color_loss_total / max(weight_total, 1e-8)

    height_loss = F.cross_entropy(outputs["height_logits"], batch["target_h_class"])
    width_loss = F.cross_entropy(outputs["width_logits"], batch["target_w_class"])
    shape_loss = height_loss + width_loss

    # Halt target: best recursion step by number of correct cells in this forward pass.
    correct_stack = torch.stack(step_corrects, dim=1).float()  # [B, T]
    best_step = correct_stack.argmax(dim=1)  # [B]
    halt_loss = F.cross_entropy(outputs["step_q_logits"], best_step)

    total = (
        cfg.color_loss_weight * color_loss_total
        + cfg.shape_loss_weight * shape_loss
        + cfg.halt_loss_weight * halt_loss
    )

    return {
        "loss": total,
        "color_loss": color_loss_total.detach(),
        "shape_loss": shape_loss.detach(),
        "halt_loss": halt_loss.detach(),
    }


@torch.no_grad()
def select_outputs(outputs: Dict[str, Any], use_q: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns selected color logits, height logits, width logits, selected step.
    """
    B = outputs["step_q_logits"].shape[0]
    if use_q:
        steps = outputs["step_q_logits"].argmax(dim=1)
    else:
        steps = torch.full((B,), len(outputs["step_color_logits"]) - 1, device=outputs["step_q_logits"].device, dtype=torch.long)

    color = []
    height = []
    width = []
    for b in range(B):
        s = int(steps[b].item())
        color.append(outputs["step_color_logits"][s][b])
        height.append(outputs["step_height_logits"][s][b])
        width.append(outputs["step_width_logits"][s][b])

    return torch.stack(color), torch.stack(height), torch.stack(width), steps


@torch.no_grad()
def compute_metrics(outputs: Dict[str, Any], batch: Dict[str, torch.Tensor], cfg: Config, use_q: bool = True) -> Dict[str, float]:
    color_logits, height_logits, width_logits, selected_steps = select_outputs(outputs, use_q=use_q)
    pred = color_logits.argmax(dim=-1)
    target = batch["target_grid"]
    mask = batch["target_mask"]

    correct = ((pred == target) & mask).sum().item()
    total = mask.sum().item()
    cell_acc = correct / max(total, 1)

    exacts = []
    for b in range(pred.shape[0]):
        exacts.append(float(torch.equal(pred[b][mask[b]], target[b][mask[b]])))

    ph = height_logits.argmax(dim=-1)
    pw = width_logits.argmax(dim=-1)
    shape_acc = ((ph == batch["target_h_class"]) & (pw == batch["target_w_class"])).float().mean().item()

    return {
        "cell_acc": cell_acc,
        "exact_match": sum(exacts) / max(len(exacts), 1),
        "shape_acc": shape_acc,
        "selected_step_avg": selected_steps.float().mean().item(),
        "correct_over_900_avg": per_sample_correct(pred, target, mask).float().mean().item(),
    }


# ============================================================
# Train / eval
# ============================================================

def move_batch_to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def train_one_epoch(model, loader, optimizer, ema: Optional[EMA], cfg: Config, epoch: int):
    model.train()
    start = time.perf_counter()
    sums = {"loss": 0.0, "color_loss": 0.0, "shape_loss": 0.0, "halt_loss": 0.0, "cell_acc": 0.0, "exact_match": 0.0, "shape_acc": 0.0}
    n = 0

    for batch in loader:
        batch = move_batch_to_device(batch, cfg.device)
        outputs = model(batch)
        losses = compute_loss(outputs, batch, cfg)

        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        if cfg.grad_clip and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update(model)

        metrics = compute_metrics(outputs, batch, cfg, use_q=True)
        for k in ["loss", "color_loss", "shape_loss", "halt_loss"]:
            sums[k] += float(losses[k].item())
        for k in ["cell_acc", "exact_match", "shape_acc"]:
            sums[k] += metrics[k]
        n += 1

    elapsed = time.perf_counter() - start
    out = {k: v / max(n, 1) for k, v in sums.items()}
    out["epoch_seconds"] = elapsed
    print(
        f"epoch {epoch:03d} train | time {elapsed:.2f}s | loss {out['loss']:.4f} | "
        f"color {out['color_loss']:.4f} | shape {out['shape_loss']:.4f} | halt {out['halt_loss']:.4f} | "
        f"cell_acc {out['cell_acc']:.4f} | exact {out['exact_match']:.4f} | shape_acc {out['shape_acc']:.4f}"
    )
    return out


@torch.no_grad()
def evaluate(model, loader, ema: Optional[EMA], cfg: Config, name: str):
    if ema is not None:
        ema.apply_shadow(model)
    model.eval()
    start = time.perf_counter()
    sums = {"loss": 0.0, "color_loss": 0.0, "shape_loss": 0.0, "halt_loss": 0.0, "cell_acc": 0.0, "exact_match": 0.0, "shape_acc": 0.0, "selected_step_avg": 0.0}
    n = 0

    for batch in loader:
        batch = move_batch_to_device(batch, cfg.device)
        outputs = model(batch)
        losses = compute_loss(outputs, batch, cfg)
        metrics = compute_metrics(outputs, batch, cfg, use_q=True)
        for k in ["loss", "color_loss", "shape_loss", "halt_loss"]:
            sums[k] += float(losses[k].item())
        for k in ["cell_acc", "exact_match", "shape_acc", "selected_step_avg"]:
            sums[k] += metrics[k]
        n += 1

    elapsed = time.perf_counter() - start
    out = {k: v / max(n, 1) for k, v in sums.items()}
    out["eval_seconds"] = elapsed
    print(
        f"{name} | time {elapsed:.2f}s | loss {out['loss']:.4f} | color {out['color_loss']:.4f} | "
        f"shape {out['shape_loss']:.4f} | halt {out['halt_loss']:.4f} | cell_acc {out['cell_acc']:.4f} | "
        f"exact {out['exact_match']:.4f} | shape_acc {out['shape_acc']:.4f} | sel_step {out['selected_step_avg']:.2f}"
    )
    if ema is not None:
        ema.restore(model)
    return out


# ============================================================
# Plotting
# ============================================================

@torch.no_grad()
def save_eval_plots(model, dataset, ema: Optional[EMA], cfg: Config):
    if ema is not None:
        ema.apply_shadow(model)
    model.eval()

    out_dir = Path(cfg.plot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(dataset) if cfg.max_eval_plots is None else min(len(dataset), cfg.max_eval_plots)

    print(f"Saving {total} eval plots to {out_dir.resolve()}")

    for idx in range(total):
        item = dataset[idx]
        batch = collate_task_context([item])
        batch = move_batch_to_device(batch, cfg.device)
        outputs = model(batch)
        color_logits, height_logits, width_logits, selected_steps = select_outputs(outputs, use_q=True)
        pred_canvas = color_logits.argmax(dim=-1)[0].cpu()

        target = item["target_grid"]
        target_mask = item["target_mask"]
        correct = ((pred_canvas == target) & target_mask).sum().item()

        input_h = int(item["input_h"])
        input_w = int(item["input_w"])
        output_h = int(item["output_h"])
        output_w = int(item["output_w"])
        pred_h = int(height_logits.argmax(dim=-1).item()) + 1
        pred_w = int(width_logits.argmax(dim=-1).item()) + 1
        pred_h = max(1, min(pred_h, cfg.max_rows))
        pred_w = max(1, min(pred_w, cfg.max_cols))

        # Display model output cropped to true shape for diagnostic clarity.
        query_in = item["query_input_grid"][:input_h, :input_w]
        pred = pred_canvas[:output_h, :output_w]
        actual = target[:output_h, :output_w]

        n_support = min(len(item["support"]), 2)
        ncols = 3 + 2 * n_support
        fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
        if ncols == 1:
            axes = [axes]

        ax_i = 0
        for s_i in range(n_support):
            sx = torch.tensor(item["support"][s_i]["input"], dtype=torch.long)
            sy = torch.tensor(item["support"][s_i]["output"], dtype=torch.long)
            axes[ax_i].imshow(sx, cmap=ARC_CMAP, norm=ARC_NORM)
            axes[ax_i].set_title(f"Demo {s_i} input")
            ax_i += 1
            axes[ax_i].imshow(sy, cmap=ARC_CMAP, norm=ARC_NORM)
            axes[ax_i].set_title(f"Demo {s_i} output")
            ax_i += 1

        panels = [
            ("Query input", query_in),
            (f"Model output\nsel step {int(selected_steps[0])}, pred shape {pred_h}x{pred_w}", pred),
            (f"Actual output\ntrue shape {output_h}x{output_w}", actual),
        ]
        for title, grid in panels:
            axes[ax_i].imshow(grid, cmap=ARC_CMAP, norm=ARC_NORM)
            axes[ax_i].set_title(title)
            ax_i += 1

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)

        task_id = item["task_id"]
        q_idx = int(item["query_idx"])
        fig.suptitle(f"task={task_id} | query={q_idx} | correct={correct}/900")
        plt.tight_layout()
        fname = f"{task_id}_query{q_idx}_correct{correct}-900.png"
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)

        if (idx + 1) % 50 == 0 or idx + 1 == total:
            print(f"Saved {idx + 1}/{total}")

    if ema is not None:
        ema.restore(model)


# ============================================================
# Main
# ============================================================

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_recursions", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--max_demo_pairs", type=int, default=8)
    p.add_argument("--include_test_with_outputs", action="store_true")
    p.add_argument("--save_path", type=str, default="checkpoints/coord_color_trm_plus.pt")
    p.add_argument("--plot_dir", type=str, default="eval_plots_plus")
    p.add_argument("--max_eval_plots", type=int, default=100)
    p.add_argument("--all_eval_plots", action="store_true")
    p.add_argument("--skip_plots", action="store_true")
    p.add_argument("--no_ema", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max_train_tasks", type=int, default=None)
    p.add_argument("--max_eval_tasks", type=int, default=None)
    return p


def main():
    args = build_argparser().parse_args()
    cfg = Config(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        n_heads=args.n_heads,
        n_recursions=args.n_recursions,
        learning_rate=args.lr,
        max_seq_len=args.max_seq_len,
        max_demo_pairs=args.max_demo_pairs,
        save_path=args.save_path,
        plot_dir=args.plot_dir,
        max_eval_plots=None if args.all_eval_plots else args.max_eval_plots,
        use_ema=not args.no_ema,
        seed=args.seed,
    )
    set_seed(cfg.seed)

    Path(cfg.save_path).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ARC coordinate-color TRM-plus model")
    print("=" * 80)
    print(asdict(cfg))

    train_ds = ARCTaskContextDataset(
        cfg.data_dir, cfg.train_split, cfg,
        include_test_with_outputs=args.include_test_with_outputs,
        max_tasks=args.max_train_tasks,
    )
    eval_ds = ARCTaskContextDataset(
        cfg.data_dir, cfg.eval_split, cfg,
        puzzle_id_map=train_ds.puzzle_id_map,
        include_test_with_outputs=args.include_test_with_outputs,
        max_tasks=args.max_eval_tasks,
    )

    num_puzzles = len(train_ds.puzzle_id_map)
    print(f"Train samples: {len(train_ds)}")
    print(f"Eval samples: {len(eval_ds)}")
    print(f"Puzzle embeddings: {num_puzzles}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=(cfg.device == "cuda"),
        collate_fn=collate_task_context,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=(cfg.device == "cuda"),
        collate_fn=collate_task_context,
    )

    model = TRMPlusARCModel(cfg, num_puzzles=num_puzzles).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    ema = EMA(model, cfg.ema_decay) if cfg.use_ema else None

    params = sum(p.numel() for p in model.parameters())
    print(f"Device: {cfg.device}")
    print(f"Parameters: {params:,}")
    print()

    best_exact = -1.0
    total_start = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, ema, cfg, epoch)
        eval_metrics = evaluate(model, eval_loader, ema, cfg, name=f"epoch {epoch:03d} eval")

        if eval_metrics["exact_match"] > best_exact:
            best_exact = eval_metrics["exact_match"]
            if ema is not None:
                ema.apply_shadow(model)
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": asdict(cfg),
                "puzzle_id_map": train_ds.puzzle_id_map,
                "epoch": epoch,
                "best_exact": best_exact,
                "train_metrics": train_metrics,
                "eval_metrics": eval_metrics,
            }, cfg.save_path)
            if ema is not None:
                ema.restore(model)
            print(f"Saved best checkpoint to {cfg.save_path}")
        print()

    elapsed = time.perf_counter() - total_start
    print(f"Training complete. Total time {elapsed:.2f}s. Best eval exact={best_exact:.4f}")

    # Load best checkpoint for plotting.
    ckpt = torch.load(cfg.save_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model_state_dict"])

    if not args.skip_plots:
        save_eval_plots(model, eval_ds, None, cfg)
    else:
        print("Skipping plots because --skip_plots was used.")


if __name__ == "__main__":
    main()
