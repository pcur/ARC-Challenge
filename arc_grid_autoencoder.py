"""
arc_grid_autoencoder.py

First-pass ARC-AGI grid autoencoder.

Goal:
    Train a small denoising convolutional autoencoder that takes an ARC grid,
    pads it to 30x30, and learns a representative latent embedding.

Input:
    ARC-AGI JSON files, either:
        data/training/*.json
        data/evaluation/*.json

Each ARC JSON file should look like:
    {
        "train": [{"input": [[...]], "output": [[...]]}, ...],
        "test":  [{"input": [[...]], "output": [[...]]}, ...]
    }

Example usage:
    python arc_grid_autoencoder.py \
        --data_dir /path/to/ARC-AGI/data/training \
        --epochs 50 \
        --batch_size 64 \
        --latent_dim 128 \
        --save_path arc_grid_ae.pt \
        --embedding_out arc_grid_embeddings.pt

Notes:
    - ARC colors are categorical tokens 0-9.
    - Padding cells use PAD_ID = 10.
    - The reconstruction loss ignores PAD_ID cells.
    - The decoder predicts only ARC colors 0-9, not PAD_ID.
    - The encoder embedding z can later be injected into TRM.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


MAX_H = 30
MAX_W = 30
NUM_COLORS = 10
PAD_ID = 10
VOCAB_SIZE = 11  # colors 0-9 plus PAD_ID


# -----------------------------
# Data loading
# -----------------------------

@dataclass
class ARCGridItem:
    task_id: str
    split: str          # "train" or "test"
    pair_index: int
    side: str           # "input" or "output"
    grid: List[List[int]]


class ARCGridDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        include_train_pairs: bool = True,
        include_test_pairs: bool = True,
        include_inputs: bool = True,
        include_outputs: bool = True,
        max_tasks: int | None = None,
        denoise_prob: float = 0.05,
    ):
        self.data_dir = Path(data_dir)
        self.include_train_pairs = include_train_pairs
        self.include_test_pairs = include_test_pairs
        self.include_inputs = include_inputs
        self.include_outputs = include_outputs
        self.max_tasks = max_tasks
        self.denoise_prob = denoise_prob

        self.items: List[ARCGridItem] = []
        self._load_items()

    def _load_items(self) -> None:
        json_files = sorted(self.data_dir.glob("*.json"))
        if self.max_tasks is not None:
            json_files = json_files[: self.max_tasks]

        for path in json_files:
            task_id = path.stem
            with open(path, "r", encoding="utf-8") as f:
                task = json.load(f)

            for split_name in ["train", "test"]:
                if split_name == "train" and not self.include_train_pairs:
                    continue
                if split_name == "test" and not self.include_test_pairs:
                    continue

                pairs = task.get(split_name, [])
                for pair_index, pair in enumerate(pairs):
                    if self.include_inputs and "input" in pair:
                        self.items.append(
                            ARCGridItem(
                                task_id=task_id,
                                split=split_name,
                                pair_index=pair_index,
                                side="input",
                                grid=pair["input"],
                            )
                        )
                    if self.include_outputs and "output" in pair:
                        self.items.append(
                            ARCGridItem(
                                task_id=task_id,
                                split=split_name,
                                pair_index=pair_index,
                                side="output",
                                grid=pair["output"],
                            )
                        )

        if len(self.items) == 0:
            raise ValueError(f"No ARC grids found in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        padded, mask = pad_grid(item.grid)

        # Target is the clean grid.
        target = padded.clone()

        # Input can be slightly corrupted for denoising.
        noisy = corrupt_grid(padded, mask, prob=self.denoise_prob)

        return {
            "x": noisy,                  # [30, 30], values 0-10
            "target": target,            # [30, 30], values 0-10
            "mask": mask,                # [30, 30], True for real cells
            "task_id": item.task_id,
            "split": item.split,
            "pair_index": item.pair_index,
            "side": item.side,
        }


def pad_grid(grid: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads an ARC grid to [30, 30].

    Returns:
        padded: LongTensor [30, 30], color ids 0-9, PAD_ID elsewhere
        mask: BoolTensor [30, 30], True where original grid exists
    """
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0

    if h > MAX_H or w > MAX_W:
        raise ValueError(f"ARC grid too large: {h}x{w}")

    padded = torch.full((MAX_H, MAX_W), PAD_ID, dtype=torch.long)
    mask = torch.zeros((MAX_H, MAX_W), dtype=torch.bool)

    for r in range(h):
        for c in range(w):
            value = int(grid[r][c])
            if not (0 <= value <= 9):
                raise ValueError(f"Invalid ARC color value: {value}")
            padded[r, c] = value
            mask[r, c] = True

    return padded, mask


def corrupt_grid(x: torch.Tensor, mask: torch.Tensor, prob: float = 0.05) -> torch.Tensor:
    """
    Randomly corrupts real cells, not padding cells.

    This makes the model a denoising autoencoder:
        corrupted grid -> clean grid

    Corruption strategy:
        With probability prob, replace a real cell with a random ARC color 0-9.
    """
    if prob <= 0:
        return x.clone()

    x_noisy = x.clone()
    random_mask = (torch.rand_like(x_noisy.float()) < prob) & mask
    random_colors = torch.randint(low=0, high=NUM_COLORS, size=x_noisy.shape)
    x_noisy[random_mask] = random_colors[random_mask]
    return x_noisy


def collate_arc(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "x": torch.stack([b["x"] for b in batch], dim=0),
        "target": torch.stack([b["target"] for b in batch], dim=0),
        "mask": torch.stack([b["mask"] for b in batch], dim=0),
        "task_id": [b["task_id"] for b in batch],
        "split": [b["split"] for b in batch],
        "pair_index": [b["pair_index"] for b in batch],
        "side": [b["side"] for b in batch],
    }


# -----------------------------
# Model
# -----------------------------

class ARCGridAutoEncoder(nn.Module):
    def __init__(self, emb_dim: int = 32, latent_dim: int = 128):
        super().__init__()
        self.emb_dim = emb_dim
        self.latent_dim = latent_dim

        self.color_embedding = nn.Embedding(VOCAB_SIZE, emb_dim, padding_idx=PAD_ID)
        C = [64, 128]
        # Encoder: [B, emb_dim, 30, 30] -> [B, latent_dim]
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(emb_dim, C[0], kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d( C[0], C[0], kernel_size=3, stride=2, padding=1),   # 30 -> 15
            nn.GELU(),
            nn.Conv2d(C[0], C[1], kernel_size=3, stride=2, padding=1),  # 15 -> 8
            nn.GELU(),
            nn.Conv2d(C[1], C[1], kernel_size=3, stride=2, padding=1), # 8 -> 4
            nn.GELU(),
        )

        self.encoder_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.GELU(),
            nn.Linear(512, latent_dim),
        )

        # Decoder: [B, latent_dim] -> [B, 10, 30, 30]
        self.decoder_mlp = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.GELU(),
            nn.Linear(512, 128 * 4 * 4),
            nn.GELU(),
        )

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(C[1], C[1], kernel_size=4, stride=2, padding=1), # 4 -> 8
            nn.GELU(),
            nn.ConvTranspose2d(C[1], C[0], kernel_size=3, stride=2, padding=1, output_padding=0), # 8 -> 15
            nn.GELU(),
            nn.ConvTranspose2d(C[0], C[0], kernel_size=4, stride=2, padding=1), # 15 -> 30
            nn.GELU(),
            nn.Conv2d(C[0], NUM_COLORS, kernel_size=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: LongTensor [B, 30, 30], values 0-10

        Returns:
            z: FloatTensor [B, latent_dim]
        """
        # [B, 30, 30, emb_dim]
        x_emb = self.color_embedding(x)

        # [B, emb_dim, 30, 30]
        x_emb = x_emb.permute(0, 3, 1, 2).contiguous()

        h = self.encoder_cnn(x_emb)
        z = self.encoder_mlp(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: FloatTensor [B, latent_dim]

        Returns:
            logits: FloatTensor [B, 10, 30, 30]
        """
        h = self.decoder_mlp(z)
        h = h.view(z.size(0), 128, 4, 4)
        logits = self.decoder_cnn(h)
        return logits

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encode(x)
        logits = self.decode(z)
        return {"z": z, "logits": logits}


# -----------------------------
# Training / evaluation
# -----------------------------

def reconstruction_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy over real ARC cells only.

    Args:
        logits: [B, 10, 30, 30]
        target: [B, 30, 30], values 0-10
        mask: [B, 30, 30], True for real cells
    """
    # CrossEntropyLoss wants target [B, H, W] and logits [B, C, H, W].
    per_cell = F.cross_entropy(logits, target.clamp(0, 9), reduction="none")
    loss = per_cell[mask].mean()
    return loss


@torch.no_grad()
def reconstruction_accuracy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)  # [B, 30, 30]
    correct = (pred == target) & mask
    total = mask.sum().item()
    if total == 0:
        return 0.0
    return correct.sum().item() / total


def train_one_epoch(model, loader, optimizer, device) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0
    total_real_cells = 0
    total_pred_real_cells = 0
    total_grad_norm = 0.0
    total_z_norm = 0.0
    total_z_std = 0.0
    total_color_entropy = 0.0

    for batch in loader:
        x = batch["x"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)

        out = model(x)
        logits = out["logits"]
        z = out["z"]

        loss = reconstruction_loss(logits, target, mask)
        acc = reconstruction_accuracy(logits, target, mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            pred_real_cells = pred[mask]
            color_counts = torch.bincount(pred_real_cells, minlength=NUM_COLORS).float()
            color_probs = color_counts / color_counts.sum().clamp_min(1.0)
            color_entropy = -(color_probs * (color_probs + 1e-8).log()).sum().item()

            total_real_cells += int(mask.sum().item())
            total_pred_real_cells += int(pred_real_cells.numel())
            total_z_norm += z.norm(dim=1).mean().item()
            total_z_std += z.std(dim=0).mean().item()
            total_color_entropy += color_entropy
            total_grad_norm += float(grad_norm)

        total_loss += loss.item()
        total_acc += acc
        total_batches += 1

    denom = max(total_batches, 1)
    return {
        "loss": total_loss / denom,
        "acc": total_acc / denom,
        "grad_norm": total_grad_norm / denom,
        "z_norm": total_z_norm / denom,
        "z_std": total_z_std / denom,
        "pred_color_entropy": total_color_entropy / denom,
        "real_cells_per_batch": total_real_cells / denom,
    }


@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0
    total_z_norm = 0.0
    total_z_std = 0.0
    total_color_entropy = 0.0
    total_confidence = 0.0

    for batch in loader:
        x = batch["x"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)

        out = model(x)
        logits = out["logits"]
        z = out["z"]

        loss = reconstruction_loss(logits, target, mask)
        acc = reconstruction_accuracy(logits, target, mask)

        probs = logits.softmax(dim=1)
        confidence = probs.max(dim=1).values[mask].mean().item()

        pred = logits.argmax(dim=1)
        pred_real_cells = pred[mask]
        color_counts = torch.bincount(pred_real_cells, minlength=NUM_COLORS).float()
        color_probs = color_counts / color_counts.sum().clamp_min(1.0)
        color_entropy = -(color_probs * (color_probs + 1e-8).log()).sum().item()

        total_loss += loss.item()
        total_acc += acc
        total_z_norm += z.norm(dim=1).mean().item()
        total_z_std += z.std(dim=0).mean().item()
        total_color_entropy += color_entropy
        total_confidence += confidence
        total_batches += 1

    denom = max(total_batches, 1)
    return {
        "loss": total_loss / denom,
        "acc": total_acc / denom,
        "z_norm": total_z_norm / denom,
        "z_std": total_z_std / denom,
        "pred_color_entropy": total_color_entropy / denom,
        "confidence": total_confidence / denom,
    }


@torch.no_grad()
def extract_embeddings(model, dataset, device, out_path: str | Path, batch_size: int = 128) -> None:
    """
    Saves one embedding per grid item.

    Output format:
        torch.save({
            "embeddings": Tensor [N, latent_dim],
            "metadata": List[dict],
        }, out_path)
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_arc)
    model.eval()

    all_z = []
    metadata = []

    for batch in loader:
        x = batch["target"].to(device)  # use clean grids for exported embeddings
        out = model(x)
        z = out["z"].detach().cpu()
        all_z.append(z)

        for task_id, split, pair_index, side in zip(
            batch["task_id"], batch["split"], batch["pair_index"], batch["side"]
        ):
            metadata.append({
                "task_id": task_id,
                "split": split,
                "pair_index": int(pair_index),
                "side": side,
            })

    embeddings = torch.cat(all_z, dim=0)
    payload = {
        "embeddings": embeddings,
        "metadata": metadata,
    }
    torch.save(payload, out_path)
    print(f"Saved embeddings to {out_path}")
    print(f"Embedding tensor shape: {tuple(embeddings.shape)}")


def save_checkpoint(model, optimizer, args, path: str | Path, epoch: int, val_metrics: Dict[str, float]) -> None:
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_metrics": val_metrics,
        "args": vars(args),
        "constants": {
            "MAX_H": MAX_H,
            "MAX_W": MAX_W,
            "NUM_COLORS": NUM_COLORS,
            "PAD_ID": PAD_ID,
            "VOCAB_SIZE": VOCAB_SIZE,
        },
    }
    torch.save(ckpt, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/training")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--emb_dim", type=int, default=32)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--denoise_prob", type=float, default=0.05)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="arc_grid_ae.pt")
    parser.add_argument("--embedding_out", type=str, default="arc_grid_embeddings.pt")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    full_dataset = ARCGridDataset(
        data_dir=args.data_dir,
        include_train_pairs=True,
        include_test_pairs=True,
        include_inputs=True,
        include_outputs=True,
        max_tasks=args.max_tasks,
        denoise_prob=args.denoise_prob,
    )

    print(f"Loaded {len(full_dataset)} grid items")

    val_size = max(1, int(len(full_dataset) * args.val_fraction))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_arc,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_arc,
        num_workers=0,
    )

    model = ARCGridAutoEncoder(
        emb_dim=args.emb_dim,
        latent_dim=args.latent_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        epoch_seconds = time.time() - epoch_start_time
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"[{timestamp}] "
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"time {epoch_seconds:.1f}s | "
            f"lr {lr:.2e} | "
            f"train loss {train_metrics['loss']:.4f} | "
            f"train acc {train_metrics['acc']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} | "
            f"val acc {val_metrics['acc']:.4f} | "
            f"val conf {val_metrics['confidence']:.4f} | "
            f"grad {train_metrics['grad_norm']:.3f} | "
            f"z_norm {val_metrics['z_norm']:.3f} | "
            f"z_std {val_metrics['z_std']:.3f} | "
            f"color_H {val_metrics['pred_color_entropy']:.3f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(model, optimizer, args, args.save_path, epoch, val_metrics)
            print(f"  saved best checkpoint to {args.save_path}")

    # Reload best checkpoint before extracting embeddings.
    ckpt = torch.load(args.save_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    extract_embeddings(
        model=model,
        dataset=full_dataset,
        device=device,
        out_path=args.embedding_out,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

