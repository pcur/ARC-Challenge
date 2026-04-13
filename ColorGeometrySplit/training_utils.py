"""
training_utils.py
=================
Shared utilities for all VAE training loops.

  GPUMemoryGuard  — monitors VRAM usage, warns if over threshold
  EarlyStopping   — stops training when val loss diverges
"""

import torch


class GPUMemoryGuard:
    """
    Monitors GPU VRAM and warns if usage exceeds max_fraction.
    A 3090 has 24GB VRAM — 80% cap = 19.2GB.
    Does not crash training, just logs a warning so batch_size can be reduced.
    """

    def __init__(self, device, max_fraction: float = 0.80):
        self.device       = device
        self.max_fraction = max_fraction
        self.total        = torch.cuda.get_device_properties(device).total_memory
        self.warned       = False

    def check(self):
        used     = torch.cuda.memory_reserved(self.device)
        fraction = used / self.total
        if fraction > self.max_fraction and not self.warned:
            gb_used  = used  / 1e9
            gb_total = self.total / 1e9
            print(f"\n  [VRAM WARNING] {gb_used:.1f}/{gb_total:.1f} GB "
                  f"({fraction*100:.1f}%) — consider reducing batch_size")
            self.warned = True

    def summary_str(self) -> str:
        """Short string for the training log, e.g. '62%'."""
        used = torch.cuda.memory_reserved(self.device)
        pct  = used / self.total * 100
        return f"{pct:.0f}%"


class EarlyStopping:
    """
    Stops training when val loss has not improved for `patience` consecutive
    epochs AND the current val loss exceeds best * divergence_threshold.

    The threshold prevents premature stopping on short noisy plateaus — it
    only fires when the model is genuinely diverging, not just flat.

    Parameters
    ----------
    patience             : epochs without improvement before checking divergence
    divergence_threshold : val loss must exceed best * this to trigger stop
                           1.05 = 5% worse than best triggers stop
    """

    def __init__(self, patience: int = 20, divergence_threshold: float = 1.00):
        self.patience             = patience
        self.divergence_threshold = divergence_threshold
        self.best_val_loss        = float("inf")
        self.counter              = 0

    def step(self, val_loss: float) -> bool:
        """Call once per epoch. Returns True if training should stop."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.counter       = 0
            return False

        self.counter += 1
        diverging = val_loss > self.best_val_loss * self.divergence_threshold
        return self.counter >= self.patience and diverging
