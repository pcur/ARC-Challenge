"""
spatial_color_ae.py
===================
SpatialColorAE — pixel-level graph autoencoder that maintains spatial
resolution throughout. No global bottleneck.

Key insight
-----------
Previous ColorVAE compressed 400 pixels → 512-dim vector → reconstruct 400
pixels. A 512-dim vector cannot hold the spatial arrangement of 400 pixels.

This model keeps per-node embeddings (N, hidden_dim) throughout:
  - Encoder: GATv2 message passing → (N, hidden_dim) node embeddings
  - Decoder: per-node MLP heads → color logits + fg/bg logits per pixel

No global pooling. No bottleneck. No decoder expansion trick.
Each pixel's embedding has full access to its local neighbourhood context.

For the transform model (TRM-style):
  H_in  (N, d)  →  cross-attention  →  H_out (N, d)  →  color logits
  The transform operates on the full spatial embedding set, not a single vector.
  Variable N is handled via padding masks in MultiheadAttention.

Architecture
------------
  SpatialGATEncoder : GATv2 layers → (N, hidden_dim) per-node embeddings
  ColorHead         : per-node MLP → (N, 10) color logits
  FGHead            : per-node MLP → (N, 1)  fg/bg logit
  SpatialColorAE    : encoder + both heads

Loss
----
  Weighted color CE : all real pixels, background weight=0.1
  FG/BG BCE         : all real pixels

Dependencies:
    pip install torch torch_geometric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


# ─────────────────────────────────────────────────────────────────────────────
# SPATIAL GAT ENCODER  —  no global pool, returns per-node embeddings
# ─────────────────────────────────────────────────────────────────────────────

class SpatialGATEncoder(nn.Module):
    """
    GATv2 encoder that returns per-node embeddings (N, hidden_dim).
    No global pooling — spatial resolution is fully preserved.

    Node input  : color one-hot (10) + row_norm (1) + col_norm (1) = 12 dims
    Edge input  : same_color (1) + dx (1) + dy (1) = 3 dims

    The row/col positional features are always present (not dropped) so
    the model always knows where each pixel is spatially.
    """

    def __init__(
        self,
        node_in_dim: int   = 12,
        edge_in_dim: int   = 3,
        hidden_dim: int    = 128,
        num_heads: int     = 4,
        num_layers: int    = 4,
        dropout: float     = 0.1,
        color_dropout_p: float = 0.3,  # zero color features for some nodes
    ):
        super().__init__()
        self.hidden_dim      = hidden_dim
        self.dropout         = dropout
        self.color_dropout_p = color_dropout_p

        self.edge_proj = nn.Linear(edge_in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = node_in_dim
        for _ in range(num_layers):
            conv = GATv2Conv(
                in_channels  = in_dim,
                out_channels = hidden_dim // num_heads,
                heads        = num_heads,
                edge_dim     = hidden_dim,
                dropout      = dropout,
                concat       = True,
            )
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim

    def forward(self, x, edge_index, edge_attr):
        """
        Parameters
        ----------
        x          : (N, 12)  node features — color one-hot + row/col position
        edge_index : (2, E)
        edge_attr  : (E, 3)   same_color + dx + dy

        Returns
        -------
        h : (N, hidden_dim)  per-node spatial embeddings
        """
        # color dropout — zero color dims for some nodes during training
        if self.training and self.color_dropout_p > 0:
            x = x.clone()
            mask = torch.rand(x.size(0), device=x.device) < self.color_dropout_p
            x[mask, :10] = 0.0   # zero color one-hot, keep position features

        e = F.silu(self.edge_proj(edge_attr))   # (E, hidden_dim)

        h = x
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index, edge_attr=e)
            h_new = norm(h_new)
            h_new = F.silu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new if h.shape == h_new.shape else h_new

        return h   # (N, hidden_dim)


# ─────────────────────────────────────────────────────────────────────────────
# PER-NODE HEADS
# ─────────────────────────────────────────────────────────────────────────────

class ColorHead(nn.Module):
    """Per-node color prediction: (N, hidden_dim) → (N, num_colors)."""

    def __init__(self, hidden_dim: int = 128, num_colors: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_colors),
        )

    def forward(self, h):
        return self.net(h)   # (N, 10)


class FGHead(nn.Module):
    """Per-node fg/bg prediction: (N, hidden_dim) → (N, 1) logit."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, h):
        return self.net(h)   # (N, 1)


# ─────────────────────────────────────────────────────────────────────────────
# SPATIAL COLOR AUTOENCODER
# ─────────────────────────────────────────────────────────────────────────────

class SpatialColorAE(nn.Module):
    """
    Spatial pixel-level color autoencoder.

    No global bottleneck — per-node embeddings preserved throughout.
    Each pixel can reconstruct its own color from its local neighbourhood.

    Outputs
    -------
    color_logits : (N_total, 10)   flat across all graphs in batch
    fg_logits    : (N_total, 1)    flat across all graphs in batch
    h            : (N_total, hidden_dim)  node embeddings (for transform model)

    Note: outputs are flat (not padded to max_nodes) — use batch.ptr to
    split per graph if needed.
    """

    def __init__(
        self,
        node_in_dim: int      = 12,
        edge_in_dim: int      = 3,
        hidden_dim: int       = 128,
        num_heads: int        = 4,
        num_layers: int       = 4,
        dropout: float        = 0.1,
        color_dropout_p: float = 0.3,
        num_colors: int       = 10,
    ):
        super().__init__()

        self.encoder = SpatialGATEncoder(
            node_in_dim    = node_in_dim,
            edge_in_dim    = edge_in_dim,
            hidden_dim     = hidden_dim,
            num_heads      = num_heads,
            num_layers     = num_layers,
            dropout        = dropout,
            color_dropout_p = color_dropout_p,
        )
        self.color_head = ColorHead(hidden_dim, num_colors)
        self.fg_head    = FGHead(hidden_dim)

    def encode(self, x, edge_index, edge_attr):
        """Graph → per-node embeddings (N, hidden_dim)."""
        return self.encoder(x, edge_index, edge_attr)

    def decode(self, h):
        """Per-node embeddings → color logits + fg logits."""
        return self.color_head(h), self.fg_head(h)

    def forward(self, x, edge_index, edge_attr):
        h            = self.encode(x, edge_index, edge_attr)
        color_logits, fg_logits = self.decode(h)
        return {
            "color_logits" : color_logits,   # (N_total, 10)
            "fg_logits"    : fg_logits,       # (N_total, 1)
            "h"            : h,               # (N_total, hidden_dim)
        }


# ─────────────────────────────────────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────────────────────────────────────

def spatial_color_loss(
    color_logits,           # (N_total, 10)   flat
    fg_logits,              # (N_total, 1)    flat
    target_color,           # (N_total,)      long — true color indices
    color_weight: float    = 1.0,
    fg_weight: float       = 1.0,
    bg_class_weight: float = 0.1,
):
    """
    Per-pixel reconstruction loss — flat tensors, no padding needed.

    color CE : weighted cross-entropy, background (0) weight = bg_class_weight
    fg BCE   : binary foreground/background classification
    """
    num_colors    = color_logits.size(-1)
    class_weights = torch.ones(num_colors, device=color_logits.device)
    class_weights[0] = bg_class_weight

    color_loss = F.cross_entropy(color_logits, target_color, weight=class_weights)

    target_fg = (target_color != 0).float().unsqueeze(-1)
    fg_loss   = F.binary_cross_entropy_with_logits(fg_logits, target_fg)

    total = color_weight * color_loss + fg_weight * fg_loss
    return total, {
        "color_loss" : color_loss.item(),
        "fg_loss"    : fg_loss.item(),
    }
