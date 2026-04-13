"""
spatial_geom_ae.py
==================
SpatialGeomAE — object-level graph autoencoder that maintains per-node
embeddings throughout. No global bottleneck.

Same principle as SpatialColorAE — no global pooling, no single latent
vector. Each node gets its own 128-dim embedding from GATv2 message
passing, and predicts its own geometry features directly from that embedding.

Encoder input
-------------
Node features (geometry only, color zeroed):
  [0:12]  geometry — area, centroid_row, centroid_col, bbox×4,
                     width, height, density, aspect_ratio, is_single_pixel

Edge features (spatial/geometric only):
  [0:4]   dx, dy, manhattan, euclidean
  [4:7]   same_row, same_col, same_area   (binary spatial flags)
  [7:9]   area_ratio_ab, area_ratio_ba
  Total: 9 dims  (color-related edge dims excluded)

Decoder output (per node)
--------------------------
  node_geom    : (N, 12)  continuous geometry — MSE loss
  edge_spatial : predicted via pairwise MLP on node embedding pairs

Architecture
------------
  SpatialGeomEncoder  : GATv2 layers → (N, hidden_dim) per-node embeddings
  GeomRegressionHead  : per-node MLP → (N, 12) geometry predictions
  SpatialGeomAE       : encoder + regression head

Loss
----
  MSE on geometry features (normalised scale)
  No KL, no bottleneck, no reconstruction expansion.

Dependencies:
    pip install torch torch_geometric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


# ─────────────────────────────────────────────────────────────────────────────
# SINUSOIDAL POSITIONAL ENCODING
# ─────────────────────────────────────────────────────────────────────────────

def sinusoidal_position_encoding(coords, d_model: int = 32):
    """
    Encode a set of 1D coordinates using sinusoidal functions at multiple
    frequencies. Same principle as transformer positional encodings.

    coords  : (N,) float tensor — normalised coordinate values in [0, 1]
    d_model : number of encoding dimensions (must be even)

    Returns : (N, d_model) — sinusoidal encoding

    At each frequency i, we compute:
        sin(coords * 2π * freq_i)
        cos(coords * 2π * freq_i)
    Frequencies are log-spaced from 1.0 to max_freq, giving the model
    sensitivity at both coarse and fine spatial scales.
    """
    assert d_model % 2 == 0
    device     = coords.device
    half       = d_model // 2
    max_freq   = 32.0   # covers up to 32 cycles across the grid
    freqs      = torch.exp(
        torch.linspace(0, torch.log(torch.tensor(max_freq)), half, device=device)
    )                                              # (half,)
    angles     = coords.unsqueeze(-1) * freqs * 2 * 3.141592653589793  # (N, half)
    return torch.cat([angles.sin(), angles.cos()], dim=-1)             # (N, d_model)


# ─────────────────────────────────────────────────────────────────────────────
# SPATIAL GEOM ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class SpatialGeomEncoder(nn.Module):
    """
    GATv2 encoder for object-level geometry graphs.
    Returns per-node embeddings (N, hidden_dim) — no global pooling.

    Node input  : geometry features (12 dims) + sinusoidal position encoding
                  of centroid_row and centroid_col (2 × pos_enc_dim dims)
    Edge input  : spatial/geometric edge features (9 dims)

    Sinusoidal positional encoding
    ------------------------------
    Centroid row and col are encoded at multiple frequencies before being
    concatenated to the node features. This gives the model strong, non-
    learnable spatial anchors at both coarse and fine scales that persist
    through message passing — unlike the raw centroid values which get
    mixed with relational features across layers and lose absolute grounding.
    """

    def __init__(
        self,
        node_in_dim: int = 12,
        edge_in_dim: int = 9,
        hidden_dim: int  = 128,
        num_heads: int   = 4,
        num_layers: int  = 4,
        dropout: float   = 0.1,
        pos_enc_dim: int = 16,   # sinusoidal dims per coordinate (row + col)
    ):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.dropout     = dropout
        self.pos_enc_dim = pos_enc_dim

        self.edge_proj = nn.Linear(edge_in_dim, hidden_dim)

        # first GAT layer takes geometry + positional encoding
        # centroid_row is dim 1, centroid_col is dim 2 (of the 12 geom dims)
        # we append 2 * pos_enc_dim sinusoidal dims to the node features
        augmented_node_dim = node_in_dim + 2 * pos_enc_dim

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = augmented_node_dim
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
        x          : (N, 12)  normalised geometry node features
                     centroid_row = x[:, 1], centroid_col = x[:, 2]
        edge_index : (2, E)
        edge_attr  : (E, 9)   normalised spatial edge features

        Returns
        -------
        h : (N, hidden_dim)  per-node spatial embeddings
        """
        # ── sinusoidal positional encoding ───────────────────────────────────
        # centroid_row and centroid_col are already normalised (from norm stats)
        # we re-normalise to [0,1] range using sigmoid for the encoding input
        # so the sinusoidal functions cover the full [0,1] period cleanly
        row_enc = sinusoidal_position_encoding(
            torch.sigmoid(x[:, 1]), self.pos_enc_dim
        )   # (N, pos_enc_dim)
        col_enc = sinusoidal_position_encoding(
            torch.sigmoid(x[:, 2]), self.pos_enc_dim
        )   # (N, pos_enc_dim)

        # concatenate positional encodings to node features
        h = torch.cat([x, row_enc, col_enc], dim=-1)   # (N, 12 + 2*pos_enc_dim)

        # ── GAT message passing ───────────────────────────────────────────────
        if edge_attr is not None and edge_attr.numel() > 0 and edge_attr.dim() >= 2:
            e = F.silu(self.edge_proj(edge_attr))
        else:
            e = None

        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index, edge_attr=e)
            h_new = norm(h_new)
            h_new = F.silu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new if h.shape == h_new.shape else h_new

        return h   # (N, hidden_dim)


# ─────────────────────────────────────────────────────────────────────────────
# PER-NODE GEOMETRY REGRESSION HEAD
# ─────────────────────────────────────────────────────────────────────────────

class GeomRegressionHead(nn.Module):
    """Per-node geometry prediction: (N, hidden_dim) → (N, node_geom_dim)."""

    def __init__(self, hidden_dim: int = 128, node_geom_dim: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_geom_dim),
        )

    def forward(self, h):
        return self.net(h)   # (N, 12)


# ─────────────────────────────────────────────────────────────────────────────
# PAIRWISE SPATIAL EDGE HEAD
# Predicts same_row / same_col / same_area for each pair of nodes.
# Only evaluated for edges that exist in the graph — not all pairs.
# ─────────────────────────────────────────────────────────────────────────────

class EdgeSpatialHead(nn.Module):
    """
    Predicts binary spatial edge flags for existing edges.
    Input: [h_src ‖ h_dst] → (E, 3) logits for same_row, same_col, same_area
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),   # same_row, same_col, same_area logits
        )

    def forward(self, h, edge_index):
        """
        h          : (N, hidden_dim)
        edge_index : (2, E)
        Returns    : (E, 3) logits
        """
        src, dst  = edge_index[0], edge_index[1]
        edge_feat = torch.cat([h[src], h[dst]], dim=-1)   # (E, 2*hidden_dim)
        return self.net(edge_feat)


# ─────────────────────────────────────────────────────────────────────────────
# SPATIAL GEOM AUTOENCODER
# ─────────────────────────────────────────────────────────────────────────────

class SpatialGeomAE(nn.Module):
    """
    Spatial object-level geometry autoencoder.

    No global bottleneck — per-node embeddings preserved throughout.
    Each node predicts its own geometry from its local neighbourhood context.

    Outputs (flat tensors — no padding)
    ------------------------------------
    node_geom    : (N_total, 12)   predicted geometry features
    edge_spatial : (E_total, 3)    predicted spatial edge flags (logits)
    h            : (N_total, hidden_dim)  node embeddings for transform model
    """

    def __init__(
        self,
        node_in_dim: int   = 12,
        edge_in_dim: int   = 9,
        hidden_dim: int    = 128,
        num_heads: int     = 4,
        num_layers: int    = 4,
        dropout: float     = 0.1,
        pos_enc_dim: int   = 16,   # sinusoidal dims per coordinate
        node_geom_dim: int = 12,
    ):
        super().__init__()

        self.encoder   = SpatialGeomEncoder(
            node_in_dim = node_in_dim,
            edge_in_dim = edge_in_dim,
            hidden_dim  = hidden_dim,
            num_heads   = num_heads,
            num_layers  = num_layers,
            dropout     = dropout,
            pos_enc_dim = pos_enc_dim,
        )
        self.geom_head = GeomRegressionHead(hidden_dim, node_geom_dim)
        self.edge_head = EdgeSpatialHead(hidden_dim)

    def encode(self, x, edge_index, edge_attr):
        """Geometry graph → per-node embeddings (N, hidden_dim)."""
        return self.encoder(x, edge_index, edge_attr)

    def decode(self, h, edge_index):
        """Per-node embeddings → geometry predictions + edge spatial flags."""
        return self.geom_head(h), self.edge_head(h, edge_index)

    def forward(self, x, edge_index, edge_attr):
        h                      = self.encode(x, edge_index, edge_attr)
        node_geom, edge_spatial = self.decode(h, edge_index)
        return {
            "node_geom"    : node_geom,     # (N_total, 12)
            "edge_spatial" : edge_spatial,  # (E_total, 3)
            "h"            : h,             # (N_total, hidden_dim)
        }


# ─────────────────────────────────────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────────────────────────────────────

def spatial_geom_loss(
    node_geom,           # (N_total, 12)  predicted
    edge_spatial,        # (E_total, 3)   predicted logits
    target_node_geom,    # (N_total, 12)  true normalised geometry
    target_edge_spatial, # (E_total, 3)   true binary flags
    geom_weight: float         = 1.0,
    edge_spatial_weight: float = 1.0,
):
    """
    Per-node geometry MSE + per-edge spatial BCE.
    Handles empty edge tensors gracefully (single-node graphs).
    """
    geom_loss = F.mse_loss(node_geom, target_node_geom)

    if edge_spatial_weight > 0 and edge_spatial.numel() > 0:
        spatial_loss = F.binary_cross_entropy_with_logits(
            edge_spatial, target_edge_spatial
        )
    else:
        spatial_loss = torch.tensor(0.0, device=node_geom.device)

    total = geom_weight * geom_loss + edge_spatial_weight * spatial_loss
    return total, {
        "geom_loss"    : geom_loss.item(),
        "spatial_loss" : spatial_loss.item(),
    }