import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1.  GRAPH DECODER  (hybrid v2)
# ─────────────────────────────────────────────────────────────────────────────

class GraphDecoder(nn.Module):
    """
    Hybrid graph decoder v2.

    v2 changes vs graph_decoder_hybrid.py:
        1. Added existence_head — predicts whether each node slot is a real
           object or padding. Equivalent to the fix in graph_decoder3.py.
        2. Added bbox_head — predicts (min_r, min_c, max_r, max_c) per node,
           normalised by 30.0. This enables graph_to_grid_from_predictions()
           in hybrid_object2.py to place shapes correctly at inference time.

    Decoder outputs:
        color_logits      (B, N, 10)       — node color, CE loss
        node_shape        (B, N, 100)      — flattened 10×10 mask, MSE/BCE loss
        edge_feats        (B, N, N, 5)     — reassembled edge features
        edge_binary       (B, N, N, 3)     — binary edge logits
        existence_logits  (B, N)           — node existence, BCE loss   ← v2
        bbox              (B, N, 4)        — normalised bbox coords      ← v2
    """

    def __init__(
        self,
        max_nodes: int,
        latent_dim: int     = 256,
        num_colors: int     = 10,
        node_shape_dim: int = 100,
        edge_out_dim: int   = 5,
        hidden_dim: int     = 256,
    ):
        super().__init__()
        self.max_nodes      = max_nodes
        self.num_colors     = num_colors
        self.node_shape_dim = node_shape_dim
        self.edge_out_dim   = edge_out_dim

        # Shared trunk: z → rich hidden representation
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),   nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim*2), nn.SiLU(),
            nn.Linear(hidden_dim*2, hidden_dim*2), nn.SiLU(),
        )
        trunk_out = hidden_dim * 2

        # ── v2: existence head ────────────────────────────────────────────
        self.existence_head = nn.Sequential(
            nn.Linear(trunk_out, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, max_nodes),
        )

        # ── color head ────────────────────────────────────────────────────
        self.color_head = nn.Sequential(
            nn.Linear(trunk_out, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, max_nodes * num_colors),
        )

        # ── shape head (flattened 10×10 mask) ────────────────────────────
        self.shape_head = nn.Sequential(
            nn.Linear(trunk_out, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, max_nodes * node_shape_dim),
        )

        # ── v2: bbox head (min_r, min_c, max_r, max_c) normalised ─────────
        self.bbox_head = nn.Sequential(
            nn.Linear(trunk_out, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, max_nodes * 4),
        )

        # ── edge heads ────────────────────────────────────────────────────
        self.edge_cont_dim   = 2   # dx, dy
        self.edge_binary_dim = 3   # touching, same_row, same_col

        self.edge_cont_head = nn.Sequential(
            nn.Linear(trunk_out, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, max_nodes * max_nodes * self.edge_cont_dim),
        )
        self.edge_binary_head = nn.Sequential(
            nn.Linear(trunk_out, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, max_nodes * max_nodes * self.edge_binary_dim),
        )

    def forward(self, z):
        B = z.size(0)
        N = self.max_nodes
        h = self.trunk(z)                                       # (B, trunk_out)

        # ── v2: existence ─────────────────────────────────────────────────
        existence_logits = self.existence_head(h)               # (B, N)

        # ── node heads ────────────────────────────────────────────────────
        color_logits = self.color_head(h).view(B, N, self.num_colors)
        node_shape   = self.shape_head(h).view(B, N, self.node_shape_dim)

        # ── v2: bbox ──────────────────────────────────────────────────────
        bbox = self.bbox_head(h).view(B, N, 4)                  # (B, N, 4)

        # ── edge heads ────────────────────────────────────────────────────
        edge_cont   = self.edge_cont_head(h).view(B, N, N, self.edge_cont_dim)
        edge_binary = self.edge_binary_head(h).view(B, N, N, self.edge_binary_dim)

        # Reassemble: [dx, dy, touching, same_row, same_col]
        edge_feats = torch.cat([edge_cont, edge_binary], dim=-1)  # (B, N, N, 5)

        return (
            color_logits,     # (B, N, 10)
            node_shape,       # (B, N, 100)
            edge_feats,       # (B, N, N, 5)
            edge_binary,      # (B, N, N, 3)  logits
            existence_logits, # (B, N)         ← v2
            bbox,             # (B, N, 4)      ← v2
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  LOSS FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def gat_vae_loss(
    out: dict,
    target_color,         # (B, N)          long   node color class
    target_node_shape,    # (B, N, 100)     float  flattened shape mask
    target_edge_cont,     # (B, N, N, 2)    float  dx, dy (normalised)
    target_edge_binary,   # (B, N, N, 3)    float  touching, same_row, same_col
    target_existence,     # (B, N)          float  1=real node, 0=pad   ← v2
    target_bbox,          # (B, N, 4)       float  normalised bbox       ← v2
    kl_weight: float          = 1e-3,
    color_weight: float       = 1.0,
    shape_weight: float       = 1.0,
    edge_cont_weight: float   = 0.1,   # v2: reduced — was 1.0, caused NaN
    edge_bin_weight: float    = 1.0,
    existence_weight: float   = 1.0,   # v2
    bbox_weight: float        = 1.0,   # v2
):
    """
    v2 changes vs graph_decoder_hybrid.py:
        1. All node/edge losses are masked by target_existence so padded
           slots don't contribute (same fix as graph_decoder3.py).
        2. Added existence BCE loss (unmasked — model must learn padding).
        3. Added bbox MSE loss (masked to real nodes only).
        4. edge_cont_weight reduced to 0.1 — dx/dy MSE was the primary
           source of NaN explosions on larger grids.
    """
    B, N, C = out["color_logits"].shape

    # ── existence BCE (all slots — model must learn which are real) ───────
    existence_loss = F.binary_cross_entropy_with_logits(
        out["existence_logits"], target_existence,
    )

    # ── node mask ─────────────────────────────────────────────────────────
    node_mask    = target_existence.view(B * N).bool()   # (B*N,)
    node_mask_2d = target_existence.bool()               # (B, N)

    # ── color CE (real nodes only) ────────────────────────────────────────
    color_logits_flat = out["color_logits"].view(B*N, C)
    target_color_flat = target_color.view(B*N)
    if node_mask.any():
        color_loss = F.cross_entropy(color_logits_flat[node_mask],
                                     target_color_flat[node_mask])
    else:
        color_loss = color_logits_flat.sum() * 0.0

    # ── shape MSE (real nodes only) ───────────────────────────────────────
    shape_flat        = out["node_shape"].view(B*N, -1)
    target_shape_flat = target_node_shape.view(B*N, -1)
    if node_mask.any():
        shape_loss = F.mse_loss(shape_flat[node_mask], target_shape_flat[node_mask])
    else:
        shape_loss = shape_flat.sum() * 0.0

    # ── v2: bbox MSE (real nodes only) ────────────────────────────────────
    bbox_flat        = out["bbox"].view(B*N, 4)
    target_bbox_flat = target_bbox.view(B*N, 4)
    if node_mask.any():
        bbox_loss = F.mse_loss(bbox_flat[node_mask], target_bbox_flat[node_mask])
    else:
        bbox_loss = bbox_flat.sum() * 0.0

    # ── edge mask (both endpoints real) ──────────────────────────────────
    edge_mask = node_mask_2d.unsqueeze(2) & node_mask_2d.unsqueeze(1)  # (B,N,N)

    # ── edge continuous MSE (real edges only) ────────────────────────────
    pred_edge_cont = out["edge_feats"][..., :2]           # dx, dy
    if edge_mask.any():
        edge_cont_loss = F.mse_loss(pred_edge_cont[edge_mask],
                                    target_edge_cont[edge_mask])
    else:
        edge_cont_loss = pred_edge_cont.sum() * 0.0

    # ── edge binary BCE (real edges only) ────────────────────────────────
    edge_mask_exp = edge_mask.unsqueeze(-1).expand_as(out["edge_binary_logits"])
    if edge_mask_exp.any():
        edge_bin_loss = F.binary_cross_entropy_with_logits(
            out["edge_binary_logits"][edge_mask_exp],
            target_edge_binary[edge_mask_exp],
        )
    else:
        edge_bin_loss = out["edge_binary_logits"].sum() * 0.0

    # ── KL divergence ─────────────────────────────────────────────────────
    kl_loss = -0.5 * torch.mean(
        1 + out["log_var"] - out["mu"].pow(2) - out["log_var"].exp()
    )

    total = (
        existence_weight * existence_loss
      + color_weight     * color_loss
      + shape_weight     * shape_loss
      + bbox_weight      * bbox_loss
      + edge_cont_weight * edge_cont_loss
      + edge_bin_weight  * edge_bin_loss
      + kl_weight        * kl_loss
    )

    return total, {
        "existence_loss" : existence_loss.item(),
        "color_loss"     : color_loss.item(),
        "shape_loss"     : shape_loss.item(),
        "bbox_loss"      : bbox_loss.item(),
        "edge_cont_loss" : edge_cont_loss.item(),
        "edge_bin_loss"  : edge_bin_loss.item(),
        "kl_loss"        : kl_loss.item(),
    }
