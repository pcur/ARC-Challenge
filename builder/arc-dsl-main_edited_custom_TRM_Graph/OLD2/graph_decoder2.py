import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1.  GRAPH DECODER
# ─────────────────────────────────────────────────────────────────────────────

class GraphDecoder(nn.Module):
    """
    FIX v2: Added a node existence head.

    Previously the decoder always predicted exactly max_nodes nodes and the
    loss treated every padded slot the same as a real node. This meant the
    model was penalised for not correctly reconstructing phantom objects that
    don't exist in the graph.

    The new existence head outputs a logit per node slot. During loss
    computation, only slots marked as real (existence == 1) contribute to
    the color, geometry and edge losses. BCE loss on the existence head itself
    is always computed over all max_nodes slots.
    """

    def __init__(
        self,
        max_nodes: int,
        latent_dim: int    = 256,
        num_colors: int    = 10,
        node_geom_dim: int = 12,
        edge_out_dim: int  = 12,
        hidden_dim: int    = 256,
    ):
        super().__init__()
        self.max_nodes     = max_nodes
        self.num_colors    = num_colors
        self.node_geom_dim = node_geom_dim
        self.edge_out_dim  = edge_out_dim

        # Shared trunk: z → rich hidden representation
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2), nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.SiLU(),
        )
        trunk_out = hidden_dim * 2

        # FIX v2: Node existence head — logit per slot, sigmoid+BCE in loss.
        # This lets the model learn how many real nodes a graph has rather than
        # always predicting max_nodes objects.
        self.existence_head = nn.Sequential(
            nn.Linear(trunk_out, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, max_nodes),   # (B, max_nodes) logits
        )

        # Color head: → (max_nodes, num_colors) logits [no activation — CE handles it]
        self.color_head = nn.Sequential(
            nn.Linear(trunk_out, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, max_nodes * num_colors),
        )

        # Geometry head: → (max_nodes, node_geom_dim) continuous values
        self.geom_head = nn.Sequential(
            nn.Linear(trunk_out, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, max_nodes * node_geom_dim),
        )

        # Edge feature heads — split by type:
        #   continuous : dx, dy, manhattan_dist, euclidean_dist, area_ratio_ab, area_ratio_ba
        #   binary     : same_color, touching, bbox_overlap, same_row, same_col, same_area
        self.edge_cont_dim   = 6   # indices [0,1,2,3,10,11]
        self.edge_binary_dim = 6   # indices [4,5,6,7,8,9]

        self.edge_cont_head = nn.Sequential(
            nn.Linear(trunk_out, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, max_nodes * max_nodes * self.edge_cont_dim),
        )

        # Binary head outputs logits — sigmoid + BCE applied in the loss
        self.edge_binary_head = nn.Sequential(
            nn.Linear(trunk_out, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, max_nodes * max_nodes * self.edge_binary_dim),
        )

    def forward(self, z):
        B = z.size(0)
        h = self.trunk(z)                                                   # (B, trunk_out)

        # FIX v2: existence logits — one per node slot
        existence_logits = self.existence_head(h)                           # (B, max_nodes)

        color_logits = self.color_head(h).view(B, self.max_nodes, self.num_colors)
        node_geom    = self.geom_head(h).view(B, self.max_nodes, self.node_geom_dim)

        # Edge continuous: dims [0,1,2,3] + [10,11]
        edge_cont   = self.edge_cont_head(h).view(
            B, self.max_nodes, self.max_nodes, self.edge_cont_dim)
        # Edge binary logits: dims [4,5,6,7,8,9]
        edge_binary = self.edge_binary_head(h).view(
            B, self.max_nodes, self.max_nodes, self.edge_binary_dim)

        # Reassemble into original 12-dim ordering for easy comparison with input:
        # [dx, dy, manhattan, euclidean, same_color, touching, bbox_overlap,
        #  same_row, same_col, same_area, area_ratio_ab, area_ratio_ba]
        edge_feats = torch.cat([
            edge_cont[..., :4],    # dx, dy, manhattan, euclidean
            edge_binary,           # same_color … same_area  (logits)
            edge_cont[..., 4:],    # area_ratio_ab, area_ratio_ba
        ], dim=-1)                                                          # (B, N, N, 12)

        return color_logits, node_geom, edge_feats, edge_binary, existence_logits


# ─────────────────────────────────────────────────────────────────────────────
# 2.  LOSS FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def gat_vae_loss(
    out: dict,
    target_color,           # (B, max_nodes)                long   — class indices 0–9
    target_node_geom,       # (B, max_nodes, 12)            float  — continuous geometry
    target_edge_cont,       # (B, max_nodes, max_nodes, 6)  float  — dx,dy,manhattan,euclidean,ratio_ab,ratio_ba
    target_edge_binary,     # (B, max_nodes, max_nodes, 6)  float  — 0/1 flags
    target_existence,       # (B, max_nodes)                float  — 1.0 for real nodes, 0.0 for padding
    kl_weight: float        = 1e-3,
    color_weight: float     = 1.0,
    geom_weight: float      = 1.0,
    edge_cont_weight: float = 1.0,
    edge_bin_weight: float  = 1.0,
    existence_weight: float = 1.0,
):
    """
    FIX v2: All node-level and edge-level losses are now masked so that only
    real node slots (target_existence == 1) contribute. Padded slots are
    excluded. The existence head has its own unmasked BCE loss.

    target_existence is a float tensor of shape (B, max_nodes) where:
        1.0 = this slot corresponds to a real object in the graph
        0.0 = this slot is padding (no object)
    """
    B, N, C = out["color_logits"].shape

    # ── existence: BCE over all slots (no masking — model must learn padding) ─
    existence_loss = F.binary_cross_entropy_with_logits(
        out["existence_logits"], target_existence
    )

    # ── build flat node mask for masked losses ───────────────────────────────
    # node_mask: (B*N,) — True for real nodes
    node_mask = target_existence.view(B * N).bool()   # (B*N,)

    # ── color: cross-entropy on real nodes only ──────────────────────────────
    color_logits_flat = out["color_logits"].view(B * N, C)  # (B*N, C)
    target_color_flat = target_color.view(B * N)            # (B*N,)

    if node_mask.any():
        color_loss = F.cross_entropy(
            color_logits_flat[node_mask],
            target_color_flat[node_mask],
        )
    else:
        color_loss = color_logits_flat.sum() * 0.0  # safe zero gradient

    # ── continuous geometry: MSE on real nodes only ──────────────────────────
    node_geom_flat = out["node_geom"].view(B * N, -1)       # (B*N, geom_dim)
    target_geom_flat = target_node_geom.view(B * N, -1)     # (B*N, geom_dim)

    if node_mask.any():
        geom_loss = F.mse_loss(
            node_geom_flat[node_mask],
            target_geom_flat[node_mask],
        )
    else:
        geom_loss = node_geom_flat.sum() * 0.0

    # ── edge mask: only include edges where BOTH src and dst are real ────────
    # node_mask reshaped to (B, N) for broadcasting
    node_mask_2d = target_existence.bool()                  # (B, N)
    # edge is real if both endpoints are real: outer product per batch item
    edge_mask = node_mask_2d.unsqueeze(2) & node_mask_2d.unsqueeze(1)  # (B, N, N)

    # ── edge continuous: MSE on real edges only ──────────────────────────────
    pred_edge_cont = torch.cat([
        out["edge_feats"][..., :4],
        out["edge_feats"][..., 10:],
    ], dim=-1)                                              # (B, N, N, 6)

    if edge_mask.any():
        edge_cont_loss = F.mse_loss(
            pred_edge_cont[edge_mask],
            target_edge_cont[edge_mask],
        )
    else:
        edge_cont_loss = pred_edge_cont.sum() * 0.0

    # ── edge binary: BCE with logits on real edges only ──────────────────────
    # edge_mask is (B, N, N) — expand to (B, N, N, 6) for indexing
    edge_mask_expanded = edge_mask.unsqueeze(-1).expand_as(out["edge_binary_logits"])

    if edge_mask_expanded.any():
        edge_bin_loss = F.binary_cross_entropy_with_logits(
            out["edge_binary_logits"][edge_mask_expanded],
            target_edge_binary[edge_mask_expanded],
        )
    else:
        edge_bin_loss = out["edge_binary_logits"].sum() * 0.0

    # ── KL divergence ────────────────────────────────────────────────────────
    kl_loss = -0.5 * torch.mean(
        1 + out["log_var"] - out["mu"].pow(2) - out["log_var"].exp()
    )

    total = (
        existence_weight  * existence_loss
      + color_weight      * color_loss
      + geom_weight       * geom_loss
      + edge_cont_weight  * edge_cont_loss
      + edge_bin_weight   * edge_bin_loss
      + kl_weight         * kl_loss
    )

    return total, {
        "existence_loss" : existence_loss.item(),
        "color_loss"     : color_loss.item(),
        "geom_loss"      : geom_loss.item(),
        "edge_cont_loss" : edge_cont_loss.item(),
        "edge_bin_loss"  : edge_bin_loss.item(),
        "kl_loss"        : kl_loss.item(),
    }
