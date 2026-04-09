import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1.  GRAPH DECODER
# ─────────────────────────────────────────────────────────────────────────────

class GraphDecoder(nn.Module):

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
        self.trunk = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.SiLU(), 
                                   nn.Linear(hidden_dim, hidden_dim * 2), nn.SiLU(),
                                   nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.SiLU())
        trunk_out = hidden_dim * 2

        # Color head: → (max_nodes, num_colors) logits [no activation — CE handles it]
        self.color_head = nn.Sequential(nn.Linear(trunk_out, hidden_dim), nn.SiLU(), 
                                        nn.Linear(hidden_dim, max_nodes * num_colors))

        # Geometry head: → (max_nodes, node_geom_dim) continuous values
        self.geom_head = nn.Sequential(nn.Linear(trunk_out, hidden_dim), nn.SiLU(),
                                       nn.Linear(hidden_dim, max_nodes * node_geom_dim))

        # Edge feature heads — split by type:
        #   continuous : dx, dy, manhattan_dist, euclidean_dist, area_ratio_ab, area_ratio_ba
        #   binary     : same_color, touching, bbox_overlap, same_row, same_col, same_area
        self.edge_cont_dim   = 6   # indices [0,1,2,3,10,11]
        self.edge_binary_dim = 6   # indices [4,5,6,7,8,9]

        self.edge_cont_head = nn.Sequential(nn.Linear(trunk_out, hidden_dim), nn.SiLU(),
                                            nn.Linear(hidden_dim, max_nodes * max_nodes * self.edge_cont_dim))

        # Binary head outputs logits — sigmoid + BCE applied in the loss
        self.edge_binary_head = nn.Sequential(nn.Linear(trunk_out, hidden_dim), nn.SiLU(), 
                                              nn.Linear(hidden_dim, max_nodes * max_nodes * self.edge_binary_dim))

    def forward(self, z):
        B = z.size(0)
        h = self.trunk(z)                                           # (B, trunk_out)

        color_logits = self.color_head(h).view(B, self.max_nodes, self.num_colors)
        node_geom    = self.geom_head(h).view(B, self.max_nodes, self.node_geom_dim)

        # Edge continuous: dims [0,1,2,3] + [10,11]
        edge_cont   = self.edge_cont_head(h).view(B, self.max_nodes, self.max_nodes, self.edge_cont_dim)
        # Edge binary logits: dims [4,5,6,7,8,9]
        edge_binary = self.edge_binary_head(h).view(B, self.max_nodes, self.max_nodes, self.edge_binary_dim)

        # Reassemble into original 12-dim ordering for easy comparison with input:
        # [dx, dy, manhattan, euclidean, same_color, touching, bbox_overlap,
        #  same_row, same_col, same_area, area_ratio_ab, area_ratio_ba]
        edge_feats = torch.cat([
            edge_cont[..., :4],    # dx, dy, manhattan, euclidean
            edge_binary,           # same_color … same_area  (logits)
            edge_cont[..., 4:],    # area_ratio_ab, area_ratio_ba
        ], dim=-1)                                                  # (B, N, N, 12)

        return color_logits, node_geom, edge_feats, edge_binary


# ─────────────────────────────────────────────────────────────────────────────
# 2.  LOSS FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def gat_vae_loss(
    out: dict,
    target_color,           # (B, max_nodes)                long   — class indices 0–9
    target_node_geom,       # (B, max_nodes, 12)            float  — continuous geometry
    target_edge_cont,       # (B, max_nodes, max_nodes, 6)  float  — dx,dy,manhattan,euclidean,ratio_ab,ratio_ba
    target_edge_binary,     # (B, max_nodes, max_nodes, 6)  float  — 0/1 flags
    kl_weight: float        = 1e-3,
    color_weight: float     = 1.0,
    geom_weight: float      = 1.0,
    edge_cont_weight: float = 1.0,
    edge_bin_weight: float  = 1.0,
):
    B, N, C = out["color_logits"].shape

    # ── color: cross-entropy ────────────────────────────────────────────────
    color_loss = F.cross_entropy(
        out["color_logits"].view(B * N, C),
        target_color.view(B * N),
    )

    # ── continuous geometry: MSE ─────────────────────────────────────────────
    geom_loss = F.mse_loss(out["node_geom"], target_node_geom)

    # ── edge continuous: MSE ─────────────────────────────────────────────────
    pred_edge_cont = torch.cat([
        out["edge_feats"][..., :4],
        out["edge_feats"][..., 10:],
    ], dim=-1)
    edge_cont_loss = F.mse_loss(pred_edge_cont, target_edge_cont)

    # ── edge binary: BCE with logits ─────────────────────────────────────────
    edge_bin_loss = F.binary_cross_entropy_with_logits(
        out["edge_binary_logits"], target_edge_binary
    )

    # ── KL divergence ────────────────────────────────────────────────────────
    kl_loss = -0.5 * torch.mean(
        1 + out["log_var"] - out["mu"].pow(2) - out["log_var"].exp()
    )

    total = (
        color_weight     * color_loss
      + geom_weight      * geom_loss
      + edge_cont_weight * edge_cont_loss
      + edge_bin_weight  * edge_bin_loss
      + kl_weight        * kl_loss
    )

    return total, {
        "color_loss"     : color_loss.item(),
        "geom_loss"      : geom_loss.item(),
        "edge_cont_loss" : edge_cont_loss.item(),
        "edge_bin_loss"  : edge_bin_loss.item(),
        "kl_loss"        : kl_loss.item(),
    }