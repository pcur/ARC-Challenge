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
            nn.Linear(latent_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2), nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.SiLU()
        )
        trunk_out = hidden_dim * 2

        # Color head: → (max_nodes, num_colors) logits [no activation — CE handles it]
        self.color_head = nn.Sequential(
            nn.Linear(trunk_out, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, max_nodes * num_colors)
        )

        # Shape head: → (max_nodes, node_shape_dim) continuous / mask values
        #
        # For the hybrid builder, this corresponds to the flattened padded
        # 10x10 object shape mask.
        self.shape_head = nn.Sequential(
            nn.Linear(trunk_out, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, max_nodes * node_shape_dim)
        )

        # Edge feature heads — split by type:
        #   continuous : dx, dy
        #   binary     : touching, same_row, same_col
        self.edge_cont_dim   = 2   # indices [0,1]
        self.edge_binary_dim = 3   # indices [2,3,4]

        self.edge_cont_head = nn.Sequential(
            nn.Linear(trunk_out, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, max_nodes * max_nodes * self.edge_cont_dim)
        )

        # Binary head outputs logits — sigmoid + BCE applied in the loss
        self.edge_binary_head = nn.Sequential(
            nn.Linear(trunk_out, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, max_nodes * max_nodes * self.edge_binary_dim)
        )

    def forward(self, z):
        B = z.size(0)
        h = self.trunk(z)                                           # (B, trunk_out)

        color_logits = self.color_head(h).view(B, self.max_nodes, self.num_colors)
        node_shape   = self.shape_head(h).view(B, self.max_nodes, self.node_shape_dim)

        # Edge continuous: dims [0,1]
        edge_cont   = self.edge_cont_head(h).view(B, self.max_nodes, self.max_nodes, self.edge_cont_dim)

        # Edge binary logits: dims [2,3,4]
        edge_binary = self.edge_binary_head(h).view(B, self.max_nodes, self.max_nodes, self.edge_binary_dim)

        # Reassemble into HYBRID 5-dim ordering for easy comparison with input:
        # [dx, dy, touching, same_row, same_col]
        edge_feats = torch.cat([
            edge_cont,             # dx, dy
            edge_binary,           # touching, same_row, same_col (logits)
        ], dim=-1)                                                    # (B, N, N, 5)

        return color_logits, node_shape, edge_feats, edge_binary


# ─────────────────────────────────────────────────────────────────────────────
# 2.  LOSS FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def gat_vae_loss(
    out: dict,
    target_color,           # (B, max_nodes)                long   — class indices 0–9
    target_node_shape,      # (B, max_nodes, 100)           float  — flattened padded shape mask
    target_edge_cont,       # (B, max_nodes, max_nodes, 2)  float  — dx, dy
    target_edge_binary,     # (B, max_nodes, max_nodes, 3)  float  — touching, same_row, same_col
    kl_weight: float        = 1e-3,
    color_weight: float     = 1.0,
    shape_weight: float     = 1.0,
    edge_cont_weight: float = 1.0,
    edge_bin_weight: float  = 1.0,
):
    B, N, C = out["color_logits"].shape

    # ── color: cross-entropy ────────────────────────────────────────────────
    color_loss = F.cross_entropy(
        out["color_logits"].view(B * N, C),
        target_color.view(B * N),
    )

    # ── node shape: MSE ─────────────────────────────────────────────────────
    #
    # The hybrid builder stores the padded flattened shape mask in the node
    # feature vector. We reconstruct that here as a continuous target.
    shape_loss = F.mse_loss(out["node_shape"], target_node_shape)

    # ── edge continuous: MSE ────────────────────────────────────────────────
    #
    # HYBRID continuous edge targets:
    #   [dx, dy]
    pred_edge_cont = out["edge_feats"][..., :2]
    edge_cont_loss = F.mse_loss(pred_edge_cont, target_edge_cont)

    # ── edge binary: BCE with logits ────────────────────────────────────────
    #
    # HYBRID binary edge targets:
    #   [touching, same_row, same_col]
    edge_bin_loss = F.binary_cross_entropy_with_logits(
        out["edge_binary_logits"], target_edge_binary
    )

    # ── KL divergence ────────────────────────────────────────────────────────
    kl_loss = -0.5 * torch.mean(
        1 + out["log_var"] - out["mu"].pow(2) - out["log_var"].exp()
    )

    total = (
        color_weight     * color_loss
      + shape_weight     * shape_loss
      + edge_cont_weight * edge_cont_loss
      + edge_bin_weight  * edge_bin_loss
      + kl_weight        * kl_loss
    )

    return total, {
        "color_loss"     : color_loss.item(),
        "shape_loss"     : shape_loss.item(),
        "edge_cont_loss" : edge_cont_loss.item(),
        "edge_bin_loss"  : edge_bin_loss.item(),
        "kl_loss"        : kl_loss.item(),
    }