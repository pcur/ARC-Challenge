import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1.  GRAPH DECODER
# ─────────────────────────────────────────────────────────────────────────────

class GraphDecoder(nn.Module):
    """
    v3: Added three new heads for cell-level reconstruction.

    New heads:
        cell_coord_head  — predicts (row, col) for each cell slot of each node
                           Output: (B, max_nodes, max_cells, 2)  continuous
        cell_color_head  — predicts color logits per cell slot
                           Output: (B, max_nodes, max_cells, num_colors)
        cell_mask_head   — predicts whether each cell slot is real or padding
                           Output: (B, max_nodes, max_cells)  logits

    Together these three heads give a complete path from z → grid:
        1. existence_head  → which node slots are real
        2. cell_mask_head  → which cell slots within each node are real
        3. cell_coord_head → where each real cell lives (row, col)
        4. cell_color_head → what color each real cell has

    The original geometry/edge heads are kept so your classmates can still
    use the graph-level reconstruction path.
    """

    def __init__(
        self,
        max_nodes: int,
        max_cells_per_node: int = 40,
        latent_dim: int         = 256,
        num_colors: int         = 10,
        node_geom_dim: int      = 12,
        edge_out_dim: int       = 12,
        hidden_dim: int         = 256,
    ):
        super().__init__()
        self.max_nodes          = max_nodes
        self.max_cells          = max_cells_per_node
        self.num_colors         = num_colors
        self.node_geom_dim      = node_geom_dim
        self.edge_out_dim       = edge_out_dim

        # ── shared trunk: z → rich hidden representation ─────────────────
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),   nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim*2), nn.SiLU(),
            nn.Linear(hidden_dim*2, hidden_dim*2), nn.SiLU(),
        )
        trunk_out = hidden_dim * 2

        # ── node existence head (v2) ──────────────────────────────────────
        self.existence_head = nn.Sequential(
            nn.Linear(trunk_out, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, max_nodes),
        )

        # ── color head (v2) ───────────────────────────────────────────────
        self.color_head = nn.Sequential(
            nn.Linear(trunk_out, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, max_nodes * num_colors),
        )

        # ── geometry head (v2) ────────────────────────────────────────────
        self.geom_head = nn.Sequential(
            nn.Linear(trunk_out, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, max_nodes * node_geom_dim),
        )

        # ── edge heads (v2) ───────────────────────────────────────────────
        self.edge_cont_dim   = 6
        self.edge_binary_dim = 6

        self.edge_cont_head = nn.Sequential(
            nn.Linear(trunk_out, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, max_nodes * max_nodes * self.edge_cont_dim),
        )
        self.edge_binary_head = nn.Sequential(
            nn.Linear(trunk_out, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, max_nodes * max_nodes * self.edge_binary_dim),
        )

        # ── v3: cell coordinate head ──────────────────────────────────────
        # Predicts (row, col) for each cell slot of each node.
        # Output shape: (B, max_nodes, max_cells, 2)
        # We use a two-stage MLP: first expand to per-node hidden states,
        # then project to cell coords. This keeps parameters manageable
        # compared to a flat max_nodes*max_cells*2 output.
        self.cell_node_proj = nn.Sequential(
            nn.Linear(trunk_out, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, max_nodes * hidden_dim // 4),
        )
        node_hidden = hidden_dim // 4

        self.cell_coord_head = nn.Sequential(
            nn.Linear(node_hidden, hidden_dim // 4), nn.SiLU(),
            nn.Linear(hidden_dim // 4, max_cells_per_node * 2),
        )

        # ── v3: cell color head ───────────────────────────────────────────
        # Predicts color logits for each cell slot of each node.
        # Output shape: (B, max_nodes, max_cells, num_colors)
        self.cell_color_head = nn.Sequential(
            nn.Linear(node_hidden, hidden_dim // 4), nn.SiLU(),
            nn.Linear(hidden_dim // 4, max_cells_per_node * num_colors),
        )

        # ── v3: cell mask head ────────────────────────────────────────────
        # Predicts which cell slots are real vs padding (logits).
        # Output shape: (B, max_nodes, max_cells)
        self.cell_mask_head = nn.Sequential(
            nn.Linear(node_hidden, hidden_dim // 4), nn.SiLU(),
            nn.Linear(hidden_dim // 4, max_cells_per_node),
        )

    def forward(self, z):
        B  = z.size(0)
        N  = self.max_nodes
        MC = self.max_cells
        h  = self.trunk(z)                                      # (B, trunk_out)

        # ── existing heads ────────────────────────────────────────────────
        existence_logits = self.existence_head(h)               # (B, N)
        color_logits     = self.color_head(h).view(B, N, self.num_colors)
        node_geom        = self.geom_head(h).view(B, N, self.node_geom_dim)

        edge_cont   = self.edge_cont_head(h).view(B, N, N, self.edge_cont_dim)
        edge_binary = self.edge_binary_head(h).view(B, N, N, self.edge_binary_dim)

        edge_feats = torch.cat([
            edge_cont[..., :4],
            edge_binary,
            edge_cont[..., 4:],
        ], dim=-1)                                              # (B, N, N, 12)

        # ── v3: per-node hidden states for cell heads ─────────────────────
        node_hidden_dim = h.shape[-1] // 4  # = hidden_dim // 2 after trunk doubles
        node_states = self.cell_node_proj(h).view(B, N, -1)    # (B, N, node_hidden)

        # ── v3: cell coordinate predictions ──────────────────────────────
        # (B, N, max_cells, 2) — raw continuous row/col predictions
        cell_coords = self.cell_coord_head(node_states).view(B, N, MC, 2)

        # ── v3: cell color predictions ────────────────────────────────────
        # (B, N, max_cells, num_colors) — logits, CE loss applied in loss fn
        cell_color_logits = self.cell_color_head(node_states).view(
            B, N, MC, self.num_colors)

        # ── v3: cell mask predictions ─────────────────────────────────────
        # (B, N, max_cells) — logits, BCE loss applied in loss fn
        cell_mask_logits = self.cell_mask_head(node_states).view(B, N, MC)

        return (
            color_logits,        # (B, N, 10)          node-level color
            node_geom,           # (B, N, 12)          node geometry
            edge_feats,          # (B, N, N, 12)       edge features
            edge_binary,         # (B, N, N, 6)        binary edge logits
            existence_logits,    # (B, N)              node existence
            cell_coords,         # (B, N, MC, 2)       cell row/col    ← v3
            cell_color_logits,   # (B, N, MC, C)       cell color      ← v3
            cell_mask_logits,    # (B, N, MC)          cell existence  ← v3
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  LOSS FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def gat_vae_loss(
    out: dict,
    target_color,           # (B, N)          long   node color class
    target_node_geom,       # (B, N, 12)      float  node geometry
    target_edge_cont,       # (B, N, N, 6)    float  continuous edge feats
    target_edge_binary,     # (B, N, N, 6)    float  binary edge flags
    target_existence,       # (B, N)          float  1=real node, 0=pad
    target_cell_coords,     # (B, N, MC, 2)   float  cell row/col      ← v3
    target_cell_colors,     # (B, N, MC)      long   cell color class  ← v3
    target_cell_mask,       # (B, N, MC)      float  1=real cell, 0=pad← v3
    kl_weight: float            = 1e-3,
    color_weight: float         = 1.0,
    geom_weight: float          = 1.0,
    edge_cont_weight: float     = 1.0,
    edge_bin_weight: float      = 1.0,
    existence_weight: float     = 1.0,
    cell_coord_weight: float    = 1.0,    # v3
    cell_color_weight: float    = 1.0,    # v3
    cell_mask_weight: float     = 1.0,    # v3
):
    """
    v3: Added three cell-level loss terms.

    Cell losses are doubly masked:
        - outer mask: target_existence (only real nodes contribute)
        - inner mask: target_cell_mask (only real cells within each node)

    This means a padded node slot contributes nothing to cell losses,
    and a padded cell slot within a real node also contributes nothing.
    """
    B, N, C = out["color_logits"].shape
    MC      = out["cell_mask_logits"].shape[-1]

    # ── existence BCE (all slots) ─────────────────────────────────────────
    existence_loss = F.binary_cross_entropy_with_logits(
        out["existence_logits"], target_existence,
    )

    # ── node masks ───────────────────────────────────────────────────────
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

    # ── geometry MSE (real nodes only) ────────────────────────────────────
    geom_flat        = out["node_geom"].view(B*N, -1)
    target_geom_flat = target_node_geom.view(B*N, -1)
    if node_mask.any():
        geom_loss = F.mse_loss(geom_flat[node_mask], target_geom_flat[node_mask])
    else:
        geom_loss = geom_flat.sum() * 0.0

    # ── edge mask (both endpoints real) ──────────────────────────────────
    edge_mask = node_mask_2d.unsqueeze(2) & node_mask_2d.unsqueeze(1)  # (B,N,N)

    pred_edge_cont = torch.cat([out["edge_feats"][..., :4],
                                out["edge_feats"][..., 10:]], dim=-1)
    if edge_mask.any():
        edge_cont_loss = F.mse_loss(pred_edge_cont[edge_mask],
                                    target_edge_cont[edge_mask])
    else:
        edge_cont_loss = pred_edge_cont.sum() * 0.0

    edge_mask_exp = edge_mask.unsqueeze(-1).expand_as(out["edge_binary_logits"])
    if edge_mask_exp.any():
        edge_bin_loss = F.binary_cross_entropy_with_logits(
            out["edge_binary_logits"][edge_mask_exp],
            target_edge_binary[edge_mask_exp],
        )
    else:
        edge_bin_loss = out["edge_binary_logits"].sum() * 0.0

    # ── v3: cell mask BCE ─────────────────────────────────────────────────
    # Doubly masked: only compute for real node slots.
    # Within those, cell_mask_head learns which cell slots are real.
    cell_mask_logits_flat = out["cell_mask_logits"].view(B*N, MC)
    target_cell_mask_flat = target_cell_mask.view(B*N, MC)
    if node_mask.any():
        cell_mask_loss = F.binary_cross_entropy_with_logits(
            cell_mask_logits_flat[node_mask],
            target_cell_mask_flat[node_mask],
        )
    else:
        cell_mask_loss = cell_mask_logits_flat.sum() * 0.0

    # ── v3: cell coordinate MSE ───────────────────────────────────────────
    # Doubly masked: real nodes AND real cells within them.
    # flat combined mask: (B*N, MC) → select real-node rows, then real cells
    cell_coord_pred_flat  = out["cell_coords"].view(B*N, MC, 2)
    target_cell_coord_flat = target_cell_coords.view(B*N, MC, 2)
    target_cell_mask_bool  = target_cell_mask.view(B*N, MC).bool()

    if node_mask.any():
        # Select real-node rows first
        coord_pred_real  = cell_coord_pred_flat[node_mask]    # (real_N, MC, 2)
        coord_tgt_real   = target_cell_coord_flat[node_mask]  # (real_N, MC, 2)
        cell_mask_real   = target_cell_mask_bool[node_mask]   # (real_N, MC)
        # Expand mask to (real_N, MC, 2) for coordinate indexing
        cell_mask_real_2 = cell_mask_real.unsqueeze(-1).expand_as(coord_pred_real)
        if cell_mask_real_2.any():
            cell_coord_loss = F.mse_loss(coord_pred_real[cell_mask_real_2],
                                         coord_tgt_real[cell_mask_real_2])
        else:
            cell_coord_loss = coord_pred_real.sum() * 0.0
    else:
        cell_coord_loss = out["cell_coords"].sum() * 0.0

    # ── v3: cell color CE ─────────────────────────────────────────────────
    # Doubly masked: real nodes AND real cells.
    cell_color_logits_flat = out["cell_color_logits"].view(B*N, MC, C)
    target_cell_colors_flat = target_cell_colors.view(B*N, MC)

    if node_mask.any():
        ccl_real  = cell_color_logits_flat[node_mask]   # (real_N, MC, C)
        cct_real  = target_cell_colors_flat[node_mask]  # (real_N, MC)
        cmr       = target_cell_mask_bool[node_mask]    # (real_N, MC)
        if cmr.any():
            cell_color_loss = F.cross_entropy(
                ccl_real[cmr],    # (real_cells, C)
                cct_real[cmr],    # (real_cells,)
            )
        else:
            cell_color_loss = ccl_real.sum() * 0.0
    else:
        cell_color_loss = out["cell_color_logits"].sum() * 0.0

    # ── KL divergence ─────────────────────────────────────────────────────
    kl_loss = -0.5 * torch.mean(
        1 + out["log_var"] - out["mu"].pow(2) - out["log_var"].exp()
    )

    total = (
        existence_weight   * existence_loss
      + color_weight       * color_loss
      + geom_weight        * geom_loss
      + edge_cont_weight   * edge_cont_loss
      + edge_bin_weight    * edge_bin_loss
      + cell_mask_weight   * cell_mask_loss
      + cell_coord_weight  * cell_coord_loss
      + cell_color_weight  * cell_color_loss
      + kl_weight          * kl_loss
    )

    return total, {
        "existence_loss"  : existence_loss.item(),
        "color_loss"      : color_loss.item(),
        "geom_loss"       : geom_loss.item(),
        "edge_cont_loss"  : edge_cont_loss.item(),
        "edge_bin_loss"   : edge_bin_loss.item(),
        "cell_mask_loss"  : cell_mask_loss.item(),
        "cell_coord_loss" : cell_coord_loss.item(),
        "cell_color_loss" : cell_color_loss.item(),
        "kl_loss"         : kl_loss.item(),
    }
