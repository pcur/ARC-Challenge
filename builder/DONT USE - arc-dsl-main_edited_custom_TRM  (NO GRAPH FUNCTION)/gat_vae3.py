import torch
import torch.nn as nn

from gat_encoder import GATEncoder, EncoderTrunk, VAEBottleneck
from graph_decoder3 import GraphDecoder, gat_vae_loss


# ─────────────────────────────────────────────────────────────────────────────
# GATVAE  v3
# ─────────────────────────────────────────────────────────────────────────────

class GATVAE(nn.Module):
    """
    v3: Forward pass now returns the three new cell-level decoder outputs:
        cell_coords       — (B, max_nodes, max_cells, 2)
        cell_color_logits — (B, max_nodes, max_cells, num_colors)
        cell_mask_logits  — (B, max_nodes, max_cells)

    These, combined with existence_logits, give a complete path from
    z → predicted grid via graph_to_grid_from_predictions() in
    custom_object3.py.

    The encoder side is unchanged. VAE weights should be frozen when
    training the downstream ARC transformer.
    """

    def __init__(
        self,
        max_nodes: int,
        max_cells_per_node: int = 40,
        node_in_dim: int        = 22,
        edge_in_dim: int        = 12,
        gat_hidden: int         = 128,
        gat_heads: int          = 4,
        gat_layers: int         = 3,
        gat_dropout: float      = 0.1,
        trunk_depth: int        = 4,
        trunk_dropout: float    = 0.1,
        latent_dim: int         = 256,
        dec_hidden: int         = 256,
        num_colors: int         = 10,
        node_geom_dim: int      = 12,
    ):
        super().__init__()

        self.max_nodes          = max_nodes
        self.max_cells_per_node = max_cells_per_node
        self.latent_dim         = latent_dim

        self.gat_encoder = GATEncoder(
            node_in_dim=node_in_dim, edge_in_dim=edge_in_dim,
            hidden_dim=gat_hidden,   num_heads=gat_heads,
            num_layers=gat_layers,   dropout=gat_dropout,
        )
        self.encoder_trunk = EncoderTrunk(
            hidden_dim=self.gat_encoder.readout_dim,
            num_layers=trunk_depth,
            dropout=trunk_dropout,
        )
        self.bottleneck = VAEBottleneck(
            input_dim=self.gat_encoder.readout_dim,
            latent_dim=latent_dim,
        )
        self.graph_decoder = GraphDecoder(
            max_nodes=max_nodes,
            max_cells_per_node=max_cells_per_node,
            latent_dim=latent_dim,
            num_colors=num_colors,
            node_geom_dim=node_geom_dim,
            edge_out_dim=edge_in_dim,
            hidden_dim=dec_hidden,
        )

    # ── encoder ──────────────────────────────────────────────────────────
    def encode(self, x, edge_index, edge_attr, batch):
        """Graph → (z, mu, log_var)."""
        graph_emb      = self.gat_encoder(x, edge_index, edge_attr, batch)
        graph_emb      = self.encoder_trunk(graph_emb)
        z, mu, log_var = self.bottleneck(graph_emb)
        return z, mu, log_var

    # ── decoder ──────────────────────────────────────────────────────────
    def decode(self, z):
        """z → all decoder outputs (8 tensors)."""
        return self.graph_decoder(z)

    # ── full forward ─────────────────────────────────────────────────────
    def forward(self, x, edge_index, edge_attr, batch):
        z, mu, log_var = self.encode(x, edge_index, edge_attr, batch)

        (color_logits,
         node_geom,
         edge_feats,
         edge_binary_logits,
         existence_logits,
         cell_coords,           # v3
         cell_color_logits,     # v3
         cell_mask_logits,      # v3
        ) = self.decode(z)

        return {
            # ── node level ────────────────────────────────────────────────
            "color_logits"       : color_logits,       # (B, N, 10)
            "node_geom"          : node_geom,           # (B, N, 12)
            "existence_logits"   : existence_logits,    # (B, N)
            # ── edge level ────────────────────────────────────────────────
            "edge_feats"         : edge_feats,          # (B, N, N, 12)
            "edge_binary_logits" : edge_binary_logits,  # (B, N, N, 6)
            # ── cell level (v3) ───────────────────────────────────────────
            "cell_coords"        : cell_coords,         # (B, N, MC, 2)
            "cell_color_logits"  : cell_color_logits,   # (B, N, MC, C)
            "cell_mask_logits"   : cell_mask_logits,    # (B, N, MC)
            # ── VAE ───────────────────────────────────────────────────────
            "z"                  : z,                   # (B, latent_dim)
            "mu"                 : mu,
            "log_var"            : log_var,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from torch_geometric.data import Data, Batch

    torch.manual_seed(42)

    MAX_NODES = 30
    MAX_CELLS = 40

    def fully_connected_graph(n_nodes):
        idx        = torch.arange(n_nodes)
        src        = idx.repeat_interleave(n_nodes)
        dst        = idx.repeat(n_nodes)
        mask       = src != dst
        edge_index = torch.stack([src[mask], dst[mask]])
        n_edges    = edge_index.size(1)
        return Data(
            x          = torch.randn(n_nodes, 22),
            edge_index = edge_index,
            edge_attr  = torch.randn(n_edges, 12),
        )

    batch = Batch.from_data_list([
        fully_connected_graph(8),
        fully_connected_graph(12),
    ])

    model = GATVAE(max_nodes=MAX_NODES, max_cells_per_node=MAX_CELLS)
    model.train()

    out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

    print("=== GATVAE v3 forward pass ===")
    for k, v in out.items():
        print(f"  {k:22s}: {tuple(v.shape)}")

    B = 2
    target_color      = torch.randint(0, 10, (B, MAX_NODES))
    target_node_geom  = torch.randn(B, MAX_NODES, 12)
    target_edge_cont  = torch.randn(B, MAX_NODES, MAX_NODES, 6)
    target_edge_bin   = (torch.rand(B, MAX_NODES, MAX_NODES, 6) > 0.5).float()
    target_existence  = torch.zeros(B, MAX_NODES)
    target_existence[0, :8]  = 1.0
    target_existence[1, :12] = 1.0

    # v3 cell targets
    target_cell_coords = torch.randint(0, 30, (B, MAX_NODES, MAX_CELLS, 2)).float()
    target_cell_colors = torch.randint(0, 10, (B, MAX_NODES, MAX_CELLS))
    target_cell_mask   = torch.zeros(B, MAX_NODES, MAX_CELLS)
    # first 5 cells real for each real node
    for b in range(B):
        n_real = int(target_existence[b].sum().item())
        target_cell_mask[b, :n_real, :5] = 1.0

    loss, breakdown = gat_vae_loss(
        out,
        target_color, target_node_geom,
        target_edge_cont, target_edge_bin,
        target_existence,
        target_cell_coords,   # v3
        target_cell_colors,   # v3
        target_cell_mask,     # v3
    )

    print(f"\n=== Loss ===")
    print(f"  total : {loss.item():.4f}")
    for k, v in breakdown.items():
        print(f"  {k:18s}: {v:.4f}")
    print("\n✅  Sanity check passed.")
