import torch
import torch.nn as nn

from gat_encoder_hybrid import GATEncoder, EncoderTrunk, VAEBottleneck
from graph_decoder_hybrid2 import GraphDecoder, gat_vae_loss


# ─────────────────────────────────────────────────────────────────────────────
# GATVAE  hybrid v2
# ─────────────────────────────────────────────────────────────────────────────

class GATVAE(nn.Module):
    """
    Hybrid Graph Attention VAE v2.

    v2 changes vs gat_vae_hybrid.py:
        - forward() now returns existence_logits and bbox from the decoder.
        - decode() unpacks 6 outputs instead of 4.

    Node format:   110 dims (10 color + 100 shape mask)
    Edge format:   5 dims  (dx, dy, touching, same_row, same_col)
    """

    def __init__(
        self,
        max_nodes: int,
        node_in_dim: int    = 110,
        edge_in_dim: int    = 5,
        gat_hidden: int     = 128,
        gat_heads: int      = 4,
        gat_layers: int     = 3,
        gat_dropout: float  = 0.1,
        trunk_depth: int    = 4,
        trunk_dropout: float = 0.1,
        latent_dim: int     = 256,
        dec_hidden: int     = 256,
        num_colors: int     = 10,
        node_shape_dim: int = 100,
    ):
        super().__init__()

        self.max_nodes   = max_nodes
        self.latent_dim  = latent_dim

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
            latent_dim=latent_dim,
            num_colors=num_colors,
            node_shape_dim=node_shape_dim,
            edge_out_dim=edge_in_dim,
            hidden_dim=dec_hidden,
        )

    def encode(self, x, edge_index, edge_attr, batch):
        graph_emb      = self.gat_encoder(x, edge_index, edge_attr, batch)
        graph_emb      = self.encoder_trunk(graph_emb)
        z, mu, log_var = self.bottleneck(graph_emb)
        return z, mu, log_var

    def decode(self, z):
        """z → (color_logits, node_shape, edge_feats, edge_binary, existence_logits, bbox)"""
        return self.graph_decoder(z)

    def forward(self, x, edge_index, edge_attr, batch):
        z, mu, log_var = self.encode(x, edge_index, edge_attr, batch)

        (color_logits,
         node_shape,
         edge_feats,
         edge_binary_logits,
         existence_logits,   # v2
         bbox,               # v2
        ) = self.decode(z)

        return {
            "color_logits"       : color_logits,       # (B, N, 10)
            "node_shape"         : node_shape,          # (B, N, 100)
            "edge_feats"         : edge_feats,          # (B, N, N, 5)
            "edge_binary_logits" : edge_binary_logits,  # (B, N, N, 3)
            "existence_logits"   : existence_logits,    # (B, N)       ← v2
            "bbox"               : bbox,                # (B, N, 4)    ← v2
            "z"                  : z,
            "mu"                 : mu,
            "log_var"            : log_var,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from torch_geometric.data import Data, Batch

    torch.manual_seed(42)
    MAX_NODES = 20

    def fully_connected_graph(n_nodes):
        idx        = torch.arange(n_nodes)
        src        = idx.repeat_interleave(n_nodes)
        dst        = idx.repeat(n_nodes)
        mask       = src != dst
        edge_index = torch.stack([src[mask], dst[mask]])
        n_edges    = edge_index.size(1)
        return Data(
            x          = torch.randn(n_nodes, 110),
            edge_index = edge_index,
            edge_attr  = torch.randn(n_edges, 5),
        )

    batch = Batch.from_data_list([
        fully_connected_graph(8),
        fully_connected_graph(12),
    ])

    model = GATVAE(max_nodes=MAX_NODES)
    model.train()
    out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

    print("=== GATVAE hybrid v2 forward pass ===")
    for k, v in out.items():
        print(f"  {k:22s}: {tuple(v.shape)}")

    B = 2
    target_color      = torch.randint(0, 10, (B, MAX_NODES))
    target_node_shape = torch.randn(B, MAX_NODES, 100)
    target_edge_cont  = torch.randn(B, MAX_NODES, MAX_NODES, 2)
    target_edge_bin   = (torch.rand(B, MAX_NODES, MAX_NODES, 3) > 0.5).float()
    target_existence  = torch.zeros(B, MAX_NODES)
    target_existence[0, :8]  = 1.0
    target_existence[1, :12] = 1.0
    target_bbox = torch.rand(B, MAX_NODES, 4)

    loss, breakdown = gat_vae_loss(
        out, target_color, target_node_shape,
        target_edge_cont, target_edge_bin,
        target_existence, target_bbox,
    )

    print(f"\n=== Loss ===")
    print(f"  total : {loss.item():.4f}")
    for k, v in breakdown.items():
        print(f"  {k:18s}: {v:.4f}")
    print("\n✅  Sanity check passed.")
