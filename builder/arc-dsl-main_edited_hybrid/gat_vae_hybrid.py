import torch
import torch.nn as nn

from gat_encoder_hybrid import GATEncoder, EncoderTrunk, VAEBottleneck
from graph_decoder_hybrid import GraphDecoder, gat_vae_loss


# ─────────────────────────────────────────────────────────────────────────────
# GATVAE
# ─────────────────────────────────────────────────────────────────────────────

class GATVAE(nn.Module):
    """
    Full Graph Attention Variational Autoencoder for the HYBRID builder.

    Parameters
    ----------
    max_nodes      : maximum number of nodes per graph (required)
    node_in_dim    : node feature dimension (default 110)
    edge_in_dim    : edge feature dimension (default 5)
    gat_hidden     : hidden dim inside GATEncoder (default 128)
    gat_heads      : number of attention heads (default 4)
    gat_layers     : number of GATv2Conv layers (default 3)
    gat_dropout    : dropout rate in GAT layers (default 0.1)
    latent_dim     : VAE latent space dimension (default 256)
    dec_hidden     : hidden dim inside GraphDecoder (default 256)
    num_colors     : ARC color classes (default 10)
    node_shape_dim : flattened padded node shape size (default 100)

    HYBRID NODE FORMAT
    ------------------
    node_in_dim = 110
        - first 10 dims  : object-level color multi-hot
        - next 100 dims  : flattened padded 10x10 shape mask

    HYBRID EDGE FORMAT
    ------------------
    edge_in_dim = 5
        - [dx, dy, touching, same_row, same_col]
    """

    def __init__(
        self,
        max_nodes: int,
        node_in_dim: int = 110,
        edge_in_dim: int = 5,
        gat_hidden: int = 128,
        gat_heads: int = 4,
        gat_layers: int = 3,
        gat_dropout: float = 0.1,
        trunk_depth: int = 4,       # residual MLP layers between GAT and bottleneck
        trunk_dropout: float = 0.1,
        latent_dim: int = 256,
        dec_hidden: int = 256,
        num_colors: int = 10,
        node_shape_dim: int = 100,
    ):
        super().__init__()

        # --------------------------------------------------------------------
        # Encoder
        # --------------------------------------------------------------------
        self.gat_encoder = GATEncoder(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=gat_hidden,
            num_heads=gat_heads,
            num_layers=gat_layers,
            dropout=gat_dropout,
        )

        # Residual MLP trunk — refines the graph embedding before the bottleneck.
        # Operates at readout_dim (gat_hidden * 2) throughout so residuals are clean.
        self.encoder_trunk = EncoderTrunk(
            hidden_dim=self.gat_encoder.readout_dim,
            num_layers=trunk_depth,
            dropout=trunk_dropout,
        )

        # --------------------------------------------------------------------
        # Variational bottleneck
        # --------------------------------------------------------------------
        self.bottleneck = VAEBottleneck(
            input_dim=self.gat_encoder.readout_dim,
            latent_dim=latent_dim,
        )

        # --------------------------------------------------------------------
        # Decoder
        # --------------------------------------------------------------------
        # For HYBRID:
        #   - node colors are categorical (10 classes)
        #   - node shapes are continuous/binary mask values of length 100
        #   - edge outputs are 5 dims total:
        #       continuous: [dx, dy]
        #       binary    : [touching, same_row, same_col]
        self.graph_decoder = GraphDecoder(
            max_nodes=max_nodes,
            latent_dim=latent_dim,
            num_colors=num_colors,
            node_shape_dim=node_shape_dim,
            edge_out_dim=edge_in_dim,
            hidden_dim=dec_hidden,
        )

    # ── encoder side ───────────────────────────────────────────────────────
    def encode(self, x, edge_index, edge_attr, batch):
        """
        Graph → (z, mu, log_var).

        This is the encoder split point.
        """
        graph_emb = self.gat_encoder(x, edge_index, edge_attr, batch)
        graph_emb = self.encoder_trunk(graph_emb)
        z, mu, log_var = self.bottleneck(graph_emb)
        return z, mu, log_var

    # ── decoder side ───────────────────────────────────────────────────────
    def decode(self, z):
        """
        z → (color_logits, node_shape, edge_feats, edge_binary_logits)
        """
        return self.graph_decoder(z)

    # ── full forward pass ───────────────────────────────────────────────────
    def forward(self, x, edge_index, edge_attr, batch):
        z, mu, log_var = self.encode(x, edge_index, edge_attr, batch)
        color_logits, node_shape, edge_feats, edge_binary_logits = self.decode(z)

        return {
            "color_logits": color_logits,          # (B, max_nodes, 10)  — categorical
            "node_shape": node_shape,              # (B, max_nodes, 100) — flattened shape mask
            "edge_feats": edge_feats,              # (B, max_nodes, N, 5) — reassembled hybrid edge output
            "edge_binary_logits": edge_binary_logits,  # (B, max_nodes, N, 3) — binary logits
            "z": z,                                # (B, latent_dim)
            "mu": mu,                              # (B, latent_dim) — for KL loss
            "log_var": log_var,                    # (B, latent_dim) — for KL loss
        }


# ─────────────────────────────────────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from torch_geometric.data import Data, Batch

    torch.manual_seed(42)

    MAX_NODES = 20   # set this to the max graph size in your dataset

    # ── fully connected graph builder ───────────────────────────────────────
    def fully_connected_graph(n_nodes):
        idx = torch.arange(n_nodes)
        src = idx.repeat_interleave(n_nodes)
        dst = idx.repeat(n_nodes)
        mask = src != dst
        edge_index = torch.stack([src[mask], dst[mask]])    # (2, N*(N-1))
        n_edges = edge_index.size(1)

        return Data(
            x=torch.randn(n_nodes, 110),      # hybrid node dim
            edge_index=edge_index,
            edge_attr=torch.randn(n_edges, 5),  # hybrid edge dim
        )

    batch = Batch.from_data_list([
        fully_connected_graph(10),
        fully_connected_graph(14),
    ])

    # ── instantiate and run ─────────────────────────────────────────────────
    model = GATVAE(
        max_nodes=MAX_NODES,
        node_in_dim=110,
        edge_in_dim=5,
        node_shape_dim=100,
    )
    model.train()

    out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

    print("=== GATVAE forward pass ===")
    for k, v in out.items():
        print(f"  {k:22s}: {tuple(v.shape)}")

    # ── fake ARC-style hybrid targets ───────────────────────────────────────
    B = 2
    target_color = torch.randint(0, 10, (B, MAX_NODES))
    target_node_shape = torch.randn(B, MAX_NODES, 100)

    # Continuous hybrid edge targets:
    #   [dx, dy]
    target_edge_cont = torch.randn(B, MAX_NODES, MAX_NODES, 2)

    # Binary hybrid edge targets:
    #   [touching, same_row, same_col]
    target_edge_bin = (torch.rand(B, MAX_NODES, MAX_NODES, 3) > 0.5).float()

    loss, breakdown = gat_vae_loss(
        out,
        target_color,
        target_node_shape,
        target_edge_cont,
        target_edge_bin,
    )

    print(f"\n=== Loss ===")
    print(f"  total : {loss.item():.4f}")
    for k, v in breakdown.items():
        print(f"  {k:16s}: {v:.4f}")

    print("\n✅  Sanity check passed.")