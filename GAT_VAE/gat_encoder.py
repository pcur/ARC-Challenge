import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool


# ─────────────────────────────────────────────────────────────────────────────
# 1.  GAT ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class GATEncoder(nn.Module):

    def __init__(self, node_in_dim: int = 22, edge_in_dim: int = 12, hidden_dim: int  = 128,
                 num_heads: int   = 4, num_layers: int  = 3, dropout: float   = 0.1):
        
        super().__init__()

        self.dropout = dropout

        # Project edge features to a dimension GATv2Conv can use
        self.edge_proj = nn.Linear(edge_in_dim, hidden_dim)

        # Layer list — first layer handles raw node features; rest use hidden_dim
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = node_in_dim
        for _ in range(num_layers):
            out_dim = hidden_dim // num_heads  # each head produces this width
            conv = GATv2Conv(in_channels=in_dim, out_channels=out_dim, heads=num_heads,
                             edge_dim=hidden_dim, dropout=dropout, concat=True)
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim

        # Graph-level readout: concat mean + max pooling → hidden_dim * 2
        self.readout_dim = hidden_dim * 2

    def forward(self, x, edge_index, edge_attr, batch):
        # Project edge features once — reused across all GAT layers
        e = F.silu(self.edge_proj(edge_attr))          # (E, hidden_dim)

        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, edge_index, edge_attr=e)   # (N, hidden_dim)
            x_new = norm(x_new)
            x_new = F.silu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            # Residual connection when shapes match (all layers except first)
            if x.shape == x_new.shape:
                x = x + x_new
            else:
                x = x_new

        # Readout: concatenate mean-pool and max-pool for richer graph summary
        graph_emb = torch.cat(
            [global_mean_pool(x, batch), global_max_pool(x, batch)],
            dim=-1,
        )                                               # (B, hidden_dim * 2)
        return graph_emb


# ─────────────────────────────────────────────────────────────────────────────
# 2.  ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class EncoderTrunk(nn.Module):

    def __init__(self, hidden_dim: int = 256, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dropout = dropout

        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = x + F.dropout(F.silu(layer(x)), p=self.dropout, training=self.training)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 3.  VAE BOTTLENECK
# ─────────────────────────────────────────────────────────────────────────────

class VAEBottleneck(nn.Module):

    def __init__(self, input_dim: int, latent_dim: int = 256):
        super().__init__()
        self.fc_mu      = nn.Linear(input_dim, latent_dim)
        self.fc_log_var = nn.Linear(input_dim, latent_dim)

    def reparameterise(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu   # deterministic at inference time

    def forward(self, graph_emb):
        mu      = self.fc_mu(graph_emb)
        log_var = self.fc_log_var(graph_emb)
        z       = self.reparameterise(mu, log_var)
        return z, mu, log_var
