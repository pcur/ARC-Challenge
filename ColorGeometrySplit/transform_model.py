"""
transform_model.py
==================
TRM-style transform model for ARC-AGI.

Architecture
------------
Given K-1 demonstration pairs as task context and a test input, predicts
the test output pixel-by-pixel through iterative refinement.

Components
----------
1. DualEncoder (frozen)
   SpatialGeomAE + SpatialColorAE encoders loaded from checkpoints.
   Produces per-node embeddings for both graph types.

2. TaskContextEncoder
   Pools demonstration pair embeddings → task context vector (1024-dim).
   Mean+max pool of each encoder output, concatenated across in/out.

3. GeomColorFusion
   Single cross-attention layer: each output pixel queries the object
   graph embeddings to enrich its representation with geometric context.
   Produces fused_in (N_pix, 256) from h_color (N_pix, 128) + h_geom (N_obj, 128).

4. OutputInitialiser
   Produces y_0 (M_pix, 256) from output pixel positions (row_norm, col_norm).
   Gives the model a spatial prior for the output grid.

5. TRMUpdateZ (weight-shared across steps)
   Updates internal reasoning latent z given x, y, z, task_context.
   Cross-attends z to fused_in and y, then processes with task_context.

6. TRMUpdateY (weight-shared across steps)
   Updates predicted output y given y and z.
   Cross-attends y to z.

7. ColorOutputHead
   Per-pixel MLP: (M_pix, 256) → (M_pix, 10) color logits.
   Reuses the frozen SpatialColorAE color head.

Loss
----
Deep supervision: CE loss at every refinement step, averaged.
Background pixels weighted at 0.1 as in the encoder training.

Dependencies:
    pip install torch torch_geometric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch


# ─────────────────────────────────────────────────────────────────────────────
# DUAL ENCODER  (frozen)
# ─────────────────────────────────────────────────────────────────────────────

class FrozenDualEncoder(nn.Module):
    """
    Loads and freezes SpatialGeomAE and SpatialColorAE encoders.
    Produces per-node embeddings for both graph types.
    """

    def __init__(self, geom_ckpt: str, color_ckpt: str, device):
        super().__init__()

        from spatial_geom_ae import SpatialGeomAE
        from spatial_color_ae import SpatialColorAE
        from train_spatial_geom import compute_norm_stats

        # ── load GeomAE ───────────────────────────────────────────────────────
        geom_state  = torch.load(geom_ckpt, map_location=device)
        geom_cfg    = geom_state["cfg"]
        self.geom_ae = SpatialGeomAE(
            hidden_dim  = geom_cfg["hidden_dim"],
            num_heads   = geom_cfg["num_heads"],
            num_layers  = geom_cfg["num_layers"],
            dropout     = geom_cfg["dropout"],
            pos_enc_dim = geom_cfg.get("pos_enc_dim", 0),
        ).to(device)
        self.geom_ae.load_state_dict(geom_state["model_state"])
        self.geom_norm = {k: v.to(device)
                         for k, v in geom_state["norm_stats"].items()}
        self.geom_hidden = geom_cfg["hidden_dim"]

        # ── load ColorAE ──────────────────────────────────────────────────────
        color_state  = torch.load(color_ckpt, map_location=device)
        color_cfg    = color_state["cfg"]
        self.color_ae = SpatialColorAE(
            hidden_dim       = color_cfg["hidden_dim"],
            num_heads        = color_cfg["num_heads"],
            num_layers       = color_cfg["num_layers"],
            dropout          = color_cfg["dropout"],
            color_dropout_p  = 0.0,   # no dropout at inference
        ).to(device)
        self.color_ae.load_state_dict(color_state["model_state"])
        self.color_hidden = color_cfg["hidden_dim"]

        # freeze both
        for p in self.geom_ae.parameters():
            p.requires_grad = False
        for p in self.color_ae.parameters():
            p.requires_grad = False

        print(f"FrozenDualEncoder ready")
        print(f"  Geom hidden : {self.geom_hidden}")
        print(f"  Color hidden: {self.color_hidden}")

    def _normalise_geom(self, obj_batch):
        """Apply geometry normalisation matching training stats."""
        from train_spatial_geom import extract_geom_features
        x_geom, edge_geom, _ = extract_geom_features(
            obj_batch, self.geom_norm, obj_batch.x.device
        )
        return x_geom, edge_geom

    @torch.no_grad()
    def encode_geom(self, obj_batch):
        """Object graph → per-node geometry embeddings (N, geom_hidden)."""
        self.geom_ae.eval()
        x_geom, edge_geom = self._normalise_geom(obj_batch)
        return self.geom_ae.encode(x_geom, obj_batch.edge_index, edge_geom)

    @torch.no_grad()
    def encode_color(self, pix_batch):
        """Pixel graph → per-node color embeddings (N, color_hidden)."""
        self.color_ae.eval()
        return self.color_ae.encode(pix_batch.x, pix_batch.edge_index,
                                    pix_batch.edge_attr)

    def get_color_head(self):
        """Return the frozen color head for decoding."""
        return self.color_ae.color_head


# ─────────────────────────────────────────────────────────────────────────────
# MEAN+MAX POOL
# ─────────────────────────────────────────────────────────────────────────────

def mean_max_pool(h, ptr):
    """
    Mean+max pool per-graph node embeddings using batch ptr.

    h   : (N_total, d)  flat node embeddings
    ptr : (B+1,)        batch pointer from PyG Batch

    Returns (B, 2d) — mean and max concatenated per graph.
    """
    B  = ptr.size(0) - 1
    d  = h.size(-1)
    out = torch.zeros(B, 2 * d, device=h.device)
    for i in range(B):
        s, e    = ptr[i].item(), ptr[i+1].item()
        if s == e:
            continue
        nodes   = h[s:e]
        out[i]  = torch.cat([nodes.mean(0), nodes.max(0).values], dim=-1)
    return out   # (B, 2d)


# ─────────────────────────────────────────────────────────────────────────────
# TASK CONTEXT ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class TaskContextEncoder(nn.Module):
    """
    Encodes K-1 demonstration pairs into a single task context vector.

    For each demo pair:
      pool(h_geom_in)  → (2*geom_hidden,)   = (256,)
      pool(h_color_in) → (2*color_hidden,)  = (256,)
      pool(h_geom_out) → (256,)
      pool(h_color_out)→ (256,)
      pair_embed = concat → (1024,)

    task_context = mean(pair_embeds) → (1024,)
    Then projected to context_dim via MLP.
    """

    def __init__(
        self,
        geom_hidden: int    = 128,
        color_hidden: int   = 128,
        context_dim: int    = 256,
    ):
        super().__init__()
        pair_dim = 4 * 2 * max(geom_hidden, color_hidden)  # 4 * 256 = 1024
        self.pair_dim   = 4 * (2 * geom_hidden + 2 * color_hidden) // 2
        # actual: 2*geom + 2*color + 2*geom + 2*color = 4*(geom+color) = 4*256 = 1024
        raw_dim = 2 * geom_hidden * 2 + 2 * color_hidden * 2  # 1024

        self.proj = nn.Sequential(
            nn.LayerNorm(raw_dim),
            nn.Linear(raw_dim, context_dim * 2),
            nn.SiLU(),
            nn.Linear(context_dim * 2, context_dim),
        )

    def forward(self, pair_embeds):
        """
        pair_embeds : (K, 1024)  — one row per demonstration pair
        Returns     : (context_dim,)  — aggregated task context
        """
        task_vec = pair_embeds.mean(dim=0)   # (1024,)
        return self.proj(task_vec)            # (context_dim,)


# ─────────────────────────────────────────────────────────────────────────────
# GEOM-COLOR FUSION
# ─────────────────────────────────────────────────────────────────────────────

class GeomColorFusion(nn.Module):
    """
    Fuses pixel-level color embeddings with object-level geometry embeddings
    via cross-attention. Each pixel queries the object graph.

    h_color (N_pix, color_hidden) + h_geom (N_obj, geom_hidden)
    → fused  (N_pix, color_hidden + geom_hidden) = (N_pix, 256)
    """

    def __init__(
        self,
        color_hidden: int = 128,
        geom_hidden: int  = 128,
        num_heads: int    = 4,
        dropout: float    = 0.1,
    ):
        super().__init__()
        self.color_hidden = color_hidden
        self.geom_hidden  = geom_hidden

        # project geom to same dim as color for cross-attention
        self.geom_proj = nn.Linear(geom_hidden, color_hidden)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim   = color_hidden,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,
        )
        self.norm = nn.LayerNorm(color_hidden)

    def forward(self, h_color, h_geom):
        """
        h_color : (N_pix, color_hidden)
        h_geom  : (N_obj, geom_hidden)
        Returns : (N_pix, color_hidden + geom_hidden)
        """
        # project geom embeddings to color space
        g      = self.geom_proj(h_geom)   # (N_obj, color_hidden)

        # cross-attention: pixels query geometry
        # add batch dim for MultiheadAttention
        q      = h_color.unsqueeze(0)     # (1, N_pix, color_hidden)
        k = v  = g.unsqueeze(0)           # (1, N_obj, color_hidden)
        attn_out, _ = self.cross_attn(q, k, v)
        geom_ctx    = self.norm(attn_out.squeeze(0))   # (N_pix, color_hidden)

        return torch.cat([h_color, geom_ctx], dim=-1)  # (N_pix, 256)


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT INITIALISER
# ─────────────────────────────────────────────────────────────────────────────

class OutputInitialiser(nn.Module):
    """
    Initialises y_0 from output pixel positions.

    Each output pixel at (row_norm, col_norm) gets a learned embedding
    that provides a spatial prior without assuming any specific color.
    """

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, out_dim // 2),
            nn.SiLU(),
            nn.Linear(out_dim // 2, out_dim),
        )

    def forward(self, pix_graph):
        """
        pix_graph : PyG Data with x[:, 10] = row_norm, x[:, 11] = col_norm
        Returns   : (N_pix, out_dim)
        """
        pos = pix_graph.x[:, 10:12]   # (N_pix, 2) row_norm + col_norm
        return self.mlp(pos)


# ─────────────────────────────────────────────────────────────────────────────
# TRM UPDATE NETWORKS
# ─────────────────────────────────────────────────────────────────────────────

class TRMUpdateZ(nn.Module):
    """
    Updates internal reasoning latent z.
    Weight-shared across all refinement steps.

    z_{t+1} = f(z_t, cross_attn(z_t, fused_in), cross_attn(z_t, y_t),
                task_context)
    """

    def __init__(
        self,
        dim: int         = 256,
        context_dim: int = 256,
        num_heads: int   = 4,
        dropout: float   = 0.1,
    ):
        super().__init__()
        self.attn_x = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                             batch_first=True)
        self.attn_y = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                             batch_first=True)

        self.norm_x = nn.LayerNorm(dim)
        self.norm_y = nn.LayerNorm(dim)
        self.norm_z = nn.LayerNorm(dim)

        # task context injected via addition after projection
        self.ctx_proj = nn.Linear(context_dim, dim)

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, z, fused_in, y, task_context):
        """
        z            : (N_pix, dim)
        fused_in     : (N_pix, dim)
        y            : (N_pix, dim)
        task_context : (dim,) or (1, dim)
        """
        # cross-attend z to input
        z_q    = z.unsqueeze(0)
        x_kv   = fused_in.unsqueeze(0)
        y_kv   = y.unsqueeze(0)

        z_x, _ = self.attn_x(z_q, x_kv, x_kv)
        z      = z + self.norm_x(z_x.squeeze(0))

        z_y, _ = self.attn_y(z_q, y_kv, y_kv)
        z      = z + self.norm_y(z_y.squeeze(0))

        # inject task context
        ctx = self.ctx_proj(task_context)
        if ctx.dim() == 1:
            ctx = ctx.unsqueeze(0)
        z   = z + ctx

        z = self.norm_z(z)
        z = z + self.ffn(z)
        return z


class TRMUpdateY(nn.Module):
    """
    Updates predicted output y from internal latent z.
    Weight-shared across all refinement steps.

    y_{t+1} = g(y_t, cross_attn(y_t, z_t))
    """

    def __init__(self, dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn   = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                             batch_first=True)
        self.norm_a = nn.LayerNorm(dim)
        self.norm_y = nn.LayerNorm(dim)
        self.ffn    = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, y, z):
        """
        y : (N_pix, dim)
        z : (N_pix, dim)
        """
        y_q    = y.unsqueeze(0)
        z_kv   = z.unsqueeze(0)
        ya, _  = self.attn(y_q, z_kv, z_kv)
        y      = y + self.norm_a(ya.squeeze(0))
        y      = self.norm_y(y)
        y      = y + self.ffn(y)
        return y


# ─────────────────────────────────────────────────────────────────────────────
# FULL TRM TRANSFORM MODEL
# ─────────────────────────────────────────────────────────────────────────────

class TRMTransform(nn.Module):
    """
    TRM-style transform model for ARC-AGI.

    Takes encoded input graph embeddings + task context,
    iteratively refines a predicted output pixel sequence,
    and decodes to color logits at each step for deep supervision.

    Parameters
    ----------
    geom_hidden  : hidden dim of SpatialGeomAE (default 128)
    color_hidden : hidden dim of SpatialColorAE (default 128)
    fused_dim    : dimension after geom-color fusion (default 256)
    context_dim  : task context vector dimension (default 256)
    num_heads    : attention heads in TRM updates (default 4)
    dropout      : dropout in TRM (default 0.1)
    num_colors   : ARC color classes (default 10)
    """

    def __init__(
        self,
        geom_hidden: int  = 128,
        color_hidden: int = 128,
        fused_dim: int    = 256,
        context_dim: int  = 256,
        num_heads: int    = 4,
        dropout: float    = 0.1,
        num_colors: int   = 10,
    ):
        super().__init__()
        self.fused_dim   = fused_dim
        self.context_dim = context_dim

        raw_context_dim  = 2 * (2 * geom_hidden + 2 * color_hidden)  # 1024

        self.task_ctx_encoder = TaskContextEncoder(
            geom_hidden  = geom_hidden,
            color_hidden = color_hidden,
            context_dim  = context_dim,
        )
        self.fusion      = GeomColorFusion(color_hidden, geom_hidden,
                                           num_heads, dropout)
        self.initialiser = OutputInitialiser(fused_dim)

        # weight-shared update networks
        self.update_z = TRMUpdateZ(fused_dim, context_dim, num_heads, dropout)
        self.update_y = TRMUpdateY(fused_dim, num_heads, dropout)

        # color prediction head
        self.color_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, fused_dim),
            nn.SiLU(),
            nn.Linear(fused_dim, num_colors),
        )

    def encode_pair(self, h_geom, h_color_in, h_geom_out, h_color_out,
                    ptr_geom_in, ptr_color_in, ptr_geom_out, ptr_color_out):
        """Pool one demonstration pair into a (1024,) embedding."""
        p_gi  = mean_max_pool(h_geom,       ptr_geom_in)    # (1, 256)
        p_ci  = mean_max_pool(h_color_in,   ptr_color_in)   # (1, 256)
        p_go  = mean_max_pool(h_geom_out,   ptr_geom_out)   # (1, 256)
        p_co  = mean_max_pool(h_color_out,  ptr_color_out)  # (1, 256)
        return torch.cat([p_gi, p_ci, p_go, p_co], dim=-1)  # (1, 1024)

    def forward(
        self,
        h_geom_in,      # (N_obj, 128)   test input geometry embeddings
        h_color_in,     # (N_pix, 128)   test input color embeddings
        task_context,   # (context_dim,) aggregated from demonstrations
        out_pix_graph,  # PyG Data       output pixel graph (for positions)
        num_steps: int  = 4,
    ):
        """
        Returns list of color_logits (N_pix, 10), one per refinement step.
        Use last for inference, all for deep supervision loss.
        """
        device = h_color_in.device

        # fuse geometry and color for input
        fused_in = self.fusion(h_color_in, h_geom_in)  # (N_pix, 256)

        # initialise output sequence from positions
        out_pix_graph = out_pix_graph.to(device)
        y = self.initialiser(out_pix_graph)             # (M_pix, 256)
        z = torch.zeros_like(y)                         # (M_pix, 256)

        all_logits = []
        for _ in range(num_steps):
            z = self.update_z(z, fused_in, y, task_context)
            y = self.update_y(y, z)
            all_logits.append(self.color_head(y))       # (M_pix, 10)

        return all_logits
