


import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        # One projection that creates Q, K, and V together
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)

        # Final output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None, is_causal: bool = False):
        """
        x: [batch_size, seq_len, d_model]
        attn_mask: optional mask broadcastable to attention scores
        is_causal: if True, prevents attending to future tokens

        returns: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Project once, then split into q, k, v
        # qkv: [B, T, 3 * d_model]
        qkv = self.qkv_proj(x)

        # Reshape to separate Q, K, V and heads
        # qkv: [B, T, 3, num_heads, head_dim]
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)

        # Permute so Q/K/V comes first
        # qkv: [3, B, num_heads, T, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # Split into q, k, v
        q, k, v = qkv[0], qkv[1], qkv[2]
        # each is [B, num_heads, T, head_dim]

        # Scaled dot-product attention
        # output: [B, num_heads, T, head_dim]
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        # Merge heads back together
        # [B, num_heads, T, head_dim] -> [B, T, num_heads, head_dim]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()

        # [B, T, num_heads, head_dim] -> [B, T, d_model]
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # Final linear projection
        out = self.out_proj(attn_output)

        return out