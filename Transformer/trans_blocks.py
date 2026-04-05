

import math
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
    




class SinusoidalPositionalEncoding(nn.Module):
    """Original sine/cosine positional encoding from 'Attention Is All You Need'.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding to input embeddings.

        Args:
            x: Tensor of shape (batch, seq_len, d_model).
        Returns:
            Tensor of shape (batch, seq_len, d_model) with positional info added.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)











