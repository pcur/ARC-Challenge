import torch

from  trans_blocks import SinusoidalPositionalEncoding, MultiHeadSelfAttention
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


