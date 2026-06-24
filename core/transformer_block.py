import torch
import torch.nn as nn 
import torch.nn.functional as F
from core.attention import MultiHeadSelfAttention 

class SwiGLU(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # In SwiGLU, it's common to use 8/3 of the hidden dim for the gate
        # but for simplicity and context alignment, we'll use 4x growth
        self.w1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.w2 = nn.Linear(embed_dim, 4 * embed_dim)
        self.w3 = nn.Linear(4 * embed_dim, embed_dim)
        self.w3.resid_proj = True

    def forward(self, x):
        # Swish(xW1) * xW2 (Gated Unit)
        gate = F.silu(self.w1(x)) * self.w2(x)
        return self.w3(gate)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = SwiGLU(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

if __name__ == '__main__':
    x = torch.randn(2, 8, 64)
    block = TransformerBlock(64, 4)
    out = block(x)
    print("output block shape", out.shape)