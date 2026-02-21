import torch
import torch.nn as nn 
from core.attention import MultiHeadSelfAttention 

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
                                  nn.Linear(embed_dim, 4 * embed_dim),
                                  nn.GELU(),
                                  nn.Linear(4 * embed_dim, embed_dim)
                                )

    def forward(self, x):

        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

if __name__ == '__main__':
    x = torch.randn(2, 8, 64)

    block = TransformerBlock(64, 4)
    
    out = block(x)
    print("output block shape", out.shape)