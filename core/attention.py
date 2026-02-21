import torch
import torch.nn as nn 
import torch.nn.functional as F 
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):

        B, T, C = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1,2)

        scores = Q @ K.transpose(-2, -1)

        scores = scores / math.sqrt(self.head_dim)

        mask = torch.tril(torch.ones(T, T, device = x.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(scores, dim = -1)

        out = weights @ V

        out = out.transpose(1,2).contiguous().view(B, T, C)

        return self.out_proj(out)

if __name__ == "__main__":
    x = torch.randn(2, 8, 64)
    attn = MultiHeadSelfAttention(embed_dim = 64, num_heads = 4)


    out = attn(x)
    print("attn output shape", out.shape)





