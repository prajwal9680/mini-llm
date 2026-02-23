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
        
        # GPT-3 Upgrade: Remove dropouts for cleaner signal
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj.resid_proj = True
        
        # ALiBi: Precompute slopes
        # Slopes for heads: 2^(-8/n), 2^(-8*2/n), ..., 2^(-8)
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = (2**(-2**-(math.log2(n)-3)))
                ratio = start
                return [start * ratio**i for i in range(n)]
            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                # Approximate for non-power-of-2 (though our setup is typically 8 or 12 heads)
                return get_slopes_power_of_2(2**math.ceil(math.log2(n)))[:n]

        slopes = torch.tensor(get_slopes(num_heads)).view(1, num_heads, 1, 1)
        self.register_buffer("slopes", slopes)

        # Buffer-based Causal Mask (Efficiency)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(1024, 1024)).view(1, 1, 1024, 1024)
        )

    def forward(self, x):
        B, T, C = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Pillar 2: Scaled Dot Product
        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        # GPT-3 Upgrade: ALiBi (Attention with Linear Biases)
        # Correct ALiBi: distance only into the past
        pos = torch.arange(T, device=x.device)
        rel = pos.view(T, 1) - pos.view(1, T)  # (T, T)
        rel = rel.clamp(min=0).float()         # only past distance

        alibi_bias = self.slopes * (-rel.view(1, 1, T, T))
        scores = scores + alibi_bias

        # Efficient Mask Usage
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)

        out = weights @ V
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)

        return out

if __name__ == "__main__":
    x = torch.randn(2, 8, 64)
    attn = MultiHeadSelfAttention(embed_dim=64, num_heads=4)
    out = attn(x)
    print("attn output shape", out.shape)