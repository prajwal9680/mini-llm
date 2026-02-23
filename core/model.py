import torch
import torch.nn as nn
import torch.nn.functional as F
from core.transformer_block import TransformerBlock

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len):
        super().__init__()
        self.num_layers = num_layers
        self.token_emb = nn.Embedding(vocab_size, embed_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
        
        self.apply(self._init_weights)
        self.head.weight = self.token_emb.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Normal init
            std = 0.02
            # GPT-3 Upgrade: Residual Projection Scaling
            # Scales all residual projections (out_proj and w3)
            if hasattr(module, 'resid_proj'):
                std *= (2 * self.num_layers) ** -0.5
            
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_emb(idx)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            # Shift logits and targets for causal LM: tokens at i predict i+1
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=-100
            )
        return logits, loss

if __name__ == "__main__":
    model = MiniGPT(100, 64, 4, 2, 16)
    idx = torch.randint(0, 100, (2, 8))
    logits = model(idx)
    print("logits shape", logits.shape)