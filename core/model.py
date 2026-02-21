import torch 
import torch.nn as nn 
from core.transformer_block import TransformerBlock


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len):
        super().__init__()
        
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)

        self.blocks = nn.ModuleList([ TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)])

        self.ln_f = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx):
        B, T = idx.shape

        positions = torch.arange(0, T, device = idx.device).unsqueeze(0)

        x = self.token_emb(idx) + self.pos_emb(positions)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        return logits

if __name__ == "__main__":
    model = MiniGPT(100, 64, 4, 2, 16)
    idx = torch.randint(0, 100, (2,8))
    logits = model(idx)
    print("logits shape", logits.shape)
    

        

        