import os
import sys

# Dynamic path detection
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
print(f"Project root: {project_root}")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tiktoken
from core.model import MiniGPT
import os

# Configuration
embed_dim = 320
num_heads = 8
num_layers = 8
block_size = 384
batch_size = 24
max_iters = 15000
eval_interval = 1000
learning_rate = 4e-3

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Tokenizer
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab

def encode(text):
    return enc.encode(text)

def decode(tokens):
    return enc.decode(tokens)

# Data path
data_path = os.path.join(project_root, "openweb_tokens.pt")

# Kaggle-specific: check if dataset is in input directory if not found
if not os.path.exists(data_path):
    # Search for openweb_tokens.pt in /kaggle/input
    for root, dirs, files in os.walk('/kaggle/input'):
        if 'openweb_tokens.pt' in files:
            data_path = os.path.join(root, 'openweb_tokens.pt')
            print(f"Found dataset at: {data_path}")
            break

# Build dataset if needed
if not os.path.exists(data_path):
    print("Building dataset...")
    from finetune.dataset import build_dataset
    build_dataset(data_path, max_examples=30000)

# Load data
print("Loading dataset...")
data = torch.load(data_path)
print(f"Total tokens: {len(data):,}")
n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]

print(f"Train tokens: {len(train_data):,}")
print(f"Val tokens: {len(val_data):,}")

# Batch loader
def get_batch(split):
    dataset = train_data if split == 'train' else val_data
    ix = torch.randint(len(dataset) - block_size, (batch_size,))
    x = torch.stack([dataset[i:i+block_size] for i in ix])
    y = torch.stack([dataset[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# Model
print("Initializing model...")
model = MiniGPT(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    max_seq_len=block_size
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Evaluation
@torch.no_grad()
def estimate_loss():
    losses = {}
    model.eval()
    
    for split in ["train", "val"]:
        total_loss = 0
        for _ in range(10):
            xb, yb = get_batch(split)
            logits = model(xb)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.reshape(B*T, C), yb.reshape(B*T))
            total_loss += loss.item()
        losses[split] = total_loss / 10
    
    model.train()
    return losses

# Generate
@torch.no_grad()
def generate(model, start_text, max_new_tokens=100, temperature=0.8, top_k=40):
    model.eval()
    start_tokens = encode(start_text)
    idx = torch.tensor([start_tokens], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]
        logits = logits / max(temperature, 1e-8)

        if top_k is not None:
            k = min(top_k, vocab_size)
            v, _ = torch.topk(logits, k)
            logits[logits < v[:, [-1]]] = -float('inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)

    return decode(idx[0].tolist())

# Training loop
print("Starting training...")
scaler = torch.cuda.amp.GradScaler()

for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {step}: train_loss={losses['train']:.4f}, val_loss={losses['val']:.4f}")
        
        if step > 0:
            sample = generate(model, start_text='The', max_new_tokens=50)
            print(f"Sample: {sample}\n")
    
    xb, yb = get_batch("train")
    
    with torch.cuda.amp.autocast():
        logits = model(xb)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.reshape(B*T, C), yb.reshape(B*T))

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

print("Training complete!")

# Final generation
print("\n" + "="*50)
print("FINAL GENERATION:")
print("="*50)
final_text = generate(model, start_text='Once upon a time', max_new_tokens=200)
print(final_text)

# Save model
torch.save(model.state_dict(), '/kaggle/working/minigpt_final.pt')
print("\nModel saved to /kaggle/working/minigpt_final.pt")