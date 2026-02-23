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
# Safer scaling for ~16GB GPUs (e.g. RTX A4000)
# (Attention memory scales with block_size^2, so keep context modest.)
embed_dim = 768
num_heads = 12
num_layers = 12
# Pillar 1 & 7: Context expansion and Extended training
block_size = 512
batch_size = 2
grad_accumulation_steps = 60 # Effective batch size = 2 * 60 = 120
max_iters = 40000 
eval_interval = 500
learning_rate = 6e-4

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Pillar 8: Hardware Acceleration (A6000 / Ampere+)
if device == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("TF32 Acceleration Enabled")

# Tokenizer
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab

def encode(text):
    return enc.encode(text)

def decode(tokens):
    return enc.decode(tokens)

# Data path (expects prebuilt tensor dataset)
data_path = os.path.join(project_root, "openweb_tokens.pt")

print("Loading prebuilt dataset...")
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
    x = torch.stack([dataset[i:i+block_size] for i in ix]).long()
    y = x.clone() # Model handles shifting internally
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

checkpoint_path = 'minigpt_checkpoint.pt'
start_step = 0

# Auto-resume if checkpoint exists
if os.path.exists(checkpoint_path):
    print(f"Found checkpoint at {checkpoint_path}. Resuming...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    # Optional: load optimizer state if you want to be 100% precise
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_step = checkpoint['step']
    print(f"Resuming from step {start_step}")
else:
    print("No checkpoint found. Starting from scratch.")

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], 
    lr=learning_rate, 
    fused=True if device == 'cuda' else False
)

# Pillar 3: Warmup + Cosine Decay Scheduler
import math
warmup_steps = 2000

def get_lr(step):
    # 1. Linear warmup
    if step < warmup_steps:
        return learning_rate * step / warmup_steps
    # 2. Cosine decay
    progress = (step - warmup_steps) / (max_iters - warmup_steps)
    return learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

# Pillar 4: Perplexity & Improved Validation
@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {}
    for split in ["train", "val"]:
        log_losses = []
        for _ in range(20): # More samples for stability
            xb, yb = get_batch(split)
            logits, loss = model(xb, targets=yb)
            log_losses.append(loss.item())
        avg_loss = sum(log_losses) / len(log_losses)
        losses[split] = avg_loss
        losses[f"{split}_ppl"] = math.exp(avg_loss)
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
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        logits = logits / max(temperature, 1e-8)

        if top_k is not None:
            k = min(top_k, vocab_size)
            v, _ = torch.topk(logits, k=50)
            logits[logits < v[:, [-1]]] = -float('inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)

        # Stop condition: If model starts hallucinating User prompt
        decoded_so_far = decode(idx[0].tolist())
        if "### User:" in decoded_so_far:
            # Cut off the hallucinated part
            final_text = decoded_so_far.split("### User:")[0]
            return final_text

    return decode(idx[0].tolist())

# Training loop
print("Starting training...")
scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None

for step in range(start_step, max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {step}: train_loss={losses['train']:.4f}, val_loss={losses['val']:.4f}")
        
        # Periodic Save (Every eval_interval)
        print(f"Saving checkpoint at step {step}...")
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': losses['val'],
        }, checkpoint_path)

        if step > 0:
            sample = generate(model, start_text='The', max_new_tokens=50)
            print(f"Sample: {sample}\n")
    
    # Pillar 3: Update Learning Rate
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # Gradient Accumulation Loop
    optimizer.zero_grad()
    accum_loss = 0
    for _ in range(grad_accumulation_steps):
        xb, yb = get_batch("train")
        
        # Use BF16 for Ampere GPUs (Faster/Stable)
        autocast_context = torch.amp.autocast('cuda', dtype=torch.bfloat16) if device == 'cuda' else torch.cuda.amp.autocast(enabled=False)
        
        with autocast_context:
            logits, loss = model(xb, targets=yb)
            loss = loss / grad_accumulation_steps # Scale loss
        
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        accum_loss += loss.item()

    # Step Optimizer
    if scaler:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # scheduler.step() # REMOVED: using manual get_lr logic above

print("Training complete!")

# Final generation
print("\n" + "="*50)
print("FINAL GENERATION:")
print("="*50)
final_text = generate(model, start_text='Once upon a time', max_new_tokens=200)
print(final_text)

# Save model
torch.save(model.state_dict(), 'minigpt_final.pt')
print("\nModel saved to minigpt_final.pt")