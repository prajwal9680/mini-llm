import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tiktoken
from datasets import load_dataset

# Project paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from core.model import MiniGPT
from finetune.dataset import InstructionDataset
import math

# ---- LORA INFRASTRUCTURE (Local Wrapper to keep core/ clean) ----
class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank=8, lora_alpha=16):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank

        # Freeze base layer
        for p in self.base_layer.parameters():
            p.requires_grad = False
        
        # LoRA matrices
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        self.lora_A = nn.Parameter(torch.zeros((in_features, rank)))
        self.lora_B = nn.Parameter(torch.zeros((rank, out_features)))
        
        # Pillar 5: Initialize LoRA B to zeros (Important for stability)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Result = Base(x) + (x @ A @ B) * scaling
        return self.base_layer(x) + (x @ self.lora_A @ self.lora_B) * self.scaling

def apply_lora(model, rank=8):
    """Injects LoRA into Q and V projections of the model blocks."""
    for block in model.blocks:
        block.attn.q_proj = LoRALinear(block.attn.q_proj, rank=rank)
        block.attn.v_proj = LoRALinear(block.attn.v_proj, rank=rank)
    return model

# ✅ Step 4 — LR / Epoch Settings
learning_rate = 2e-5
num_epochs = 2
batch_size = 16 # Optimized for A6000
block_size = 1024
lora_rank = 8
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Hardware Acceleration
if device == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
vocab_size = tokenizer.n_vocab

# Model
print("Initializing model...")
model = MiniGPT(
    vocab_size=vocab_size,
    embed_dim=768,
    num_heads=12,
    num_layers=12,
    max_seq_len=block_size
).to(device)

# Load Pre-trained Weights (Step 6000 Checkpoint)
checkpoint_path = 'minigpt_checkpoint.pt'
if os.path.exists(checkpoint_path):
    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Load model state dict (ignoring optimizer/step etc)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Base weights loaded successfully!")
else:
    print("Warning: No pre-training checkpoint found. Fine-tuning from scratch!")

# Apply LoRA wrapping (STEP 2: Injected Trainable Matrices)
print(f"Applying LoRA (rank={lora_rank}) to attention layers...")
model = apply_lora(model, rank=lora_rank).to(device)

print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Dataset
print("Loading dataset...")
raw_data = load_dataset("databricks/databricks-dolly-15k", split="train")
dataset = InstructionDataset(raw_data, tokenizer, block_size=block_size)
dataloader = DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=6, # Multiprocessing boot
    pin_memory=True, # High speed host-to-device transfer
    persistent_workers=True # Keep workers alive between epochs
)

# ✅ Step 1 — Check Masking Live
# ---- SANITY CHECK: MASKING ----
x, y = dataset[0]

print("First 40 input tokens:")
print(x[:40])

print("First 40 label tokens:")
print(y[:40])

# exit()  # Remove this after verifying

# Optimizer: Filter only trainable parameters
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=2e-5,
    weight_decay=0.1,
    fused=True if device == 'cuda' else False
)

# ---- VISUALIZATION & OVERFIT TEST ----
def visualize_masking(dataset, tokenizer, num_samples=1):
    print("\n--- Label Masking Visualization (First 50 tokens) ---")
    x, y = dataset[0]
    tokens = x.tolist()
    labels = y.tolist()
    for i in range(min(50, len(tokens))):
        t, l = tokens[i], labels[i]
        if t == 50256 and i > 0: break # Stop at first pad if possible
        token_str = tokenizer.decode([t]).replace("\n", "\\n")
        label_str = "KEEP" if l != -100 else "MASK"
        print(f"Token: {token_str:15} | Label: {label_str}")

def overfit_test(model, dataset, optimizer, device, num_iters=30):
    print("\n--- Starting Single-Sample Overfit Test ---")
    model.train()
    x, y = dataset[0]
    x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
    for i in range(num_iters):
        logits, loss = model(x, targets=y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f"Overfit Iter {i}, Loss: {loss.item():.4f}")
    print("Overfit test completed. Loss should drop significantly.\n")

# Run Verification
visualize_masking(dataset, tokenizer)
# overfit_test(model, dataset, optimizer, device) # Uncomment to run active overfit test

# Training Loop
model.train()
step = 0
for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    for input_ids, labels in dataloader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16) if device == 'cuda' else torch.cuda.amp.autocast(enabled=False):
            logits, loss = model(input_ids, targets=labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ✅ Step 2 — Print First 200 Loss Values
        if step < 200:
            print(f"Step {step}: {loss.item()}")
        
        step += 1

# ✅ Step 3 — Quick Functional Test After Training
# ---- QUICK FUNCTIONAL TEST ----
model.eval()

prompt = "### User:\nAdd two numbers: 4 and 6\n\n### Assistant:\n"
input_ids = tokenizer.encode(prompt)
input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

# Simple greedy generation for testing
generated = input_ids

for _ in range(50):
    with torch.no_grad():
        logits, _ = model(generated)

    next_token_logits = logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    generated = torch.cat((generated, next_token), dim=1)

decoded = tokenizer.decode(generated[0].tolist())
print(f"Generated text:\n{decoded}")

# ✅ Step 4 — Save LoRA Adapters Only
def save_lora_only(model, path):
    print(f"Saving LoRA adapters to {path}...")
    # Only save parameters that require grad (the adapters)
    lora_state_dict = {k: v for k, v in model.state_dict().items() if 'lora_' in k}
    torch.save(lora_state_dict, path)
    print("Adapters saved successfully!")

lora_output_path = 'lora_sft_epoch_2.pt'
save_lora_only(model, lora_output_path)

print("Model output test run and saving completed.")
