# Mini-LLM Final Launch-Ready Codebase (GPT-3 Hybrid - 500 INR Gold Standard)



## File: README.md

``




``n

## File: chat.py

``
python


import os
import sys
import torch
import torch.nn.functional as F
import tiktoken

# Project paths
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from core.model import MiniGPT

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = 'minigpt_checkpoint.pt' # Base pre-trained model
lora_path = 'lora_sft_epoch_2.pt'         # SFT Adapters

# Hyperparameters (Matching pre-training config)
embed_dim = 768
num_heads = 12
num_layers = 12
block_size = 1024 # Pillar 1: Context Length Expansion

# Tokenizer
enc = tiktoken.get_encoding("gpt2")

def encode(text):
    return enc.encode(text)

def decode(tokens):
    return enc.decode(tokens)

# ---- LORA INJECTION LOGIC (Mirroring Training) ----
class LoRALinear(torch.nn.Module):
    def __init__(self, base_layer, rank=8, lora_alpha=16):
        super().__init__()
        self.base_layer = base_layer
        self.scaling = lora_alpha / rank
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        self.lora_A = torch.nn.Parameter(torch.zeros((in_features, rank)))
        self.lora_B = torch.nn.Parameter(torch.zeros((rank, out_features)))

        # Pillars 5: Identical initialization to Training code
        import math
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.base_layer(x) + (x @ self.lora_A @ self.lora_B) * self.scaling

def apply_lora(model, rank=8):
    for block in model.blocks:
        block.attn.q_proj = LoRALinear(block.attn.q_proj, rank=rank)
        block.attn.v_proj = LoRALinear(block.attn.v_proj, rank=rank)
    return model

# Setup Model
def load_chat_model():
    model = MiniGPT(
        vocab_size=enc.n_vocab,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=block_size
    )

    # 1. Load Base Weights
    if os.path.exists(checkpoint_path):
        print(f"Loading Base Brain: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    
    # 2. Inject LoRA Adapters
    model = apply_lora(model).to(device)

    # 3. Load LoRA weights
    if os.path.exists(lora_path):
        print(f"Loading Chat Adapters: {lora_path}")
        model.load_state_dict(torch.load(lora_path, map_location=device), strict=False)
    
    model.eval()
    return model

@torch.no_grad()
def generate_response(model, prompt, max_new_tokens=150, temperature=0.7, top_k=40):
    input_tokens = encode(prompt)
    idx = torch.tensor([input_tokens], dtype=torch.long, device=device)
    
    # We'll track where the new tokens start
    prompt_len = len(input_tokens)
    
    for _ in range(max_new_tokens):
        # Pillar 6: Context Truncation (Safety)
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        
        # Pillar 6: Stochastic Sampling + Top-K
        logits = logits[:, -1, :] / max(temperature, 1e-8)
        
        # Top-K Filtering
        v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)

        # Hallucination Guard: Stop if model tries to speak for User
        # Only check the NEWLY generated part
        new_tokens = idx[0][prompt_len:].tolist()
        decoded_new = decode(new_tokens)
        
        if "### User:" in decoded_new:
            # Cut off the hallucinated part and return
            clean_new = decoded_new.split("### User:")[0]
            return clean_new.strip()
        
        # Stop at EOT
        if next_token.item() == 50256:
            break

    # Return only the newly generated tokens, decoded
    return decode(idx[0][prompt_len:].tolist()).replace("<|endoftext|>", "").strip()

def start_chat():
    print("\n--- MiniGPT Chat Interface (Session 2/3) ---")
    print("Type 'exit' to quit. Use 'reset' to clear conversation history.\n")
    
    model = load_chat_model()
    
    # System Prompt (GPT-3 Pillar 4)
    system_prompt = "You are a helpful, creative, and honest AI assistant."
    history = f"System: {system_prompt}\n\n"
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit': break
        if user_input.lower() == 'reset':
            history = f"System: {system_prompt}\n\n"
            print("Conversation history cleared.")
            continue

        # Format for SFT model
        current_prompt = f"{history}### User:\n{user_input}\n\n### Assistant:\n"
        
        print("Assistant: ", end="", flush=True)
        response = generate_response(model, current_prompt)
        print(response)
        
        # Update history
        history += f"### User:\n{user_input}\n\n### Assistant:\n{response}\n\n"
        
        # Keeping history within context window (GPT-3 Pillar 4)
        # Better Truncation: Keep System Prompt + last 4 turns
        history_tokens = encode(history)
        if len(history_tokens) > (block_size - 150):
            print("(Context full: trimming history)")
            turns = history.split("### User:")
            # Always keep first part (System prompt)
            system_part = turns[0]
            # Keep last 3 turns if available
            recent_turns = turns[-3:]
            history = system_part + "### User:" + "### User:".join(recent_turns)

if __name__ == "__main__":
    start_chat()
``n

## File: configs\base_config.py

``
python


``n

## File: configs\lora_config.py

``
python


``n

## File: configs\train_config.py

``
python


``n

## File: core\attention.py

``
python


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
``n

## File: core\model.py

``
python


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
``n

## File: core\tokenizer.py

``
python


``n

## File: core\transformer_block.py

``
python


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
``n

## File: finetune\dataset.py

``
python


from datasets import load_dataset
import torch
from torch.utils.data import Dataset
import tiktoken
import os
import shutil
import numpy as np

class InstructionDataset(Dataset):
    """
    Expects data in unified format:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    def __init__(self, data, tokenizer, block_size):
        self.data = data
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.pad_id = getattr(self.tokenizer, "pad_token_id", 50256)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item.get("messages", [])
        
        # Fallback for Dolly-style single turn data (unification)
        if not messages and "instruction" in item:
            messages = [
                {"role": "user", "content": item["instruction"] + ("\n" + item["input"] if item.get("input") else "")},
                {"role": "assistant", "content": item["output"]}
            ]

        all_tokens = []
        all_labels = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            # Format: ### Role:\nContent\n\n
            prefix = f"### {role.capitalize()}:\n"
            suffix = "\n\n"
            
            prefix_tokens = self.tokenizer.encode(prefix)
            content_tokens = self.tokenizer.encode(content)
            suffix_tokens = self.tokenizer.encode(suffix)
            
            if role == "assistant":
                # Add EOT token to Assistant response to teach it to STOP
                # Format: content + EOT
                EOT_ID = 50256
                turn_content_tokens = content_tokens + [EOT_ID]
                turn_tokens = prefix_tokens + turn_content_tokens + suffix_tokens
                
                # Mask prefix and suffix, keep content + EOT
                turn_labels = ([-100] * len(prefix_tokens)) + turn_content_tokens + ([-100] * len(suffix_tokens))
            else:
                turn_tokens = prefix_tokens + content_tokens + suffix_tokens
                # Mask entire user turn
                turn_labels = [-100] * len(turn_tokens)
            
            all_tokens.extend(turn_tokens)
            all_labels.extend(turn_labels)

        # Truncate from the BACK to keep the most recent signal (Assistant Answer)
        if len(all_tokens) > self.block_size:
            all_tokens = all_tokens[-self.block_size:]
            all_labels = all_labels[-self.block_size:]

        # Padding
        pad_len = self.block_size - len(all_tokens)
        if pad_len > 0:
            all_tokens += [self.pad_id] * pad_len
            all_labels += [-100] * pad_len

        return torch.tensor(all_tokens), torch.tensor(all_labels)









def build_dataset(output_path, max_examples=30000):
    print(f"Building dataset with {max_examples} examples...")
    
    dataset = load_dataset(
        "openwebtext",
        split="train",
        streaming=True
    )

    enc = tiktoken.get_encoding("gpt2")
    tokens_acc = []
    total_tokens = 0

    for i, example in enumerate(dataset):
        if i >= max_examples:
            break
        
        if i % 1000 == 0:
            print(f"Processed {i} examples, total tokens: {total_tokens}")
        
        text = example["text"]
        if len(text) > 0:
            encoded = enc.encode(text)
            if len(encoded) > 0:
                tokens_acc += encoded + [50256]
                total_tokens += len(encoded) + 1
    
    # --- BLOCK: StackExchange technical knowledge ---
    print("Adding StackExchange technical knowledge...")
    stack = load_dataset("HuggingFaceH4/stack-exchange-preferences", split="train")

    added = 0
    max_stack = 60000   # Optimized ratio (~10%)

    for example in stack:
        if added >= max_stack:
            break

        # Robust mapping: detect field names
        q = example.get("question") or example.get("prompt") or ""
        a = example.get("answer") or example.get("response") or ""
        text = q + " " + a

        encoded = enc.encode(text)
        if len(encoded) > 0:
            tokens_acc += encoded + [50256]
            total_tokens += len(encoded) + 1
            added += 1
  
    print(f"Converting {total_tokens} tokens to tensor (int32)...")
    tokens = torch.tensor(tokens_acc, dtype=torch.int32)
    
    print(f"Saving {len(tokens)} tokens to {output_path}")
    torch.save(tokens, output_path)

    # Clean up Hugging Face cache to save disk space on Kaggle
    cache_dir = os.path.expanduser('~/.cache/huggingface')
    if os.path.exists(cache_dir):
        print(f"Cleaning up cache at {cache_dir}...")
        shutil.rmtree(cache_dir, ignore_errors=True)
        print("âœ… Cache cleared!")

    return tokens

# MOCK DATA TEST
mock_data = [{"instruction": "Explain AI", "input": "", "output": "AI is..."}]
tokenizer = tiktoken.get_encoding("gpt2")
test_ds = InstructionDataset(mock_data, tokenizer, block_size=128)

x, y = test_ds[0] # Correct: index the INSTANCE
print(f"Input tokens: {y[:10]}")
``n

## File: finetune\evaluate.py

``
python


``n

## File: finetune\finetune_gpt2_lora.py

``
python


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

# âœ… Step 4 â€” LR / Epoch Settings
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

# âœ… Step 1 â€” Check Masking Live
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

        # âœ… Step 2 â€” Print First 200 Loss Values
        if step < 200:
            print(f"Step {step}: {loss.item()}")
        
        step += 1

# âœ… Step 3 â€” Quick Functional Test After Training
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

# âœ… Step 4 â€” Save LoRA Adapters Only
def save_lora_only(model, path):
    print(f"Saving LoRA adapters to {path}...")
    # Only save parameters that require grad (the adapters)
    lora_state_dict = {k: v for k, v in model.state_dict().items() if 'lora_' in k}
    torch.save(lora_state_dict, path)
    print("Adapters saved successfully!")

lora_output_path = 'lora_sft_epoch_2.pt'
save_lora_only(model, lora_output_path)

print("Model output test run and saving completed.")
``n

## File: lora\apply_lora.py

``
python


``n

## File: lora\lora.py

``
python


``n

## File: lora\lora_utils.py

``
python


``n

## File: requirements.txt

``




torch
tiktoken
datasets
``n

## File: training\generate.py

``
python


``n

## File: training\train_minigpt.py

``
python


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
# Maximum Safe Scaling for Kaggle T4 (15GB VRAM)
# This is equivalent to "GPT-3 Small" (approx 125M parameters)
embed_dim = 768
num_heads = 12
num_layers = 12
# Pillar 1 & 7: Context expansion and Extended training
block_size = 1024
batch_size = 4  # User specified
grad_accumulation_steps = 30 # Effective batch size = 4 * 30 = 120
max_iters = 60000 # Gold Standard (500 INR target)
eval_interval = 1000 # More efficient for longer run
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
    build_dataset(data_path, max_examples=120000)

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
torch.save(model.state_dict(), '/kaggle/working/minigpt_final.pt')
print("\nModel saved to /kaggle/working/minigpt_final.pt")
``n

## File: training\utils.py

``
python


``n
