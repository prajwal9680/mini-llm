from core.model import MiniGPT
import os
import sys
import torch
import torch.nn.functional as F
import tiktoken

# Project paths
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)


# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = 'minigpt_checkpoint.pt'  # Base pre-trained model
lora_path = 'lora_sft_epoch_2.pt'         # SFT Adapters

# Hyperparameters (Matching pre-training config)
embed_dim = 768
num_heads = 12
num_layers = 12
block_size = 1024  # Pillar 1: Context Length Expansion

# Tokenizer
enc = tiktoken.get_encoding("gpt2")


def encode(text):
    return enc.encode(text)


def decode(tokens):
    return enc.decode(tokens)

# ---- LORA INJECTION LOGIC (Mirroring Training) ----


class LoRALinear(torch.nn.Module):
    def __init__(self, base_layer, rank=16, lora_alpha=16):
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


def apply_lora(model, rank=16):
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
        model.load_state_dict(torch.load(
            lora_path, map_location=device), strict=False)

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
        if user_input.lower() == 'exit':
            break
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
            history = system_part + "### User:" + \
                "### User:".join(recent_turns)


if __name__ == "__main__":
    start_chat()
