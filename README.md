# MiniGPT — GPT-3 Style Language Model from Scratch

A GPT-3 style transformer language model built entirely from scratch in PyTorch, featuring modern architectural upgrades including ALiBi positional encoding, SwiGLU activations, and LoRA fine-tuning.

Built as a personal deep learning project while studying CSE at RVCE.

---

## Architecture

```
MiniGPT (~110M parameters)
├── Token Embedding          (vocab: 50,257 GPT-2 tokens)
├── 12x Transformer Blocks
│   ├── LayerNorm
│   ├── Multi-Head Self-Attention (12 heads, 768 dim)
│   │   └── ALiBi positional bias (no positional embeddings)
│   ├── LayerNorm
│   └── SwiGLU Feed-Forward  (768 → 3072 → 768)
└── Output LM Head           (weight-tied with token embedding)
```

### Key Design Choices

| Feature | Description |
|---|---|
| **ALiBi** | Attention with Linear Biases — replaces positional embeddings for better length generalization |
| **SwiGLU** | Gated activation (used in LLaMA, PaLM) instead of standard GELU MLP |
| **Residual Scaling** | `std *= (2 * num_layers)^-0.5` on residual projections for stable deep training |
| **LoRA** | Low-Rank Adaptation on Q and V projections for efficient instruction fine-tuning |
| **Weight Tying** | Input embedding and output LM head share weights (GPT-2 style) |

---

## Project Structure

```
mini-llm-local/
├── core/
│   ├── model.py              # MiniGPT model definition
│   ├── attention.py          # Multi-head self-attention with ALiBi
│   └── transformer_block.py  # TransformerBlock with SwiGLU
├── finetune/
│   ├── finetune_gpt2_lora.py # LoRA supervised fine-tuning script
│   └── dataset.py            # Instruction dataset with label masking
├── chat/
│   └── chat.py               # Interactive chat interface
├── configs/                  # Training hyperparameter configs
├── training/                 # Pre-training scripts
├── lora/                     # LoRA adapter utilities
├── requirements.txt
└── README.md
```

---

## Training Details

**Pre-training:**
- Dataset: OpenWebText + StackExchange
- Context length: 1024 tokens
- Optimizer: AdamW with cosine LR schedule

**Fine-tuning (SFT with LoRA):**
- Dataset: Databricks Dolly-15k
- LoRA rank: 8, alpha: 16
- Applied to: Q and V projections in all 12 attention layers
- Trainable params: ~1.2M out of 110M (only LoRA matrices)
- Label masking: user turns masked, only assistant responses trained on

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Chat with the model (requires pre-trained checkpoint)
python chat/chat.py
```

**Chat commands:**
- `exit` — quit
- `reset` — clear conversation history

---

## Requirements

- Python 3.8+
- PyTorch (CUDA recommended)
- `tiktoken`

```bash
pip install torch tiktoken
```

---

## What I Learned

- Implementing multi-head self-attention from scratch
- How ALiBi works as a drop-in replacement for positional embeddings
- Why SwiGLU outperforms standard GELU in practice
- LoRA: freezing base weights and training only low-rank adapter matrices
- Instruction tuning with proper label masking (user turns masked)
- Residual projection scaling for training stability
