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

        # Fallback for Dolly-style single turn data
        if not messages and "instruction" in item:
            user_text = item["instruction"]

            if item.get("input"):
                user_text += "\n" + item["input"]
            if item.get("context"):
                user_text += "\n" + item["context"]

            assistant_text = item.get("output") or item.get("response") or ""

            messages = [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text}
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
                turn_labels = ([-100] * len(prefix_tokens)) + \
                    turn_content_tokens + ([-100] * len(suffix_tokens))
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
    stack = load_dataset(
        "HuggingFaceH4/stack-exchange-preferences", split="train")

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
        print("✅ Cache cleared!")

    return tokens


# MOCK DATA TEST
mock_data = [{"instruction": "Explain AI", "input": "", "output": "AI is..."}]
tokenizer = tiktoken.get_encoding("gpt2")
test_ds = InstructionDataset(mock_data, tokenizer, block_size=128)

x, y = test_ds[0]  # Correct: index the INSTANCE
print(f"Input tokens: {y[:10]}")
