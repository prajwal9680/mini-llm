from datasets import load_dataset
import tiktoken
import torch

def build_dataset(output_path):
    print("Building dataset (streaming)...")

    dataset = load_dataset(
        "openwebtext",
        split="train[:2%]",
        streaming=True
    )

    enc = tiktoken.get_encoding("gpt2")

    token_chunks = []

    for example in dataset:
        encoded = enc.encode(example["text"])
        token_chunks.append(torch.tensor(encoded, dtype=torch.long))

    tokens = torch.cat(token_chunks)

    torch.save(tokens, output_path)

    print("Saved:", len(tokens))