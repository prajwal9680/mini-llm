from datasets import load_dataset
import tiktoken
import torch

def build_dataset(output_path):
    print("Building dataset (streaming)...")

    dataset = load_dataset(
        "openwebtext",
        split="train",
        streaming=True
    )

    enc = tiktoken.get_encoding("gpt2")

    token_chunks = []
    max_examples = 200_000   # ~2% approx (adjust if needed)

    for i, example in enumerate(dataset):
        if i >= max_examples:
            break
        encoded = enc.encode(example["text"])
        token_chunks.append(torch.tensor(encoded, dtype=torch.long))

    tokens = torch.cat(token_chunks)

    torch.save(tokens, output_path)

    print("Saved:", len(tokens))