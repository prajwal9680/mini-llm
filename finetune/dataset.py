from datasets import load_dataset
import tiktoken
import torch

def build_dataset(output_path):
    print("Building dataset (streaming)...")

    dataset = load_dataset("openwebtext", split="train[:5%]")

    enc = tiktoken.get_encoding("gpt2")

    tokens = []

    for example in dataset:
        tokens.extend(enc.encode(example["text"]))

    tokens = torch.tensor(tokens, dtype=torch.long)

    torch.save(tokens, output_path)

    print("Saved:", len(tokens))