from datasets import load_dataset
import tiktoken
import torch

def build_dataset(output_path):
    print("Building dataset...")

    dataset = load_dataset("openwebtext", split="train[:5%]")
    all_text = "\n".join(dataset["text"])

    enc = tiktoken.get_encoding("gpt2")
    tokens = torch.tensor(enc.encode(all_text), dtype=torch.long)

    torch.save(tokens, output_path)

    print("Saved:", len(tokens))