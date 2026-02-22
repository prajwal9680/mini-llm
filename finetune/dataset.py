
from datasets import load_dataset
import tiktoken
import torch
import os

def build_dataset(output_path, max_examples=30000):
    print(f"Building dataset with {max_examples} examples...")
    
    dataset = load_dataset(
        "openwebtext",
        split="train",
        streaming=True,
        trust_remote_code=True
    )

    enc = tiktoken.get_encoding("gpt2")
    token_chunks = []
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
                token_chunks.append(torch.tensor(encoded, dtype=torch.long))
                total_tokens += len(encoded)

    print(f"Concatenating {len(token_chunks)} chunks...")
    tokens = torch.cat(token_chunks)
    
    print(f"Saving {len(tokens)} tokens to {output_path}")
    torch.save(tokens, output_path)
    return tokens

if __name__ == "__main__":
    build_dataset("openweb_tokens.pt")