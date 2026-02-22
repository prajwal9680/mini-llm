
from datasets import load_dataset
import tiktoken
import torch
import os
import shutil

def build_dataset(output_path, max_examples=30000):
    print(f"Building dataset with {max_examples} examples...")
    
    dataset = load_dataset(
        "openwebtext",
        split="train",
        streaming=True
    )

    enc = tiktoken.get_encoding("gpt2")
    import array
    tokens_acc = array.array('I')
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
                # Add <|endoftext|> token (50256) between documents
                tokens_acc.extend(encoded)
                tokens_acc.append(50256)
                total_tokens += len(encoded) + 1

    print(f"Converting {total_tokens} tokens to tensor...")
    tokens = torch.tensor(tokens_acc, dtype=torch.long)
    
    print(f"Saving {len(tokens)} tokens to {output_path}")
    torch.save(tokens, output_path)

    # Clean up Hugging Face cache to save disk space on Kaggle
    cache_dir = os.path.expanduser('~/.cache/huggingface')
    if os.path.exists(cache_dir):
        print(f"Cleaning up cache at {cache_dir}...")
        shutil.rmtree(cache_dir, ignore_errors=True)
        print("✅ Cache cleared!")

    return tokens

if __name__ == "__main__":
    build_dataset("openweb_tokens.pt", max_examples=1000000)