import torch

ckpt = torch.load("minigpt_checkpoint.pt", map_location="cpu")

print(type(ckpt))
print(ckpt.keys() if isinstance(ckpt, dict) else "Not a dict")