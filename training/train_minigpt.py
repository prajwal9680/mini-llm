import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import requests

from core.model import MiniGPT

batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
embed_dim = 256
num_heads = 8
num_layers = 8
 
device = "cuda" if torch.cuda.is_available() else "cpu"

#----------------------------------------------------------
#LOAD_DATA(TINY SHAKESPEAR)
#----------------------------------------------------------

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = requests.get(url).text

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i, ch in enumerate(chars) }
itos = {i:ch for ch, i in stoi.items()}

def encode (s):
    return [stoi[ch] for ch in s]

def decode(indices):
    return "".join([itos[i] for i in indices])

data = torch.tensor(encode(text), dtype = torch.long)

n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]

#-----------------------------------------------------------
#BATCH_LOADER
#------------------------------------------------------------

def get_batch(split):
    dataset = train_data if split == 'train' else val_data
    ix = torch.randint(len(dataset) - block_size, (batch_size,))
    x = torch.stack([dataset[i : i+block_size] for i in ix])
    y = torch.stack([dataset[i+1 : i+block_size+1] for i in ix])

    return x.to(device), y.to(device)





#----------------------------------------------------------------
#MODEL
#----------------------------------------------------------------

model = MiniGPT(vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, max_seq_len=block_size).to(device)
optimizer = optim.AdamW(model.parameters(), lr = learning_rate)

#----------------------------------------------------------------------
#EVALUATION
#-------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    losses = {}
    model.eval()
    

    for split in ["train", "val"]:
        total_loss = 0
        for _ in range(20):
            xb, yb = get_batch(split)
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))

            total_loss += loss.item()
        losses[split] = total_loss / 20

    model.train()
    return losses

#---------------------------------------------------------------------------------
#GENERATE
#----------------------------------------------------------------------------------
@torch.no_grad()
def generate(model, start_token, max_new_tokens=300, temperature=1.0, top_k=None):
    model.eval()
    idx = torch.tensor([[stoi[start_token]]], device=device)

    vocab_size = len(stoi)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        logits = logits[:,-1,:]
        logits = logits/max(temperature, 1e-8)

        if top_k is not None:
            k = min(top_k, vocab_size)
            v, _ = torch.topk(logits, k)
            logits[logits < v[:, [-1]]] = -float('inf')

        probs = F.softmax(logits, dim = -1)
        next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_token), dim=1)

    return decode(idx[0].tolist())



#----------------------------------------------------------------------------
#TRAINING_LOOP
#----------------------------------------------------------------------------
for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step : {step} | train_loss : {losses['train']:.4f} | val_loss : {losses['val']:.4f}")
        sample = generate(model, start_token='\n', max_new_tokens=200, temperature=0.8, top_k=40)
        print(sample)
    xb, yb = get_batch("train")

    logits = model(xb)
    loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
            
