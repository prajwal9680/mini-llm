import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import requests

from core.model import MiniGPT

batch_size = 64
block_size = 512
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
embed_dim = 256
num_heads = 8
num_layers = 8
 
device = "cuda" if torch.cuda.is_available() else "cpu"





import tiktoken
enc = tiktoken.get_encoding("gpt2")
def encode( text):
    return enc.encode(text)

def decode(tokens):
    return enc.decode(tokens)

vocab_size = enc.n_vocab

#----------------------------------------------------------
#LOAD_DATA(TINY SHAKESPEAR)
#----------------------------------------------------------


data = torch.load("openweb_tokens.pt")

n = int(0.95 * len(data))

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
            with torch.cuda.amp.autocast():
                logits = model(xb)
            
                B, T, C = logits.shape
                loss = F.cross_entropy(logits.reshape(B*T, C), yb.reshape(B*T))

                total_loss += loss.item()
        losses[split] = total_loss / 20

    model.train()
    return losses

#---------------------------------------------------------------------------------
#GENERATE
#----------------------------------------------------------------------------------
@torch.no_grad()
def generate(model, start_text, max_new_tokens=300, temperature=1.0, top_k=None):
    model.eval()

    start_tokens = encode(start_text)
    idx = torch.tensor([start_tokens],dtype=torch.long, device=device)

    

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

scaler = torch.cuda.amp.GradScaler()

for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step : {step} | train_loss : {losses['train']:.4f} | val_loss : {losses['val']:.4f}")
        sample = generate(model, start_text='\n', max_new_tokens=200, temperature=0.8, top_k=40)
        print(sample)
    xb, yb = get_batch("train")

    with torch.cuda.amp.autocast():
        logits = model(xb)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.reshape(B*T, C), yb.reshape(B*T))

    optimizer.zero_grad()
    scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    scaler.step(optimizer)
    scaler.update()
 

            
