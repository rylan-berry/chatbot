import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from model import *
from vocabulary_aid import *
import json

#Loads merges and special tokens
with open("merges.json", "r", encoding="utf-8") as f:
    meta = json.load(f)
merges = {tuple(map(int, k.split(','))): v for k, v in meta["merges"].items()}
special_tokens = meta["spec_tokens"]
vocabSize = 256 + len(merges) + len(special_tokens)

#Loads model and pretrained weights
m = BigramLanguageModel(vocabSize).to(device)
m.load_state_dict(torch.load("model.pt", map_location=device))
print("Loaded pretrained model.")

#converts text to utf-8, then itterates through it using the known merges list to merge byte pairs.
def encode(text):
  bytes = list(text.encode("utf-8"))
  tokens = []
  i = 0

  while i < len(text):
     specTok = False

     for special, token_id in special_tokens.items():
            spec_bytes =  list(special.encode("utf-8"))
            spec_len = len(spec_bytes)
            if bytes[i:i + spec_len] == spec_bytes:
                tokens.append(token_id)
                i += spec_len
                specTok = True
                break
     if specTok:
         continue
     tokens.append(bytes[i])
     i+=1
        
    

  while len(tokens) >= 2:
    stats = get_stats(tokens)
    pair = min(stats, key=lambda p: merges.get(p, float("inf")))
    if pair not in merges:
      break # no more merges
    idx = merges[pair]
    tokens = merge(tokens, pair, idx)
  return tokens

#Loads fine-tuning data
with open("finetuneData.txt", "r", encoding = "utf-8") as f:
    text = f.read()


data = encode(text)
data = torch.tensor(data, dtype=torch.long)

n = int(0.9 * len(data))
trainData = data[:n]
valData = data[n:]

def get_batch(split):
    data = trainData if split == 'train' else valData
    ix = torch.randint(len(data) - blockSize, (batchSize,))
    x = torch.stack([data[i:i+blockSize] for i in ix])
    y = torch.stack([data[i+1:i+blockSize+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y

@torch.no_grad()
def estimateLoss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(evalIters)
        for k in range(evalIters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

#Sets fine-tuning hyperparameters
maxIters = 1000
evalInterval = 100
lRate = 1e-6
optimizer = torch.optim.AdamW(m.parameters(), lr=lRate)

#Fine-tuning loop
for iter in range(maxIters):
    if iter % evalInterval == 0:
        losses = estimateLoss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#Saves the fine-tuned model
torch.save(m.state_dict(), "finetuned_model.pt")
print("Fine-tuned model saved.")
