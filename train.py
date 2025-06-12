from model import *
import torch
import torch.nn as nn
from torch.nn import functional as F
import json

maxIters = 5000
evalInterval = 500
lRate = 3e-4

torch.manual_seed(1337)
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


vocab = sorted(list(set(text)))
vocabSize = len(vocab)

stoi = {ch:i for i,ch in enumerate(vocab)}
itos = {i:ch for i,ch in enumerate(vocab)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
trainData = data[:n]
valData = data[n:]

def get_batch(split):
    data = trainData if split == 'train' else valData
    ix = torch.randint(len(data) - blockSize, (batchSize,))
    x = torch.stack([data[i:i+blockSize] for i in ix])
    y = torch.stack([data[i+1:i+blockSize+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y

m = BigramLanguageModel(vocabSize).to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=lRate)#learn function'

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

for iter in range(maxIters):

    if iter % evalInterval == 0:
        losses = estimateLoss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(m.state_dict(), "model.pt")
print("Model saved as model.pt")
