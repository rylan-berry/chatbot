from model import *
from vocabulary_aid import *
import torch
import torch.nn as nn
from torch.nn import functional as F
import json
import numpy as np

#model training vars
maxIters = 10000
evalInterval = 500
lRate = 3e-4

torch.manual_seed(1337)

with open("merges.json", "r", encoding="utf-8") as f:
    meta = json.load(f)
merges = {tuple(map(int, k.split(','))): v for k, v in meta["merges"].items()}
specTokens = meta["spec_tokens"]
vocabSize = 256 + len(merges) + len(specTokens)


data = np.fromfile("train_data.bin", dtype=np.int64)  # or int32 if you used that
data = torch.tensor(data, dtype=torch.long)

n = int(0.1*len(data))
trainData = data[n:]
valData = data[:n]

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
