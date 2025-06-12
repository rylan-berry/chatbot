from model import *
from vocabulary_aid import *
import torch
import torch.nn as nn
from torch.nn import functional as F
import json

#vocab training vars
vocabSize = 512
numMerges = vocabSize - 256 #256 b/c there's 256 options for standard utf-8

#model training vars
maxIters = 5000
evalInterval = 500
lRate = 3e-3

torch.manual_seed(1337)
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

txt = text[:int(0.1*len(text))] #uses 10% of input as vocab training
#utf-8 conversion
tokens = txt.encode("utf-8")
tokens = list(map(int, tokens))

#itterates throught the list for a given ammount of times, places all those merges into a merges array
ids = list(tokens)
merges = {} #(int, int)->int
for i in range(numMerges):
  stats = get_stats(ids)
  pair = max(stats, key=stats.get)
  idx = 256 + i
  print(f"merging {pair} into new token {idx}")
  ids = merge(ids, pair, idx)
  merges[pair] = idx

#exporting merges
merges_json = {f"{a},{b}": idx for (a, b), idx in merges.items()}
with open("merges.json", "w", encoding="utf-8") as f:
    json.dump(merges_json, f, indent=2)

#creates a dicitonary where a given token is turned into it's bytes
vocab = {idx: bytes([idx]) for idx in range(256)}
for(p0,p1), idx in merges.items():
  vocab[idx] = vocab[p0] + vocab[p1]

#uses vocab to unmerge each item in ids, which is then uses the utf-8 decode to turn it into text
def decode(ids):
  tokens = b"".join(vocab[idx] for idx in ids)
  text = tokens.decode("utf-8", errors="replace")
  return text

#converts test to utf-8, then itterates through it using the known merges list to merge byte pairs.
def encode(text):
  tokens = list(text.encode("utf-8"))
  while len(tokens) >= 2:
    stats = get_stats(tokens)
    pair = min(stats, key=lambda p: merges.get(p, float("inf")))
    if pair not in merges:
      break # no more merges
    idx = merges[pair]
    tokens = merge(tokens, pair, idx)
  return tokens

#vocab = sorted(list(set(text)))
#vocabSize = len(vocab)

#stoi = {ch:i for i,ch in enumerate(vocab)}
#itos = {i:ch for i,ch in enumerate(vocab)}
#encode = lambda s: [stoi[c] for c in s]
#decode = lambda l: ''.join([itos[i] for i in l])

#with open("vocab.json", "w", encoding="utf-8") as f:
#    json.dump(vocab, f, ensure_ascii=False, indent=2)

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
