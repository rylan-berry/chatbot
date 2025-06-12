from model import *
from vocabulary_aid import *
import torch
import torch.nn as nn
from torch.nn import functional as F
import json


with open("merges.json", "r", encoding="utf-8") as f:
    merges_json = json.load(f)
merges = {tuple(map(int, k.split(','))): v for k, v in merges_json.items()}

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


m = BigramLanguageModel(len(vocab))
m.load_state_dict(torch.load("model.pt", map_location=device))
m = m.to(device)
m.eval()

text = input("Start the text: ")

context = torch.tensor([encode(text)], dtype=torch.long, device=device)
generated = m.generate(context, maxNewTokens=500)[0].tolist()
print(decode(generated))