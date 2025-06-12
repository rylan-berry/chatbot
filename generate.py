from model import *
import torch
import torch.nn as nn
from torch.nn import functional as F
import json


with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)


stoi = {ch:i for i,ch in enumerate(vocab)}
itos = {i:ch for i,ch in enumerate(vocab)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])


m = BigramLanguageModel(len(vocab))
m.load_state_dict(torch.load("model.pt", map_location=device))
m = m.to(device)
m.eval()

context = torch.zeros((1,1), dtype=torch.long, device=device)
generated = m.generate(context, maxNewTokens=500)[0].tolist()
print(decode(generated))