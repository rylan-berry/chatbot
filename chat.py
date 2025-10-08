from model import *
from vocabulary_aid import *
import torch
import torch.nn as nn
from torch.nn import functional as F
import json

with open("merges.json", "r", encoding="utf-8") as f:
    meta = json.load(f)
merges = {tuple(map(int, k.split(','))): v for k, v in meta["merges"].items()}
specTokens = meta["spec_tokens"]

#creates a dicitonary where a given token is turned into it's bytes
vocab = {idx: bytes([idx]) for idx in range(256)}
for(p0,p1), idx in merges.items():
  vocab[idx] = vocab[p0] + vocab[p1]
vocab[specTokens["<!ENDDOC>"]] = b"<!ENDDOC>"
vocab[specTokens["<!ENDPROMPT>"]] = b"<!ENDPROMPT>"

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
m.load_state_dict(torch.load("finetuned_model.pt", map_location=device))
m = m.to(device)
m.eval()


inp = input("User: ")
context = torch.tensor([],dtype=torch.long,device=device)
pGen = torch.tensor([],dtype=torch.long,device=device)
while(inp != ""):
    inp += " "
    encInp =encode(inp)
    encInp.append(specTokens["<!ENDPROMPT>"])
    encInp = torch.tensor(encInp,dtype=torch.long,device=device)
    context = torch.cat((context, pGen, torch.tensor([specTokens["<!ENDDOC>"]],dtype=torch.long,device=device), encInp),dim=0)
    inLen = len(context)

    generated = m.generate(context, maxNewTokens=512)[0]
    pGen = generated[inLen:]
    print("Bot: " + decode(pGen.tolist())) #print out should NOT include context
    inp = input("User: ")
print("End of conversation")