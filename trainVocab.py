from vocabulary_aid import *
import json

#vocab training vars
vocabSize = 4352
numMerges = vocabSize - 256 #256 b/c there's 256 bytes in standard utf-8
specialTokens = {
   "<!ENDDOC>":vocabSize,
   "<!ENDPROMPT>":vocabSize+1,
}

with open('inputVocab.txt', 'r', encoding='utf-8') as f:
    text = f.read()

tokens = text.encode("utf-8")
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
    json.dump({
       "merges":merges_json, 
       "spec_tokens":specialTokens,
       }, f, indent=2)