from vocabulary_aid import *
import requests
from bs4 import BeautifulSoup
import os
import json
import torch
import re

totalDocs = 100

with open("merges.json", "r", encoding="utf-8") as f:
    meta = json.load(f)
merges = {tuple(map(int, k.split(','))): v for k, v in meta["merges"].items()}
specialTokens = meta["spec_tokens"]
#converts text to utf-8, then itterates through it using the known merges list to merge byte pairs.
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

def urlGet(i):
   i = i+1
   return (f"https://www.gutenberg.org/cache/epub/{i}/pg{i}.txt")

def extract_gutenberg_content(text):
    # Extracts the main content from a Project Gutenberg ebook.
    # Returns a cleaned string or None if no valid section is found.

    start_pattern = re.compile(r'\*\*\*\s*START OF (THE )?PROJECT GUTENBERG EBOOK.*?\*\*\*', re.IGNORECASE)
    end_pattern = re.compile(r'\*\*\*\s*END OF (THE )?PROJECT GUTENBERG EBOOK.*?\*\*\*', re.IGNORECASE)

    start_match = start_pattern.search(text)
    end_match = end_pattern.search(text)

    if start_match and end_match:
        start = start_match.end()
        end = end_match.start()
        main_text = text[start:end].strip()
        return main_text
    else:
        # Fallback
        return None
    
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # normalize whitespace
    text = text.strip()
    return text

with open("train_data.bin", "ab") as f:
    for i in range(totalDocs):
        try:
            res = requests.get(urlGet(i))
            soup = BeautifulSoup(res.text, "html.parser")
            text =  soup.get_text(separator=' ', strip=True)
            
            text = clean_text(extract_gutenberg_content(text))

            print(i)
            print(len(text))
            print(text[:100])

            tokens = encode(text)
            tokens.append(specialTokens["<!ENDDOC>"])
            
            tensor = torch.tensor(tokens, dtype=torch.long)
            f.write(tensor.numpy().tobytes())
        except Exception as e:
           print(f"Error with {urlGet(i)}: {e}")
   

