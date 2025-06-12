#merges all the pairs that are the equal to input pair from ids, and is merged into index idx
def merge(ids, pair, idx):
  newids = []
  i = 0
  while i < len(ids):
    if i < len(ids)-1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      newids.append(idx)
      i+=2
    else:
      newids.append(ids[i])
      i+=1
  return newids


#gets all the occurances of pairs
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
      counts[pair] = counts.get(pair, 0)+1
    return counts