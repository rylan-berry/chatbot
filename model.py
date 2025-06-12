import torch
import torch.nn as nn
from torch.nn import functional as F

batchSize = 64
blockSize = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'

evalIters = 200
nEmbed = 384
nLayer = 6
nHead = 6
dropout = 0.2





class Head(nn.Module):
    """ one head of slef-attention """
    def __init__(self, headSize):
        super().__init__()
        self.key = nn.Linear(nEmbed, headSize, bias=False)
        self.query = nn.Linear(nEmbed, headSize, bias=False)
        self.value = nn.Linear(nEmbed, headSize, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(blockSize, blockSize)))

        self.dropout = nn.Dropout(dropout)
    

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of slef-attention """

    def __init__(self, numHeads, headSize):
        super().__init__()
        self.heads = nn.ModuleList([Head(headSize) for _ in range(numHeads)])
        self.proj = nn.Linear(numHeads * headSize, nEmbed)
        self.dropout = nn.Dropout(dropout)
    

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity"""

    def __init__(self, nEmbed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nEmbed, 4 * nEmbed),
            nn.ReLU(),
            nn.Linear(4 * nEmbed, nEmbed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, nEmbed, nHead):
        super().__init__()
        headSize = nEmbed // nHead
        self.sa = MultiHeadAttention(nHead, headSize)
        self.ffwd = FeedForward(nEmbed)
        self.ln1 = nn.LayerNorm(nEmbed)
        self.ln2 = nn.LayerNorm(nEmbed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):


    def __init__(self, vocabSize):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocabSize, nEmbed)
        self.position_embedding_table = nn.Embedding(blockSize, nEmbed)
        self.blocks = nn.Sequential(*[Block(nEmbed, nHead=nHead) for _ in range(nLayer)])
        self.ln_f = nn.LayerNorm(nEmbed)
        self.lm_head = nn.Linear(nEmbed, vocabSize)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tokEmb = self.token_embedding_table(idx)
        posEmb = self.position_embedding_table(torch.arange(T, device=device))
        x = tokEmb + posEmb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)



        if targets is None:
            loss = None
        else: 
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, maxNewTokens):

        for _ in range(maxNewTokens):
            idxCond = idx[:, -blockSize:]
            logits, loss = self(idxCond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = -1)

            idxNext = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idxNext), dim=1)
        return idx