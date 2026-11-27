import torch
import torch.nn as nn
from torch.nn import functional as F

# Shape letters used in the comments below:
#   B = batch size       (how many examples go through at once, e.g. 256)
#   T = time / length    (tokens per example: 15 for split, 11 for k1, 9 for k2, 8 for direct)
#   C = channels         (numbers in each token's vector, n_embd = 64)
#   A = answer vocab     (output scores per position: 7, 10, 100 or 1999)

# Model size: held fixed across all formats (paper, Appendix A.3: GPT-2, emb 64, 2 layers, 4 heads)
n_embd = 64   # C: length of the vector that represents each token
n_layer = 2   # how many Blocks are stacked
n_head = 4    # attention heads per Block (each gets head_size = 64 / 4 = 16)
block_size = 16  # longest full sequence is 16 tokens (8 prompt + 8 split answer); also the size of the position table
dropout = 0.0  # not specified in CLAUDE.md: check the paper's Appendix A.3

class Head(nn.Module):
    """ one head of self-attention: lets each position look at earlier positions """

    def __init__(self, head_size):
        super().__init__()
        # each layer reads all 64 numbers of a token and compresses them to head_size (16)
        self.key = nn.Linear(n_embd, head_size, bias=False)    # "what do I contain?"
        self.query = nn.Linear(n_embd, head_size, bias=False)  # "what am I looking for?"
        self.value = nn.Linear(n_embd, head_size, bias=False)  # "what I hand over if you attend to me"
        # lower-triangular matrix of ones: row t has ones for positions 0..t (the past) and zeros for the future
        # register_buffer = saved with the model but not a trainable parameter
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.droupout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        # compute attention scores ("affinities"): how much should position i look at position j?
        #scaled so that the dot products don't grow too large (this scales by C = 64; the usual choice is head_size, but it works)
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,head_size) @ (B,head_size,T) ---> (B,T,T)
        # causal mask: future positions get -inf so softmax turns them into weight 0
        # (position 7, the "=", can see positions 0..7 but never the answer it has to predict)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T) each row now adds up to 1
        wei = self.droupout(wei) # (B,T,T)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,head_size)
        out = wei @ v # (B,T,T) @ (B,T,head_size) ---> (B,T,head_size): each position = weighted mix of the values it attends to
        return out

class MultiHeadAttention(nn.Module):
    """ several heads in parallel; each can learn a different looking pattern """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)  # mixes the heads' outputs back into 64 numbers
        self.droupout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) #concatenating over channel dimension: 4 x (B,T,16) -> (B,T,64)
        return self.droupout(self.proj(out))

class FeedForward(nn.Module):
    """ processes each position on its own (attention moved info BETWEEN positions, this works WITHIN one) """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),   # 64 -> 256
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),   # 256 -> 64
            nn.Dropout(dropout)
        )


    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ one transformer block: attention, then feed-forward. Shape stays (B,T,C) throughout. """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd //n_head  # 64 // 4 = 16 numbers per head
        self.sa = MultiHeadAttention(n_head, head_size)  # sa = self-attention
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)  # layer norm: rescales each token's 64 numbers to a steady range
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x+ self.sa(self.ln1(x)) #residual connections by doing x+ (add the result onto the input, don't replace it)
        x = x+self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
  def __init__(self, n_in, A):
    # n_in = number of input ids the model can read (prompt tokens + answer tokens)
    # A    = number of answer tokens it can output (this is m, the choices per step)
    super().__init__()
    self.token_embedding_table = nn.Embedding(n_in, n_embd)  # one row per input id (prompt + answer tokens)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)  # one row per position 0..15, tells the model WHERE a token is
    self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)])  # the stack of Blocks
    self.ln_f = nn.LayerNorm(n_embd) #final layer norm
    self.lm_head = nn.Linear(n_embd, A)  # one score per answer token only

  def forward(self, idx, targets=None):
    # idx: (B,T) integer ids of the input tokens
    B, T = idx.shape

    tok_emb = self.token_embedding_table(idx) # (B,T,C): each id looked up as a 64-number vector
    pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C): one vector per position
    x = tok_emb + pos_emb # (B,T,C): "this token, at this place" (pos_emb is added to every example in the batch)
    x = self.blocks(x)    # (B,T,C): each position now also holds information from earlier positions
    x = self.ln_f(x)      # (B,T,C)
    logits = self.lm_head(x) # (B,T,A): at every position, one score per answer token = the model's guess for the NEXT token


    if targets is None:
      loss = None
    else:
      # flatten (B,T) into one long list of B*T positions so cross_entropy can compare row by row
      B, T, C = logits.shape  # here C is A, the number of scores per position
      logits = logits.view(B*T, C)   # (B*T, A)
      targets = targets.view(B*T)    # (B*T,) one correct answer id per position, or -100 meaning "skip this position"
      # small when the right token gets a high score; -100 targets are ignored by default
      # computed on CPU because nll_loss segfaults on MPS with torch 2.0.0 + macOS 26 (gradients still flow back to the GPU)
      loss = F.cross_entropy(logits.cpu(), targets.cpu())
    return logits, loss

  def generate(self,idx, max_new_tokens):
    #idx = (B,T) array of ids in the current context (for us: the prompt)
    for _ in range(max_new_tokens):
      #get prediction
      logits, loss = self(idx[:, -block_size:]) #only use last block_size tokens as context
      #focus only on last one: only the final position predicts the next token
      logits = logits[:,-1, :] # becomes (B,A)
      #get probabilities
      probs = F.softmax(logits, dim = -1)
      #sample from probabilities (random pick, weighted by probs; for exact-match eval we will want argmax instead)
      idx_next = torch.multinomial(probs, num_samples = 1)
      #append sampled index to the running sequence
      idx = torch.cat((idx, idx_next), dim = 1)
    return idx
