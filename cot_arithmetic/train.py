
import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_layer = 6
n_head = 6
dropout = 0.2

torch.manual_seed(1337)


#wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in characters: ", len(text))

print(text[:1000])

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

#tokenizing all the words
stoi ={ ch:i for i,ch in enumerate(chars)} #iterate thru chars and map each char to its index in chars
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])


data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9*len(data)) # training data is 90% of total
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,)) #batch_size number of random offsets selcted between 0 and data length - block_size
  x = torch.stack([data[i:i+block_size] for i in ix]) #for each offset in ix (size 4 rn), 1d tensor is made and stacked for the tokens for chars at index i to i+block_size
  y = torch.stack([data[i+1:i+block_size + 1] for i in ix]) #same as above but offset forward by 1 to be the targets
  x, y = x.to(device), y.to(device)
  return x, y

#estimate_loss
@torch.no_grad()
def estimate_loss():
  out = {}
  m.eval() #set model to evaluation mode
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      logits, loss = m(X,Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  m.train() #set model back to training mode
  return out


model = BigramLanguageModel()
m = model.to(device)


optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate)


for iter in range(max_iters):
  
  if iter % eval_interval == 0:
    losses = estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
  
  xb,yb = get_batch('train')

  logits, loss = m(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()


context = torch.zeros((1,1), dtype = torch.long, device = device) #initial token
print(decode(m.generate(context,max_new_tokens = 500)[0].tolist()))


