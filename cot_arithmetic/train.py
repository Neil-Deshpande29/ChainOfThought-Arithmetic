from cot_arithmetic.tokenizer import Tokenizer
from cot_arithmetic.data import sample_problems, make_example
from cot_arithmetic.model import GPT
import torch
from torch import nn
from torch.nn import functional as F

#hyperparameters (paper, Appendix A.3). Model size lives in model.py.
fmt = "k1"  # "split", "k1", "k2" or "direct": the only thing that changes between runs
batch_size = 256
learning_rate = 2.5e-3
epochs = 20
warmup_frac = 0.1  # 10% warmup, then linear decay
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
n_train = 10_000  # PLACEHOLDER: size of D is still an open decision
n_test = 2_000  # PLACEHOLDER
seed = 0  # same seed gives the same D for every format

torch.manual_seed(seed)

# D: lists of (a, b) operand pairs. train and test never overlap, and the same seed gives the same D for every format
train_data, val_data = sample_problems(n_train, n_test, seed=seed)
tok = Tokenizer(fmt)  # builds the id maps for this format (answer ids 0..A-1, prompt ids after)

def build_tensors(pairs):
    # for each (a, b): make_example gives (prompt tokens, answer tokens) as strings,
    # encode_example turns them into (x, y) lists of ids (y has -100 at prompt positions)
    examples = [tok.encode_example(*make_example(a, b, fmt)) for a, b in pairs]
    xs, ys = zip(*examples)  # list of N (x, y) pairs -> all the xs, all the ys
    return torch.tensor(xs), torch.tensor(ys)  # each (N, T): N problems, T tokens each (T is fixed within a format)

X_train, Y_train = build_tensors(train_data)  # (n_train, T)
X_test, Y_test = build_tensors(val_data)      # (n_test, T)



def get_batch(X, Y):
    # randperm shuffles the row numbers 0..N-1; the first batch_size of them are unique random rows
    idx = torch.randperm(X.shape[0])[:batch_size]
    # X[idx] picks those rows: (N, T) -> (B, T), B = batch_size. .to(device) puts them where the model lives
    return X[idx].to(device), Y[idx].to(device)

# n_in embedding rows (all ids the model reads), A output scores (answer tokens only)
model = GPT(tok.n_in, tok.A).to(device)


# Learning rate schedule: linear warmup from 0 up to learning_rate over the first 10% of steps,
# then linear decay back down to 0 at the last step
steps_per_epoch = n_train // batch_size  # batches needed to see (about) every training problem once
total_steps = epochs * steps_per_epoch
warmup_steps = int(warmup_frac * total_steps)

def get_lr(step):
    if step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps  # ramp up
    # fraction of the decay phase completed: 0 right after warmup, 1 at the last step
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return learning_rate * (1 - progress)  # ramp down

@torch.no_grad()  # no gradients needed when we only measure
def test_loss():
    model.eval()  # eval mode (turns dropout off)
    _, loss = model(X_test.to(device), Y_test.to(device))  # the whole test set in one pass: (n_test, T)
    model.train()  # back to train mode
    return loss.item()

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(total_steps):
    # set this step's learning rate on the optimizer
    lr = get_lr(step)
    for group in optimizer.param_groups:
        group['lr'] = lr

    xb, yb = get_batch(X_train, Y_train)  # (B, T) inputs and targets
    _, loss = model(xb, yb)               # loss only counts answer positions (targets of -100 are skipped)
    optimizer.zero_grad(set_to_none=True) # clear the previous step's gradients
    loss.backward()                       # compute gradients
    optimizer.step()                      # update the weights

    if step % steps_per_epoch == 0 or step == total_steps - 1:  # print once per epoch
        print(f"step {step}/{total_steps}  epoch {step // steps_per_epoch}  lr {lr:.5f}  "
              f"train loss {loss.item():.4f}  test loss {test_loss():.4f}")

@torch.no_grad()  # no gradients needed when we only measure
def exact_match(x, y):
      model.eval()
      seq = x[:, :8].to(device)             # start: the prompts, (N, 8)
      true = y[:, 7:].to(device)            # the correct answers, (N, n_ans)
      n_ans = true.shape[1]
      
      for i in range(n_ans):
        logits, _ = model(seq)                 # (N, 8+i, A)
        last = logits[:, -1, :]                     # (N, A): the last position's scores
        next_id = torch.argmax(last, dim=-1)       # (N,) the predicted
        seq = torch.cat((seq, next_id[:, None]), dim=1)  # append the predicted id to the sequence
  
      pred = seq[:, 8:]                      # (N, n_ans) the predicted answer ids
      correct = (pred == true).all(dim=1)    # (N,) True
      model.train()  # back to train mode
      return correct.float().mean().item()    # fraction of problems that were exactly correct

print(f"exact-match accuracy on the test set ({fmt}): {exact_match(X_test, Y_test):.4f}")
