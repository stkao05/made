import argparse
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from made import MADE


class MINST(Dataset):
  def __init__(self, data):
    self.data = data

  def __getitem__(self, index):
    return self.data[index]

  def __len__(self):
    return len(self.data)


def nll(x, p):
  return -1 * ((x * p.log()) + (1 - x) * (1 - p).log()).sum(dim=1)


def nll_loss(x, p, smoothing=None):
  if smoothing:
    x = x * (1 - smoothing) + (1 - x) * smoothing

  return -1 * ((x * p.log()) + (1 - x) * (1 - p).log()).sum(dim=1).mean()


@torch.no_grad()
def eval_split(ds, model, device):
  dl = DataLoader(ds, batch_size=128)
  lossi = []

  for x in dl:
    x = x.to(device=device)
    logits = model(x)
    p = F.sigmoid(logits)

    loss = nll_loss(x, p)
    lossi.append(loss.item())

  return torch.tensor(lossi).mean().item()


def train_loop(model, optim, train_ds, valid_ds, epoch, device):
  train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

  for i in range(epoch):
    start = time.time()

    for x in train_dl:
      x = x.to(device=device)
      optim.zero_grad()
      logits = model(x)
      p = F.sigmoid(logits)

      loss = nll_loss(x, p, smoothing=0.005)
      loss.backward()
      norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
      optim.step()

    tr_loss = eval_split(train_ds, model, device)
    va_loss = eval_split(valid_ds, model, device)
    end = time.time()
    print(f"epoch {i+1} | train: {tr_loss:.2f} | valid: {va_loss:.2f} | norm: {norm:.2f} | time: {end - start:.2f}")


def parse_int_list(s):
    try:
        return list(map(int, s.split(',')))
    except ValueError:
        raise argparse.ArgumentTypeError('Invalid comma-separated list of integers')


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-s', type=parse_int_list, help='hidden size, common separated')
  parser.add_argument('-e', type=int, help='epoch', default=5)
  parser.add_argument('-f', help='output path of the model weight', default=".data/mnist_made.py")

  args = parser.parse_args()
  hidden_sizes = args.s
  epoch = args.e
  output_path = args.f

  with np.load('.data/binarized_mnist.npz') as f:
    train_ds = MINST(torch.from_numpy(f["train_data"]).float())
    valid_ds = MINST(torch.from_numpy(f["valid_data"]).float())

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = MADE(input_size=784, hidden_sizes=hidden_sizes, output_size=784)
  model.to(device=device)
  optim = torch.optim.Adam(model.parameters())

  train_loop(model, optim, train_ds, valid_ds, epoch=epoch, device=device)
  torch.save(model.state_dict(), output_path)


if __name__ == '__main__':
  main()