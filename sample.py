import argparse
import random
import torch
import torch.nn.functional as F
from made import MADE
import matplotlib.pyplot as plt

@torch.no_grad()
def sample(model, num_sample=1):
    device = model.residual.weight.device
    x = torch.zeros(num_sample, model.input_size, device=device) # (B, D)
    x[:, model.input_order[0]] = random.randint(0, 1)

    index_order = list(enumerate(model.input_order))
    index_order.sort(key=lambda x: x[1])

    for i in range(1, model.input_size):
      idx, _ = index_order[i]
      p = F.sigmoid(model.forward(x)) # (B, D)
      x[:, idx] = torch.bernoulli(p[:, idx])

    return x


def parse_int_list(s):
    try:
        return list(map(int, s.split(',')))
    except ValueError:
        raise argparse.ArgumentTypeError('Invalid comma-separated list of integers')


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-s', type=parse_int_list, help='hidden size, common separated')
  parser.add_argument('-f', help='path of the saved model weight')
  args = parser.parse_args()
  hidden_sizes = args.s
  weight_path = args.f

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = MADE(input_size=784, hidden_sizes=hidden_sizes, output_size=784)
  model.load_state_dict(torch.load(weight_path))
  model.to(device=device)

  samples = sample(model, 20)
  fig, axs = plt.subplots(4, 5, figsize=(6, 4))
  for i, ax in enumerate(axs.flat):
    ax.imshow(samples[i].cpu().numpy().reshape(28, 28), cmap="gray")
    ax.axis("off")
  plt.show()
  plt.savefig('./.data/sample.png', bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
  main()