import random
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
return mask of size (d_in, d_out), where d_in, d_out = len(mk_in), len(mk_out)
"""
def init_mask(mk_in, mk_out, eq=True):
    mk_out = torch.tensor(mk_out).view(-1, 1)  # reshape for broadcasting
    mk_in = torch.tensor(mk_in).view(1, -1)

    if eq:
        mask = mk_out >= mk_in
    else:
        mask = mk_out > mk_in

    return mask.float().T


class MaskedLinear(nn.Linear):
  def __init__(self, in_features, out_features, mask, bias=True):
    super().__init__(in_features, out_features, bias)
    self.register_buffer("mask", mask)

  """
  x: (batch_size, in_features)
  """
  def forward(self, x):
    return F.linear(x, self.weight * self.mask.T, self.bias)


class MADE(nn.Module):

  def __init__(self, input_size, hidden_sizes, output_size, natural_order=True):
    super().__init__()
    assert output_size % input_size == 0
    self.input_size = input_size

    mk = [0]
    mk[0] = [_ for _ in range(input_size)]
    if not natural_order:
      random.shuffle(mk[0])

    for n in hidden_sizes:
      mk.append([random.randint(0, input_size - 2) for _ in range(n)])

    mk.append(mk[0] * (output_size // input_size))

    self.input_order = mk[0]
    dim = [input_size] + hidden_sizes + [output_size]
    layers = []

    for i in range(len(dim)):
      if i == 0: continue
      last = i == len(dim) - 1

      mask = init_mask(mk[i-1], mk[i], eq=(not last))
      layers.append(MaskedLinear(dim[i-1], dim[i], mask))
      layers.append(nn.ReLU())

    layers.pop() # remove ReLU() for the last layer
    self.layers = nn.Sequential(*layers)

    mask_r = init_mask(mk[0], mk[-1], eq=False)
    self.residual = MaskedLinear(input_size, output_size, mask_r, bias=False) # unsure if residual weight should be shared


  """
  x: (batch_size, input_size)
  return logit in shape of (batch_size, output_size)
  """
  def forward(self, x):
    r = self.residual(x)
    return self.layers(x) + r


if __name__ == '__main__':
  """
  check if model satisfies auto-regressive property
  """
  input_d, hidden_n, output_d = 100, 50, 200
  model = MADE(input_d, [hidden_n], output_d)
  data = torch.randint(low=0, high=2, size=(1, input_d)).float()

  for i in range(output_d):
    x = data.clone().detach().requires_grad_(True)
    xh = model(x)
    xh_i = xh[0, i]
    xh_i.backward()

    j = i % input_d
    dep = x.grad.nonzero().T[0,:]
    ok = dep <= j
    ok = ok.all()

    if not ok:
      print(f"error: not autoregressive at dim: {i}")