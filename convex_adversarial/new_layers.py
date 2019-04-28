import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def apply_on_last_n_dim(tensor, fn, n):
  initial_sizes = tensor.size()
  new_batch_size = int(np.prod(initial_sizes[:-n]))
  new_shape = torch.Size([new_batch_size]) + initial_sizes[-n:]
  tensor = tensor.contiguous().view(*new_shape)
  tensor = fn(tensor)
  output_size = initial_sizes[:-n] + tensor.size()[1:]
  return tensor.view(output_size)

class Window(nn.Module):

  def __init__(self, i, j, size, input_shape):
    super(Window, self).__init__()
    self.i = i
    self.j = j
    self.size = size
    self.input_shape = input_shape

  def forward(self, x):
    fn = lambda t: t[:, :, self.i : self.i + self.size, self.j: self.j + self.size]
    return apply_on_last_n_dim(x, fn, 3)

class AddBias(nn.Module):

  def __init__(self, bias):
    super(AddBias, self).__init__()
    self.bias = bias

  def forward(self, x):
    return x + self.bias


class ExtractCliques(nn.Module):

  def __init__(self, clique_size, input_size):
    super(ExtractCliques, self).__init__()
    self.clique_size = clique_size
    self.input_size = input_size

  def forward(self, x):
    def fn(z):
      batch_size = z.size(0)
      patches = F.unfold(z, self.clique_size, padding=0)
      return patches.view(batch_size, self.clique_size, self.clique_size, -1) \
                    .permute(0, 3, 1, 2).contiguous()
    return apply_on_last_n_dim(x, fn, n=3)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

