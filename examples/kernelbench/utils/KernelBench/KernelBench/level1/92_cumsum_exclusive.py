import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs an exclusive cumulative sum (does not include the current element).

    Parameters:
        dim (int): The dimension along which to perform the exclusive cumulative sum.
    """

    def __init__(self, dim):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x):
        cumsum = torch.cumsum(x.narrow(dim=self.dim, start=0, length=x.size(self.dim)-1), dim=self.dim)
        return torch.cat((torch.zeros_like(x.select(self.dim, 0).unsqueeze(self.dim)), cumsum), dim=self.dim)

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    return [torch.rand(batch_size, *input_shape)]

def get_init_inputs():
    return [dim]
