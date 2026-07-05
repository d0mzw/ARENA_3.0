# %%
import json
import sys
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path

import einops
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from IPython.display import display
from jaxtyping import Float, Int
from PIL import Image
from rich import print as rprint
from rich.table import Table
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm.notebook import tqdm

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part2_cnns"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

MAIN = __name__ == "__main__"

import part2_cnns.tests as tests
import part2_cnns.utils as utils
from plotly_utils import line

# %%

class ReLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        # m = nn.ReLU()
        # return m(x)

        return t.maximum(x, t.tensor(0))

tests.test_relu(ReLU)

# %%
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        """
        A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Each float in the weight and bias tensors are drawn independently from the
        # uniform distribution on the interval: [−1/√Nin, 1/√Nin] where Nin is the
        # number of input

        # Scale factor
        sf = 1 / np.sqrt(in_features)

        # Samples uniform random between [0, 1), with shape (out_features, in_features)
        sample = t.rand(out_features, in_features)

        # Shifts [0, 1) to [0, 2] through 2*samples, then [-1, 1) through (2*samples) - 1
        sample = (2 * sample) - 1

        weight = sf * sample

        self.weight = nn.Parameter(weight)
        self.bias = None

        if bias:
            bias = sf * (2 * t.rand(out_features) - 1)
            self.bias = nn.Parameter(bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (*, in_features)
        Return: shape (*, out_features)
        """
        
        # e.g. in_features=3, out_features=2
        # self.weight.shape = (out_features, in_features) = (2, 3)

        # [[w00, w01, w02],    <- row 0 (produces output 0)
        #  [w10, w11, w12]]    <- row 1 (produces output 1)

        # The output is 2 numbers:
        # out0 = w00*x0 + w01*x1 + w02*x2    (row 0 of weight, dotted with x)
        # out1 = w10*x0 + w11*x1 + w12*x2    (row 1 of weight, dotted with x)

        x = einops.einsum(x, self.weight, "... in_features, out_features in_features -> ... out_features")
        
        if self.bias is not None:
            x += self.bias

        return x


    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


tests.test_linear_parameters(Linear, bias=False)
tests.test_linear_parameters(Linear, bias=True)
tests.test_linear_forward(Linear, bias=False)
tests.test_linear_forward(Linear, bias=True)

# %%
