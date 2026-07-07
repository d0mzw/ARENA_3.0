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

class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()

        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: Tensor) -> Tensor:

        # A batch of 64 images, each 3 channels, 28×28 pixels
        # input.shape = (64, 3, 28, 28)
        # dim_0 = 64, dim_1 = 3, dim_2 = 28, dim_3 = 28
        shape = input.shape

        # Get start & end dims, handling negative indexing for end dim
        # start_dim = 1, end_dim = -1
        start_dim = self.start_dim

        # end_dim = 4 + (-1) = 3
        end_dim = self.end_dim if self.end_dim >= 0 else len(shape) + self.end_dim

        # Get the shapes to the left / right of flattened dims, as well as size of flattened middle
        
        # shape[:1] = (64,)
        shape_left = shape[:start_dim]

        # shape[4:] = ()
        shape_right = shape[end_dim + 1 :]
        
        # shape[1:4] = (3, 28, 28) are the dims we're flattening
        # t.tensor((3, 28, 28)) turns that tuple into a tensor [3, 28, 28]
        # t.prod multiplies all its elements together: 3 * 28 * 28 = 2352
        # .item() pulls that out as int: shape_middle = 2352
        shape_middle = t.prod(t.tensor(shape[start_dim : end_dim + 1])).item()


        # shape_left      = (64,)
        # (shape_middle,) = (2352,)
        # shape_right     = ()
        return t.reshape(input, shape_left + (shape_middle,) + shape_right)

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["start_dim", "end_dim"]])
# %%

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = Flatten()
        self.linear1 = Linear(in_features=28 * 28, out_features=100)
        self.relu = ReLU()
        self.linear2 = Linear(in_features=100, out_features=10)


    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.relu(self.linear1(self.flatten(x))))


tests.test_mlp_module(SimpleMLP)
tests.test_mlp_forward(SimpleMLP)

# %%

# Transform = per-image preprocessing pipeline, applied to every image.
MNIST_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(0.1307, 0.3081),

    ]
)


def get_mnist(trainset_size: int = 10_000, testset_size: int = 1_000) -> tuple[Subset, Subset]:
    """Returns a subset of MNIST training data."""

    # Get original datasets, which are downloaded to "./data" for future use
    mnist_trainset = datasets.MNIST(exercises_dir / "data", train=True, download=True, transform=MNIST_TRANSFORM)
    mnist_testset = datasets.MNIST(exercises_dir / "data", train=False, download=True, transform=MNIST_TRANSFORM)

    # Return a subset of the original datasets
    mnist_trainset = Subset(mnist_trainset, indices=range(trainset_size))
    mnist_testset = Subset(mnist_testset, indices=range(testset_size))

    return mnist_trainset, mnist_testset


mnist_trainset, mnist_testset = get_mnist()

# DataLoader = batches + shuffles the dataset. Dataset gives 1 item; loader gives groups of 64.
# shuffle=True for train (randomize order each epoch), False for test (order irrelevant).
mnist_trainloader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)
mnist_testloader = DataLoader(mnist_testset, batch_size=64, shuffle=False)

# Get the first batch of test data, by starting to iterate over `mnist_testloader`
for img_batch, label_batch in mnist_testloader:
    print(f"{img_batch.shape=}\n{label_batch.shape=}\n")
    break

# Get the first datapoint in the test set, by starting to iterate over `mnist_testset`
for img, label in mnist_testset:
    print(f"{img.shape=}\n{label=}\n")
    break

t.testing.assert_close(img, img_batch[0])
assert label == label_batch[0].item()

# %%
device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
print(device)


# %%
model = SimpleMLP().to(device)

# Number of samples in each batch
batch_size = 128

# An epoch is one complete pass through your entire training dataset
epochs = 10

mnist_trainset, _ = get_mnist()
mnist_trainloader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)

# The optimizer, this is what actually updates the weights. model.parameters() gives it all 
# the trainable weights (this is why nn.Parameter mattered, so the optimizer can find them). 
# Adam is a specific update algorithm (a smart version of gradient descent). lr=1e-3 is the 
# learning rate, how big each update step is.
optimizer = t.optim.Adam(model.parameters(), lr=1e-3)

loss_list = []

for epoch in range(epochs):

    # pbar is tqdm(trainloader), the dataloader wrapped in a progress bar. Each inner 
    # iteration gives one batch: imgs (128, 1, 28, 28) and labels (128,).
    pbar = tqdm(mnist_trainloader)

    for imgs, labels in pbar:
        # Move data to device, perform forward pass
        imgs, labels = imgs.to(device), labels.to(device)

        # Feed the batch through the model, get predictions. logits are the raw output 
        # scores, shape (128, 10), for each of 128 images, 10 scores (one per digit 0-9). 
        # Calling model(imgs) runs the forward methods you built:
        # (Flatten → Linear → ReLU → Linear)
        logits = model(imgs)

        # Calculate loss, perform backward pass
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Update logs & progress bar
        loss_list.append(loss.item())
        pbar.set_postfix(epoch=f"{epoch + 1}/{epochs}", loss=f"{loss:.3f}")

line(
    loss_list,
    x_max=epochs * len(mnist_trainset),
    labels={"x": "Examples seen", "y": "Cross entropy loss"},
    title="SimpleMLP training on MNIST",
    width=700,
)

# %%
print("cuda available:", t.cuda.is_available())   # ROCm reports as 'cuda' in torch
print("device count:", t.cuda.device_count())
print("device name:", t.cuda.get_device_name(0) if t.cuda.is_available() else "none")
# %%

# import time

# def benchmark_matmul(N, iters=20, dtype=t.float16):
#     a = t.randn(N, N, device=device, dtype=dtype)
#     b = t.randn(N, N, device=device, dtype=dtype)

#     for _ in range(5):          # shorter warmup
#         c = a @ b
#     t.cuda.synchronize()

#     start = time.time()
#     for _ in range(iters):
#         c = a @ b
#     t.cuda.synchronize()
#     elapsed = time.time() - start

#     tflops = 2 * N**3 * iters / elapsed / 1e12
#     print(f"N={N:>6}  {str(dtype):>16}  {tflops:>7.1f} TFLOP/s  ({elapsed:.2f}s / {iters} iters)")
#     return tflops

# print("fp16:")
# for N in [4096, 8192, 16384]:
#     benchmark_matmul(N, dtype=t.float16)

# print("\nbf16:")
# for N in [4096, 8192, 16384]:
#     benchmark_matmul(N, dtype=t.bfloat16)

# print("\nfp32 (small size only, it's the slow path):")
# benchmark_matmul(4096, dtype=t.float32)   # small N so it doesn't drag

# print(f"\nPeak VRAM: {t.cuda.max_memory_allocated()/1e9:.1f}GB")

# %%
@dataclass
class SimpleMLPTrainingArgs:
    """
    Defining this class implicitly creates an __init__ method, which sets arguments as below, e.g.
    self.batch_size=64. Any of these fields can also be overridden when you create an instance, e.g.
    SimpleMLPTrainingArgs(batch_size=128).
    """

    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3

def train(args: SimpleMLPTrainingArgs) -> tuple[list[float], SimpleMLP]:
    """
    Trains the model, using training parameters from the `args` object.

    Returns:
        The model, and lists of loss & accuracy.
    """

    model = SimpleMLP().to(device)

    mnist_trainset, mnist_testset = get_mnist()
    mnist_trainloader = DataLoader(mnist_trainset, batch_size=args.batch_size, shuffle=True)
    mnist_testloader = DataLoader(mnist_testset, batch_size=args.batch_size, shuffle=False)

    optimizer = t.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_list = []
    accuracy_list = []
    accuracy = 0.0

    for epoch in range(args.epochs):
        pbar = tqdm(mnist_trainloader)

        for imgs, labels in pbar:
            # Move data to device, perform forward pass
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)

            # Calculate loss, perform backward pass
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Update logs & progress bar
            loss_list.append(loss.item())
            pbar.set_postfix(epoch=f"{epoch + 1}/{args.epochs}", loss=f"{loss:.3f}")

        num_correct_classifications = 0
        for imgs, labels in mnist_testloader:
            imgs, labels = imgs.to(device), labels.to(device)
            with t.inference_mode():
                logits = model(imgs)

            # Compute num correct by comparing argmaxed logits to true labels
            predictions = t.argmax(logits, dim=1)
            num_correct_classifications += (predictions == labels).sum().item()

        # Compute & log total accuracy
        accuracy = num_correct_classifications / len(mnist_testset)
        accuracy_list.append(accuracy)

    return loss_list, accuracy_list, model

args = SimpleMLPTrainingArgs()
loss_list, accuracy_list, model = train(args)

line(
    y=[loss_list, [0.1] + accuracy_list],  # we start by assuming a uniform accuracy of 10%
    use_secondary_yaxis=True,
    x_max=args.epochs * len(mnist_trainset),
    labels={"x": "Num examples seen", "y1": "Cross entropy loss", "y2": "Test Accuracy"},
    title="SimpleMLP training on MNIST",
    width=800,
)

# %%

class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        """
        Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.

        We assume kernel is square, with height = width = `kernel_size`.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        kernel_height = kernel_size
        kernel_width = kernel_size

        sf = 1 / np.sqrt(in_channels * kernel_height * kernel_height)
        self.weight = nn.Parameter(sf * (2 * t.rand(out_channels, in_channels, kernel_height, kernel_width) - 1))


    def forward(self, x: Tensor) -> Tensor:
        """Apply the functional conv2d, which you can import."""
        return t.nn.functional.conv2d(x, self.weight, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        keys = ["in_channels", "out_channels", "kernel_size", "stride", "padding"]
        return ", ".join([f"{key}={getattr(self, key)}" for key in keys])


tests.test_conv2d_module(Conv2d)
m = Conv2d(in_channels=24, out_channels=12, kernel_size=3, stride=2, padding=1)
print(f"Manually verify that this is an informative repr: {m}")

# %%

class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: int, stride: int | None = None, padding: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        """Call the functional version of maxpool2d."""
        return F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        """Add additional information to the string representation of this class."""
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["kernel_size", "stride", "padding"]])

# %%
