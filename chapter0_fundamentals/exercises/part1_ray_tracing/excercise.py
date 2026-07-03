# %%
import os
import sys
from functools import partial
from pathlib import Path
from typing import Callable

import einops
import plotly.express as px
import plotly.graph_objects as go
import torch as t
from IPython.display import display
from ipywidgets import interact
from jaxtyping import Bool, Float
from torch import Tensor
from tqdm import tqdm

# import plotly.io as pio
# pio.renderers.default = "vscode"

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part1_ray_tracing"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part1_ray_tracing.tests as tests
from part1_ray_tracing.utils import (
    render_lines_with_plotly,
    setup_widget_fig_ray,
    setup_widget_fig_triangle,
)
from plotly_utils import imshow

MAIN = __name__ == "__main__"

# %%

def make_rays_1d(num_pixels: int, y_limit: float) -> Tensor:
    """
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is
        also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains
        (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    """
    rays = t.zeros((num_pixels, 2, 3), dtype=t.float32)
    rays[:, 1, 1] = t.linspace(-y_limit, y_limit, num_pixels)
    rays[:, 1, 0] = 1
    return rays


rays1d = make_rays_1d(9, 10.0)
fig = render_lines_with_plotly(rays1d)
# %%

# %%
# ------------------------------------------------------------
#  THE RAY
# ------------------------------------------------------------
#
#     R(u) = O + u * D
#
#        O = origin     = rays[:, 0, :]   = [0, 0, 0]
#        D = direction  = rays[:, 1, :]   = [1, y, 0]
#        u = how far along the ray
#
#     u = 0  -> O         (start)
#     u = 1  -> O + D      (one step out)
#     u = 2  -> O + 2D     (further)
#
#     RULE:  u >= 0
#
#
# ------------------------------------------------------------
#  THE SEGMENT
# ------------------------------------------------------------
#
#     P(v) = L1 + v * (L2 - L1)
#
#        L1        = one endpoint  (the start)
#        L2 - L1   = direction from L1 toward L2
#        v         = how far along the segment
#
#     v = 0    -> L1        (one end)
#     v = 0.5  -> midpoint
#     v = 1    -> L2        (other end)
#
#     RULE:  0 <= v <= 1
#
#
# ------------------------------------------------------------
#  SIDE BY SIDE
# ------------------------------------------------------------
#
#                    RAY                 SEGMENT
#     start          O = rays[:,0]       L1
#     direction      D = rays[:,1]       L2 - L1
#     param          u  in [0, inf)      v  in [0, 1]
#     equation       O + u*D             L1 + v*(L2 - L1)
#
#
# ------------------------------------------------------------
#  STEP 1:  WHERE DO THEY CROSS?
# ------------------------------------------------------------
#
#  They cross where the ray point equals the segment point:
#
#       O + u*D  =  L1 + v*(L2 - L1)
#
#  Move both unknowns (u, v) to the left:
#
#       u*D  -  v*(L2 - L1)  =  L1 - O
#
#  This is a linear system  A @ [u, v] = b  :
#
#       A = [ D , -(L2 - L1) ]      (the two directions as columns)
#       b =  L1 - O
#
#  Solve for [u, v]  (e.g. t.linalg.solve(A, b)).
#
#
# ------------------------------------------------------------
#  STEP 2:  IS THE CROSSING REAL?
# ------------------------------------------------------------
#
#       u >= 0          hit is in FRONT of the ray (not behind the origin)
#       0 <= v <= 1     hit is BETWEEN the segment endpoints (not off the ends)
#
#  Both true   -> the ray HITS the segment.
#  Either false-> the infinite lines crossed, but the real ray/segment do NOT.
# ============================================================

print(f"{rays1d=}")

def intersect_ray_1d(ray: Float[Tensor, "points dims"], segment: Float[Tensor, "points dims"]) -> bool:
    """
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    """
    # print(ray.shape)
    assert ray.shape[0] == 2    # n_points
    assert ray.shape[1] == 3    # n_dim

    # print(segment.shape)
    assert segment.shape[0] == 2    # n_points
    assert segment.shape[1] == 3    # n_dim
    
    # print(f"{ray=}")
    # print(f"{segment=}")

    O = ray[0][:2]
    D = ray[1][:2]
    # print(f"{O=}")
    # print(f"{D=}")

    L1 = segment[0][:2]
    L2 = segment[1][:2]
    # print(f"{L1=}")
    # print(f"{L2=}")

    matrix = t.stack([D, -(L2 - L1)], dim=1)
    # print(matrix)

    try:
        # matrix @ [u, v] = L1 - O
        sol = t.linalg.solve(matrix, L1 - O)
    except RuntimeError:
        return False
    
    u = sol[0].item()
    v = sol[1].item()
    return u >= 0 and 0 <= v and v <= 1

test_ray = t.tensor([
    [  0.0000,   0.0000,   0.0000],
    [  1.0000, -10.0000,   0.0000]
])

# print(f"{test_rays.shape=}")
# print(f"n_points={test_rays.shape[1]}")
# print(f"n_dim={test_rays.shape[2]}")


test_segment = t.tensor([
    [   1.0000,  -10.0000,   0.0],
    [   1.0000,   10.0000,   0.0]
])

# print(f"{test_segments.shape=}")
# print(f"n_points={test_segments.shape[1]}")
# print(f"n_dim={test_segments.shape[2]}")

intersect_ray_1d(test_ray, test_segment)

tests.test_intersect_ray_1d(intersect_ray_1d)
tests.test_intersect_ray_1d_special_case(intersect_ray_1d)

# %%

# Elementwise Logical Operations on Tensors
x = t.tensor([True, False, True, False])
y = t.tensor([True, True, False, False])

# AND
assert (x & y).equal(t.tensor([True, False, False, False]))

# OR
assert (x | y).equal(t.tensor([True, True, True, False]))

# NOT
assert (~x).equal(t.tensor([False, True, False, True]))

# Combining two conditions on the same tensor.
v = t.tensor([-0.5, 0.3, 0.8, 1.5])
assert ((v >= 0) & (v <= 1)).equal(t.tensor([False, True, True, False]))

try:
    (v >= 0) and (v <= 1)
except RuntimeError as e:
    print(f"Using 'and' on tensors raises: {e}")

# Operator precedence
v = t.tensor([1, 2, 3, 4])
assert (~(v > 2)).equal(t.tensor([True, True, False, False]))
assert not (~v > 2).equal(t.tensor([True, True, False, False]))

# %%

# einops

x = t.randn(4, 3)
print(x)

x_repeat = einops.repeat(x, 'a b -> a b c', c=2)
assert x_repeat.shape == (4, 3, 2)
print(x_repeat)

t.testing.assert_close(x_repeat[:, :, 0], x)
t.testing.assert_close(x_repeat[:, :, 1], x)

# %%

# Logical Reductions

x = t.tensor([
    [True,  False, False],
    [False, False, False],
    [True,  True,  True],
])

# over each row (left to right)
assert x.any(dim=1).equal(t.tensor([True, False, True]))
assert x.all(dim=1).equal(t.tensor([False, False, True]))

# over each colmn (top to bottom)
assert x.any(dim=0).equal(t.tensor([True, True, True]))
assert x.all(dim=0).equal(t.tensor([False, False, False]))

# over everything
assert x.any().equal(t.tensor(True))
assert x.all().equal(t.tensor(False))

# %%

# Broadcasting

B = t.ones(4, 3, 2)
# print(B)

A = t.ones(3, 2)
# print(A)

C = A + B
print(f"{C.shape=}")

print(f"{C=}")


# %%

# Indexing

D = t.ones(2)
E = t.zeros(3, 2)

print(f"{D=}")
print(f"{E=}")

# %%

print(E[[True, False, True], :].shape)
E[[True, False, True], :] = D
print(E)

# %%

D = t.ones(2)
F = t.zeros(2, 2, 2) 

print(f"{D=}")
print(f"{F=}")
# %%

print(F[[[True, True], [False, True]], :].shape)
print(F[[[True, True], [False, True]], :])

F[[[True, True], [False, True]], :] = D
print(F)
# %%

def intersect_rays_1d(
    rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]
) -> Bool[Tensor, " nrays"]:
    """
    For each ray, return True if it intersects any segment.
    """
    n_rays = rays.shape[0]
    n_segments = segments.shape[0]

    # print(f"{n_rays=}")
    # print(f"{n_segments=}")

    O = rays[:, 0, :2]
    D = rays[:, 1, :2]
    L1 = segments[:, 0, :2]
    L2 = segments[:, 1, :2]

    # print(f"{O=}")
    # print(f"{D=}")
    # print(f"{L1=}")
    # print(f"{L2=}")

    O = einops.repeat(O, "n_rays d -> n_rays n_segments d", n_segments=n_segments)
    D = einops.repeat(D, "n_rays d -> n_rays n_segments d", n_segments=n_segments)
    L1 = einops.repeat(L1, "n_segments d -> n_rays n_segments d", n_rays=n_rays)
    L2 = einops.repeat(L2, "n_segments d -> n_rays n_segments d", n_rays=n_rays)

    # print(f"{O=}")
    # print(f"{D=}")
    # print(f"{L1=}")
    # print(f"{L2=}")

    A = t.stack([D, -(L2 - L1)], dim=-1)

    dets = t.linalg.det(A)
    singular = dets.abs() < 1e-8
    A[singular] = t.eye(2)

    sol = t.linalg.solve(A, L1 - O)
    u = sol[..., 0]
    v = sol[..., 1]

    intersects = (u >= 0) & (v >= 0) & (v <= 1) & (~singular)
    return intersects.any(dim=1)

# print(f"{rays1d=}")

test_segments = t.tensor([
    [[1.0,  -10.0, 0.0],
     [1.0,   10.0, 0.0]],

    [[2.0,   -5.0, 0.0],
     [2.0,    5.0, 0.0]],
])

intersect_rays_1d(rays1d, test_segments)
tests.test_intersect_rays_1d(intersect_rays_1d)
tests.test_intersect_rays_1d_special_case(intersect_rays_1d)


# %%

y = t.tensor([1, 2])
print(einops.repeat(y, "y -> (y z)", z=3))

z = t.tensor([5, 6, 7])
print(einops.repeat(z, "z -> (y z)", y=2))

# %%

def make_rays_2d(num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float) -> Float[Tensor, "nrays 2 3"]:
    """
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    """

    n_pixels = num_pixels_y * num_pixels_z
    
    y_list = t.linspace(-y_limit, y_limit, num_pixels_y)
    print(f"{y_list=}")

    z_list = t.linspace(-z_limit, z_limit, num_pixels_z)
    print(f"{z_list=}")

    rays = t.zeros((n_pixels, 2, 3), dtype=t.float32)
    
    # x how far ray shoots out
    rays[:, 1, 0] = 1
    
    y_grid = einops.repeat(y_list, "y -> (y z)", z=num_pixels_z)
    print(y_grid)

    rays[:, 1, 1] = y_grid

    z_grid = einops.repeat(z_list, "z -> (y z)", y=num_pixels_y)
    print(z_grid)

    rays[:, 1, 2] = z_grid
    
    print(rays)

    return rays


make_rays_2d(2, 3, 0.3, 0.3)
rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
render_lines_with_plotly(rays_2d)

# %%

one_triangle = t.tensor([[0, 0, 0], [4, 0.5, 0], [2, 3, 0]])
A, B, C = one_triangle
x, y, z = one_triangle.T

fig: go.FigureWidget = setup_widget_fig_triangle(x, y, z)
display(fig)


@interact(u=(-0.5, 1.5, 0.01), v=(-0.5, 1.5, 0.01))
def update(u=0.0, v=0.0):
    P = A + u * (B - A) + v * (C - A)
    fig.update_traces({"x": [P[0]], "y": [P[1]]}, 2)

# %%

Point = Float[Tensor, "points=3"]

def triangle_ray_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    """
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the ray and the triangle intersect.
    """
    print(f"{A=}")
    print(f"{A.shape=}")
    print(f"{B=}")
    print(f"{B.shape=}")
    print(f"{C=}")
    print(f"{C.shape=}")

    print(f"{O=}")
    print(f"{O.shape=}")
    print(f"{D=}")
    print(f"{D.shape=}")

    matrix = t.stack([-D, (B - A), (C - A)], dim=1)
    print(matrix)

    try:
        sol = t.linalg.solve(matrix, O - A)
    except RuntimeError:
        return False
    print(sol)

    s = sol[0]
    u = sol[1]
    v = sol[2]

    return ((0 <= u) & (0 <= v) & ((u + v) <= 1) & (s >= 0)).item()

tests.test_triangle_ray_intersects(triangle_ray_intersects)

# %%
