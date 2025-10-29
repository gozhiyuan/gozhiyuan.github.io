---
layout: post
title: Triton Introduction 💻
subtitle: 
categories: Large-Language-Model GPU-Acceleration
tags: [Blog]
banner: "/assets/images/banners/yuanpang-wa-iceburg2.jpg"
---


# Triton Introduction

Here we use Triton to implement the weighted sum kernel (both forward and backward pass) as an example. The implementation is taking from the [assignment 2](https://github.com/stanford-cs336/assignment2-systems/blob/main/cs336_spring2025_assignment2_systems.pdf) of the Stanford CS336 lecture.

```python
import triton
import triton.language as tl
import torch
from einops import rearrange
from triton import cdiv
import time


@triton.jit
def weighted_sum_fwd(
    x_ptr, weight_ptr, # Input pointers
    output_ptr, # Output pointer
    x_stride_row, x_stride_dim, # Strides tell us how to move one element in each axis of a tensor
    weight_stride_dim, # Likely 1
    output_stride_row, # Likely 1
    ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr, # Tile shapes must be known at compile time
):
    # Each instance will compute the weighted sum of a tile of rows of x.
    # `tl.program_id` gives us a way to check which thread block we're running in
    row_tile_idx = tl.program_id(0)

    # Block pointers give us a way to select from an ND region of memory
    # and move our selection around.
    # The block pointer must know:
    # - The pointer to the first element of the tensor
    # - The overall shape of the tensor to handle out-of-bounds access
    # - The strides of each dimension to use the memory layout properly
    # - The ND coordinates of the starting block, i.e., "offsets"
    # - The block shape to use load/store at a time
    # - The order of the dimensions in memory from major to minor
    # axes (= np.argsort(strides)) for optimizations, especially useful on H100

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(ROWS, D,),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(ROWS,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    # Initialize a buffer to write to
    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        # Load the current block pointer
        # Since ROWS_TILE_SIZE might not divide ROWS, and D_TILE_SIZE might not divide D,
        # we need boundary checks for both dimensions
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero") # (ROWS_TILE_SIZE, D_TILE_SIZE)
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero") # (D_TILE_SIZE,)

        # Compute the weighted sum of the row.
        output += tl.sum(row * weight[None, :], axis=1)

        # Move the pointers to the next tile.
        # These are (rows, columns) coordinate deltas
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE)) # Move by D_TILE_SIZE in the last dimension
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,)) # Move by D_TILE_SIZE

    # Write output to the output block pointer (a single scalar per row).
    # Since ROWS_TILE_SIZE might not divide ROWS, we need boundary checks
    tl.store(output_block_ptr, output, boundary_check=(0,))


@triton.jit
def weighted_sum_backward(
    x_ptr, weight_ptr, # Input
    grad_output_ptr, # Grad input
    grad_x_ptr, partial_grad_weight_ptr, # Grad outputs
    stride_xr, stride_xd,
    stride_wd,
    stride_gr,
    stride_gxr, stride_gxd,
    stride_gwb, stride_gwd,
    NUM_ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)
    n_row_tiles = tl.num_programs(0)

    # Inputs
    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr,
        shape=(NUM_ROWS,), strides=(stride_gr,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D,), strides=(stride_xr, stride_xd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,), strides=(stride_wd,),
        offsets=(0,), block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,
        shape=(NUM_ROWS, D,), strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    partial_grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr,
        shape=(n_row_tiles, D,), strides=(stride_gwb, stride_gwd),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0),
    )

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,), padding_option="zero") # (ROWS_TILE_SIZE,)

        # Outer product for grad_x
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero") # (D_TILE_SIZE,)
        grad_x_row = grad_output[:, None] * weight[None, :]
        tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0, 1))

        # Reduce as many rows as possible for the grad_weight result
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero") # (ROWS_TILE_SIZE, D_TILE_SIZE)
        grad_weight_row = tl.sum(row * grad_output[:, None], axis=0, keep_dims=True)
        tl.store(partial_grad_weight_block_ptr, grad_weight_row, boundary_check=(1,)) # Never out of bounds for dim 0

        # Move the pointers to the next tile along D
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE))
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))


class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # Cache x and weight to be used in the backward pass, when we
        # only receive the gradient wrt. the output tensor, and
        # need to compute the gradients wrt. x and weight.
        D, output_dims = x.shape[-1], x.shape[:-1]  # Reshape input tensor to 2D
        input_shape = x.shape
        x = rearrange(x, "... d -> (...) d")

        ctx.save_for_backward(x, weight)

        assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

        ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16 # Roughly 16 loops through the embedding dimension
        ctx.ROWS_TILE_SIZE = 16 # Each thread processes 16 batch elements at a time
        ctx.input_shape = input_shape

        # Need to initialize empty result tensor. Note that these elements are not necessarily 0!
        y = torch.empty(output_dims, device=x.device)

        # Launch our kernel with n instances in our 1D grid.
        n_rows = y.numel()
        weighted_sum_fwd[(cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
            x, weight,
            y,
            x.stride(0), x.stride(1),
            weight.stride(0),
            y.stride(0),
            ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE, D_TILE_SIZE=ctx.D_TILE_SIZE,
        )

        return y.view(input_shape[:-1])


    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        ROWS_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE # These don't have to be the same
        n_rows, D = x.shape

        # Our strategy is for each thread block to first write to a partial buffer,
        # then we reduce over this buffer to get the final gradient.
        partial_grad_weight = torch.empty((cdiv(n_rows, ROWS_TILE_SIZE), D), device=x.device, dtype=x.dtype)
        grad_x = torch.empty_like(x)

        weighted_sum_backward[(cdiv(n_rows, ROWS_TILE_SIZE),)](
            x, weight,
            grad_out,
            grad_x, partial_grad_weight,
            x.stride(0), x.stride(1),
            weight.stride(0),
            grad_out.stride(0),
            grad_x.stride(0), grad_x.stride(1),
            partial_grad_weight.stride(0), partial_grad_weight.stride(1),
            NUM_ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE, D_TILE_SIZE=D_TILE_SIZE,
        )
        grad_weight = partial_grad_weight.sum(axis=0)
        return grad_x, grad_weight


def triton_weighted_sum(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute weighted sum along last dimension using Triton autograd function.

    x: (..., D)
    weight: (D,)
    returns: (...,)
    """
    return WeightedSumFunc.apply(x, weight)


def torch_weighted_sum(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Reference PyTorch implementation of weighted sum along last dimension."""
    return torch.tensordot(x, weight, dims=([-1], [0]))


def _benchmark(fn, *args, warmup: int = 5, iters: int = 50) -> float:
    """Benchmark a CUDA function returning the average milliseconds per call."""
    # Warm-up
    for _ in range(warmup):
        out = fn(*args)
        if torch.is_tensor(out):
            out = out.sum()
        torch.cuda.synchronize()
    # Timed
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
        if torch.is_tensor(out):
            out = out.sum()
        torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) * 1000.0 / iters


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device != 'cuda':
        print("CUDA not available; Triton kernels require CUDA. Exiting.")
        return

    # Problem sizes
    ROWS = 1 << 16  # 65,536 rows
    D = 1024        # embedding dim

    # Inputs
    x = torch.randn(ROWS, D, device=device, dtype=torch.float32)
    weight = torch.randn(D, device=device, dtype=torch.float32)

    # Correctness check
    y_ref = torch_weighted_sum(x, weight)
    y_tri = triton_weighted_sum(x, weight)
    max_abs_err = (y_ref - y_tri).abs().max().item()
    print(f"Max abs error: {max_abs_err:.3e}")

    # Performance
    ms_ref = _benchmark(torch_weighted_sum, x, weight)
    ms_tri = _benchmark(triton_weighted_sum, x, weight)
    print(f"PyTorch: {ms_ref:.3f} ms | Triton: {ms_tri:.3f} ms | Speedup: {ms_ref / ms_tri:.2f}x")


if __name__ == '__main__':
    main()
```

## 1. Understanding `make_block_ptr` Parameters

### 1.1 Concrete Example

Imagine you have a matrix `x` with shape `(1000, 512)` where:
- **ROWS = 1000** (total rows in the full tensor)
- **D = 512** (embedding dimension, total columns)

Now, you can't process all 1000 rows at once in a single GPU thread block, so you split the work into **tiles**:
- **ROWS_TILE_SIZE = 16** (process 16 rows per thread block)
- **D_TILE_SIZE = 64** (process 64 columns at a time in a loop)

### 1.2 Parameter Breakdown

Here's what each parameter in `make_block_ptr` means:

#### 1.2.1 **`shape=(ROWS, D)`** - The FULL tensor dimensions
```python
shape=(1000, 512)  # The complete tensor size
```
This tells Triton the global shape for **boundary checking** (so it doesn't read out-of-bounds).

#### 1.2.2 **`strides=(x_stride_row, x_stride_dim)`** - Memory layout
```python
# For a contiguous tensor x[1000, 512]:
x_stride_row = 512   # Jump 512 elements to move to next row
x_stride_dim = 1     # Jump 1 element to move to next column
```
Strides tell you how many elements to skip in memory to move 1 step in each dimension.

#### 1.2.3 **`offsets=(row_tile_idx * ROWS_TILE_SIZE, 0)`** - Starting position
```python
# If row_tile_idx = 3:
offsets=(3 * 16, 0) = (48, 0)  # Start at row 48, column 0
```
This is the **coordinate** in the full tensor where this thread block starts reading.

#### 1.2.4 **`block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE)`** - How much to load at once
```python
block_shape=(16, 64)  # Load a 16×64 tile each time
```
This is the size of the data block you'll load with `tl.load()`.

#### 1.2.5 **`order=(1, 0)`** - Memory ordering
```python
order=(1, 0)  # Dimension 1 (D) is contiguous, dimension 0 (ROWS) has larger stride
```
This is `np.argsort(strides)` - helps Triton optimize memory access patterns.

### 1.3 Visual Example

Let's trace through **thread block 3** (`row_tile_idx = 3`):

```
Full tensor x: [1000 rows × 512 cols]

┌─────────────────────────────────────┐
│                                     │
│  rows 0-15    (tile 0)             │
│  rows 16-31   (tile 1)             │
│  rows 32-47   (tile 2)             │
│  rows 48-63   (tile 3) ← YOU ARE HERE!
│  rows 64-79   (tile 4)             │
│  ...                               │
│  rows 992-999 (tile 62)            │
└─────────────────────────────────────┘

Tile 3 processes rows 48-63:
- offsets = (48, 0) → start at row 48
- block_shape = (16, 64) → load 16 rows × 64 cols
- Loop 8 times (512/64) to cover all columns:
  - Iteration 0: cols 0-63
  - Iteration 1: cols 64-127
  - ... (using advance() to move the pointer)
```

### 1.4 Why So Many Parameters?

The separation allows **flexibility**:
- **ROWS, D**: Global context (for bounds checking)
- **ROWS_TILE_SIZE, D_TILE_SIZE**: Performance tuning (how big are your tiles?)
- **row_tile_idx**: Parallelization (which tile does THIS thread handle?)
- **strides**: Memory layout (works with transposed/strided tensors)

### 1.5 Summary Table

| Parameter | Value (example) | Meaning |
|-----------|----------------|---------|
| `ROWS` | 1000 | Total rows in full tensor |
| `ROWS_TILE_SIZE` | 16 | Rows processed per thread block |
| `row_tile_idx` | 3 | Which tile (0-62) this thread handles |
| `offsets` | (48, 0) | Starting coordinate = `row_tile_idx × ROWS_TILE_SIZE` |
| `x_stride_row` | 512 | Memory jump between rows |
| `block_shape` | (16, 64) | Size of data loaded per `tl.load()` |

Does this help clarify the distinction between global dimensions, tile sizes, and positioning? The key is that **ROWS/D describe the full tensor**, while **ROWS_TILE_SIZE/D_TILE_SIZE describe how you chunk the work**.

## 2. Advanced Concepts: Memory Layout and Access Patterns

### 2.1 Parallelization Strategy: Different Rows, Loop Over Columns

Exactly right. Look at the pattern:

```python
# Each thread block handles DIFFERENT rows:
offsets=(row_tile_idx * ROWS_TILE_SIZE, 0)
#        ↑ different per thread            ↑ all start at column 0

# Thread 0: offsets=(0, 0)   → rows 0-15
# Thread 1: offsets=(16, 0)  → rows 16-31  
# Thread 2: offsets=(32, 0)  → rows 32-47
# ... parallel execution

# Then EACH thread loops over columns:
for i in range(tl.cdiv(D, D_TILE_SIZE)):  # Loop over column chunks
    x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))  # Move right
    #                                  ↑ don't change rows
    #                                     ↑ move D_TILE_SIZE columns
```

This is a common pattern: **parallelize across rows, serialize across columns**.

### 2.2 Strides vs Offsets - The Key Difference

**Offsets** = logical coordinates (which element?)  
**Strides** = memory jumps (how to get there?)

#### Memory Layout Example

Here's actual memory for a `(4, 3)` tensor:

```python
tensor = [[10, 20, 30],    # Row 0
          [40, 50, 60],    # Row 1  
          [70, 80, 90],    # Row 2
          [11, 22, 33]]    # Row 3

# In memory (contiguous row-major):
[10, 20, 30, 40, 50, 60, 70, 80, 90, 11, 22, 33]
 ↑0  ↑1  ↑2  ↑3  ↑4  ↑5  ↑6  ↑7  ↑8  ↑9  ↑10 ↑11

strides = (3, 1)  # stride_row=3, stride_col=1
```

To access element at `(row, col)`:
```python
memory_index = row * stride_row + col * stride_col

# Example: element at (2, 1) = 80
memory_index = 2 * 3 + 1 * 1 = 7  ✓
```

#### How They Work Together

```python
# You want to start at logical position (row=2, col=1)
offsets = (2, 1)

# Triton calculates the memory address:
actual_address = base_ptr + (2 * x_stride_row) + (1 * x_stride_dim)
actual_address = base_ptr + (2 * 3) + (1 * 1)  
actual_address = base_ptr + 7  # Points to element 80!
```

**Key point**: 
- **Offsets** say "I want row 2, column 1" (logical)
- **Strides** say "to get there, jump 7 elements in memory" (physical)

### 2.3 Order Parameter - Memory Contiguity

The `order` parameter tells Triton which dimensions are **most contiguous** in memory.

#### What "Contiguous" Means

```python
# Row-major (C-style, PyTorch default):
tensor[4, 3] in memory: [row0_col0, row0_col1, row0_col2, row1_col0, ...]
                         ↑ adjacent elements are in the same row
strides = (3, 1)
order = (1, 0)  # dimension 1 (cols) is most contiguous (stride=1)
                # dimension 0 (rows) is less contiguous (stride=3)

# Column-major (Fortran-style):
tensor[4, 3] in memory: [row0_col0, row1_col0, row2_col0, row3_col0, row0_col1, ...]
                         ↑ adjacent elements are in the same column
strides = (1, 4)
order = (0, 1)  # dimension 0 (rows) is most contiguous (stride=1)
                # dimension 1 (cols) is less contiguous (stride=4)
```

#### How to Calculate Order

```python
order = tuple(np.argsort(strides))

# Example 1 (row-major):
strides = (3, 1)
np.argsort([3, 1]) = [1, 0]  # Index 1 has smallest stride
order = (1, 0)

# Example 2 (column-major):
strides = (1, 4)
np.argsort([1, 4]) = [0, 1]  # Index 0 has smallest stride
order = (0, 1)
```

#### Why It Matters

Triton uses `order` to optimize memory coalescing on GPUs:

```python
# With order=(1, 0), Triton knows dimension 1 is contiguous
# So when loading block_shape=(16, 64), it will:
# - Load 64 consecutive elements per row (coalesced reads)
# - Skip by stride_row=512 between rows

# Good:  Load columns within a row → sequential memory access
# Bad:   Load rows within a column → scattered memory access
```

### 2.4 Putting It All Together

```python
x_block_ptr = tl.make_block_ptr(
    x_ptr,                              # Start of memory
    shape=(1000, 512),                  # Full tensor dimensions (for bounds)
    strides=(512, 1),                   # How to navigate memory
    offsets=(48, 0),                    # Start at logical position row 48
    block_shape=(16, 64),               # Load 16×64 tiles
    order=(1, 0),                       # Cols are contiguous
)

# What happens:
# 1. Starting memory address = x_ptr + 48*512 + 0*1 = x_ptr + 24576
# 2. Load 16 rows × 64 cols from that position
# 3. Because order=(1, 0), Triton knows to read 64 consecutive 
#    elements per row for optimal GPU memory access
# 4. After advance((0, 64)), new offset is (48, 64)
#    → new address = x_ptr + 48*512 + 64*1 = x_ptr + 24640
```

#### Quick Test of Understanding

```python
# Transposed tensor: x.T with shape (512, 1000)
strides = (1, 512)  # Now cols have stride 512!
order = (0, 1)      # Dimension 0 is most contiguous

# Non-contiguous slice: x[:, ::2] (every other column)
shape = (1000, 256)
strides = (512, 2)  # Stride 2 in last dimension!
order = (1, 0)      # Still (1, 0) because 2 < 512
```

## 3. Understanding the Forward Pass Kernel

### 3.1 The For Loop - Processing Columns in Chunks

```python
for i in range(tl.cdiv(D, D_TILE_SIZE)):
    # tl.cdiv is "ceiling division" = ⌈D / D_TILE_SIZE⌉
```

#### Example Setup
```python
ROWS = 100, D = 512
ROWS_TILE_SIZE = 16, D_TILE_SIZE = 64

# This thread handles rows 32-47 (row_tile_idx = 2)
# Need to process all 512 columns, but can only do 64 at a time

iterations = ceil(512 / 64) = 8 iterations

# Iteration 0: columns 0-63
# Iteration 1: columns 64-127
# Iteration 2: columns 128-191
# ...
# Iteration 7: columns 448-511
```

#### What Happens Each Iteration

```python
# === ITERATION 0 (columns 0-63) ===
row = tl.load(x_block_ptr, ...)       # Load x[32:48, 0:64]   → shape (16, 64)
weight = tl.load(weight_block_ptr, ...)# Load weight[0:64]     → shape (64,)

output += tl.sum(row * weight[None, :], axis=1)  # Accumulate partial sum

x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))  # Move to columns 64-127
weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))

# === ITERATION 1 (columns 64-127) ===
row = tl.load(x_block_ptr, ...)       # Load x[32:48, 64:128] → shape (16, 64)
weight = tl.load(weight_block_ptr, ...)# Load weight[64:128]   → shape (64,)

output += tl.sum(row * weight[None, :], axis=1)  # Accumulate more

# ... continue for 8 iterations total
```

### 3.2 Boundary Checks and Padding

#### Why Do We Need Them?

Dimensions often don't divide evenly:

```python
# Problem: What if D = 500 and D_TILE_SIZE = 64?
iterations = ceil(500 / 64) = 8

# Iteration 0-6: OK (columns 0-63, 64-127, ..., 384-447)
# Iteration 7:   columns 448-511 BUT we only have 500 columns!
#                Trying to read columns 500-511 = OUT OF BOUNDS! 💥
```

#### How Padding Solves It

```python
row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
#                          ↑ check dim 0 (rows) and dim 1 (columns)
#                                                 ↑ pad with zeros if OOB
```

**Concrete Example:**
```python
D = 500, D_TILE_SIZE = 64
# Iteration 7: trying to load columns 448-511

# Without boundary_check: 💥 CRASH (read invalid memory)

# With boundary_check=(0, 1), padding_option="zero":
#   Columns 448-499: read actual data
#   Columns 500-511: return 0 (padded)
# Result shape still (16, 64), but last 12 values are 0
```

#### Which Dimensions to Check?

```python
# x is 2D:
row = tl.load(x_block_ptr, boundary_check=(0, 1))
#                                          ↑  ↑
#                          check rows ────┘  └── check columns

# weight is 1D:
weight = tl.load(weight_block_ptr, boundary_check=(0,))
#                                                  ↑
#                             only check dimension 0
```

### 3.3 Broadcasting with `weight[None, :]`

This is NumPy-style **shape manipulation** for broadcasting.

#### The Problem

```python
row    shape: (16, 64)   # 16 rows, 64 values per row
weight shape: (64,)      # 64 weights

# We want: row * weight (element-wise multiply each row by weight)
# But shapes don't match! ❌
```

#### The Solution - Add a Dimension

```python
weight[None, :] 
# Before: shape (64,)
# After:  shape (1, 64)

# Now broadcasting works:
row          (16, 64)
weight[None, :]  (1, 64)  ← broadcasts to (16, 64)
─────────────────────────
result       (16, 64)
```

#### Visual Example

```python
row = [[1, 2, 3, 4],
       [5, 6, 7, 8]]        # shape (2, 4)

weight = [10, 20, 30, 40]   # shape (4,)

weight[None, :] = [[10, 20, 30, 40]]  # shape (1, 4)

# Broadcasting repeats the row:
[[10, 20, 30, 40],    # row 0 gets multiplied by [10, 20, 30, 40]
 [10, 20, 30, 40]]    # row 1 gets multiplied by [10, 20, 30, 40]

result = [[1*10,  2*20,  3*30,  4*40],
          [5*10,  6*20,  7*30,  8*40]]

result = [[10, 40, 90, 160],
          [50, 120, 210, 320]]

# Then sum along axis=1 (across columns):
output = [10+40+90+160, 50+120+210+320] = [300, 700]
```

#### Why axis=1?

```python
tl.sum(row * weight[None, :], axis=1)
#                              ↑
#              sum across dimension 1 (columns)

# Input shape:  (16, 64)
# After axis=1: (16,)  ← sum each row to a single number
```

### 3.4 Complete Example Walkthrough

Let's trace **one thread block** processing a simple case:

```python
# Setup:
ROWS = 100, D = 200
ROWS_TILE_SIZE = 4, D_TILE_SIZE = 100
row_tile_idx = 2  # This thread handles rows 8-11

# Data:
x = torch.randn(100, 200)
weight = torch.randn(200)

# Goal: compute output[8:12] = sum(x[8:12, :] * weight, axis=1)
```

#### Iteration 0 (columns 0-99)

```python
# Load data
row = x[8:12, 0:100]        # shape (4, 100)
weight_chunk = weight[0:100] # shape (100,)

# Broadcast and multiply
weight_chunk[None, :] # shape (1, 100) → broadcasts to (4, 100)
product = row * weight_chunk[None, :]  # shape (4, 100)

# Example values:
# product = [[0.5, -1.2, ..., 0.8],  ← row 8
#            [2.1,  0.3, ..., -0.5], ← row 9
#            [-0.7, 1.8, ..., 1.2],  ← row 10
#            [0.9, -0.4, ..., 0.6]]  ← row 11

# Sum across columns (axis=1)
partial_sum = tl.sum(product, axis=1)  # shape (4,)
# partial_sum = [sum(row 8's 100 values),
#                sum(row 9's 100 values),
#                sum(row 10's 100 values),
#                sum(row 11's 100 values)]
# Example: [45.2, -12.3, 67.8, 23.1]

output = [45.2, -12.3, 67.8, 23.1]  # Initialize

# Advance pointers
x_block_ptr → now points to columns 100-199
weight_block_ptr → now points to weight[100:200]
```

#### Iteration 1 (columns 100-199)

```python
# Load next chunks
row = x[8:12, 100:200]       # shape (4, 100)
weight_chunk = weight[100:200] # shape (100,)

# Compute and accumulate
product = row * weight_chunk[None, :]
partial_sum = tl.sum(product, axis=1)  # shape (4,)
# Example: [12.5, 34.7, -8.9, 15.6]

output += partial_sum  # Accumulate!
# output = [45.2+12.5, -12.3+34.7, 67.8-8.9, 23.1+15.6]
# output = [57.7, 22.4, 58.9, 38.7]
```

#### Final Result

```python
# After all iterations, output contains the weighted sum for rows 8-11
tl.store(output_block_ptr, output, boundary_check=(0,))
# Writes [57.7, 22.4, 58.9, 38.7] to global output[8:12]
```

### 3.5 Boundary Check on Store

```python
tl.store(output_block_ptr, output, boundary_check=(0,))
```

**Why check on write?**

```python
ROWS = 98, ROWS_TILE_SIZE = 16
# Thread block 6: handles rows 96-111
# But rows 98-111 don't exist!

# With boundary_check=(0,):
#   output[96:98] → write normally
#   output[98:111] → skip write (out of bounds)
```

### 3.6 Summary

| Concept | Purpose | Example |
|---------|---------|---------|
| **For loop** | Process columns in chunks | 8 iterations for D=512, tile=64 |
| **boundary_check** | Avoid reading/writing OOB memory | Check when tiles exceed tensor size |
| **padding_option="zero"** | Fill OOB values with 0 | Last tile reads past D |
| **weight[None, :]** | Add dimension for broadcasting | (64,) → (1, 64) → broadcasts |
| **axis=1** | Sum across columns | (16, 64) → (16,) |
| **output +=** | Accumulate across iterations | Total sum across all column chunks |

## 4. PyTorch + Triton Integration

### 4.1 Why Rearrange: `"... d -> (...) d"`

#### The Problem: Arbitrary Input Shapes

```python
# Function signature: weighted_sum(x, weight) → sum along last dimension
# But x could have ANY number of dimensions!

x1 = torch.randn(100, 512)           # 2D: (batch, D)
x2 = torch.randn(32, 10, 512)        # 3D: (batch, seq, D)
x3 = torch.randn(8, 4, 20, 512)      # 4D: (B, heads, seq, D)

# The Triton kernel only handles 2D: (rows, D)
# Solution: Flatten everything except last dimension
```

#### What Rearrange Does

```python
x = rearrange(x, "... d -> (...) d")
#              ↑           ↑
#     "any dims, D"   "flatten, D"
```

**Concrete Examples:**

```python
# Example 1: 2D input
x = torch.randn(100, 512)           # shape (100, 512)
x_reshaped = rearrange(x, "... d -> (...) d")
# Result: (100, 512)  ← no change!

# Example 2: 3D input
x = torch.randn(32, 10, 512)        # shape (32, 10, 512)
x_reshaped = rearrange(x, "... d -> (...) d")
# Result: (320, 512)  ← flattened 32*10 = 320 rows

# Example 3: 4D input
x = torch.randn(8, 4, 20, 512)      # shape (8, 4, 20, 512)
x_reshaped = rearrange(x, "... d -> (...) d")
# Result: (640, 512)  ← flattened 8*4*20 = 640 rows
```

#### Why Save Original Shape?

```python
input_shape = x.shape               # Save BEFORE reshaping
x = rearrange(x, "... d -> (...) d") # Flatten to 2D

# ... run Triton kernel (2D) ...

return y.view(input_shape[:-1])     # Restore original shape (except last dim)
```

**Example:**
```python
# Input:  x.shape = (32, 10, 512)
# After rearrange: (320, 512)
# Kernel output: y.shape = (320,)
# Return: y.view(32, 10) = (32, 10)  ← matches input structure!
```

### 4.2 `ctx.save_for_backward` - What Happens Behind the Scenes

#### What It Does

```python
ctx.save_for_backward(x, weight)
```

This tells PyTorch's autograd system: **"I'll need these tensors during the backward pass"**

#### Behind the Scenes

```python
# PyTorch internally does something like:
class Context:
    def save_for_backward(self, *tensors):
        self._saved_tensors = []
        for t in tensors:
            # Keep the tensor alive (prevent garbage collection)
            # Store a reference that can be retrieved later
            self._saved_tensors.append(t)
    
    @property
    def saved_tensors(self):
        return tuple(self._saved_tensors)
```

#### Why We Need It

**The autograd flow:**
```python
# === FORWARD PASS ===
y = weighted_sum(x, weight)  # Compute output
# x and weight go out of scope... but we need them for backward!

# === BACKWARD PASS (later) ===
# Given: grad_output (gradient wrt y)
# Need: grad_x, grad_weight
# But we need the ORIGINAL x and weight to compute these!

# grad_x = grad_output @ weight  ← need weight!
# grad_weight = x.T @ grad_output ← need x!
```

**Concrete Example:**
```python
# Forward:
x = [[1, 2, 3],
     [4, 5, 6]]        # shape (2, 3)
weight = [10, 20, 30]  # shape (3,)

y = weighted_sum(x, weight) = [1*10 + 2*20 + 3*30, 4*10 + 5*20 + 6*30]
                             = [140, 320]

# Backward (later):
grad_output = [1.0, 2.0]  # Given

# To compute grad_x:
grad_x = grad_output[:, None] * weight[None, :]
       = [[1.0], [2.0]] * [[10, 20, 30]]
       = [[10, 20, 30],
          [20, 40, 60]]
# ↑ We needed the original 'weight' to compute this!

# To compute grad_weight:
grad_weight = x.T @ grad_output
            = [[1, 4],     [[1.0],      [[9.0],
               [2, 5],  @   [2.0]]   =   [12.0],
               [3, 6]]                   [15.0]]
# ↑ We needed the original 'x' to compute this!
```

### Memory Consideration

```python
ctx.save_for_backward(x, weight)
# ⚠️ These tensors stay in GPU memory until backward completes!
# Trade-off: Memory cost vs. recomputation cost
```

### 4.3 Why `ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16`

#### What `next_power_of_2` Does

```python
triton.next_power_of_2(D)  # Finds smallest 2^n >= D

# Examples:
next_power_of_2(100) = 128   # 2^7
next_power_of_2(512) = 512   # 2^9 (already power of 2)
next_power_of_2(1000) = 1024 # 2^10
next_power_of_2(2048) = 2048 # 2^11
```

#### Why Power of 2?

GPUs work most efficiently with power-of-2 tile sizes:
- Better memory coalescing
- Efficient warp utilization (GPUs have 32 threads/warp)
- Easier to optimize by compilers

#### The Formula Explained

```python
ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16
#                 ↑ round up to power of 2    ↑ divide by 16
```

**Goal:** Make ~16 iterations through the loop

```python
# For D = 512:
D_TILE_SIZE = next_power_of_2(512) // 16 = 512 // 16 = 32
iterations = ceil(512 / 32) = 16  ✓

# For D = 1000:
D_TILE_SIZE = next_power_of_2(1000) // 16 = 1024 // 16 = 64
iterations = ceil(1000 / 64) = 16  ✓

# For D = 100:
D_TILE_SIZE = next_power_of_2(100) // 16 = 128 // 16 = 8
iterations = ceil(100 / 8) = 13  ✓ (close to 16)
```

#### Why ~16 Iterations?

Balance between:
- **Too few iterations** (large tiles): 
  - More data per load
  - But may not fit in fast SRAM
  - Wastes registers if tile is too big
  
- **Too many iterations** (small tiles):
  - Loop overhead dominates
  - More pointer arithmetic
  - Less work per iteration

**16 is a heuristic** that works well across various D values.

#### Why Save to `ctx`?

```python
ctx.D_TILE_SIZE = ...
ctx.ROWS_TILE_SIZE = ...
ctx.input_shape = ...
```

The backward pass needs **the same tile sizes** to call the backward kernel:

```python
def backward(ctx, grad_out):
    # Retrieve saved values
    x, weight = ctx.saved_tensors
    ROWS_TILE_SIZE = ctx.ROWS_TILE_SIZE  # ← Need same tiling!
    D_TILE_SIZE = ctx.D_TILE_SIZE
    
    # Call backward kernel with matching configuration
    weighted_sum_backward[(cdiv(n_rows, ROWS_TILE_SIZE),)](
        ...,
        ROWS_TILE_SIZE=ROWS_TILE_SIZE,
        D_TILE_SIZE=D_TILE_SIZE,
    )
```

### 4.4 The Kernel Launch Syntax

```python
weighted_sum_fwd[(cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
    # Grid size ↑                                   ↑ Function call
    x, weight, y,  # Arguments
    ...
)
```

#### Grid Dimensions

```python
grid = (cdiv(n_rows, ctx.ROWS_TILE_SIZE),)
#       ↑ number of thread blocks

# Example: n_rows = 1000, ROWS_TILE_SIZE = 16
grid = (ceil(1000 / 16),) = (63,)
# Launch 63 parallel thread blocks
# Block 0 handles rows 0-15
# Block 1 handles rows 16-31
# ...
# Block 62 handles rows 992-999 (only 8 rows, boundary check!)
```

#### Triton Launch Syntax

```python
kernel_function[grid_dimensions](arguments)
#               ↑ tuple         ↑ regular function call

# Grid can be 1D, 2D, or 3D:
kernel[10,]                 # 1D: 10 blocks
kernel[10, 20]              # 2D: 10x20 = 200 blocks
kernel[10, 20, 5]           # 3D: 10x20x5 = 1000 blocks
```

## 5. Backward Pass Implementation

### 5.1 Mathematical Background

First, let's understand what gradients we're computing:

```python
# Forward: y = weighted_sum(x, weight)
# y[i] = sum_j (x[i,j] * weight[j])

# Given: grad_output (gradient wrt y)
# Need to compute:
# 1. grad_x[i,j] = ∂L/∂x[i,j]
# 2. grad_weight[j] = ∂L/∂weight[j]
```

#### Gradient Formulas (Chain Rule)

```python
# For grad_x:
# ∂L/∂x[i,j] = ∂L/∂y[i] * ∂y[i]/∂x[i,j]
#            = grad_output[i] * weight[j]

# For grad_weight:
# ∂L/∂weight[j] = sum_i (∂L/∂y[i] * ∂y[i]/∂weight[j])
#               = sum_i (grad_output[i] * x[i,j])
```

#### Concrete Example

```python
# Forward:
x = [[1, 2, 3],
     [4, 5, 6]]        # shape (2, 3)
weight = [10, 20, 30]  # shape (3,)

y = [1*10 + 2*20 + 3*30,   # y[0] = 140
     4*10 + 5*20 + 6*30]   # y[1] = 320

# Backward (given grad_output):
grad_output = [1.0, 2.0]  # shape (2,)

# grad_x[i,j] = grad_output[i] * weight[j]
grad_x = [[1.0 * 10, 1.0 * 20, 1.0 * 30],     # row 0
          [2.0 * 10, 2.0 * 20, 2.0 * 30]]      # row 1
       = [[10, 20, 30],
          [20, 40, 60]]

# grad_weight[j] = sum_i (grad_output[i] * x[i,j])
grad_weight[0] = 1.0 * 1 + 2.0 * 4 = 9.0
grad_weight[1] = 1.0 * 2 + 2.0 * 5 = 12.0
grad_weight[2] = 1.0 * 3 + 2.0 * 6 = 15.0
grad_weight = [9.0, 12.0, 15.0]
```

### 5.2 The Parallelization Challenge

#### Problem: Gradient for `weight` Needs Reduction Across ALL Rows

```python
# grad_x: Independent per row → Easy to parallelize!
# Each thread block can compute its own rows

# grad_weight: Needs sum across ALL rows → Hard to parallelize!
# grad_weight[j] = sum over ALL i (grad_output[i] * x[i,j])
#                  ↑ all thread blocks contribute to same result
```

#### Solution: Partial Buffers + Final Reduction

```python
# Strategy:
# 1. Each thread block computes a PARTIAL gradient for its rows
# 2. Store partial results in a buffer
# 3. Sum the partial results (on CPU or GPU)

partial_grad_weight = torch.empty((n_thread_blocks, D), ...)
#                                  ↑ one row per thread block

# Thread 0: partial_grad_weight[0, :] = sum over rows 0-15
# Thread 1: partial_grad_weight[1, :] = sum over rows 16-31
# ...

# Final: grad_weight = partial_grad_weight.sum(axis=0)
```

### 5.3 Step-by-Step Walkthrough

#### Setup

```python
# Forward computed:
x = torch.randn(64, 200)      # shape (64, 200)
weight = torch.randn(200)     # shape (200,)
y = weighted_sum(x, weight)   # shape (64,)

# Backward receives:
grad_output = torch.randn(64) # shape (64,)

# Configuration:
ROWS_TILE_SIZE = 16
D_TILE_SIZE = 64
n_thread_blocks = ceil(64 / 16) = 4

# Allocate outputs:
grad_x = torch.empty(64, 200)
partial_grad_weight = torch.empty(4, 200)  # 4 thread blocks × 200 dims
```

#### Thread Block 0 Execution (rows 0-15)

Let's trace what **thread block 0** does:

```python
row_tile_idx = 0  # This is thread block 0
n_row_tiles = 4   # Total 4 thread blocks

# Initialize pointers to handle rows 0-15
grad_output_block_ptr → grad_output[0:16]
x_block_ptr → x[0:16, 0:64]  # Start at columns 0-63
weight_block_ptr → weight[0:64]
grad_x_block_ptr → grad_x[0:16, 0:64]
partial_grad_weight_block_ptr → partial_grad_weight[0, 0:64]
#                                                    ↑ row 0 of buffer
```

##### Iteration 0 (columns 0-63)

```python
# === Load inputs ===
grad_output = grad_output[0:16]    # shape (16,)
# Example: [0.5, -1.2, 0.8, ..., 0.3]  (16 values)

weight = weight[0:64]              # shape (64,)
# Example: [10, 20, 30, ..., 15]  (64 values)

row = x[0:16, 0:64]                # shape (16, 64)
# 16 rows × 64 values each

# === Compute grad_x (outer product) ===
grad_x_row = grad_output[:, None] * weight[None, :]
#            ↑ shape (16, 1)        ↑ shape (1, 64)
#            Result: (16, 64)

# Example:
# grad_output[:, None] = [[0.5],
#                         [-1.2],
#                         [0.8],
#                         ...]       shape (16, 1)
#
# weight[None, :] = [[10, 20, 30, ..., 15]]  shape (1, 64)
#
# Broadcasting:
# grad_x_row[0, :] = [0.5*10, 0.5*20, 0.5*30, ..., 0.5*15]
# grad_x_row[1, :] = [-1.2*10, -1.2*20, -1.2*30, ..., -1.2*15]
# ... (16 rows total)

tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0, 1))
# Writes grad_x[0:16, 0:64] ✓

# === Compute partial grad_weight (reduction) ===
grad_weight_row = tl.sum(row * grad_output[:, None], axis=0, keep_dims=True)
#                        ↑ (16, 64)  ↑ (16, 1)
#                        Result after broadcast: (16, 64)
#                        After sum(axis=0): (1, 64)

# Detailed:
# row * grad_output[:, None]:
# [[x[0,0]*0.5, x[0,1]*0.5, ..., x[0,63]*0.5],
#  [x[1,0]*-1.2, x[1,1]*-1.2, ..., x[1,63]*-1.2],
#  ...
#  [x[15,0]*0.3, x[15,1]*0.3, ..., x[15,63]*0.3]]
#
# Sum down columns (axis=0):
# grad_weight_row[0] = x[0,0]*0.5 + x[1,0]*-1.2 + ... + x[15,0]*0.3
# grad_weight_row[1] = x[0,1]*0.5 + x[1,1]*-1.2 + ... + x[15,1]*0.3
# ...
# grad_weight_row[63] = x[0,63]*0.5 + x[1,63]*-1.2 + ... + x[15,63]*0.3
#
# This is the contribution of rows 0-15 to weight gradients 0-63!

tl.store(partial_grad_weight_block_ptr, grad_weight_row, boundary_check=(1,))
# Writes partial_grad_weight[0, 0:64] ✓

# === Advance pointers ===
x_block_ptr → x[0:16, 64:128]
weight_block_ptr → weight[64:128]
grad_x_block_ptr → grad_x[0:16, 64:128]
partial_grad_weight_block_ptr → partial_grad_weight[0, 64:128]
```

##### Iteration 1 (columns 64-127)

```python
# Same process for next 64 columns
grad_output = grad_output[0:16]    # Same as before (doesn't change)
weight = weight[64:128]
row = x[0:16, 64:128]

# Compute and store grad_x[0:16, 64:128]
grad_x_row = grad_output[:, None] * weight[None, :]
tl.store(grad_x_block_ptr, grad_x_row, ...)

# Compute and store partial_grad_weight[0, 64:128]
grad_weight_row = tl.sum(row * grad_output[:, None], axis=0, keep_dims=True)
tl.store(partial_grad_weight_block_ptr, grad_weight_row, ...)

# Advance pointers...
```

##### Iteration 2 (columns 128-191)

```python
# Same for columns 128-191
# ... (continues for all D_TILE_SIZE chunks)
```

#### Parallel Execution of Other Thread Blocks

While thread block 0 handles rows 0-15, **other blocks run in parallel**:

```python
# Thread Block 1 (row_tile_idx = 1):
# - Computes grad_x[16:32, :]
# - Writes partial_grad_weight[1, :] (sum over rows 16-31)

# Thread Block 2 (row_tile_idx = 2):
# - Computes grad_x[32:48, :]
# - Writes partial_grad_weight[2, :] (sum over rows 32-47)

# Thread Block 3 (row_tile_idx = 3):
# - Computes grad_x[48:64, :]
# - Writes partial_grad_weight[3, :] (sum over rows 48-63)
```

#### After Kernel Completes

```python
# grad_x is fully computed ✓
grad_x.shape = (64, 200)

# partial_grad_weight has contributions from each block
partial_grad_weight.shape = (4, 200)
# Row 0: contribution from rows 0-15
# Row 1: contribution from rows 16-31
# Row 2: contribution from rows 32-47
# Row 3: contribution from rows 48-63

# Final step: Sum to get complete gradient
grad_weight = partial_grad_weight.sum(axis=0)
#             ↑ sum down columns
# Result shape: (200,)

# grad_weight[j] = partial[0,j] + partial[1,j] + partial[2,j] + partial[3,j]
#                = (sum over rows 0-15 of grad_output[i]*x[i,j])
#                + (sum over rows 16-31 of grad_output[i]*x[i,j])
#                + (sum over rows 32-47 of grad_output[i]*x[i,j])
#                + (sum over rows 48-63 of grad_output[i]*x[i,j])
#                = sum over ALL rows of grad_output[i]*x[i,j] ✓
```

### 5.4 Key Design Details

#### 1. Why `grad_output[:, None]`?

```python
grad_output.shape = (16,)
grad_output[:, None].shape = (16, 1)  # Add dimension for broadcasting

# For grad_x:
grad_output[:, None] * weight[None, :]
# (16, 1) × (1, 64) → broadcasts to (16, 64) ✓

# For grad_weight:
row * grad_output[:, None]
# (16, 64) × (16, 1) → broadcasts to (16, 64) ✓
```

#### 2. Why `keep_dims=True`?

```python
grad_weight_row = tl.sum(row * grad_output[:, None], axis=0, keep_dims=True)
#                                                             ↑
# Without keep_dims: result shape (64,)
# With keep_dims: result shape (1, 64)

# We need (1, 64) because partial_grad_weight_block_ptr expects 2D:
# block_shape=(1, D_TILE_SIZE)
#              ↑ dimension 0 is size 1
```

#### 3. The `partial_grad_weight_block_ptr` Setup

```python
partial_grad_weight_block_ptr = tl.make_block_ptr(
    partial_grad_weight_ptr,
    shape=(n_row_tiles, D,),  # (4, 200) full buffer
    strides=(stride_gwb, stride_gwd),
    offsets=(row_tile_idx, 0),  # Each block writes to DIFFERENT row
    #        ↑ block 0 writes row 0, block 1 writes row 1, etc.
    block_shape=(1, D_TILE_SIZE),  # Write 1 row at a time
    order=(1, 0),
)

# Thread 0: offsets=(0, 0) → writes to partial_grad_weight[0, :]
# Thread 1: offsets=(1, 0) → writes to partial_grad_weight[1, :]
# Thread 2: offsets=(2, 0) → writes to partial_grad_weight[2, :]
# Thread 3: offsets=(3, 0) → writes to partial_grad_weight[3, :]
```

#### 4. Boundary Checks

```python
# grad_x and x have same checks:
boundary_check=(0, 1)  # Check both rows and columns

# partial_grad_weight:
boundary_check=(1,)  # Only check dimension 1 (columns)
# Comment says: "Never out of bounds for dim 0"
# Why? Because we allocated exactly n_row_tiles rows,
# and each block writes to exactly one row (its row_tile_idx)
```

### 5.5 Complete Numerical Example

Let's trace a tiny example:

```python
# Setup
x = [[1, 2],
     [3, 4]]           # shape (2, 2)
weight = [10, 20]     # shape (2,)
grad_output = [1, 2]  # shape (2,)

ROWS_TILE_SIZE = 1  # Process 1 row per block
D_TILE_SIZE = 2     # Process all columns at once
n_blocks = 2

# === Thread Block 0 (row 0) ===
grad_output_chunk = [1]         # grad_output[0:1]
row_chunk = [[1, 2]]            # x[0:1, 0:2]
weight_chunk = [10, 20]         # weight[0:2]

# Compute grad_x[0, :]
grad_x[0, :] = [1] * [10, 20] = [1*10, 1*20] = [10, 20]

# Compute partial_grad_weight[0, :]
partial_grad_weight[0, :] = sum([[1, 2]] * [[1], axis=0
#                                           = sum([[1*1, 2*1]], axis=0)
#                                           = sum([[1, 2]], axis=0)
#                                           = [1, 2]

# === Thread Block 1 (row 1) ===
grad_output_chunk = [2]         # grad_output[1:2]
row_chunk = [[3, 4]]            # x[1:2, 0:2]
weight_chunk = [10, 20]         # weight[0:2]

# Compute grad_x[1, :]
grad_x[1, :] = [2] * [10, 20] = [2*10, 2*20] = [20, 40]

# Compute partial_grad_weight[1, :]
partial_grad_weight[1, :] = sum([[3, 4]] * [[2]], axis=0)
#                           = sum([[3*2, 4*2]], axis=0)
#                           = sum([[6, 8]], axis=0)
#                           = [6, 8]

# === Final Reduction ===
partial_grad_weight = [[1, 2],
                       [6, 8]]

grad_weight = partial_grad_weight.sum(axis=0)
            = [1+6, 2+8]
            = [7, 10]

# === Verify Correctness ===
# grad_weight[0] = grad_output[0]*x[0,0] + grad_output[1]*x[1,0]
#                = 1*1 + 2*3 = 7 ✓

# grad_weight[1] = grad_output[0]*x[0,1] + grad_output[1]*x[1,1]
#                = 1*2 + 2*4 = 10 ✓
```

### 5.6 Summary: Why This Design?

| Aspect | Reason |
|--------|--------|
| **Parallel grad_x** | Each row independent → easy parallelization |
| **Partial buffers** | grad_weight needs sum across rows → use reduce pattern |
| **Outer product** | grad_x = broadcast multiply (efficient on GPU) |
| **Inner product** | grad_weight = reduce sum (accumulate partial results) |
| **Final .sum(axis=0)** | Combine partial gradients from all thread blocks |
| **Same tiling** | Reuse forward's tile sizes for memory efficiency |

### 5.7 Performance Consideration

```python
# Why not just use atomic adds for grad_weight?
# atomic_add(grad_weight[j], grad_output[i] * x[i,j])

# Problems:
# 1. Atomic operations are SLOW (serialization)
# 2. Many threads competing for same memory location
# 3. GPU throughput collapses

# Our approach:
# 1. Each thread block works independently (no contention)
# 2. Final sum is a single reduction (fast)
# 3. Much better GPU utilization
```

## 6. Memory Management and Gradient Flow

### 6.1 Where Gradients Live

#### The Key Point: Gradients Stay on GPU

```python
# All tensors are on GPU throughout:
x = torch.randn(100, 512, device='cuda')      # GPU ✓
weight = torch.randn(512, device='cuda')      # GPU ✓
y = weighted_sum(x, weight)                   # GPU ✓

# During backward:
loss = y.sum()
loss.backward()

# grad_output is created on GPU ✓
# grad_x is created on GPU ✓
# grad_weight is created on GPU ✓

# Everything stays on GPU - no CPU↔GPU transfers!
```

### 6.2 What `make_block_ptr` Actually Does

**`make_block_ptr` does NOT transfer data!** It just creates a "view" or "handle" to access existing GPU memory.

```python
# This is what happens:

# 1. PyTorch creates grad_output tensor on GPU
grad_output = torch.randn(100, device='cuda')
# Memory allocated on GPU at address, say, 0x7f8a00000000

# 2. Pass the pointer to Triton kernel
weighted_sum_backward[(n_blocks,)](
    ...,
    grad_output,  # ← Pass the GPU memory address
    ...
)

# 3. Inside kernel: make_block_ptr creates a "sliding window"
grad_output_block_ptr = tl.make_block_ptr(
    grad_output_ptr,  # ← This is already a GPU address!
    shape=(NUM_ROWS,),
    offsets=(row_tile_idx * ROWS_TILE_SIZE,),
    block_shape=(ROWS_TILE_SIZE,),
    order=(0,),
)
# This just says: "I want to read from this GPU memory,
# starting at offset X, reading Y elements at a time"
```

**Analogy:** Think of `make_block_ptr` like array slicing in Python - creating a view doesn't copy data, it just points to a slice of memory. Similarly, `make_block_ptr` creates a "view" into GPU memory with no data movement.

### 6.3 The Complete Gradient Flow

Let me trace the full backpropagation to show where each gradient comes from:

```python
# === Forward Pass ===
x = torch.randn(100, 512, device='cuda')     # On GPU
weight = torch.randn(512, device='cuda')     # On GPU
y = weighted_sum(x, weight)                  # On GPU, shape (100,)

loss = some_loss_function(y)                 # On GPU, scalar
loss.backward()  # ← Start backpropagation

# === Backward Pass (from top to bottom) ===

# Step 1: Loss gradient (computed by PyTorch)
# ∂loss/∂loss = 1.0 (always)

# Step 2: Gradient wrt y (computed by loss function's backward)
grad_y = loss_function.backward()            # On GPU, shape (100,)
#        ↑ This is created by the PREVIOUS layer's backward

# Step 3: Call weighted_sum backward
# grad_y becomes "grad_output" for weighted_sum
grad_x, grad_weight = weighted_sum.backward(grad_output=grad_y)
#                                           ↑ Input from upstream
#       ↑ Outputs to downstream
```

#### Inside `weighted_sum.backward`

```python
def backward(ctx, grad_output):
    # grad_output is RECEIVED from upstream
    # It's already on GPU! Shape (n_rows,)
    
    x, weight = ctx.saved_tensors
    # x and weight were saved during forward
    # They're already on GPU!
    
    n_rows, D = x.shape
    
    # === ALLOCATE new tensors for outputs ===
    grad_x = torch.empty_like(x)  # Allocate GPU memory for grad_x
    partial_grad_weight = torch.empty((n_blocks, D), device=x.device)
    #                                 ↑ Allocate GPU memory for partial buffer
    
    # === Launch kernel ===
    # We pass GPU pointers to the kernel:
    weighted_sum_backward[(n_blocks,)](
        x,                  # GPU pointer (input, saved from forward)
        weight,             # GPU pointer (input, saved from forward)
        grad_output,        # GPU pointer (INPUT from upstream)
        grad_x,             # GPU pointer (OUTPUT, will be filled)
        partial_grad_weight,# GPU pointer (OUTPUT, will be filled)
        ...
    )
    
    # grad_x and partial_grad_weight are now filled by kernel
    
    grad_weight = partial_grad_weight.sum(axis=0)  # GPU operation
    
    return grad_x, grad_weight  # Return to downstream layers
```

### 6.4 Inside the Kernel: All Pointers Point to GPU Memory

```python
@triton.jit
def weighted_sum_backward(
    x_ptr,                # GPU memory address
    weight_ptr,           # GPU memory address  
    grad_output_ptr,      # GPU memory address (from upstream!)
    grad_x_ptr,           # GPU memory address (empty, to fill)
    partial_grad_weight_ptr,  # GPU memory address (empty, to fill)
    ...
):
    # All pointers are GPU addresses!
    
    # Create block pointers (just slicing, no data movement):
    
    # For INPUT from upstream:
    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr,  # ← Already on GPU, received from upstream
        shape=(NUM_ROWS,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )
    # This says: "Read from grad_output_ptr starting at offset X"
    
    # For OUTPUT to downstream:
    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,  # ← Empty GPU memory allocated in backward()
        shape=(NUM_ROWS, D),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    # This says: "Write to grad_x_ptr starting at offset Y"
    
    # Read from GPU (no CPU involved):
    grad_output = tl.load(grad_output_block_ptr, ...)
    
    # Compute on GPU:
    grad_x_row = grad_output[:, None] * weight[None, :]
    
    # Write to GPU (no CPU involved):
    tl.store(grad_x_block_ptr, grad_x_row, ...)
```

### 6.5 Visual Memory Diagram

```
GPU Memory:
┌──────────────────────────────────────────────────┐
│                                                  │
│  x:            [saved during forward]            │
│  weight:       [saved during forward]            │
│  y:            [computed during forward]         │
│                                                  │
│  grad_output:  [received from upstream layer]    │  ← INPUT
│                                                  │
│  grad_x:       [allocated empty, filled by kernel]  ← OUTPUT
│  partial_grad_weight: [allocated empty, filled]  │  ← OUTPUT (temp)
│  grad_weight:  [computed by summing partial]     │  ← OUTPUT
│                                                  │
└──────────────────────────────────────────────────┘

CPU Memory:
┌──────────────────────────────────────────────────┐
│  (empty - no gradient data here!)                │
└──────────────────────────────────────────────────┘

All operations happen on GPU!
make_block_ptr just creates "views" into GPU memory.
```

### 6.6 Why Create Block Pointers for Both Input and Output?

You create block pointers for **both** because:

1. **Input gradients** (`grad_output`): Need to **read** from upstream
2. **Output gradients** (`grad_x`, `grad_weight`): Need to **write** for downstream

```python
# INPUT block pointers (read from):
grad_output_block_ptr  # Read gradient from upstream layer
x_block_ptr            # Read saved input from forward
weight_block_ptr       # Read saved weight from forward

# OUTPUT block pointers (write to):
grad_x_block_ptr       # Write gradient to pass to downstream
partial_grad_weight_block_ptr  # Write partial gradient (temp buffer)
```

### 6.7 Common Misconception vs Reality

| Misconception | Reality |
|--------------|---------|
| `make_block_ptr` allocates memory | No, memory already allocated by PyTorch |
| `make_block_ptr` transfers CPU↔GPU | No, all data stays on GPU |
| `make_block_ptr` copies data | No, it just creates a "view" or "handle" |
| Need to "pass gradient to GPU" | Gradients are created on GPU directly |

### 6.8 When Do CPU↔GPU Transfers Happen?

Transfers only happen when you explicitly request them:

```python
# GPU → CPU (explicit):
x_cpu = x.cpu()           # Transfer to CPU
x_numpy = x.cpu().numpy() # Transfer to CPU, convert to numpy

# CPU → GPU (explicit):
x_gpu = x_cpu.cuda()      # Transfer to GPU
x_gpu = torch.tensor(np_array, device='cuda')  # Create on GPU

# During training (everything on GPU):
for batch in dataloader:
    x, y = batch
    x, y = x.cuda(), y.cuda()  # ← Transfer input batch once
    
    output = model(x)          # GPU
    loss = criterion(output, y)  # GPU
    loss.backward()            # GPU (all gradients on GPU!)
    optimizer.step()           # GPU
    
# No more transfers until next batch!
```

### 6.9 Summary: Gradient Flow and Block Pointers

**Key Takeaways:**

1. **`grad_output` is received from upstream** - created by the previous layer's backward pass
2. **It's already on GPU** - no CPU↔GPU transfers during backpropagation
3. **`make_block_ptr` does NOT transfer data** - it just creates a pointer/view to access GPU memory
4. **All gradients stay on GPU** throughout the entire backward pass
5. **We create block pointers to:**
   - **Read** inputs (x, weight, grad_output) from saved tensors and upstream
   - **Write** outputs (grad_x, partial_grad_weight) for downstream layers

The block pointers are just a convenient way to access different parts of GPU memory in a structured way - think of them as "smart pointers" that understand tensor layouts and tiling!

### 6.10 GPU Memory Hierarchy and Data Movement

**`make_block_ptr` doesn't transfer data AT ALL** - not between CPU↔GPU, and not between different GPU memory types. It's purely a metadata structure that describes:
- Where data lives (pointer to global memory)
- How to access it (shape, strides, offsets)

#### GPU Memory Types

GPUs have multiple memory types:

```
┌─────────────────────────────────────────┐
│  GPU Die (Streaming Multiprocessor)     │
│                                          │
│  ┌────────────────┐                     │
│  │   Registers    │ ← Fastest (~1 cycle)│
│  │  (per thread)  │                     │
│  └────────────────┘                     │
│         ↕                                │
│  ┌────────────────┐                     │
│  │ Shared Memory  │ ← Fast (~20 cycles) │
│  │ (SRAM, per SM) │   ~100 KB           │
│  └────────────────┘                     │
└──────────┬───────────────────────────────┘
           ↕
  ┌────────────────┐
  │ Global Memory  │ ← Slow (~200-400 cycles)
  │ (HBM/GDDR)     │   GBs of memory
  └────────────────┘
```

#### How Data Actually Moves

The **data transfer happens with `tl.load()` and `tl.store()`**, not `make_block_ptr`:

```python
# Backward pass example:

# 1. make_block_ptr just creates a "view" - NO data movement
grad_output_block_ptr = tl.make_block_ptr(
    grad_output_ptr,  # Points to global memory
    ...
)

# 2. tl.load() actually moves data: Global Memory → Registers
grad_output = tl.load(grad_output_block_ptr, ...)
#             ↑ THIS is where data moves!
# Data flow: HBM → (cache) → Registers

# 3. Compute happens in registers
grad_x_row = grad_output[:, None] * weight[None, :]
#            ↑ All computation in registers

# 4. tl.store() moves data back: Registers → Global Memory
tl.store(grad_x_block_ptr, grad_x_row, ...)
#        ↑ THIS writes back to HBM
```

#### Detailed Data Flow in Backward Pass

```python
@triton.jit
def weighted_sum_backward(...):
    row_tile_idx = tl.program_id(0)
    
    # === Phase 1: Setup (NO data movement) ===
    grad_output_block_ptr = tl.make_block_ptr(...)  # Just metadata
    x_block_ptr = tl.make_block_ptr(...)            # Just metadata
    weight_block_ptr = tl.make_block_ptr(...)       # Just metadata
    
    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        # === Phase 2: Load from Global Memory → Registers ===
        
        # HBM (global) → L2 cache → L1 cache → Registers
        grad_output = tl.load(grad_output_block_ptr, ...)  
        # Shape (16,) now in registers
        
        weight = tl.load(weight_block_ptr, ...)
        # Shape (64,) now in registers
        
        row = tl.load(x_block_ptr, ...)
        # Shape (16, 64) now in registers
        
        # === Phase 3: Compute in Registers ===
        # All these operations happen in registers/shared memory
        grad_x_row = grad_output[:, None] * weight[None, :]  # (16, 64)
        grad_weight_row = tl.sum(row * grad_output[:, None], axis=0)  # (1, 64)
        
        # === Phase 4: Store back to Global Memory ===
        # Registers → L1 cache → L2 cache → HBM (global)
        tl.store(grad_x_block_ptr, grad_x_row, ...)
        tl.store(partial_grad_weight_block_ptr, grad_weight_row, ...)
        
        # Advance pointers (just update metadata, no data movement)
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
```

#### What Triton Does Automatically

Triton's compiler automatically manages:

1. **Registers**: Thread-local variables
2. **Shared Memory**: For reductions and thread communication within a block
3. **Coalescing**: Combines small memory requests into large ones
4. **Caching**: Uses L1/L2 cache automatically

**Example: Reduction Uses Shared Memory**

```python
# When you do:
grad_weight_row = tl.sum(row * grad_output[:, None], axis=0)

# Triton internally:
# 1. Each thread computes partial sums (in registers)
# 2. Uses shared memory to communicate between threads
# 3. Does tree reduction in shared memory
# 4. Final result goes to registers → then stored to global memory
```

#### Memory Access Pattern in Backward Pass

```
Thread Block 0 (rows 0-15):

Iteration 0:
┌──────────────┐
│ Global Mem   │
│ (HBM)        │
└──────┬───────┘
       │ tl.load()
       ↓
┌──────────────┐
│ Registers    │  ← grad_output[0:16], weight[0:64], x[0:16, 0:64]
│              │  
│ Compute:     │  ← grad_x_row = grad_output[:, None] * weight[None, :]
│              │  ← grad_weight = tl.sum(...)
└──────┬───────┘
       │ tl.store()
       ↓
┌──────────────┐
│ Global Mem   │  ← Write grad_x[0:16, 0:64]
│ (HBM)        │  ← Write partial_grad_weight[0, 0:64]
└──────────────┘
```

#### Why Tiling Matters for Performance

**Bad: Many small loads**
```python
# This would be slow (not what Triton does):
for i in range(16):
    for j in range(64):
        val = load_one_element(i, j)  # 16*64 = 1024 global mem accesses!
        compute(val)
```

**Good: Tiled loads (what Triton does)**
```python
# Much faster:
row = tl.load(x_block_ptr, ...)  # Load entire (16, 64) tile at once
#     ↑ Coalesced memory access
#     ↑ Amortizes latency
compute(row)  # All in registers
```

### 6.11 Why the For Loop Is Efficient

You might wonder: "There are many loads/stores in the for loop - isn't that inefficient?"

**Answer: No! The for loop with tiling is actually MUCH more efficient than alternatives.**

#### Why We Can't Load Everything at Once

```python
# What we'd LIKE to do (but can't):
x = tl.load(x_ptr)  # Load ALL of x (64, 200)
weight = tl.load(weight_ptr)  # Load ALL weight (200,)
result = compute(x, weight)  # Compute once
tl.store(output_ptr, result)

# Problem: 64 * 200 * 4 bytes = 51 KB of data
# Plus weight: 200 * 4 = 0.8 KB
# Plus intermediate results, etc.
# 
# But we only have:
# - Registers: ~64 KB per SM (shared by many threads)
# - Shared Memory: ~100 KB per SM (shared by all threads in block)
# 
# If we try to load too much → not enough resources!
```

#### Resource Constraints

Each GPU Streaming Multiprocessor (SM) has limited resources:

```
Typical GPU (e.g., A100):
┌─────────────────────────────────┐
│ Per SM Resources:                │
│ - Registers: 65,536 × 32-bit    │ ← ~256 KB total
│ - Shared Memory: 164 KB          │
│ - Max threads: 2048              │
│                                  │
│ Must be shared among ALL threads!│
└─────────────────────────────────┘

If you try to use too many registers per thread:
→ Fewer threads can run simultaneously
→ Lower occupancy
→ Worse performance (can't hide memory latency)
```

#### Tiling Makes It Efficient

The for loop with tiling is actually a **good trade-off**:

**Bad: Load everything (doesn't fit)**
```python
# 64 rows × 200 cols = 12,800 floats = 51 KB
# Won't fit well in registers for many threads
x_all = tl.load(...)  # ❌ Too much data!
```

**Bad: Load one element at a time**
```python
# 200 iterations, each loading 1 element
for j in range(200):  # ❌ Way too many iterations!
    val = load_one_float(j)  # Terrible memory coalescing
    result += val * weight[j]
# This would be MUCH slower!
```

**Good: Tiled loading (what we do)**
```python
# Only ~3-16 iterations, each loading a tile
for i in range(tl.cdiv(200, 64)):  # ✓ Just 4 iterations for D=200, tile=64
    row = tl.load(x_block_ptr, ...)  # Load 16×64 = 1024 floats
    weight_chunk = tl.load(weight_block_ptr, ...)  # Load 64 floats
    
    # Do lots of computation with this data
    output += tl.sum(row * weight_chunk[None, :], axis=1)
    # 16 * 64 = 1024 multiplies + reductions per iteration
```

#### Arithmetic Intensity: Key to Performance

**Arithmetic Intensity** = (# of operations) / (# of bytes loaded)

```python
# Iteration 0 of the loop:

# Memory traffic:
# - Load x[0:16, 0:64]: 1024 floats = 4 KB
# - Load weight[0:64]: 64 floats = 256 bytes
# - Store output accumulation (just updates, stays in registers)
# Total: ~4.25 KB loaded

# Computation:
# - row * weight: 16 × 64 = 1024 multiplies
# - sum(axis=1): 1024 - 16 = 1008 additions
# - output +=: 16 additions
# Total: ~2048 operations

# Arithmetic intensity = 2048 ops / 4.25 KB ≈ 480 ops/KB

# This is GOOD! GPU memory bandwidth ~1-2 TB/s
# GPU compute: ~300 TFLOPS
# We want high arithmetic intensity to be compute-bound, not memory-bound
```

#### Memory Coalescing: The Secret Sauce

Each `tl.load()` in the loop benefits from **coalesced access**:

```python
# Thread block with 16 threads, each handling 1 row:

# When loading x[0:16, 0:64]:
# Thread 0: reads x[0, 0:64]  ← addresses 0-63
# Thread 1: reads x[1, 0:64]  ← addresses 512-575
# ...
# Thread 15: reads x[15, 0:64] ← addresses 7680-7743

# GPU combines these into large memory transactions
# Instead of 16 separate requests, it issues:
# - One 32-byte transaction for x[0,0:8]
# - One 32-byte transaction for x[0,8:16]
# - ... etc

# This is MUCH faster than 16 × 64 = 1024 individual loads!
```

#### Cache Reuse Between Iterations

```python
# The loop also benefits from L1/L2 cache:

Iteration 0: Load x[0:16, 0:64]    → might bring x[0:16, 0:127] into cache
Iteration 1: Load x[0:16, 64:128]  → cache hit! Already in L1/L2
Iteration 2: Load x[0:16, 128:192] → might be in cache if prefetched
```

#### Real Performance Numbers

**Scenario: Process 64 rows × 512 dims**

| Strategy | Iterations | Data/iter | Coalescing | Estimated Time |
|----------|-----------|-----------|------------|----------------|
| **Load all** | 1 | 128 KB | ❌ Won't fit | N/A (OOM) |
| **One element** | 512 | 4 bytes | ❌ Poor | ~100 μs |
| **Tile=32** | 16 | 2 KB | ✓ Good | ~5 μs |
| **Tile=64** | 8 | 4 KB | ✓ Good | ~3 μs |
| **Tile=128** | 4 | 8 KB | ✓ Good | ~2.5 μs |

The tiled approach (8-16 iterations) is **30-40× faster** than naive element-by-element!

#### Why ~16 Iterations?

```python
ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16

# Target ~16 iterations because:
# - Few enough iterations (low loop overhead)
# - Small enough tiles (fit in registers)
# - Large enough tiles (good memory coalescing)
# - Balance compute vs memory
```

### 6.12 Summary: Memory and Performance

| Concept | Reality |
|---------|---------|
| **make_block_ptr** | Just metadata, NO data movement |
| **tl.load()** | Global memory → Registers (via cache) |
| **tl.store()** | Registers → Global memory (via cache) |
| **Computation** | Happens in registers |
| **Shared memory** | Used automatically for reductions/sync |
| **Caching** | L1/L2 cache used automatically |
| **For loop iterations** | 8-16 iterations is optimal (not 1, not 512) |
| **Coalescing** | Adjacent threads access adjacent memory |
| **Arithmetic intensity** | High ops/byte ratio → compute-bound |

#### Compared to Manual CUDA

If you wrote this in raw CUDA, you'd have to manually:

```cuda
__shared__ float shared_data[TILE_SIZE];  // Declare shared memory

// Manually load to shared memory
shared_data[threadIdx.x] = global_data[blockIdx.x * TILE_SIZE + threadIdx.x];
__syncthreads();  // Wait for all threads

// Use shared memory
float result = compute(shared_data[...]);

// Write back to global
global_output[...] = result;
```

**With Triton**: It handles all this automatically! You just write:
```python
data = tl.load(...)  # Triton figures out optimal memory usage
result = compute(data)
tl.store(..., result)
```

The beauty of Triton is that it abstracts away the memory hierarchy while still generating efficient code that uses shared memory, coalescing, etc. under the hood.