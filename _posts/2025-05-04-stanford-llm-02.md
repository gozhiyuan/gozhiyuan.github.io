---
layout: post
title: Language Modeling Resource Accounting
subtitle: Language Modeling from Scratch Lecture 2
categories: Stanford-LLM-From-Scratch-2025
tags: [llm]
banner: "/assets/images/banners/yuanpang-wa-iceburg2.jpg"
---

# Language Model Training - PyTorch Primitives & Resource Accounting


This blog summarizes key concepts from Stanford CS336 Lecture 2, focusing on PyTorch primitives, efficient resource accounting (memory and compute), and foundational elements of training deep learning models from scratch.

[Course link](https://stanford-cs336.github.io/spring2025/)


## 1. Core Objectives and Motivation

The lecture emphasizes the practical and intuitive understanding needed to build and train language models efficiently. Efficiency translates directly into cost:

> "Efficiency is the name of the game. And to be efficient you have to know exactly how many flops you're actually expending..."

**Key motivating questions:**
- **Training Time**: How long to train a 70B parameter transformer on 15T tokens with 1,024 H100s? → ~144 days
- **Max Model Size**: What’s the largest model trainable on 8 H100s with AdamW? → ~40B parameters (HBM limited)


## 2. PyTorch Primitives: Tensors as Building Blocks

Tensors are pointers to allocated memory with metadata like shape and stride.

### 2.1 Floating-Point Formats in Deep Learning (float32, float16, bfloat16, fp8)


#### Float32 (FP32)
- **Bits**: 32 bits → 1 sign, 8 exponent, 23 fraction
- **Alias**: FP32, Single Precision
- **Memory Usage**: 4 bytes per element  
  - Example: 4×8 matrix = 128 bytes; GPT-3 FFN matrix ≈ 2.3 GB
- **Standard**: Gold standard in computing; default in PyTorch
- **Implications**:
  - Safe and stable for training
  - Memory-intensive
  - Slower on GPUs like H100 compared to lower precision formats


#### Float16 (FP16)
- **Bits**: 16 bits → 1 sign, 5 exponent, 10 fraction
- **Alias**: Half Precision
- **Memory Usage**: 2 bytes per element (half of float32)
- **Dynamic Range**: Limited → prone to underflow/overflow  
  - Example: `torch.tensor([1e-8], dtype=torch.float16)` → 0
- **Implications**:
  - Reduces memory, but causes numerical instability
  - Not recommended for training large models unless managed carefully


#### Bfloat16 (BF16)
- **Bits**: 16 bits → 1 sign, 8 exponent, 7 fraction
- **Alias**: Brain Float (Google Brain, 2018)
- **Memory Usage**: 2 bytes per element (like FP16)
- **Dynamic Range**: Same as float32
- **Resolution**: Worse than float32, but acceptable for DL
- **Implications**:
  - Better stability than FP16
  - Widely used for training on TPUs and H100 GPUs
  - Preferred for mixed precision training
  - Despite high theoretical throughput, MFU may be lower than expected


#### FP8 (8-bit)
- **Bits**: 8 bits  
- **Variants**:
  - **E4M3**: 4 exponent bits, 3 mantissa → better resolution  
  - **E5M2**: 5 exponent bits, 2 mantissa → better range
- **Memory Usage**: 1 byte per element
- **Hardware Support**: H100 GPUs only
- **Implications**:
  - Very fast, highly memory-efficient
  - Training is unstable without specialized techniques
  - Still experimental, but promising for future DL efficiency

#### Mixed Precision Training

**Trade-off**:
- High precision (e.g., float32): Accuracy & stability, but costly
- Low precision (e.g., BF16, FP8): Efficient, but unstable if misused

**Best Practice**:
- **Use float32** for:
  - Parameters
  - Optimizer states
- **Use BF16 / FP8** for:
  - Forward pass (activations)

**Tools**:
- PyTorch AMP (Automatic Mixed Precision): Handles casting automatically

**Inference**:
- Post-training quantization often allows aggressive precision reduction, improving speed and memory footprint with minimal accuracy loss.



### 2.2 Tensor Location

- Default: CPU  
- For GPU: `.to("cuda:0")` or use `device='cuda'`  
- Data transfer is expensive

### 2.3 Tensor Views and Operations

What are tensors in PyTorch? PyTorch tensors are pointers into allocated memory with metadata describing how to get to any element of the tensor.

```python
def tensor_storage():
    x = torch.tensor([
        [0., 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
    ])
    To go to the next row (dim 0), skip 4 elements in storage.
    assert x.stride(0) == 4
    To go to the next column (dim 1), skip 1 element in storage.
    assert x.stride(1) == 1
    To find an element:
    r, c = 1, 2
    index = r * x.stride(0) + c * x.stride(1)  # @inspect index
    assert index == 6
```


In PyTorch, tensors are fundamental building blocks that act as pointers into some allocated memory, accompanied by metadata that specifies how to access any element within that memory. Understanding how PyTorch handles tensors, particularly with "views" and "contiguous" memory, is crucial for efficient resource accounting in deep learning.


### 2.4 Operations for Views (No Copy)

Many PyTorch operations provide a different "view" of an existing tensor rather than creating a new copy. This means that the new tensor shares the same underlying allocated memory as the original tensor.

- **Slicing**: Accessing rows/columns (e.g., `x[0]`, `x[:, 1]`) returns views.
- **`view()` function**: Reshapes a tensor without copying, e.g., turning a 2×3 matrix into 3×2.
- **`transpose()`**: Produces a view by switching dimensions (no data copy).

- **Shared memory**: Mutating the original tensor will affect any views and vice versa.
- **Efficiency**: Views are "free" in terms of memory and compute cost since no allocation or copying is done.


### 2.5 Contiguous Tensors

The concept of "contiguous" refers to how a tensor's elements are arranged in its underlying memory.
A tensor is **contiguous** if:
- Its elements are stored **sequentially** in memory.
- Iterating through the tensor (e.g., row-by-row) maps directly to stepping through the memory array.
- Example: A 4×4 matrix has strides of `[4, 1]`, so to get to the next row (dim 0), skip 4 elements; to get to the next column (dim 1), skip 1.

- Some view operations like `transpose()` create **non-contiguous** tensors.
- Elements are no longer sequential in memory even though the tensor structure is still valid.


#### Limitations and Solutions

- **Issue**: Non-contiguous tensors may throw a `RuntimeError` if passed into functions like `view()` that assume contiguous memory layout.
- **Solution**: Use `.contiguous()` to force a copy of the data into a new, contiguous memory layout.

#### `.contiguous()`:
- Allocates new memory and copies the data in sequential layout.
- Allows safe reshaping or viewing after complex operations.

#### `.reshape()` vs. `.view()`:
- Both can reshape tensors.
- If the tensor is **non-contiguous**, `reshape()` will allocate new memory (like `.contiguous().view()`).
- Be mindful: reshaping a non-contiguous tensor incurs memory and compute overhead.


**Summary**:
- Use view operations (like slicing, view, transpose) for efficient memory use.
- Check `.is_contiguous()` if unsure about memory layout.
- Use `.contiguous()` to make data layout compatible with operations that require sequential memory.


### 2.6 Batched Matrix Multiplication in PyTorch

For tensor multiplication with multiple dimensions, PyTorch's matrix multiplication operation (using the `@` operator or `torch.matmul`) performs what's known as a **batched matrix multiplication**. This means it conceptually iterates over the **leading dimensions** of the tensors, applying standard **2D matrix multiplication** to the **innermost two dimensions**.

#### Key Concepts

- **The "Bread and Butter" Operation**:  
  Matrix multiplication is considered the *"bread and butter of deep learning"*.

- **Batching in Deep Learning**:  
  In machine learning applications, operations are generally performed in batches. For language models, this typically means performing operations "for every example in a batch and for every sequence in a batch".

- **How PyTorch Handles It**:  
  When tensors with more than two dimensions are involved, PyTorch efficiently performs batched matrix multiplication.  
  For example, if you have a tensor `x` of shape `(batch, sequence, rows, columns)` and you multiply it by a matrix `w` of shape `(columns, new_columns)`, PyTorch will perform the matrix multiplication **for each (batch, sequence) pair**.

#### Example

```python
x = torch.ones(4, 8, 16, 32)
w = torch.ones(32, 2)
y = x @ w  # shape: (4, 8, 16, 2)
```

Here, PyTorch automatically applies the matrix multiplication of shape (16, 32) @ (32, 2) across all combinations of the first two dimensions (batch and sequence). In other words, for every batch and sequence element, PyTorch multiplies x[b, s, :, :] @ w, resulting in a shape of (16, 2). Final output shape: (4, 8, 16, 2).

#### Why It Matters
- No Explicit Loops Required:
PyTorch eliminates the need to manually loop over batches or sequences. You write clean, vectorized code, and PyTorch handles the parallelization.

- Optimized for Hardware:
These operations are highly optimized and take advantage of modern hardware (e.g., Tensor Cores on GPUs), leading to much faster computations than manual implementations.

"In this case, we iterate over values of the first 2 dimensions of x and multiply by w."


### 2.7 Einops: Named Dimensions

- Avoid `transpose(-2, -1)` confusion  
- Use `einops.rearrange`, `reduce`, `einsum`

```python
def einops_einsum():
    Einsum is generalized matrix multiplication with good bookkeeping.
    Define two tensors:
    x: Float[torch.Tensor, "batch seq1 hidden"] = torch.ones(2, 3, 4)  # @inspect x
    y: Float[torch.Tensor, "batch seq2 hidden"] = torch.ones(2, 3, 4)  # @inspect y
    Old way:
    z = x @ y.transpose(-2, -1)  # batch, sequence, sequence  @inspect z
    New (einops) way:
    z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")  # @inspect z
    Dimensions that are not named in the output are summed over.
    Or can use ... to represent broadcasting over any number of dimensions:
    z = einsum(x, y, "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2")  # @inspect z
```

When discussing the efficiency of tensor operations in PyTorch, especially multi-dimensional ones that conceptually involve iteration, `torch.compile` plays a significant role in ensuring these operations are executed as efficiently as possible.

Here's how `torch.compile` relates to the discussion:

- **Optimized Code Generation**: If you are using `torch.compile`, it *"will generate the code that will use the hardware properly"*. This means `torch.compile` aims to optimize the underlying execution of your PyTorch code, leveraging the specialized hardware available.

- **Efficiency for `einsum`**: In the context of `einops` (which provides `einsum` for generalized matrix multiplication), a question was raised about whether `torch.compile` guarantees efficient compilation. The short answer given was **"yes"**. It was explained that `torch.compile` *"will figure out the best way to reduce the best order of dimensions to reduce and then use that"*. This means `torch.compile` can determine the most efficient way to perform the sum-over operations implied by `einsum` notation.

- **One-Time Optimization and Reuse**: When `einsum` is used within `torch.compile`, the optimization process *"only do that one time and then you know reuse the same implementation over and over again"*. This makes the execution *"better than anything designed by hand"*. This reiterates that while there's a conceptual "iteration" or "summing over" in multi-dimensional operations, `torch.compile` ensures the actual low-level implementation is highly optimized and performed in parallel by the hardware — not through inefficient Python loops.

- **Leveraging Tensor Cores**: The efficiency of PyTorch operations, including those optimized by `torch.compile`, relies on the use of specialized hardware like Tensor Cores. These are *"specialized hardware to do matmul"* (matrix multiplications). By default, PyTorch *"should use it"*, and `torch.compile` further ensures that the generated code effectively utilizes these specialized components.

**Summary**: While operations like batched matrix multiplication conceptually iterate over leading dimensions, `torch.compile` acts as an advanced optimization tool to ensure that this conceptual iteration is translated into highly efficient, often parallelized, low-level code that fully exploits the capabilities of GPUs and their Tensor Cores. This is why you do **not** need to write explicit for loops for higher dimensions in PyTorch, and why these operations remain **highly performant** despite their multi-dimensional nature.


## 3. Compute Accounting: FLOPs and MFU

### 3.1 FLOPs Intuition

- **GPT-3**: ~3.14e23 FLOPs  
- **GPT-4** (speculative): ~2e25 FLOPs  
- H100 (dense): ~989 TFLOP/s  

**Data Type Impact:**
- Performance improves with FP16, bfloat16, FP8
- Sparse FLOP/s are often inflated

A100 has a peak performance of 312 teraFLOP/s
H100 has a peak performance of 1979 teraFLOP/s with sparsity, 50% without
So 8 H100s for 2 weeks: total_flops = 8 * (60 * 60 * 24 * 7) * h100_flop_per_sec  # @inspect total_flops

### 3.2 FLOPs for Matrix Mults

#### Matrix Multiplication (Matmul) FLOPs

- **Importance**: Matrix multiplication is considered the "bread and butter of deep learning" and generally dominates the compute cost for large enough matrices.

- **FLOPs Rule of Thumb**:  
  For a matrix multiplication, the number of **FLOPs** is approximated as:
  $[
  2 \times B \times D \times K
  ]$
  where:
  - $( B )$: batch size (or number of data points),
  - $( D )$: input dimension,
  - $( K )$: output dimension.

  This is because for every element in the output matrix, there is typically one multiplication and one addition (2 operations per output element).

- **Example**:  
  Multiplying a $( B \times D )$ matrix $( x )$ by a $( D \times K )$ matrix $( w )$ results in a $( B \times K )$ output and costs $( 2 \times B \times D \times K )$ FLOPs.

- **Transformer Generalization**:  
  This approximation generalizes to the Transformer architecture, where most of the computation comes from matmuls in attention and feedforward layers.

- **Multi-Dimensional Tensors**:  
  For tensors like `x` of shape `(batch, sequence, 16, 32)` and `w` of shape `(32, 2)`, PyTorch performs **batched matrix multiplication**:
  - Conceptually iterates over leading dimensions `(batch, sequence)`.
  - Applies 2D matmul to the trailing dimensions `(16, 32) @ (32, 2)`.
  - Efficiently implemented in PyTorch's backend (C++/CUDA), often leveraging **Tensor Cores**.
  - No need for explicit Python for-loops.
  - `torch.compile` can further optimize this by choosing the best reduction and fusing operations.


#### Element-wise Operation FLOPs

- **Nature of Operation**:  
  Element-wise ops apply a scalar or function to **each element** of a tensor. Examples include:
  - `pow`, `sqrt`, `rsqrt`, `+`, `*`

- **FLOPs Calculation**:  
  For a tensor of shape $( m \times n )$, the number of FLOPs is:
  $[
  O(m \times n)
  ]$

- **Implication**:  
  These are generally much cheaper than matmuls and scale linearly with the number of elements.


#### Addition Operation FLOPs

- **FLOP Definition**:  
  FLOPs include both multiplications and additions.

- **Matrix Addition**:  
  Adding two $( m \times n )$ matrices element-wise requires:
  $[
  m \times n \text{ FLOPs}
  ]$

- **Equivalence**:  
  For FLOP counting, **additions are considered equal to multiplications** in cost.

#### Which is More Expensive in FLOPs: Element-wise Operations or Matrix Multiplication?

- **Matrix Multiplication (Matmul)** is generally **much more expensive** in FLOPs compared to element-wise operations.

- **Matmul FLOPs** grow **cubically** (or at least quadratically with large dimensions), since each output element involves multiple multiply-add operations.

- **Element-wise FLOPs** grow **linearly** with the number of elements because each element is processed independently with a single operation.  

Examples

1. **Matrix Multiplication**

   Multiply matrices:
   - $( A )$ of shape $( 1024 \times 512 )$
   - $( B )$ of shape $( 512 \times 256 )$

   FLOPs:
   $[
   2 \times 1024 \times 512 \times 256 = 268,435,456 \text{ FLOPs (about 268 million)}
   ]$

2. **Element-wise Operation**

   Square every element of a $( 1024 \times 512 )$ matrix:

   FLOPs:
   $[
   1024 \times 512 = 524,288 \text{ FLOPs (about 0.5 million)}
   ]$


- Matrix multiplication in the example costs **~268 million FLOPs**.
- Element-wise squaring of the same sized matrix costs only **~0.5 million FLOPs**.

**Conclusion**:  
Matrix multiplication can be **hundreds of times more expensive** than element-wise operations for typical deep learning tensor sizes. This is why matmul usually dominates training and inference computation cost.


```python
device = get_device()
x = torch.ones(B, D, device=device)
w = torch.randn(D, K, device=device)
y = x @ w
actual_num_flops = 2 * B * D * K  
# We have one multiplication (x[i][j] * w[j][k]) and one addition per (i, j, k) triple.
```

- Matmul is __Dominant cost in DL__
- FLOPs = `2 × B × D × K` for (B×D) × (D×K)  
- Forward pass: ~`2 × tokens × params` FLOPs
- Total (fwd + bwd): ~`6 × tokens × params`

### 3.3 MFU: Model FLOPs Utilization

- Matrix multiplications dominate: (2 m n p) FLOPs   
- FLOP/s depends on hardware (H100 >> A100) and data type (bfloat16 >> float32)  
- Model FLOPs utilization (MFU): (actual FLOP/s) / (promised FLOP/s)
- MFU ≥ 0.5 is good  
- Tensor cores improve matmul efficiency

## 4. Gradients and Backward Pass

### Forward Pass FLOPs

For a **simple linear model** that maps a $D$-dimensional input to $K$ outputs across $B$ data points:

- **Single Layer**:
  - Input: $x$ of shape $(B, D)$  
  - Weights: $w$ of shape $(D, K)$  
  - FLOPs:  
    $$
    2 \times B \times D \times K
    $$

- **Two-Layer Linear Model**:
  - First layer: $x \rightarrow h_1 = xW_1$, with $W_1 \in \mathbb{R}^{D \times D}$  
  - Second layer: $h_1 \rightarrow h_2 = h_1W_2$, with $W_2 \in \mathbb{R}^{D \times K}$  
  - Total forward FLOPs:  
    $$
    2 \times B \times D \times D + 2 \times B \times D \times K
    $$
  - Equivalent to:  
    $$
    2 \times B \times \text{(number of parameters)}
    $$

- **General Rule**:
  $$
  \text{Forward FLOPs} \approx 2 \times \text{batch size} \times \text{number of parameters}
  $$

### Backward Pass FLOPs

The **backward pass** computes gradients and typically costs more:

- **Gradient of $W_2$**:
  - Uses: $( h_1^T \cdot \frac{dL}{dh_2} )$  
  - FLOPs:  
    $$
    2 \times B \times D \times K
    $$

- **Gradient of $h_1$**:
  - Uses: $( \frac{dL}{dh_2} \cdot W_2^T )$  
  - FLOPs:  
    $$
    2 \times B \times D \times K
    $$

- **Gradient of $W_1$**:
  - Uses: $( x^T \cdot \frac{dL}{dh_1} )$  
  - FLOPs:  
    $$
    2 \times B \times D \times D
    $$

- **Total Backward Pass FLOPs (2-layer model)**:
  $$
  \approx 4 \times B \times \text{(number of parameters)}
  $$

- **General Rule**:
  $$
  \text{Backward FLOPs} \approx 2 \times \text{Forward FLOPs}
  $$

### Total Training Step FLOPs

- **Combined Forward + Backward Pass**:
  $$
  \text{Total FLOPs} \approx 6 \times B \times \text{(number of parameters)}
  $$
- This **"rule of six"** is used for napkin math when estimating training compute.


### Micro-Level Breakdown of One Weight Update

For a weight $w$ connecting unit $i$ to unit $j$:

1. **Forward pass**:  
   $( h^{(i)} \cdot w \rightarrow a^{(j)} )$

2. **Add to $j$'s input accumulator**:  
   $( a^{(j)} += h^{(i)} \cdot w )$

3. **Backward pass to $i$**:  
   $( \frac{dL}{da^{(j)}} \cdot w \rightarrow \frac{dL}{dh^{(i)}} )$

4. **Accumulate to $i$’s gradient**:  
   $( \frac{dL}{dh^{(i)}} += \ldots )$

5. **Compute gradient for $w$**:  
   $( \frac{dL}{dw} += h^{(i)} \cdot \frac{dL}{da^{(j)}} )$

6. **(Sneaky FLOP)** Accumulate over batch:  
   $( \text{dL/dw} += \text{contribution from example} )$

Each step involves multiply-add FLOPs. Summed across all weights and data points, this explains the total training cost.

- `loss.backward()` handles autograd  
- **Backward FLOPs ≈ 4× tokens × params**
- **Total FLOPs = 6× tokens × params** (fwd + bwd)

## 5. Building a Model: Parameters, Init, Training Loop

### 5.1 Parameters

- Stored as `nn.Parameter`, tracked for grads

### 5.2 Initialization

- Naive init → exploding values  
- Use **Xavier** or `nn.init.trunc_normal_`  

### 5.3 GPU Transfer

- `model.to(get_device())`  
- Always verify tensor/model device

### 5.4 Randomness

- Fix seeds for `torch`, `numpy`, and `random`  
- Determinism aids reproducibility

### 5.5 Data Loading

- Tokenized sequences, often NumPy arrays  
- Use `np.memmap` for TB-scale data  
- Use `pin_memory=True`, `non_blocking=True` for overlap

### 5.6 Optimizers
- **Adam**: combo of Momentum + RMSProp  
- **Memory Cost**:
  - Gradients: same size as params
  - States (Adam): ~2× params

- Use `optimizer.zero_grad(set_to_none=True)` for memory savings

The **"optimizer state"** refers to additional information that optimizers (such as AdaGrad or Adam) maintain throughout training to effectively update model parameters. These states are stored as tensors.

#### 🔧 What It Stores

- Optimizer state stores **running averages or sums** of gradients (or squared gradients).
- These help adapt the learning rate for each parameter over time.
- **Example**:  
  - In **AdaGrad**, the optimizer state stores `g2` — the sum of squared gradients.
  - In **Adam**, the state includes:
    - `m`: exponential moving average of gradients.
    - `v`: exponential moving average of squared gradients.

#### 💾 Memory Requirements

- The optimizer state contributes **significantly to memory usage** during training.
- For many optimizers, the state size is proportional to the number of parameters:
  - A model with `N` parameters will typically have `O(N)` memory for the optimizer state.
- **Napkin Math**:
  - Using `float32` (4 bytes per value):
    - Parameters: 4 bytes
    - Gradients: 4 bytes
    - Optimizer state:
      - AdaGrad: 4 bytes (1 accumulator)
      - AdamW: 8 bytes (2 accumulators: `m`, `v`)
  - **Total** (AdamW):  
    $$
    \text{Memory per parameter} = 4 + 4 + 4 + 4 = 16 \text{ bytes}
    $$
  - This rough estimate helps determine how large a model can fit into memory.

#### 🧠 Data Types and Precision

- **Float32** is typically used for:
  - Model parameters
  - Optimizer state
- Lower precision types (`bfloat16`, `float16`) can be used for:
  - Forward pass
  - Backward pass (with caution)
- **Why float32 for optimizer state**:
  - Ensures **stability** and **accurate accumulation** across training steps.
  - Important in **mixed precision training**, where:
    - Activations and intermediate computations use low precision.
    - Parameters and optimizer state remain in full precision for safety.


#### 🔁 Role in Training Loop & Checkpointing

- In each training step:
  1. Compute gradients via backward pass.
  2. Optimizer uses:
     - Gradients
     - Internal optimizer state
     - to update model parameters.
- **Checkpointing**:
  - Crucial for long training runs (e.g., large LLMs).
  - A complete checkpoint must include:
    - `model.state_dict()` (parameters)
    - `optimizer.state_dict()` (optimizer state)
  - This enables training to **resume exactly** where it left off after a crash.


#### Summary
- Optimizer state is a **critical hidden cost** in training.
- Its size and precision directly affect:
  - Memory footprint
  - Training stability
  - Checkpoint reliability


### 5.7 Total Memory Accounting (Assuming FP32)

- Parameters  
- Activations: `B × D × num_layers`  
- Gradients  
- Optimizer state  
- **Total = 4 × (params + activations + grads + optimizer_state)**

### 5.8 Training Loop

```python
model = MyModel().to(device)
optimizer = torch.optim.Adam(model.parameters())

for x, y in dataloader:
    x, y = x.to(device), y.to(device)
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
