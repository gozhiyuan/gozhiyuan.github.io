---
layout: post
title: GPUs for Deep Learning 🚀
subtitle: Language Modeling from Scratch Lecture 5
categories: Large-Language-Model
tags: [Stanford-LLM-From-Scratch-2025]
banner: "/assets/images/banners/yuanpang-wa-iceburg2.jpg"
---

# GPUs for Deep Learning 🚀

This lecture synthesizes key insights on GPUs, focusing on their architecture, performance bottlenecks, and advanced optimization techniques crucial for scaling large language models (LLMs).

- **🔥 Core Message:** While GPU computational power (especially for matrix multiplications) has scaled exponentially, memory access speed has not kept pace, making **memory the primary bottleneck**.
- **🧠 Key to Performance:** Effective GPU utilization hinges on minimizing slow global memory access and maximizing the use of fast on-chip memory.
- **🛠️ Essential Techniques:** Low-precision computation, operator fusion, recomputation, memory coalescing, and especially tiling, are essential for achieving high performance.
- **✨ Case Study: FlashAttention:** These principles are exemplified by the architecture of FlashAttention, which dramatically accelerates the attention mechanism by cleverly applying these memory-aware optimizations.

[Course link](https://stanford-cs336.github.io/spring2025/)

## 1. GPU Architecture: Optimizing for Throughput
### 1.1. CPU vs. GPU Fundamental Differences
The fundamental distinction between CPUs and GPUs lies in their design goals:

- **🐢 CPUs (Central Processing Units):** Optimize for **latency**, aiming to complete a few tasks as quickly as possible. They feature large control units, robust branching support, and large caches, but fewer processing cores.
- **🐇 GPUs (Graphics Processing Units):** Optimize for **throughput**, aiming to process a large volume of data in parallel. They possess "many tiny compute units (ALUs)" and are designed for "many many threads," making them ideal for parallelizable workloads like deep learning.

### 1.2. GPU Anatomy and Execution Model
![alt_text](/assets/images/llm-from-scratch/05/1.png "image_tooltip")
- **Streaming Multiprocessors (SMs):** GPUs contain numerous SMs, which act as independent processing units capable of executing "blocks" (jobs). An A100 GPU has 108 SMs.
- **Streaming Processors (SPs):** Each SM contains many SPs, which execute "threads" in parallel.
- **Threads, Blocks, and Warps:**
    - **Threads:** Individual units of work, executing "the same instructions but with different inputs (SIMT)."
    - **Blocks:** Groups of threads assigned to an SM, with their own shared memory.
    - **Warps:** Threads execute in groups of 32. This Single Instruction, Multiple Thread (SIMT) model means all threads in a warp execute the same instruction.

### 1.3. GPU Memory Hierarchy
Memory proximity to the SM dictates speed:

- **🥇 Registers, L1 Cache, Shared Memory:** The fastest, located inside the SM. Shared memory is crucial for data reuse within a block.
- **🥈 L2 Cache:** On-die, but outside the SM, offering reasonable speed (approx. 10x slower than L1/shared memory).
- **🥉 Global Memory (DRAM/HBM):** The slowest memory, located off-chip. Accessing global memory is significantly slower (200-300 clock cycles vs. 20 for on-SM memory).
> "Information that goes across blocks need to be read/written to global memory (slow)."

### 1.4. GPUs as Fast Matrix Multipliers
- **Programmable Shaders:** Early GPUs were leveraged for matrix multiplications through programmable shaders.
- **Tensor Cores:** Modern GPUs include specialized "Tensor cores... specialized matrix multiplication circuits."
> "Matmuls are >10x faster than other floating point ops!"
This specialization is why "if you're going to design any sort of a neural architecture... you have to have most of your workload be matrix multiplies."

### 1.5. Compute vs. Memory Scaling
A critical observation is that "FLOPs scale faster than memory – it’s hard to keep our compute units fed with data!"
- Compute capabilities have scaled at an "astoundingly fast" rate.
- Memory bandwidth has grown much slower.
This widening gap means "your bottlenecks are probably going to end up being memory." Therefore, optimizing memory movement is paramount for high performance.

![alt_text](/assets/images/llm-from-scratch/05/2.png "image_tooltip")

### 1.6. TPUs: Similar Principles
TPUs (Tensor Processing Units) share many conceptual similarities with GPUs, featuring "lightweight control, fast (big) matmul unit, fast memory."

## 2. Understanding GPU Performance Bottlenecks and Optimization
GPU performance can be complex, often characterized by the "roofline model," which distinguishes between memory-bound and compute-bound regimes. The goal is to avoid being memory-bound.

Key optimization tricks:
- Control divergence
- Low precision computation
- Operator fusion
- Recomputation
- Coalescing memory
- Tiling

### 2.1. Avoiding Memory Bottlenecks: Key Tricks
#### 2.1.1 Control Divergence (Non-Memory Issue)
In the SIMT model, all threads in a warp execute the same instruction. Conditional statements can cause "significant overhead" because threads that do not meet the condition are paused, forcing serialized execution.
> "Conditional statements within a single warp... can be really really damaging."

#### 2.1.2 Low Precision Computation (Quantization)
Using fewer bits (e.g., FP16 instead of FP32) "improves arithmetic intensity" by reducing the amount of data moved per operation.

#### 2.1.3 Operator Fusion
![alt_text](/assets/images/llm-from-scratch/05/3.png "image_tooltip")
Operator fusion is a technique to make ML workloads faster on a GPU by minimizing access to slow global memory.

Think of a GPU like a factory:
- 🏭 **Factory:** The compute units.
- 📦 **Warehouse:** The memory from which inputs are drawn.
- 🚚 **Conveyor Belts:** The finite bandwidth for transferring data.

The core issue is the **memory bottleneck**: compute capabilities scale faster than memory bandwidth.

**Problem with Naive (Non-Fused) Computation:**
- Data is repeatedly moved back and forth between the "warehouse" (memory) and the "factory" (compute unit).
- This constant "shipping back and forth" incurs significant memory overhead and leaves compute units idling.

**Solution with Fused Kernels:**
- A "fused kernel" keeps data within faster, on-chip memory (shared memory or registers) for as long as possible.
- It performs all dependent operations on a piece of data sequentially before writing the final result back to global memory.
- This dramatically reduces memory round trips.

**Example: `sin^2(x) + cos^2(x)`**
- **Naive Approach:** Would launch five separate CUDA kernels, with lots of "back and forth" data movement for intermediate results (`sin(x)`, `cos(x)`, etc.).
- **Fused Approach:** All five operations are fused into a single kernel call. Intermediate results stay in fast on-chip memory.

**Automatic Fusion:**
- Compilers like `torch.compile` can perform many fusions automatically. Using such tools is strongly encouraged!

#### 2.1.4 Recomputation (Memory-Compute Trade-off)
> "Throwing away computation can actually be optimal."
Instead of storing intermediate activations (which is memory-intensive), one can recompute them on the fly. This "trades compute which you have too much of for memory bandwidth which you had too little of." This is the same principle as gradient checkpointing.

#### 2.1.5 Coalescing Memory Accesses
Global memory is read in "burst mode."
> "Memory accesses are coalesced if all the threads (in a warp) fall within the same burst."
If threads in a warp access contiguous memory locations, the hardware can fetch much more data in a single operation, effectively increasing memory throughput.

![alt_text](/assets/images/llm-from-scratch/05/4.png "image_tooltip")

#### 2.1.6 Tiling (The Big One)
Tiling is the practice of "grouping and ordering threads to minimize global memory access." It's a crucial technique to overcome the memory bottleneck.

![alt_text](/assets/images/llm-from-scratch/05/5.png "image_tooltip")
![alt_text](/assets/images/llm-from-scratch/05/6.png "image_tooltip")

**The Core Idea of Tiling:**
- Break down a large computation (like a matrix multiplication) into smaller, manageable "tiles."
- Load these tiles into the GPU's fast shared memory.
- Perform a significant amount of computation on the tile data before writing the final results back to slow global memory.

**Tiling for Matrix Multiplication: An Example**
- **Problem with Naive Matrix Multiplication:** Each input element might be read multiple times from global memory, and memory access might not be coalesced.
- **How Tiling Improves It:**
    1.  **Cut into Tiles:** Large matrices are logically cut into smaller sub-matrices.
    2.  **Load to Shared Memory:** Tiles are loaded into the fast shared memory of an SM.
    3.  **Compute in Phases:** Partial sums for the output matrix are computed using the tiles in shared memory.
    4.  **Reuse Data:** Repeated reads for computations within a tile now access the fast shared memory, not global memory.
    5.  **Coalesced Access:** Loading an entire tile can be done with coalesced memory access, further speeding up the initial load.

**Advantages of Tiling:**
- **⬇️ Reduced Global Memory Access:** The primary benefit.
- **🔥 Improved Arithmetic Intensity:** Increases the ratio of FLOPs to memory bytes accessed.
- **ιεραρχία Better Utilization of Memory Hierarchy:** Leverages the GPU's memory hierarchy effectively.
- **🤝 Enables Coalescing:** Allows for more predictable and structured memory access patterns.

**Complexities and Challenges with Tiling:**
- **Tile Size Optimization:** Choosing the optimal tile size is critical and depends on shared memory size, coalesced access patterns, and matrix divisibility.
- **Memory Alignment and Padding:** If matrix dimensions are not multiples of the memory burst size, padding might be needed to avoid performance degradation.
- **Implementation Complexity:** Tiled algorithms are more complex to implement than naive approaches.

### 2.2. Matrix Mystery: Why Bigger Matrices and Specific Sizes are Faster
![alt_text](/assets/images/llm-from-scratch/05/8.png "image_tooltip")
The "unpredictable looking wavelike patterns" in GPU performance for matrix multiplications can be explained by:

- **Roofline Model:** As matrix size increases, performance improves due to better compute intensity (compute-bound regime).
- **Tiling Alignment and Divisibility:** Performance drops significantly when matrix dimensions are not divisible by certain numbers (e.g., 32, 16, 8). This prevents efficient tiling and coalesced memory reads.
- **Wave Quantization:** Sharp drops in performance at specific matrix sizes occur when the number of tiles exceeds the available SMs on the GPU. This forces some SMs to run low-utilization tiles, causing overall performance to plummet.
> "An A100 has 108 SMs, so it cannot execute all 120 tiles."

![alt_text](/assets/images/llm-from-scratch/05/7.png "image_tooltip")

## 3. FlashAttention: A Case Study in GPU Optimization
FlashAttention is a prime example of applying these GPU optimization principles to accelerate the transformer's attention mechanism. It tackles the challenge of "computing exact attention in sub quadratic HBM accesses."

### 3.1. Attention Computation Recap
Attention involves three matrix multiplies (Q, K, V) and a softmax operation. The challenge is the softmax, a global operation that is traditionally problematic for tiling.

### 3.2. FlashAttention's Core Techniques
- **Tiling for KQV Matrix Multiply:** Ensures that the matrix multiplications are performed with minimal global memory access.
![alt_text](/assets/images/llm-from-scratch/05/9.png "image_tooltip")

- **Incremental (Online) Softmax Computation:** Allows the softmax to be computed "tile-by-tile" by incrementally updating the maximum value and a telescoping sum. This avoids "materializ[ing] the full N^2 matrix."
- **Fusion of Exponential Operator:** The exponential operation within the softmax is fused with other operations.
- **Recomputation for Backward Pass:** Uses recomputation "tile by tile" to avoid storing the large N^2 sized softmax activations.

By integrating these techniques, FlashAttention achieves significant speedups by "think[ing] carefully about the GPU (coalescing, tiling, fusion)."

### 3.3 Online Softmax Example
Normal softmax over a sequence of scores `x_1, ..., x_n` is:
`softmax_i = exp(x_i) / Σ_j exp(x_j)`

This is problematic because you need all `x_j` to compute the denominator, and large `x` values can cause numerical instability.

**Online softmax** streams through the data, maintaining:
- `m_j`: maximum of values seen so far.
- `l_j`: sum of exponentials adjusted by the current max.

**Example — Element-wise for `[2.0, 1.0, 5.0]`**

**Step 1: Start with `2.0`**
- `m_1 = 2.0`
- `l_1 = exp(2.0 - 2.0) = 1.0`

**Step 2: Add `1.0`**
- `m_2 = 2.0` (max doesn't change)
- `l_2 = l_1 + exp(1.0 - 2.0) = 1.0 + 0.3679 = 1.3679`

**Step 3: Add `5.0` (new max!)**
- `m_3 = 5.0`
- Rescale old sum: `l_2 * exp(m_2 - m_3) = 1.3679 * exp(-3.0) = 0.0681`
- Add new term: `exp(5.0 - 5.0) = 1.0`
- `l_3 = 0.0681 + 1.0 = 1.0681`

**Final softmax values:**
`softmax = [0.0466, 0.0171, 0.9363]`

**Why This Helps in FlashAttention:**
- Avoids materializing the whole N×N attention score matrix.
- Processes each tile in fast shared GPU memory.
- Improves GPU compute utilization by keeping data on-chip.

![alt_text](/assets/images/llm-from-scratch/05/10.png "image_tooltip")
