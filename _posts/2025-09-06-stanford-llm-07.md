---
layout: post
title: LLM Training Parallelism Basics
subtitle: Language Modeling from Scratch Lecture 7
categories: Large-Language-Model Distributed-Training
tags: [Stanford-LLM-From-Scratch-2025]
banner: "/assets/images/banners/yuanpang-wa-iceburg2.jpg"
---

# LLM Training Parallelism Basics

**"Parallelism Basics"** focuses on the system complexities behind training massive language models (LMs) that **exceed a single GPU’s capacity**.  
Goals:
- Understand different **parallelization paradigms**.  
- Learn **why multiple methods** are combined.  
- See how **large-scale training** is organized.

[Course link](https://stanford-cs336.github.io/spring2025/)  
[Code Link in the next lecture](https://stanford-cs336.github.io/spring2025-lectures/?trace=var/traces/lecture_08.json)


## ⚙️ 1. Limits of Single-GPU Scaling

Training on a single GPU faces two bottlenecks:

### 🧮 Compute
- Even though GPUs are powerful, **exaflop-scale** compute (like supercomputers) is needed for huge models.

### 💾 Memory
- Large models exceed single-GPU memory.
- Training uses **≈5 copies of weights** → up to **16 bytes/parameter** due to optimizer state.
- Therefore, we use **multi-GPU, multi-machine parallelism**:
  - **Intra-node:** Fast (within one machine)
  - **Inter-node:** Slower (across machines)


## 🌐 2. Networking and Communication Basics

### 🏢 Datacenter as the Unit of Compute
What we want from multi-machine scaling: Scaling targets **linear memory and compute growth** proportional to GPU count.
- Linear memory scaling (max model params scales with num gpus)
- Linear compute scaling (model flops scale linearly with num gpus)

### 📡 Collective Communication Primitives

![alt_text](/assets/images/llm-from-scratch/07/1.png "image_tooltip")

| Primitive | Description | Approx. Cost |
|------------|--------------|---------------|
| 🟢 **All Reduce** | Sums inputs → copies to all machines | `≈ 2 × #params` |
| 📣 **Broadcast** | Sends one input to all ranks | — |
| 📦 **Reduce** | Sums → sends to one machine | — |
| 🔁 **All Gather / Reduce Scatter** | Key building blocks; `All Reduce = Reduce-Scatter + All-Gather` | Efficient in bandwidth-limited settings |

### 🧩 Hardware Design Differences

| Hardware | Networking | Description |
|-----------|-------------|-------------|
| 🎮 **GPU** | All-to-all (≤256 devices) | Fast arbitrary communication within limit |
| 🧱 **TPU** | Toroidal mesh | Efficient local communication; All-Reduce works equally well |


## 🧩 3. Parallelization Primitives

Three main ideas:
1. **Data Parallelism (DP)**
2. **Model Parallelism (MP)**  
3. **Activation Parallelism**

![alt_text](/assets/images/llm-from-scratch/07/2.png "image_tooltip")


## 4. Data Parallelism

Splits the data batch **B** across **M** machines.

### 🧰 4.1 Naïve Data Parallelism (DDP)
- ✅ Each GPU processes `B/M` samples.
- ⚡ Compute scales well.
- 📡 Communication: `2 × #params` (for gradients).
- ❌ Memory issue: each GPU replicates all parameters + state (~16 bytes/param).

**Breakdown (per parameter):**
- FP/BF16 weights: 2B  
- Gradients: 2B  
- FP32 master weights: 4B  
- Adam (m, v): 8B  
→ **Total: ~16 bytes/parameter**

---

### 4.2 ZeRO Stage 1: Optimizer State Sharding

Reduce memory overhead by sharding the **optimizer state** across GPUs.

![alt_text](/assets/images/llm-from-scratch/07/3.png "image_tooltip")

#### High-Level Idea
- **Sharding:** The optimizer state (first + second moments in Adam) is **split/sharded** across GPUs.  
- **Replication:** Each GPU still maintains a **full copy of parameters and gradients**.  
- **Responsibility:** Each GPU updates only its assigned shard of optimizer state and corresponding parameters.

#### Operational Steps
1. **Forward/Backward Pass:** Each GPU computes a full gradient on its local batch subset.  
2. **Gradient Synchronization (Reduce-Scatter):** Gradients are synchronized via Reduce-Scatter (cost: `#params`).  
   - Each GPU receives the summed gradient information for its shard.  
3. **Parameter Update:** Each GPU updates only its assigned parameters using its optimizer shard.  
4. **Parameter Synchronization (All-Gather):** Updated parameters are collected and broadcast to all GPUs (cost: `#params`).

Now, **ZeRO Stage 1** keeps the same training flow as **DDP** for most things — but **shards the optimizer state only**.

#### 🧮 Example
Say your model has **12 layers** and you train with **4 GPUs**.

| GPU | Model Parameters | Gradients | Optimizer State Responsibility |
|------|------------------|------------|--------------------------------|
| GPU 0 | ✅ full model | ✅ full grads | Layers 0–2 |
| GPU 1 | ✅ full model | ✅ full grads | Layers 3–5 |
| GPU 2 | ✅ full model | ✅ full grads | Layers 6–8 |
| GPU 3 | ✅ full model | ✅ full grads | Layers 9–11 |

Each GPU holds the optimizer slots (`m`, `v`) **only for its assigned layers**.

#### 🏃‍♂️ Forward & Backward Pass
1. **Forward pass:** Each GPU runs forward on its batch (same as DDP).  
2. **Backward pass:** Each GPU computes full gradients for the model (same as DDP).  

So far, memory usage is the same as DDP.

#### 🔁 Gradient Synchronization (Reduce-Scatter)
After backward:
- Instead of doing an **All-Reduce**, ZeRO uses **Reduce-Scatter**.  
- Each GPU gets the **summed gradient for the subset of parameters it owns** (its optimizer shard).  
  - Example: GPU0 receives gradients for Layers 0–2, GPU1 for Layers 3–5, etc.

#### ⚙️ Parameter Update
- Each GPU **updates only its assigned parameters** (the ones its optimizer shard corresponds to).  
- Other GPUs **don’t update** those layers — they’ll receive the new parameters later.

#### 📤 Parameter Synchronization (All-Gather)
After updating:
- GPUs perform an **All-Gather** to share their updated parameter shards.  
- Now, **all GPUs have the full, updated model** again.

---

### 4.3 ZeRO Stage 2: Gradient Sharding
Further reduce memory by **sharding gradients** in addition to the optimizer state.

![alt_text](/assets/images/llm-from-scratch/07/4.png "image_tooltip")

#### High-Level Idea
- **Sharding:** Both **optimizer state** and **gradients** are sharded.  
- **Challenge:** A full gradient vector cannot exist in memory.  
- **Solution:** Execute backward pass incrementally and free gradients immediately after use.

#### Operational Steps
1. **Incremental Backward Pass:** Each GPU backpropagates layer-by-layer.  
2. **Immediate Reduction:** After each layer, gradients are reduced and sent to the GPU responsible for that parameter shard.  
3. **Immediate Freeing:** Gradients are freed once no longer needed, preventing buildup of full gradient tensors.  
4. **Update and All-Gather:** Updated parameters are gathered across GPUs.

#### Key Outcome
- **Memory Scaling:** Enables extremely large models (e.g., **24.6B params** on 8×A100 80GB).  
- **Communication Cost:** ≈ `2 × #params` (similar to Stage 1).  
- **Trade-Off:** Minor extra synchronization overhead (layer-wise communication).

#### 🔹 What stays the same

Each process (GPU) still:
- Has **the full model parameters**.
- Computes **forward and backward passes** on its **own batch** (like DDP).
- Keeps **its assigned optimizer shard** (from Stage 1).

#### 🔹 What changes (vs. Stage 1)

Now, **gradients are also partitioned** after the backward pass —  
so each GPU **stores only the gradients for the parameters it is responsible for updating**.

Let’s illustrate:

| GPU | Model Parameters | Gradients | Optimizer State Responsibility |
|------|------------------|------------|-------------------------------|
| GPU 0 | ✅ full model | 🔹 Layers 0–2 only | Layers 0–2 |
| GPU 1 | ✅ full model | 🔹 Layers 3–5 only | Layers 3–5 |
| GPU 2 | ✅ full model | 🔹 Layers 6–8 only | Layers 6–8 |
| GPU 3 | ✅ full model | 🔹 Layers 9–11 only | Layers 9–11 |


#### 🏃‍♂️ Forward & Backward Pass

**Forward:**  
Each GPU computes forward on its batch — same as DDP.

**Backward:**  
Each GPU initially computes **full gradients** for the model (same as before),  
but **then ZeRO immediately shards them** across GPUs using **Reduce-Scatter**.

So, after backward, GPU0 holds only the gradients for Layers 0–2, GPU1 for Layers 3–5, etc.

This already saves memory because each GPU keeps only a slice of gradients.

1. Forward Pass
Each process runs the forward pass normally (same as DDP).

2. Backward Pass (Gradient Computation)
Each GPU computes local gradients for all model parameters — exactly like in DDP.

However, ZeRO hooks into the autograd engine so that as soon as a layer finishes its backward computation:

Those gradients are Reduce-Scattered to all GPUs.

The local copy of the full gradient is deleted immediately.

So while a single layer is computing backward, that layer’s gradients exist in full.
But across the full model, you never have all layers’ gradients in memory at once.

#### ⚙️ Parameter Update

Each GPU updates **only its assigned parameters**,  
using the gradients and optimizer state shards it owns.

Example:
- GPU0 updates Layers 0–2.
- GPU1 updates Layers 3–5.
- etc.

#### 📤 Parameter Synchronization (All-Gather)

After updating:
- GPUs perform **All-Gather** to share updated parameter shards.
- Every GPU reconstructs the **full model** for the next forward pass.

---

### 4.4 ZeRO Stage 3 (FSDP): Parameter Sharding
Achieve **maximum memory efficiency** by sharding **parameters, gradients, and optimizer states**.

![alt_text](/assets/images/llm-from-scratch/07/5.png "image_tooltip")

#### High-Level Idea
- **Sharding:** Every model component (parameters, gradients, optimizer state) is distributed.  
- **On-Demand Fetching:** Parameters are fetched via **All-Gather** just before computation and freed afterward.  
- **Overlapping:** Communication is overlapped with computation to hide latency.

#### Simplified Operational Steps
1. **Forward Pass (On-Demand All-Gather):**  
   - GPU gathers needed weights for the current layer.  
   - Performs computation.  
   - Frees gathered weights.
2. **Repeat for Each Layer:**  
   - `All-Gather → Compute → Free`.
3. **Backward Pass:**  
   - Reverse process using `All-Gather` (for weights), `Reduce-Scatter` (for gradients), and immediate freeing.


#### 🧠 Concept Summary

| Component | DDP | ZeRO-1 | ZeRO-2 | ZeRO-3 / FSDP |
|------------|------|---------|---------|----------------|
| Model Params | Full copy | Full copy | Full copy | ✅ **Sharded** |
| Gradients | Full | Full | ✅ Sharded | ✅ Sharded |
| Optimizer States | Full | ✅ Sharded | ✅ Sharded | ✅ Sharded |
| Memory Saving | 🔴 Low | 🟡 Medium | 🟢 High | 🟢🟢🟢 **Maximal** |


| GPU | Holds Parameters for | Gradients | Optimizer States |
|------|----------------------|------------|------------------|
| GPU 0 | Layers 0–2 | Layers 0–2 | Layers 0–2 |
| GPU 1 | Layers 3–5 | Layers 3–5 | Layers 3–5 |
| GPU 2 | Layers 6–8 | Layers 6–8 | Layers 6–8 |
| GPU 3 | Layers 9–11 | Layers 9–11 | Layers 9–11 |


#### 🏃‍♂️ Forward Pass

Each GPU owns only a shard of parameters, but every GPU computes the forward pass for the entire model for its mini-batch. To do this, before computing a layer, every GPU must have the full parameters of that layer, because the forward of that layer depends on all weights.

So the flow is:
- Layer 0:
    - Each GPU calls the All-Gather for layer 0’s parameters.
    - Now all GPUs have the complete layer 0 weights.
    - Each GPU performs the forward pass of layer 0 on its batch.
    - After computing, each GPU can reshard/free the parameters of layer 0.

- Layer 1:
    - Output of layer 0 (activations) is the input.
    - Each GPU calls the All-Gather for layer 1’s parameters.
    - Forward pass of layer 1 happens on every GPU, using its mini-batch activations.

✅ **Memory is used only for one layer’s full params at a time.**


#### 🔁 Backward Pass

Backward is also sequential but leverages gradient sharding:

- Step 1: Gradient Computation
    - Each GPU has its own mini-batch activations stored (or recomputed).
    - To compute the gradient of layer n, the GPU needs the full parameters for that layer (All-Gathered).
    - Compute the local gradient for that GPU’s shard.

- Step 2: Reduce-Scatter (Gradient Sharding)
    - Each GPU does not keep the full gradient. Instead: 
        - Immediately after computing a layer’s gradients Reduce-Scatter distributes the gradients to the GPUs that “own” those parameters.
        - Each GPU ends up with gradient shards only for its parameters.

- Step 3: Optimizer Update
    - Each GPU updates only the parameters it owns using:
        - Its sharded gradients
        - Its sharded optimizer state (m, v for Adam)
    - Other GPUs do not update these parameters.

- Step 4: Parameter Reshard / Free
    - After update, parameters are resharded and old full copies are freed.

FSDP also overlaps communication and computation in backward using **backward prefetching** hooks. 
✅ This prevents any GPU from holding full gradients at once.
- **Gradient Computation / Local Gradients**  
   - After backward, each GPU has a local, full gradient for that module (because parameters were unsharded for forward).
- **Reduce-Scatter**  
   - Immediately, the gradients are **Reduce-Scattered** across ranks: each GPU retains only the gradient shard for its parameter partition. The others are discarded or freed. :contentReference[oaicite:10]{index=10}  
- **Reshard or Free**  
   - Full parameter copies loaded for backward are again freed or reshared as soon as possible.
- **Optimizer Update**  
   - Each GPU updates **its shard** of parameters using its shard of gradient + optimizer state (m, v shards). No full gradient or full optimizer state is ever needed.

---

### 4.5 Compare to Layer Parallelism

🔹 What ZeRO / FSDP Does
ZeRO shards memory across GPUs, but it does not parallelize computation of layers.

- Optimizer Sharding (ZeRO Stage 1–3): Each GPU holds only the optimizer state for its parameter shard.
- Gradient Sharding (ZeRO Stage 2–3): Gradients are split across GPUs according to the parameter shards.
- Parameter Sharding (ZeRO Stage 3 / FSDP): Each GPU holds only a portion of parameters at any time.

Key point: All GPUs still compute forward and backward for all layers sequentially for their mini-batch.

🔹 Why It’s Not Layer Parallelism
Layer parallelism (like pipeline parallelism) means:
- Different GPUs compute different layers in parallel.
- GPU0 computes layer 0, GPU1 computes layer 1 simultaneously, etc.

ZeRO / FSDP does not do this:
- GPUs compute layers in the same order for their mini-batch.
- Layer computations are sequential, not spread across GPUs.

🔹 What ZeRO / FSDP is
- Memory parallelization: spreads parameters, gradients, optimizer state across GPUs.
- Communication-efficient: uses Reduce-Scatter / All-Gather to share only what’s necessary per shard.
- Data Parallel + Sharding: multiple GPUs still process different mini-batches, like normal data parallelism.

✅ So ZeRO is shard-based memory parallelism, not true layer parallelism.


## 🧱 5. Model Parallelism (MP)

Splits parameters across GPUs. Instead of syncing weights, communicates **activations**.


### 🚇 5.1 Pipeline Parallelism (PP)

**Pipeline parallelism (PP)** is a fundamental strategy used in training **large language models (LLMs)** and is categorized as a form of **model parallelism**.  
Model parallelism splits the model’s parameters across GPUs and communicates activations between them — unlike data parallelism, which splits the batch data.

![alt_text](/assets/images/llm-from-scratch/07/6.png "image_tooltip")

#### ⚙️ Core Concept and Mechanism

Pipeline parallelism involves cutting the deep neural network **along the depth dimension** (layer-wise parallel).

- **Layer Distribution:**  
  Instead of replicating the entire model on every machine, each GPU is assigned a **subset of the model’s layers**.

- **Activation Communication:**  
  During forward and backward passes, GPUs **pass activations and partial gradients** between adjacent stages (GPUs).

- **Addressing Utilization Issues:**  
  A naive implementation of layer-wise parallelism results in **poor GPU utilization** — GPUs are active only 1/N of the time (where N = number of GPUs) while waiting for other stages.

- **Micro-batches (Pipelining):**  
  PP mitigates idle time by using **micro-batches**.  
  Once GPU 1 finishes the forward pass of micro-batch 1, it sends activations to GPU 2 and **immediately starts** micro-batch 2.  
  This overlaps **communication and computation**, improving utilization.

#### 📈 Advantages and Communication Properties

Pipeline parallelism is chosen for its **memory efficiency** and **communication behavior**:

- **Memory Scaling:**  
  PP achieves **linear memory scaling** for parameters and enables models to fit when single-GPU memory is insufficient.  
  It uses **less memory** than Distributed Data Parallel (DDP).

- **Activation Reliance:**  
  PP’s communication depends solely on **activations** (`b × s × h`):  
  - `b`: batch size  
  - `s`: sequence length  
  - `h`: hidden/residual dimension  
  Communication is **point-to-point** between GPUs.

- **Use on Slower Links:**  
  PP works well across **inter-node networks** because its activation-based communication is typically lighter than FSDP or Tensor Parallelism.

#### ⚠️ Disadvantages and Performance Constraints

Despite memory benefits, PP introduces **performance complexity** and **synchronization overheads**:

- **The Bubble:**  
  PP suffers from idle time known as the **"pipeline bubble"**.  
  The ratio of bubble time (overhead) to useful compute is approximately:

  $[
  \frac{n_{micro}}{n_{stages} - 1}
  ]$

  where  
  - `n_stages`: number of pipeline stages  
  - `n_micro`: number of micro-batches  

- **Batch Size Dependency:**  
  Performance depends heavily on **large batch sizes**, which are required to keep the pipeline full and minimize bubbles.  
  PP effectively **consumes batch size** as a limited resource.

- **High Implementation Complexity:**  
  PP is challenging to implement efficiently and often requires low-level modifications to **autograd** and **queue scheduling**.  
  It’s typically rated **“NO” for ease of use**.

#### 🧠 Advanced Techniques — Zero Bubble Pipelining (DualPipe)

To reduce idle time, advanced scheduling strategies like **Zero-Bubble Pipelining (DualPipe)** have been developed.

![alt_text](/assets/images/llm-from-scratch/07/7.png "image_tooltip")

- **Goal:** Fill the idle pipeline time with useful work.  
- **Key Idea:** Split the **backward pass** into two components:
  1. **Activation backpropagation** (`z`, `x`)  
  2. **Weight gradient computation** (`W`)  

- **Rescheduling:**  
  Since computing `W` gradients is independent of activation dependencies, it can be **rescheduled** to execute during otherwise idle periods — effectively hiding the bubble.

#### 🧩 Strategic Usage in Large-Scale Training

Modern large-scale training combines PP with other parallelization techniques — often called **3D parallelism** (Data + Tensor + Pipeline).

1. **Model Fitting:**  
   - PP is typically deployed **across nodes (inter-node)** after maximizing **Tensor Parallelism** within a node (e.g., 8 GPUs).  
   - This ensures the model and activations fit into memory.

2. **Combinations in Practice:**
   - **DeepSeek V3:** 16-way PP + Expert Parallelism + ZeRO Stage 1  
   - **Yi Model:** PP + ZeRO Stage 1 + Tensor Parallelism  
   - **Llama 3:** PP during pretraining and long-context fine-tuning


✅ **In short:**  
Pipeline Parallelism is **layer-wise model parallelism** that reduces memory per GPU and overlaps compute via **micro-batching**, but requires careful scheduling to avoid **pipeline bubbles** and ensure high utilization.

---

### 🧮 5.2 Tensor Parallelism (TP)

Tensor Parallelism (TP) is a form of **model parallelism** that focuses on **parallelizing the computation within individual layers**, especially large matrix multiplications. Unlike **Pipeline Parallelism (PP)**, which splits the model *by depth (layer-wise)*, **Tensor Parallelism** splits the model *by width (tensor-wise)*.

![alt_text](/assets/images/llm-from-scratch/07/8.png "image_tooltip")

- **Matrix Decomposition:**  
  Matrix multiplication can be decomposed into smaller submatrices, allowing distributed computation across GPUs.

- **Layer Distribution:**  
  GPUs each hold a **slice of the weight tensor**.  
  For example, a weight matrix `A` can be split into `[A₁, A₂, ...]`.

#### 🧠 Forward Pass Example (MLP Layer)

If an operation is $( Y = X \cdot A )$:

1. The input `X` is **copied** to all GPUs.
2. Each GPU computes its local partial result:  
   - GPU₁ → `Y₁ = X · A₁`  
   - GPU₂ → `Y₂ = X · A₂`
3. The partial results are **combined** using an **All-Reduce** to get the final output `Z`.
4. A **synchronization barrier** ensures all GPUs are aligned per layer.

In this setting:
- Forward pass: $( f = \text{identity} )$, $( g = \text{All-Reduce} )$
- Backward pass: $( f = \text{All-Reduce} )$, $( g = \text{identity} )$

#### 🔁 Backward Pass

During backpropagation:
- Gradients are computed locally per GPU for their submatrix of `A`.
- Then an **All-Reduce** is performed to synchronize the partial gradients before the next layer backward step.

This ensures correctness while maintaining layer-wise tensor splitting.

---

#### ✅ Advantages (Pros)

1. **No Pipeline Bubble**  
   TP does not suffer from idle time — all GPUs participate in each layer’s computation simultaneously.

2. **Batch Size Independence**  
   TP works efficiently even with small batch sizes and doesn’t rely on large batches like PP.

3. **Low Implementation Complexity**  
   Easier to integrate — typically requires modifying linear layers, not full computation graphs.

#### ❌ Disadvantages (Cons)

1. **High Communication Cost**  
   Each layer’s partial outputs require **All-Reduce**, introducing heavy communication overhead:  
   $[
   \text{Cost} \approx 8 \cdot b \cdot s \cdot h \cdot \frac{n_{\text{devices}}}{n_{\text{devices}} - 1}
   ]$
   where:
   - $( b )$: batch size  
   - $( s )$: sequence length  
   - $( h )$: hidden dimension  

2. **Bandwidth Dependency**  
   Efficient TP requires **high-bandwidth, low-latency interconnects** (e.g., NVLink).  
   Performance drops drastically over slower network links.


#### 🧩 Strategic Usage

| Deployment | Strategy |
|-------------|-----------|
| **Intra-node** | Use TP **within a single node**, where GPUs have NVLink or NVSwitch. |
| **Inter-node** | Avoid or limit TP across nodes (slow interconnects). |
| **Optimal GPU count** | Typically up to **8 GPUs per node**. Beyond that, communication dominates. |
| **3D Parallelism** | Combine TP (intra-node) with PP or ZeRO (inter-node) for scaling across clusters. |

1. **Within each node** → Tensor Parallel up to 8 GPUs.  
2. **Across nodes** → Use Pipeline Parallelism (PP) or ZeRO-3/FSDP to scale further.


#### 🧾 Summary Table

| Aspect | Tensor Parallelism (TP) | Pipeline Parallelism (PP) |
|--------|--------------------------|----------------------------|
| Split Dimension | Width (within layers) | Depth (between layers) |
| Communication | Heavy (All-Reduce) | Light (point-to-point) |
| Bubble (Idle Time) | None | Present (unless zero-bubble) |
| Batch Size Requirement | Small | Large |
| Implementation | Easier | Complex |
| Best Placement | Intra-node | Inter-node |
| Memory Savings | Parameters + Gradients | Parameters + Activations |
| Combined With | SP, PP, ZeRO | TP, ZeRO |


- Tensor Parallelism splits **individual layers** across GPUs.  
- Every GPU computes **partial results** that are combined via **All-Reduce**.  
- **No idle time**, but **high communication bandwidth** required.  
- Works best **within nodes** with NVLink (≤8 GPUs).  
- Often combined with **Pipeline Parallelism** (depth) and **ZeRO/FSDP** (memory sharding) in **3D parallelism setups**.


### ⚡ 5.3. Activation Parallelism

**Activation Parallelism** is a crucial strategy for training large language models (LLMs), specifically designed to **manage and reduce activation memory**, which can become a major bottleneck in large-scale training.

As models grow and sequence lengths increase, **activation memory** becomes a serious limitation.  
While **Model Parallelism** techniques like **Pipeline Parallelism (PP)** and **Tensor Parallelism (TP)** achieve linear scaling for **parameters**, **gradients**, and **optimizer states**, they do **not** fully address the memory consumed by **activations**.

![alt_text](/assets/images/llm-from-scratch/07/9.png "image_tooltip")

Activation memory usage is **dynamic**:
- It **grows** during the **forward pass**.
- It **peaks** during the **backward pass**.
- It is **freed** afterward.

Managing this **peak usage** is essential to fit larger models and increase batch sizes.

#### The Straggler Term
Even with TP, splitting matrix multiplications in the Attention and MLP blocks, a large **unreduced activation memory term** remains:

$[
\text{Straggler Term} \approx 10 \cdot s \cdot b \cdot h
]$

Where:
- $( s )$: sequence length  
- $( b )$: batch size  
- $( h )$: hidden/residual dimension

#### Source of Unreduced Memory
This unreduced activation memory comes from **non-matrix multiplication** components such as:
- **LayerNorm:** $( 4sbh )$
- **Dropout:** $( 2sbh )$
- **Inputs to Attention and MLP layers:** $( 4sbh )$

These are **pointwise operations** not parallelized by Tensor Parallelism.

---

#### ⚙️ Implementation via Sequence Parallelism (SP)

**Sequence Parallelism (SP)** is the main implementation method for **activation parallelism**.

#### 🔍 Core Observation
Pointwise ops like **LayerNorm** and **Dropout** operate **independently** across the **sequence dimension**.  
This means:
> Each token’s computation does not depend on others — perfect for sequence-wise sharding.

#### ✂️ Sharding Mechanism
SP **splits** these operations along the **sequence axis**:
- If a sequence has **1024 tokens**, and you have **4 GPUs**,  
  → each GPU processes **256 tokens** for all pointwise ops.

This reduces activation memory per GPU by a factor of 4 (in this case).

#### 🎯 Goal: Linear Scaling
By combining **Sequence Parallelism (SP)** with **Tensor Parallelism (TP)**, the system achieves **linear scaling** in **activation memory** — now all components (parameters, gradients, activations) scale with the number of GPUs.

#### 🔁 Sequence Parallelism Communication Primitives

To coordinate the sequence sharding, SP uses **collective communication primitives**:

| Component | Forward Pass | Backward Pass |
|------------|---------------|----------------|
| **g**      | All-Gather    | Reduce-Scatter |
| **ḡ (g-bar)** | Reduce-Scatter | All-Gather     |

- **All-Gather:** Combine partial results from all GPUs (used before operations that need the full sequence).  
- **Reduce-Scatter:** Distribute and sum gradients or activations across GPUs (used after backward).

#### ⚡ Related Technique: Activation Recomputation

**Activation Recomputation** (or **checkpointing**) complements SP to further reduce memory.

#### 💡 Key Idea
Instead of storing all activations, **recompute** some during the backward pass when needed — trading compute for memory.

#### ✅ Benefits
- Reduces memory required for **quadratic attention terms**.
- Enables **larger batch sizes**, which improves **pipeline parallelism efficiency** by **hiding bubbles**.

#### ⚖️ Trade-Off
- Slightly more computation (extra forward passes).
- But total **throughput increases**, since training becomes more memory efficient.

![alt_text](/assets/images/llm-from-scratch/07/10.png "image_tooltip")


## 🔺 6. Combining Strategies — 3D Parallelism

**3D Parallelism** (sometimes extended to **4D** or **5D**) is the **standard best practice** for efficiently training extremely large language models (LLMs).  
It combines multiple parallelization techniques to balance **memory**, **compute**, and **bandwidth** constraints.

![alt_text](/assets/images/llm-from-scratch/07/11.png "image_tooltip")

The general strategy is driven by two goals:
1. **Make the model fit into memory.**
2. **Scale compute to fully utilize all available GPUs.**


### ⚙️ Simple Rules of Thumb for Combining Parallelism

The approach follows a **staged strategy**, prioritizing **bandwidth-hungry methods** on **fast interconnects first**, and then scaling outward.


### 🧠 Stage 1: Fitting the Model in Memory (Model Parallelism)

The first priority is ensuring that the **model parameters and activations fit into memory**.  
This is achieved through **Model Parallelism**.


#### 1️⃣ Maximize Tensor Parallelism (TP) Within Each Node

- **Rule:** Use **Tensor Parallelism (TP)** up to the number of GPUs per machine (e.g., `TP=8`).
- **Reasoning:**  
  TP relies on **low-latency, high-bandwidth interconnects** (e.g., NVLink) since it performs **large All-Reduce** operations per layer.  
  These connections exist within a **single node**, making TP ideal for intra-node scaling.
- **Benefit:**  
  - Linear memory scaling for **parameters**, **gradients**, and **optimizer states**.  
  - No pipeline bubble.  
  - Does **not** consume batch size.
- **Add-on: Sequence Parallelism (SP)**  
  - Applied on top of TP to achieve **full linear memory scaling for activations**, especially with long sequences.  
  - Splits **LayerNorm** and **Dropout** operations along the **sequence axis**.


#### 2️⃣ Scale Across Machines Using Pipeline Parallelism (PP) or ZeRO-3 (FSDP)

- **Rule:** Once TP is maxed out within the node, if the model still doesn’t fit, use **Pipeline Parallelism (PP)** or **ZeRO Stage 3 (FSDP)** **across nodes**.
- **Reasoning:**  
  - **PP** works well on **slower network links** because it only communicates **activations** (`b × s × h`) point-to-point,  
    rather than large collective operations.
  - **FSDP (ZeRO-3)** shards parameters, gradients, and optimizer states, giving **linear memory scaling** for all static components.
- **Trade-offs:**  
  - **PP:** High dependency on **large batch sizes** to hide synchronization overhead (the “bubble”).  
  - **FSDP:** Avoids bubbles, but incurs **3× parameter-size communication cost** per step.


### 🚀 Stage 2: Scaling Compute (Data Parallelism)

Once the model fits in memory, the next step is to **fully utilize compute** using **Data Parallelism (DP)**.

- **Rule:** Scale out with **Data Parallelism (DP)** — typically **ZeRO Stage 1 (Optimizer Sharding)**.
- **Reasoning:**  
  DP scales compute by splitting batches across machines.  
  ZeRO-1 is “free” in bandwidth-limited setups (same communication cost as naive DDP) while reducing optimizer memory.
- **Why Last?**  
  - DP works well even with **low-bandwidth** links.  
  - It efficiently brings more GPUs into play to increase total throughput.

### ⚖️ Managing Resources and Efficiency

#### 🧮 Batch Size as a Resource
- **Global batch size** is limited and must be managed carefully:
  - **PP** consumes batch size to hide the pipeline bubble.  
  - **DP** consumes batch size for scaling compute.  
  - **TP** has **no impact** on batch size.

#### 🔁 Activation Recomputation
- **Purpose:** Reduces memory by recomputing activations during backward pass (used in Flash Attention).  
- **Effect:**  
  - Frees activation memory → allows **larger batch sizes**.  
  - Larger batches improve throughput by **masking communication overheads**, especially for PP.
- **Trade-off:** Adds compute (extra FLOPs), but often yields **net throughput gains**.

#### ⚙️ Optimal Configuration
Empirical results show:
> Maximize **Tensor Parallelism (TP=8)** first,  
> then balance **Pipeline Parallelism (PP)** and **Data Parallelism (DP)**  
> for the best linear scaling of total FLOPs.


### 🌍 Real-World Examples of 3D Parallelism

| Model | Techniques Used | Notes |
|--------|------------------|-------|
| **DeepSeek V3** | PP (16-way), Expert Parallelism (64-way), ZeRO-1 | Expert parallelism adds another dimension. |
| **Yi** | ZeRO-1, TP, PP | Balanced approach across all dimensions. |
| **LLaMA 3 (405B)** | TP=8, SP, PP, DP | Ordered by required bandwidth; DP tolerates longest latency. |
| **Gemma 2** | ZeRO-3, TP + SP (Model Parallelism), DP | FSDP + sequence sharding for efficiency. |

In current best practices, Activation Parallelism (SP) is often combined with Tensor Parallelism to form a single Model Parallel dimension, as seen in the parallelism utilized by models like DeepSeek and Gemma 2, which used MP=TP+SP


### 🧩 Summary Table

| Dimension | Method | Scales | Communication | Notes |
|------------|--------|--------|----------------|--------|
| **Tensor Parallelism (TP)** | Split within layer | Intra-node | All-Reduce | Fastest, no batch cost |
| **Sequence Parallelism (SP)** | Split activations | Intra-node | Reduce/All-Gather | Complements TP |
| **Pipeline Parallelism (PP)** | Split layers | Inter-node | Point-to-point | Needs large batches |
| **Data Parallelism (DP)** | Split batches | Inter-node | All-Reduce | Easiest scaling |
| **ZeRO / FSDP** | Shard params/opt | Inter-node | All-Gather | Full memory savings |


✅ **Takeaway**

3D Parallelism =  
→ **Tensor + Sequence Parallelism (fit model intra-node)**  
→ **Pipeline or ZeRO/FSDP (fit model inter-node)**  
→ **Data Parallelism (scale compute)**  

Together, they form the foundation of **modern large-scale model training**, enabling models with **hundreds of billions of parameters** to be trained efficiently across **thousands of GPUs**.
