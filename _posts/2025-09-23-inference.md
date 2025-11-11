---
layout: post
title: Modern LLM Inference 💻
subtitle: Language Modeling from Scratch Lecture 11
categories: Large-Language-Model
tags: [Stanford-LLM-From-Scratch-2025]
banner: "/assets/images/banners/yuanpang-wa-iceburg2.jpg"
---

# 🚀 Modern LLM Inference — Workloads, Bottlenecks, and Optimization Techniques

🧠 *From defining inference to lossy/lossless acceleration and dynamic serving — a full-stack view of how modern LLM inference actually works.*

[Course link](https://stanford-cs336.github.io/spring2025/)


## 1. 🏁 Introduction: What *Is* Inference and Why It’s Hard

### ✅ What Is Inference?
Inference answers one fundamental question:

> **Given a trained model, how do we generate outputs given prompts?**

But inference appears in many real scenarios:

- 💬 Chatbots  
- 👨‍💻 Code assistants (Cursor allegedly produces **1B accepted LOC/day**)  
- 📊 Batch data processing  
- 🧪 Evaluation & benchmarking  
- 🧠 Test-time compute (CoT / self-consistency)  
- 🤖 RLHF — sampling trajectories

Training happens once.  
Inference happens **forever**, at massive scales (OpenAI serves ~**100B words/day**).

That’s why inference efficiency is existential.

### ⚡ Key Inference Metrics
1. **TTFT (Time-to-first-token)** – responsiveness  
2. **Latency** – speed of generating each token  
3. **Throughput** – total tokens/sec across all users  

Training = parallel over sequence.  
Inference = **token-by-token**, must process history → inherently sequential.

➡️ **This is what makes inference memory-limited.**


## 2. Why Transformer Inference Is Memory-Bound

The Transformer’s efficiency changes dramatically between **training** and **inference** because of how sequences are processed.

![alt_text](/assets/images/llm-from-scratch/11/1.png "image_tooltip")

### ✅ Training: Parallel Across the Entire Sequence
During supervised training, **all tokens are visible at once**.  
This allows massive tensor parallelism through matrix multiplications.

➡️ **Compute utilization is high.**  
➡️ **Training is compute-limited.**

### ✅ Inference: Sequential, Auto-Regressive Generation
During inference, tokens must be generated **one at a time**: token_t → depends on → [token_1, token_2, ..., token_t-1]  

This **sequential dependency** prevents parallelism.

➡️ **Generation cannot fully use GPU compute.**  
➡️ **Becomes memory-limited instead of compute-limited.**

### ✅ The KV Cache (Stored in HBM)
To avoid recomputing attention over the whole prefix, models store:

- B  — batch (number of sequences)  
- S  — sequence length  
- L  — number of layers  
- K  — number of attention heads  
- H  — hidden dimension  

The **KV cache** stores an H-dim vector per (B, S, L, K).  
It massively increases memory access during generation.

### ✅ Two Stages of Inference

**Prefill (Parallelizable ✅ Compute-limited)**
- Processes the prompt  
- Behaves like training  
- High arithmetic intensity → fast

**Generation (Sequential ❌ Memory-limited)**
- One token at a time  
- Requires frequent KV-cache reads  
- Low arithmetic intensity → slow


## 3. 🧮 Arithmetic Intensity: The Core Metric

### ✅ Definition
Arithmetic Intensity =  
**FLOPs / Bytes Transferred**

- **High** → compute-limited (good)  
- **Low** → memory-limited (bad)

H100 GPU has accelerator intensity ≈ **295**.  
You want **AI > 295** to stay compute-bound.


### ✅ Example: Matrix Multiplication $(X_{B×D} \times W_{D×F})$

![alt_text](/assets/images/llm-from-scratch/11/2.png "image_tooltip")

- FLOPs = $(2BDF)$  
- Arithmetic intensity ≈ **B**

So:

- If **B ≥ 300** → compute-limited ✅  
- If **B = 1** (matrix-vector multiply) → intensity = 1 ❌ *really bad*

This is exactly the situation during **inference generation**.

### 1️⃣ MLP Layers (Feed-Forward Networks)

![alt_text](/assets/images/llm-from-scratch/11/3.png "image_tooltip")

$[
\text{Intensity} \propto B \times T
]$

- **Prefill**: $(T = S)$ → large → compute-limited ✅  
- **Generation**: $(T = 1)$ → intensity = B → depends on concurrent requests  

If batch size is small, MLP becomes memory-bound.

### 2️⃣ Attention Layers (Self-Attention)

![alt_text](/assets/images/llm-from-scratch/11/4.png "image_tooltip")

$[
\text{Attention Intensity} =
\frac{S + T}{S \cdot T}
]$

#### ✅ Prefill (T = S)
$[
\frac{2S}{S^2} = O(1/S) \quad \text{(Compute-limited if S large)}
]$

#### ❌ Generation (T = 1)
$[
\frac{S + 1}{S} \approx 1
]$

**Always ≈ 1 → deeply memory-limited.**  
And importantly:

⚠️ **No dependence on batch size (B).** Batching cannot help the attention layer during generation.

---

### ✅ MLP during generation
- Intensity = **B**
- Can be improved by adding more concurrent sequences

### ❌ Attention during generation
- Intensity = **≈ 1**
- **Completely memory-bound**
- **Batching does NOT help**

Why batching fails for attention:

| Layer | Shared across batch? | Effect |
|-------|-------------------------|---------|
| **MLP** | Yes → same weights reused | B increases FLOPs → higher intensity |
| **Attention** | No → each sequence has its OWN KV | Memory reads scale with B → FLOPs scale too → B cancels out |

Thus:
> **Every sequence has its own unique KV cache.  
Batching does not increase arithmetic intensity.**


### ✅ Summary Table

| Stage | Layer Type | Arithmetic Intensity | Limitation | Does Batching Help? |
|-------|------------|----------------------|------------|----------------------|
| **Prefill** | MLP + Attention | High (B×S or S/2) | Compute-limited ✅ | ✅ Yes |
| **Generation (T=1)** | MLP | = B | Memory-limited if B small | ✅ Yes |
| **Generation (T=1)** | Attention | ≈ 1 | **Memory-limited (worst)** | ❌ No |


Inference suffers because:

- 🔥 Prefill is compute-limited → fast  
- ❄️ Generation is memory-limited → slow  
- ❌ Attention dominates and has intensity ≈ 1  
- ❌ Batching cannot fix attention’s memory costs  
- ✅ Therefore, the KV cache becomes the primary bottleneck.

This is why modern research focuses on:
- shrinking the KV cache (GQA, MLA, CLA)  
- replacing it entirely (Mamba/SSMs)  
- paging it efficiently (vLLM)  

Reducing attention’s memory traffic = **the key to fast inference**.


## 3. ⏱ Latency vs Throughput: Core Trade-off

Using Llama 2 13B on H100:

### ✅ B = 1 → Best latency
- ~8 ms/token  
- ~124 tok/sec

### ✅ B = 64 → High throughput
- Much higher throughput  
- Higher latency  
- Higher memory usage  
- KV cache grows linearly with B  

Batching improves throughput but hurts latency.

### ✅ Want both?
Run **M model replicas** → multiply throughput without hurting latency.


## 📦 4. Techniques for Shrinking the KV Cache (Motivation, Methods, and Trade-offs)

The need to shrink the **Key-Value (KV) Cache** arises from a simple but critical fact:

> ✅ **Inference—especially sequential generation—is fundamentally memory-limited, not compute-limited.**  
> ✅ **KV cache size directly determines memory bandwidth → which determines latency and throughput.**

Because the KV cache grows with sequence length, layers, heads, and hidden dimension, it becomes the **dominant memory bottleneck**.  
Thus, shrinking the KV cache is one of the most effective architectural optimizations for speeding up inference **without retraining an entirely new model**.

Below are the main techniques used to reduce KV cache size.


### 4.1 Grouped-Query Attention (GQA)

**GQA reduces the number of key/value heads, keeping the number of query heads the same.**

#### ✅ Key Ideas
- Standard Multi-Head Attention (MHA):  Query heads = # Key heads = # Value heads = N
- **GQA:**  N query heads share K key/value heads (K < N)

Each KV head serves a *group* of Q heads → **Grouped-Query Attention**.

#### ✅ KV Cache Reduction
Reducing KV heads from N → K shrinks the cache by a factor of:

$[
\text{Reduction Factor} = \frac{N}{K}
]$

#### ✅ Efficiency Gains
Fewer KV tensors → fewer memory loads → lower latency + higher throughput.

Example:  
Switching a Llama 2 13B model to a 1:5 GQA ratio enabled batch sizes like **B = 256**, dramatically improving throughput.

#### ✅ Accuracy
Nearly identical to full MHA.  
**Llama 3 adopts GQA** for this reason.

---

### 4.2️. Multi-Head Latent Attention (MLA)

![alt_text](/assets/images/llm-from-scratch/11/5.png "image_tooltip")

MLA reduces the **dimensionality** of the key and value vectors rather than reducing the number of KV heads.

**Key Ideas**
- Original KV dimension: large (e.g., 16,384)
- MLA projects KV vectors into a smaller latent dimension:  e.g., 16,384 → 512
- KV cache shrinks proportional to the dimension reduction:

$[
\text{Reduction} \propto \frac{\text{old dim}}{\text{new dim}}
]$

DeepSeek V2 uses:  512 (latent) + 64 (for RoPE compatibility) = 576 total dims


#### 4.2.1. Why MLA Compresses K/V but Not Q

In standard attention, logits are computed as:

$[
\text{logits} = Q \cdot K^T
]$

This requires Q and K to have **the same dimensionality**.

However, MLA reduces the dimensionality of **K and V**, not Q:

- **K/V stored in KV cache → major memory sink → must be compressed**
- **Q is never stored in the KV cache → compressing it saves nothing**

Therefore:

- ✅ Compressing K/V reduces memory (KV cache)
- ❌ Compressing Q does NOT reduce memory (Q is computed per token and discarded)
- ✅ But Q must be projected into the same latent dimension as K for multiplication

Thus the model keeps:

- **Q_full**: full-dimensional query (e.g., 4096)
- **K_latent**: stored in KV cache (e.g., 512 or 576)
- **V_latent**: stored in KV cache (same latent dimension)
- **Q_latent**: computed *temporarily* for attention

This leads to the next component.


#### 4.2.2. Introducing a New Q→Latent Projection Matrix

Because Q_full cannot directly multiply with K_latent:

- Q_full: 4096 dims  
- K_latent: 512 dims  
- ❌ Dot product impossible

MLA introduces a new learned projection:

$[
W_{Q \rightarrow latent} \in \mathbb{R}^{d_{full} \times d_{latent}}
]$

to produce:

$[
Q_{\text{latent}} = Q_{\text{full}} \cdot W_{Q\rightarrow latent}
]$

Now:

- Q_latent: 512 dims  
- K_latent: 512 dims  

Dot-product is valid:

$[
\text{logits} = Q_{\text{latent}} \cdot K_{\text{latent}}^T
]$

**Important:**  
✅ Q_latent is NOT stored  
✅ Only temporary  
✅ Does not increase KV cache size  
✅ MLA still “compresses only K/V” from a memory perspective


#### 4.2.3. Why K/V Can Be Compressed but Q Must Stay Full

This follows deep attention-geometry principles:

- **Q encodes the layer’s computation**  
  → must retain full expressiveness  
  → compressing Q collapses layer depth and harms accuracy heavily

- **K/V encode token memory**  
  → can be compressed without destroying computational meaning  
  → attention distribution remains workable if geometric relationships are preserved

Thus MLA design:

- ✅ Full-dim Q (keeps accuracy)  
- ✅ Compressed K/V (saves KV-cache)  
- ✅ Align via Q→latent projection


#### 4.2.4. How RoPE Works With MLA (Full Explanation)

**RoPE (Rotary Position Embedding)** applies rotations to Q and K **in their full dimension only**:

$[
Q_{\text{rope}} = \text{RoPE}(Q_{\text{full}})
]$
$[
K_{\text{rope}} = \text{RoPE}(K_{\text{full}})
]$

Why not apply RoPE directly in latent space?

- Projection destroys RoPE’s sinusoidal subspaces  
- Latent space does not preserve frequency structure  
- Positional geometry must be applied *before* dimensionality reduction

So MLA applies RoPE **before compression**:

1. Compute Q_full, K_full
2. Apply RoPE to Q_full and K_full
3. Project K_full into latent space  
4. Project Q_full into latent space (for compute only)


#### 4.2.5. Why MLA Adds Back Extra RoPE Dimensions (+64 Example)

When K_full (4096 dims) is projected into K_latent (e.g., 512 dims), the projection inevitably:

- Loses high-frequency RoPE components  
- Blurs precise rotational structure  
- Weakens relative positional encoding

DeepSeek V2 fixes this by allocating **extra RoPE-preservation dimensions**:

$[
K_{\text{latent}} = 
K_{\text{compressed}}^{(512)} \oplus K_{\text{pos}}^{(64)}
]$

- Total latent K dimension becomes: 512 compressed dims + 64 RoPE-pos dims = 576 total dims
- These 64 dims explicitly store RoPE harmonics that would otherwise be destroyed by projection.
- Thus MLA latent K = **compressed memory + preserved positional structure**.
- This keeps positional awareness strong even after compression.

Only K needs additional RoPE dimensions because:
- RoPE affects attention geometry, and this geometry depends on Q and K being in the same positional space.
- Q does NOT get compressed, so it keeps its full RoPE structure and does NOT lose positional information.
- K DOES get compressed, so it loses some high-frequency RoPE information — that must be restored.

#### 4.2.6. Complete MLA Pipeline (Concrete Numerical Example)

Assume:

- Full dimension: 4096  
- Latent dimension: 512  
- Extra RoPE positional dims: 64

**Step-by-step:**

1. **Compute full Q/K/V**
- Q_full = X ⋅ W_Q → (4096)
- K_full = X ⋅ W_K → (4096)
- V_full = X ⋅ W_V → (4096)

2. **Apply RoPE in full dimension**
- Q_rope (4096)
- K_rope (4096)

3. **Project K/V into latent**
- K_core = Linear(K_rope) → (512)
- V_latent = Linear(V_full) → (512)

4. **Extract RoPE harmonics**
- K_pos = selected positional dims → (64)

5. **Form final latent K**
- K_latent = concat(K_core (512), K_pos (64)) → (576)

6. **Project Q_rope into latent (temporary)**
- Q_latent = Linear(Q_rope) → (576)

7. **Compute attention in latent space**
- logits = Q_latent ⋅ K_latent^T
- output = softmax(logits) ⋅ V_latent


8. **KV cache stores only:**
- K_latent (576)
- V_latent (512)

9. **Q_latent is discarded** (never cached).


#### 4.2.7. Final Takeaways

- ✅ MLA compresses **K/V** for KV-cache savings  
- ✅ MLA introduces a **new Q→latent projection** for dimension alignment  
- ✅ Q remains full-dimensional for accuracy reasons  
- ✅ RoPE must be applied **before** the latent projection  
- ✅ MLA adds extra RoPE dims to recover positional information lost during compression  
- ✅ Only K/V latent are stored; Q_latent is temporary  

This design balances:

- **Memory efficiency** (smaller KV cache)  
- **Computational compatibility** (dimension alignment)  
- **Positional accuracy** (RoPE preservation)  
- **Model quality** (full-dim Q expressiveness)

MLA is therefore an elegant solution that compresses what matters (KV cache) while preserving what must remain expressive (Q).


---

### 4.3️. Cross-Layer Attention (CLA)

CLA reduces KV cache size by sharing **the same** KV projections across multiple layers.

![alt_text](/assets/images/llm-from-scratch/11/6.png "image_tooltip")

#### ✅ Key Ideas
- Standard Transformer: each layer has its own K and V projections.
- CLA: **reuse** the same KV projections across layers.

#### ✅ Analogy
- GQA = share KV across heads (within a layer)  
- **CLA = share KV across layers** (vertical sharing)

Reduced memory consumption with minimal accuracy loss, improving the perplexity-vs-KV-size trade-off.

#### ❓ Subsection: Why Cross-Layer Attention Shares K/V but **Not** Q

Cross-Layer Attention (CLA) reduces KV cache size by **sharing Key (K) and Value (V) projections across layers**, but it **does not share Query (Q) projections**.  
This design choice is intentional and rooted in the fundamental roles that Q, K, and V play in attention.

**1. Different Roles: Q = Computation, K/V = Memory**

In self-attention:

$[
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
]$

Each projection matrix has a different semantic role:

**K (Keys) and V (Values) represent sequence memory**
- Encode what each token *is*  
- Store token identity and content  
- Act like **read-only memory**  
- Should be consistent across layers  
- Safe to share across layers

**Q (Queries) represent layer-specific computation**
- Determine what each layer *wants* from memory  
- Encode how the layer interprets the sequence  
- Are unique to each depth in the network  
- Must not be shared across layers  
- Sharing Q destroys the layer hierarchy


**Why Sharing K/V Works**

K and V:
- Represent static properties of the input  
- Do not contain layer-specific meaning  
- Can be reused like a shared memory bank  
- Allow each layer to interpret the same memory differently using its own Q

Thus, sharing KV reduces cache size **without damaging the computational diversity** of the layers.


**Why Sharing Q Breaks the Model**

Sharing Q would force all layers to ask the **same question** about the memory.

This would:
- Collapse the differences between layers  
- Cause layers to behave identically  
- Destroy depth expressiveness  
- Dramatically increase perplexity  
- Make the deep Transformer behave like a repeated shallow block

Empirical results show:
- Sharing KV works well ✅  
- Sharing Q causes large accuracy drops ❌  
- Sharing QKV destroys the model ❌❌

**Analogy: Memory vs. Instructions**

- **K/V = memory**  
  Shared, stable, consistent across layers.

- **Q = instructions**  
  Each layer needs its own instructions to compute something different.

You can share memory across modules,  
but not the program code.


**Summary Table**

| Projection | Role | Share Across Layers? | Reason |
|-----------|------|----------------------|--------|
| **Q** | Layer computation / interpretation | ❌ No | Must vary per layer to maintain depth & expressiveness |
| **K** | Token identity | ✅ Yes | Represent sequence memory, not layer-specific |
| **V** | Token content | ✅ Yes | Safe shared memory |
| **All QKV** | Entire attention block | ❌ No | Collapses layers, destroys model |


**CLA shares K/V because they are shared-memory representations of the sequence, but it cannot share Q because Q encodes layer-specific computation.**  
Sharing Q would collapse all layers into the same function, destroying depth and severely harming accuracy.


### 4️.4. Local Attention

Local attention restricts each token to attend only to the most recent *K* tokens.

#### ✅ Key Ideas
Instead of full attention over all past tokens: Attend only to last K tokens


#### ✅ KV Cache Effect
Any KV entry older than the local window can be **discarded**.

This makes KV cache size:

$[
\text{O(window size)} = \textbf{O(1)}
]$

instead of:

$[
\text{O(sequence length)}
]$

#### ✅ Pros
- KV cache does **not grow** with sequence length → huge savings
- Useful for very long contexts (100k+ tokens)

#### ✅ Cons & Hybrid Layers

![alt_text](/assets/images/llm-from-scratch/11/7.png "image_tooltip")

Local attention alone harms long-range reasoning.  
Solution:

- Interleave global attention layers periodically  
  (e.g., 1 global layer for every 5–6 local layers)


### ✅ Summary: The Four Strategies for Shrinking the KV Cache

| Strategy | Main Idea | KV Cache Saved By | Trade-offs |
|----------|-----------|-------------------|-------------|
| **GQA** | Fewer KV heads | N/K reduction | Minimal accuracy loss |
| **MLA** | Smaller KV dimensions | Dim reduction (e.g., 16k→512) | Slight RoPE modifications |
| **CLA** | Share KV across layers | Multi-layer reuse | Architecture complexity |
| **Local Attention** | Attend to last K tokens only | Constant-sized KV | Weak long-range modeling; use hybrids |


Shrinking the KV cache is one of the *most powerful levers* for improving inference performance because:

- ✅ Generation is memory-bound  
- ✅ KV cache dominates memory reads  
- ✅ Reducing KV memory → directly reduces latency  
- ✅ Enables larger batch sizes → increases throughput

These “architectural shortcuts” must be carefully tuned to balance:

**speed 🏎️ vs memory 💾 vs accuracy 📉.**

But when done correctly, they unlock massive improvements in real-world LLM serving.


## 📐 5. Architectural Alternatives to Transformers

The motivation for exploring architectural alternatives arises from a fundamental limitation: **Transformers paired with autoregressive generation are inherently memory-limited during inference**.  
The architecture was originally optimized for **training efficiency**, not inference.  
As a result, the **full attention mechanism** forces models to maintain a growing **Key-Value (KV) cache**, which becomes the primary bottleneck.

To overcome these structural constraints, researchers pursue **radical architecture shifts** that *avoid or eliminate* the sequential, memory-heavy properties of Transformers. Two main directions have emerged:  
✅ **State-Space Models (SSMs)**  
✅ **Diffusion Models for Language**


### 5.1. State-Space Models (SSMs)

State-Space Models originate from **signal processing** and **control theory**, and were initially designed to model long sequences with **O(N)** (linear) complexity instead of **O(N²)**.

#### ✅ Core Motivation
The earliest successes of SSMs focused on **long-context tasks**, achieving remarkable efficiency. However, they initially struggled on general language modeling.

#### ✅ Evolution of SSMs
- **S4**  
  Early SSM models like S4 implemented classic linear dynamical systems and excelled at synthetic long-range tasks.
- **Associative Recall Problem**  
  Early SSMs failed at associative recall — the ability to retrieve a specific memory far in the past.  
  This capability is *essential* for language modeling but was missing in classical SSMs.
- **Mamba**  
  Introduced **input-dependent parameters**, allowing SSMs to dynamically adapt to sequence content.  
  Mamba reached Transformer-level quality up to the 1B scale.
- **Jamba (52B MoE)**  
  Combines SSM and Transformer layers in a **1:7 ratio**.  
  Validated that SSM+Transformer hybrids can scale to tens of billions of parameters.

#### ✅ Inference Benefit: O(1) State Instead of O(T) KV Cache
SSMs replace the **growing KV cache** (size O(T)) with a **constant-size state** (size O(1)).  
This dramatically improves inference efficiency:

- Memory stays constant  
- Latency does not grow with sequence length  
- More scalable for very long contexts

### 5.2. Linear Attention — From Softmax to Associative Kernels (O(N) Inference)

**Linear attention** replaces softmax with an **associative kernel** so we can **reorder** computation, pre-accumulate K/V once, and evaluate queries in **O(1)** per token. It’s **not equal** to softmax; it’s an **approximation** designed for speed and constant memory.

#### 🔎 What Problem Are We Solving?
Standard self-attention is:
$[
\text{Attn}(Q,K,V)=\text{softmax}(QK^\top)V
]$
- Requires forming an $(N\times N)$ score matrix (pairwise Q–K comparisons) → **O(N²)** compute and memory traffic.
- In inference, even with a KV cache, each new token still attends to **all past tokens** → **O(N)** per step, **O(N²)** total.
- Attention during generation is **memory-bound**; bandwidth dominates latency.

**Goal:** make attention **associative** so we can **reorder** and avoid building the \(N\times N\) matrix.


#### 🧠 Core Idea: Replace Softmax with an Associative Kernel
Softmax uses the exponential kernel \(e^{q^\top k}\), which is **not** associative (you can’t regroup terms).  
Linear attention approximates this kernel with one that **factorizes**:
$[
K(q,k)\approx \phi(q)^\top \phi(k)
]$
If this holds, then:
$[
\text{Attn}(Q,K,V) \approx \frac{\phi(Q)\left(\phi(K)^\top V\right)}{\phi(Q)\left(\phi(K)^\top \mathbf{1}\right)}
]$
Now we can **reorder**:
1) **Prefix accumulation:** $(S=\sum_{j=1}^N \phi(k_j)V_j^\top)$ and $(z=\sum_{j=1}^N \phi(k_j))$  
2) **Query eval:** $(\text{out}_i=\frac{\phi(q_i)S}{\phi(q_i)z})$

- **Per-token cost:** O(1)  
- **Total:** **O(N)**  
- **Memory:** constant state (no growing KV cache)

> ⚠️ This is an **approximation**, not exact softmax.


#### 🧩 Where Do the Feature Maps $(\phi(\cdot))$ Come From?
We want $(\exp(q^\top k)\approx \phi(q)^\top\phi(k))$. Common constructions:

- **Taylor / Polynomial expansion (intuition):**  
  $(e^{x}\approx 1+x+\tfrac{x^2}{2!}+\cdots)$.  
  Each $((q^\top k)^m)$ can be written as a dot product in a lifted space (e.g., $(q^{\otimes m})$ with $(k^{\otimes m})$).

- **Random Features for Softmax (Performer / FAVOR+):**  
  $(\phi(x)=\exp(-\|x\|^2/2)\exp(Wx))$, with rows of $(W)$ sampled from a Gaussian; Monte-Carlo approximates $(\exp(q^\top k))$.

- **Positive kernels (e.g., ReLU maps):**  
  Cheap $(\phi(x)=\text{ReLU}(Wx)+1)$; weaker approximation but fast.

> The **better** the feature map, the **closer** we are to softmax behavior—at higher compute for \(\phi\).


#### 🔁 Why Reordering Matters for Inference
**Softmax (exact):** must compute $(\{q_t^\top k_j\}_{j\le t})$ every step → O(t) per step → **O(N²)** total.  
**Linear attention (approx.):** maintain a **running state**:
$[
S_t=S_{t-1}+\phi(k_t)V_t^\top,\quad z_t=z_{t-1}+\phi(k_t)
]$
then
$[
\text{out}_t=\frac{\phi(q_t)S_t}{\phi(q_t)z_t}
]$
No loop over history → **O(1)** per step, **O(N)** total, **constant memory**.


#### 🧮 Minimal Pseudocode (Single Head, Causal)
```python
# Running state (initialized once)
S = 0         # shape: [d_phi, d_v]
z = 0         # shape: [d_phi]

for t in range(N):            # streaming tokens
    k_t, v_t = K[t], V[t]
    phi_k = phi(k_t)          # [d_phi]
    S += outer(phi_k, v_t)    # accumulate phi(k_t) * v_t^T
    z += phi_k                # accumulate for normalization

    q_t = Q[t]
    phi_q = phi(q_t)          # [d_phi]
    out_t = (phi_q @ S) / (phi_q @ z + 1e-9)
```

#### 🆚 Linear Attention vs. FlashAttention (Online Softmax)

**FlashAttention** — *exact softmax*, with kernel tiling/fusion to cut HBM traffic; still **O(N²)** and needs a **KV cache**. (Systems/kernel optimization.)

**Linear Attention** — *changes the math* (kernel approximation) → **O(N)**, constant state, **no KV cache**; but **approximate**.

**When to use:**
- Use **FlashAttention** when you need **exactness**.
- Use **Linear Attention** when you need **sub-quadratic scaling / streaming** with acceptable approximation error.


#### ⚖️ Quality–Speed Tradeoffs

**Pros:** **O(N)** compute, constant memory, great for very long context & streaming; RNN-like updates.  
**Cons:** Not identical to softmax; may need **hybrids** (e.g., mix **linear + local + a few global softmax** layers) to match quality.


#### 💡 Intuition Recap

- **Kernel = similarity function.** Softmax uses the exponential kernel $(e^{q^\top k})$.
- **Associativity is the key.** If $(K(q,k)=\phi(q)^\top\phi(k))$, we can **pre-sum K/V once** and **apply Q after**, yielding **O(N)**.
- **Linear attention $(\neq)$ softmax.** It’s a **principled approximation** enabling **reordering** and **linear complexity**.


#### ❓FAQ

**Q: Is linear attention just “reordering” softmax?**  
**A:** No. Reordering is only valid **after** replacing softmax with an **associative kernel** approximation.

**Q: If K,V arrive one-by-one, do we still benefit?**  
**A:** Yes. We update a **fixed-size state** $((S, z))$ once per token and never revisit earlier tokens.

**Q: Which $(\phi)$ should I use?**  
**A:** **Performer/FAVOR+ random features** are a strong default; **Taylor/polynomial** features are instructive; **ReLU-style** are fastest but weaker.

**Q: Do current frontier LLMs use linear attention?**  
**A:** Most closed, general-purpose LLMs still use **softmax attention** (often with **FlashAttention** kernel


### 5.3. Diffusion Models for Language

Diffusion models break away entirely from the Transformer paradigm by **abandoning autoregressive generation**.

#### ✅ Core Idea
Adapted from image generation:

1. Start with full-sequence noise  
2. Iteratively denoise/refine for several steps  
3. Produce the complete output sequence

#### ✅ Key Benefits
- **Parallel Token Generation**  
  All tokens are generated in parallel — no sequential dependency.
- **Iterative Refinement**  
  The model improves the sequence step-by-step instead of autoregressively.

#### ✅ Inference Benefit: Massive Throughput
Because generation is **fully parallel**, GPUs can be fully saturated:

- Higher tokens/sec than autoregressive Transformers  
- Excellent performance on coding benchmarks  
- Even with multiple refinement steps, total time can still be lower

This parallelism sidesteps the core bottlenecks of:

- Autoregressive sequence dependence  
- KV cache growth  
- Memory-bound attention


### ✅ Summary

Architectural alternatives to Transformers represent a **fundamentally different approach** to solving inference bottlenecks:

- **SSMs** remove the growing KV cache by maintaining a **constant state**  
- **Linear/local attention** reduces complexity with lightweight approximations  
- **Diffusion models** eliminate autoregression entirely by generating sequences **in parallel**

Unlike KV-cache optimization techniques like GQA or MLA, these models reimagine the **entire computation and generation process**, offering potentially dramatic improvements in inference efficiency and scalability.


## 💾 6. Quantization + Pruning

### **Quantization**
Reducing precision:
- bf16 (2 bytes)  
- int8 (1 byte)  
- int4 (0.5 bytes)  

Variants:
- LLM.int8() (handles outliers)  
- Activation-aware quant (int3, int2)  
- AWQ (SmoothQuant-style)

### **Pruning**
- Remove layers/heads/neurons  
- Retrain via distillation  
- Example: 15B → 8B with tiny accuracy loss


## 7. ✅ Speedups Without Accuracy Loss (Lossless)

### 🔮 7.1. Speculative Decoding
Use **fast draft model (p)** to generate multiple candidate tokens.  
Use **slow target model (q)** to *verify them in parallel*.

Procedure:
1. Draft model proposes tokens  
2. Target model evaluates (prefill = faster)  
3. Accept/reject using corrected probability q/p  

Used in:
- Medusa (multi-token draft heads)  
- EAGLE (draft model leverages target model features)

Extreme example:  
**70B target + 8B draft** → large speedup, no accuracy loss.


## 8. 🧵 Handling Dynamic Real-World Inference Workloads

Real-world LLM inference traffic is fundamentally different from training.  
Training processes dense blocks of tokens with fixed **Batch Size × Sequence Length**, but production workloads behave like a **ragged array**:

- Requests arrive and finish at different times  
- Many requests share prefixes (system prompts, multi-sample decoding)  
- Sequence lengths vary widely, making padding inefficient  

To handle this unpredictable, heterogeneous traffic efficiently, we need specialized **systems-level techniques**. Two of the most important are **Continuous Batching** and **Paged Attention**.


### 8.1. Continuous Batching (Iteration-Level Scheduling)

Continuous batching ensures high GPU utilization by **never waiting** for a batch to finish before adding new requests.

#### ✅ The Idea  
During decoding, generation occurs **one token per step**.  
After generating a token for all active sequences:

1. The worker **returns control** to the scheduler  
2. The scheduler **adds new incoming requests** immediately  
3. The next decode step runs with the expanded batch  

This prevents the GPU from ever running “half empty.”

#### ✅ Selective Batching for Heterogeneous Lengths

Different sequences have different lengths. Traditional batching struggles because tensors must be same-sized. Continuous batching handles this via **selective batching**:

##### **1. MLP Layers (Most Compute-Heavy)**
- MLP weights are **shared across all sequences**  
- Tokens from different sequences can be **flattened and concatenated**  
  - Example: sequences of lengths 3, 9, 5 → flattened to 17 tokens  
- All tokens are processed together in one large batch  
- Huge throughput gains because MLPs dominate compute

##### **2. Attention Layers**
- Attention **cannot be merged**, because **each sequence has its own KV cache**  
- Must process each sequence separately for attention heads  
- But because MLP is the bottleneck, selective batching still yields major gains

**Result:**  
High throughput even when requests are messy, diverse, and unpredictable.


### 8.2. Paged Attention (vLLM-Style Memory Management)

Paged Attention solves the **KV-cache fragmentation** problem using ideas from operating-system paging.

#### ✅ Why Fragmentation Happens

Traditional KV-cache allocation requires a **single contiguous block** for each request:

- **Internal fragmentation:**  
  If a user expects 512 tokens but only generates 80, the leftover KV space is wasted.
- **External fragmentation:**  
  Gaps appear between allocated regions, leaving un-usable "white space."

This leads to *massive memory waste* when many sequences have different lengths.


### ✅ The Solution: Paging the KV Cache

Paged Attention divides each sequence’s KV cache into **non-contiguous blocks** and places them wherever space is available.

#### **1. Non-Contiguous KV Blocks**
- KV cache is broken into fixed-size blocks  
- Blocks can be stored **anywhere in memory**  
- No need for one large contiguous region  
- Completely eliminates internal & external fragmentation

#### **2. Prefix Sharing (Copy-on-Write)**
Many sequences share initial prefixes:

- system prompts  
- chat histories  
- multiple decoding samples from the same prompt  

Paged Attention enables **block-level sharing** of these prefixes:

- Blocks include a **reference counter**  
- Multiple sequences point to the same KV blocks  
- When sequences diverge:
  - Only the modified block is copied  
  - Others remain shared  

This saves:
- KV memory  
- Compute (prefix need not be recomputed)  
- Latency for multi-sample generation


### 8.3. Additional vLLM Optimizations

Paged Attention is part of a broader set of optimizations enabling high-throughput serving:

- **FlashAttention / FlashDecoding kernels**  
  Highly optimized attention kernels for fast memory-bound operations  
- **Kernel fusion:**  
  Merges block read + attention into single kernels to reduce overhead  
- **CUDA Graphs:**  
  Pre-captures execution graphs to reduce kernel launch overhead  
- **Efficient scheduling:**  
  Ensures continuous batching with minimal latency


### ✅ Summary

Handling dynamic, real-world inference requires more than just model optimizations — it demands **systems engineering**:

- **Continuous batching** keeps GPUs busy despite ragged, asynchronous workloads  
- **Paged Attention** eliminates KV-cache fragmentation and allows prefix sharing  
- **Kernel-level optimizations** (FlashAttention, CUDA Graphs) reduce memory traffic & overhead  

Together, these techniques allow LLM serving frameworks like **vLLM** to achieve far higher memory efficiency and throughput on unpredictable, heterogeneous inference traffic.



## 9. ✅ Final Summary

Inference ≠ training.

Inference is:
- ✅ **sequential**
- ✅ **memory-bound**
- ✅ **dynamic**
- ✅ **KV-cache dominated**

Optimizations fall into 3 buckets:

### ✅ **Lossy**
- KV cache reduction (GQA/MLA/CLA)  
- New architectures (Mamba, linear attention)  
- Quantization / pruning  

### ✅ **Lossless**
- Speculative decoding  
- Parallel verification  
- Medusa/EAGLE  

### ✅ **System-level**
- Continuous batching  
- Paged attention  
- Multi-replica parallelism  

The biggest wins increasingly come from **rethinking architectures** (SSMs, hybrids, diffusion).  
