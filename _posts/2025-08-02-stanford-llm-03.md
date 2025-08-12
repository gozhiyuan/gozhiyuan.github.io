---
layout: post
title: LLM Architectures and Hyperparameters 🧠
subtitle: Language Modeling from Scratch Lecture 3
categories: Large-Language-Model
tags: [Stanford-LLM-From-Scratch-2025]
banner: "/assets/images/banners/yuanpang-wa-iceburg2.jpg"
---

# LLM Architectures and Hyperparameters 🧠

This lecture summarizes key architectural trends, hyperparameter choices, and stability tricks observed in modern Large Language Models (LLMs).

- **📈 Architectural Trends:** While the field is rapidly evolving, a "convergent evolution" towards "LLaMA-like" architectures is evident.
- **🔑 Key Consensus:** Widespread adoption of pre-normalization, RMSNorm, and Rotary Position Embeddings (RoPE).
- **🚀 Empirical Gains:** Gated Linear Units (*GLU) for activations have also shown consistent empirical gains.
- **👍 Rules of Thumb:** Hyperparameter choices, such as feedforward dimension ratios and aspect ratios, often adhere to surprising consensus rules of thumb.
- **💡 Recent Advancements:** Focus heavily on stability tricks, particularly for softmax operations, and attention variants like Multi-Query Attention (MQA) and Grouped-Query Attention (GQA) to optimize inference costs.

[Course link](https://stanford-cs336.github.io/spring2025/)


## 1. Core Architectural Components & Variations
The foundational "original" Transformer architecture has undergone significant modifications, with many modern LLMs sharing common advancements.

### 1.1 🔄 Normalization (LayerNorm vs. RMSNorm & Pre-vs-Post Norm)
![alt_text](/assets/images/llm-from-scratch/03/1.png "image_tooltip")
**Pre-vs-Post Norm:** This is the most consistent consensus in modern LLMs.
- Almost all models released since early GPT versions use **pre-normalization**.
- LayerNorm is applied *before* the multi-head attention and feed-forward blocks in the residual stream.
- This design choice, in contrast to BERT's post-normalization, significantly improves training stability and allows for larger learning rates by preventing gradient attenuation and spikes.
> "Almost all modern LMs use pre-norm (but BERT was post-norm)."
> "Original stated advantage– removing warmup. Today – stability and larger LRs for large networks."

**Newer Variations ("Double Norm"):**
![alt_text](/assets/images/llm-from-scratch/03/2.png "image_tooltip")
- Some recent models (Grok, Gemma 2, Olmo 2) experiment with adding a second LayerNorm *outside* the residual stream, after the attention and FFN blocks.
- This is distinct from traditional post-norm as it doesn't interfere with the main residual signal path, further enhancing stability.

**LayerNorm vs. RMSNorm:**
![alt_text](/assets/images/llm-from-scratch/03/3.png "image_tooltip")
While the original Transformer used LayerNorm (normalizing mean and variance), many modern LLMs (LLaMA family, PaLM, Chinchilla) have transitioned to RMSNorm.
RMSNorm omits the mean subtraction and bias term, resulting in:
- **🏎️ Faster Computation:** Fewer operations due to no mean calculation.
- **💾 Fewer Parameters:** No bias term to store.
- **✅ Better Performance:** RMSNorm achieves comparable or even better performance.
- **🧠 Memory Efficiency:** Crucially, RMSNorm reduces data movement, a significant factor in runtime.

- **Fewer Parameters:** RMSNorm does not add a bias term (beta) that needs to be stored and loaded from memory, unlike LayerNorm. This reduction in parameters directly contributes to less data needing to be moved.
- **Fewer Operations:** RMSNorm does not subtract the mean of the activations. While matrix multiplications are most of the FLOPs (99.8%), normalization (0.17% of FLOPs) can be 25% of the runtime due to memory movement.
- **Memory Movement Matters:** Even though the FLOPs saved by RMSNorm are small, the overall runtime is heavily influenced by memory movement. Less data movement leads to faster execution.

In practice, RMSNorm offers runtime improvements and sometimes even performance gains (lower loss). A win-win! 🏆
> "Important lesson: FLOPS are not runtime!"
> "RMSNorm can still matter due to the importance of data movement."

**General Trend: Dropping Bias Terms:**
Most modern Transformers omit bias terms in linear layers for similar reasons: memory efficiency and optimization stability.
> "People more generally drop bias terms since the compute/param tradeoffs are not great."


### 1.2 ⚡ Activations and Feedforward Networks (FFN)
**Activation Zoo:** There's a wide variety of activations (ReLU, GeLU, Swish, ELU, *GLU variants).

![alt_text](/assets/images/llm-from-scratch/03/4.png "image_tooltip")

***Gated Linear Units (GLU):***
- A significant trend is the adoption of gated activations, specifically **SwiGLU** (LLaMa, PaLM, Mistral) and **GeGLU** (T5 v1.1, LaMDA, Phi3, Gemma).
- These modify the first part of an FF layer by augmenting it with an entry-wise linear term.
> "*GLU isn’t necessary for a good model (see GPT3), but it’s probably helpful."
> "Evidence points towards somewhat consistent gains from Swi/GeGLU."

**Parameter Matching:** Gated models often use smaller dimensions for the $d_{ff}$ (by a factor of 2/3) to maintain a similar total parameter count as non-gated counterparts.

```python
class SwiGLU(nn.Module):
    def __init__(
            self, d_model: int, d_ff: int | None = None,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
        ):
        """
        SwiGLU is a variant of the GLU activation function that uses SiLU (Swish) activation.
        The feed-forward dimension d_ff is set to approximately 8/3 * d_model, rounded to the nearest multiple of 64.
        
        Args:
            d_model: int
                The input and output dimension
            d_ff: Optional[int]
                The intermediate dimension. If None, will be set to nearest multiple of 64 to 8/3 * d_model
            device: Optional[torch.device]
                The device to create the module's parameters on
            dtype: Optional[torch.dtype]
                The dtype to create the module's parameters with
        """
        super().__init__()
        
        # If d_ff not provided, compute it as nearest multiple of 64 to 8/3 * d_model
        if d_ff is None:
            d_ff = int(round(8/3 * d_model / 64) * 64)
            
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)  # First projection
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)  # Output projection
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)  # Gate projection
        self.act = nn.SiLU()

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        # Project using Linear layers (which use einsum internally)
        w1x = self.w1(x)  # shape: ... d_ff
        w3x = self.w3(x)  # shape: ... d_ff
        
        # Apply SiLU only to w1x and multiply with w3x
        activated = self.act(w1x) * w3x  # shape: ... d_ff
        
        # Project back to d_model dimension
        return self.w2(activated)  # shape: ... d_model

```


### 1.3 ⏩ Serial vs. Parallel Layers
- **Standard:** Most Transformer blocks operate serially (attention then MLP).
- **Parallelization:** A few models (GPT-J, PaLM, GPT-NeoX, Cohere Command A) compute attention and MLP in parallel, then add their outputs to the residual stream. This offers potential compute wins by allowing shared LayerNorms and fused matrix multiplies, improving GPU utilization.
> "No extremely serious ablations, but has a compute win."


### 1.4 📍 Position Embeddings
![alt_text](/assets/images/llm-from-scratch/03/5.png "image_tooltip")
**Evolution:** Position embeddings have seen diverse approaches:
- **Sine Embeddings:** (Original Transformer) Add sine and cosine functions.
- **Absolute Embeddings:** (GPT1/2/3, OPT) Add a learned position vector to the embedding.
- **Relative Embeddings:** (T5, Gopher, Chinchilla) Add a vector to the attention computation.
- **Rotary Position Embeddings (RoPE):** Most modern models (GPT-J, PaLM, LLaMA, and almost all models post-2023) have converged on RoPE.

**Core Idea of RoPE:**
![alt_text](/assets/images/llm-from-scratch/03/6.png "image_tooltip")
RoPE ensures the attention function only depends on the relative position ($i-j$) between tokens by rotating query/key vectors based on their absolute positions. Inner products are invariant to arbitrary rotations.

**Implementation:**
RoPE is applied at each attention operation by multiplying query/key inputs with sines and cosines, unlike additive sine embeddings.
![alt_text](/assets/images/llm-from-scratch/03/7.png "image_tooltip")

The core idea of RoPE is that while the individual rotation angles for each position are unique, the relative distance between positions is what determines the final attention score. For each posistion and dimension pair has it's own rotate submatrix.

**The Angles Are Not the Same 📐**
For a given dimension pair $i$, the rotation angle for position 3 is $\theta_{3,i}$, and for position 4 it's $\theta_{4,i}$. These are different because the angles depend on the absolute position.

**Individual Rotations, Collective Effect 🔄**
A vector at position $m$ is rotated based on a set of angles $\theta_{m,0}, \theta_{m,1}, \dots, \theta_{m, d/2-1}$. Each angle corresponds to a different dimension pair within the vector.

**The Dot Product Is What Matters ✨**
The magic happens when we take the dot product of a query vector from position 3 ($q_3$) and a key vector from position 1 ($k_1$). The attention score is proportional to this dot product.

The dot product of the rotated vectors simplifies to:
$$ q_3 \cdot k_1 \propto \sum_{i=0}^{d/2-1} [q_{3,2i}k_{1,2i} + q_{3,2i+1}k_{1,2i+1}] \cdot \cos(\theta_{3,i} - \theta_{1,i}) + \dots $$

Notice the term $\cos(\theta_{3,i} - \theta_{1,i})$. The crucial part is that the rotation angle difference is what's used. Since $\theta_{m,i} = m \cdot \omega_i$, the difference is:
$$ \theta_{3,i} - \theta_{1,i} = (3 \cdot \omega_i) - (1 \cdot \omega_i) = (3-1) \cdot \omega_i = 2 \cdot \omega_i $$

Now, let's look at positions 4 and 2. The angle difference is:
$$ \theta_{4,i} - \theta_{2,i} = (4 \cdot \omega_i) - (2 \cdot \omega_i) = (4-2) \cdot \omega_i = 2 \cdot \omega_i $$

Since the angle difference is the same in both cases, RoPE treats the relative position of $(3,1)$ and $(4,2)$ as identical. This allows the model to generalize effectively.

> "Rope has now many different algorithms for extrapolating context length and that's an important part of sort of the modern productionized language model but also it seems to be empirically quite effective even at fairly small scales in small context length so it's kind of won out on this position embedding battle."



## 2. Hyperparameter Consensus and Considerations
Despite the massive scale of LLMs, many hyperparameters show surprising consensus or well-defined optimal ranges.

### 2.1 📏 Feedforward Dimension ($d_{ff}$) to Model Dimension ($d_{model}$) Ratio
![alt_text](/assets/images/llm-from-scratch/03/8.png "image_tooltip")
- **Rule of Thumb:** A strong consensus exists for $d_{ff} = 4 \cdot d_{model}$ for ReLU-style FFNs.
- **GLU Variant Adjustment:** For GLU variants, this ratio is commonly scaled down to $d_{ff} = 8/3 \cdot d_{model}$ (approximately 2.66), to maintain similar parameter counts.
- **Outliers:** T5-11B famously used a 64x multiplier ($d_{ff} = 64 \cdot d_{model}$), demonstrating that radical choices can work, though T5 v1.1 later reverted to a more standard 2.5x multiplier, suggesting the extreme ratio might be suboptimal.
- **Empirical Basis:** Studies show a "basin" between 1-10 where this hyperparameter is near-optimal.


### 2.2 🧩 Head-Dim * Num-Heads to Model-Dim Ratio
![alt_text](/assets/images/llm-from-scratch/03/9.png "image_tooltip")
- **Standard Practice:** Most models adhere to the guideline that $head\_dim \cdot num\_heads = d_{model}$. This means as the number of heads increases, the dimension per head decreases proportionally, keeping the total attention parameter count fixed.
- **Exceptions:** T5 is a notable exception, using a 16x ratio.
- **Practicality:** While some theoretical arguments suggest issues with very low dimensions per head (low rank bottlenecks), in practice, models with a 1:1 ratio perform well.
> "But we don’t seem to be seeing significant ‘low rank bottlenecks’ in practice.."


### 2.3 📐 Aspect Ratio ($d_{model} / n_{layer}$)
- **Sweet Spot:** Most models show consistency, clustering around a ratio of $d_{model} / n_{layer}$ of 100-200.
- **Examples:** BLOOM (205), GPT3/OPT/Mistral/Qwen (128), LLaMA/LLaMA2 (102).
- **Trade-offs:**
    - **Deep Models:** Harder to parallelize (e.g., pipeline parallel) and can have higher latency.
    - **Wide Models:** Can leverage tensor parallelism more effectively, but require fast networking.
- **Empirical Evidence:** Studies by Kaplan et al. (2020) show an optimal aspect ratio around 100 across different model scales, with the optimum not shifting significantly with size.


### 2.4 🔤 Vocabulary Sizes
![alt_text](/assets/images/llm-from-scratch/03/10.png "image_tooltip")
- **Monolingual Models:** Typically range from 30,000-50,000 tokens (e.g., Original Transformer, GPT, LLaMA).
- **Multilingual / Production Systems:** Trend towards larger vocabularies of 100,000-250,000 tokens (e.g., mT5, PaLM, GPT-4, Command A). This accommodates diverse languages, emojis, and improves inference cost for non-English languages by packing them into fewer tokens.
> "Monolingual vocabs don’t need to be huge, but multilingual ones do."


### 2.5 🎯 Regularization (Dropout and Weight Decay)
![alt_text](/assets/images/llm-from-scratch/03/11.png "image_tooltip")
**Arguments Against Pretraining Regularization:**
- With vast datasets and typically only one pass over the corpus, overfitting during pretraining is not generally a primary concern.
> "SGD only does a single pass on a corpus (hard to memorize)."

**Practice:**
- **Dropout:** Common in older models but has largely fallen out of favor in newer open models.
- **Weight Decay:** Continues to be widely used.

**Why Weight Decay?**
![alt_text](/assets/images/llm-from-scratch/03/12.png "image_tooltip")
- Its primary effect is not to control overfitting but rather to improve optimization dynamics.
- Weight decay interacts with learning rate schedules (especially cosine decay), leading to faster convergence and better training losses, particularly towards the tail end of training when learning rates decrease.
> "It’s not to control overfitting Weight decay interacts with learning rates (cosine schedule)."
> "You still ‘regularize’ LMs but its effects are primarily on optimization dynamics."


## 3. Stability Tricks
Training very large models often encounters stability issues, particularly concerning gradient explosions. Many recent innovations focus on mitigating these.

### 3.1 🚨 Problem Areas: Softmax Operations
Softmax functions are prone to numerical instability due to exponentials and potential division by zero. This applies to both the output softmax (for token prediction) and the attention softmax.


### 3.2 🔒 Output Softmax Stability: Z-Loss
![alt_text](/assets/images/llm-from-scratch/03/13.png "image_tooltip")
- **Concept:** Introduced by PaLM, this trick adds an auxiliary loss term ($\alpha \cdot \log(Z_{xi})^2$) to the main objective. $Z_{xi}$ is the softmax normalizer (sum of exponentials of logits).
- **Purpose:** It encourages $\log(Z_{xi})$ to be close to zero, effectively forcing the normalizer $Z_{xi}$ to be close to one. This makes the softmax operation numerically more stable by preventing logits from "blowing up."
- **Adoption:** Baichuan 2, DCLM, OLMo 2, and Gemma 2 have adopted this.

**The Softmax Problem**
Suppose at the final layer of your LM, you get logits for a vocabulary of size 5:
`init logits = [12.0, 8.0, -3.0, 2.0, 0.5]`

The softmax probability for token i is:
$$ P(i) = \frac{\exp(\text{logit}_i)}{Z} $$
where
$$ Z = \sum_j \exp(\text{logit}_j) $$

**Step 1 — Calculate $Z$**
```
exp(12.0) = 162754.79
exp(8.0)  = 2980.96
exp(-3.0) = 0.0498
exp(2.0)  = 7.389
exp(0.5)  = 1.6487
```
Sum:
$$ Z = 162754.79 + 2980.96 + 0.0498 + 7.389 + 1.6487 \approx 165744.84 $$

**Step 2 — Why this is bad**
- $Z$ is huge here (165k).
- Large $Z$ means the softmax exponentials are in the unstable range (prone to floating-point overflow).
- The normalizer also varies wildly during training → causes loss spikes.

**🛠 Step 3 — Z-loss Fix**
Z-loss adds an extra penalty term:
$$ \text{Z-loss term} = \alpha \cdot (\log(Z))^2 $$

Example:
`log(Z) = log(165744.84) ≈ 12.015`

If α = 0.1:
$$ \text{Z-loss term} = 0.1 \cdot (12.015)^2 \approx 14.44 $$

**Step 4 — Combined Loss**
If your original cross-entropy loss was:
`CE loss = 2.35`

then your total loss becomes:
`Total Loss = 2.35 + 14.44 ≈ 16.79`

This big penalty tells the model:
> “Whoa — your Z is way too large. Adjust your logits so that Z stays near 1.”

**🧠 Intuition**
- **Without Z-loss:** The model can push all logits very high or low, causing instability.
- **With Z-loss:** Model learns to keep logits calibrated so $Z \approx 1$, meaning $\log(Z) \approx 0$.
This makes the final softmax more numerically stable and prevents loss explosions in large-scale training.

### 3.3 🛡️ Attention Softmax Stability: QK Norm
- **Concept:** Queries (Q) and Keys (K) are LayerNormed (or RMSNormed) before their inner product is computed for the attention softmax.
- **Purpose:** This bounds the size of the inputs to the softmax, naturally controlling its bad behaviors.
- **Origin:** This technique originated in vision and multimodal models (Dehgani 2023, Chameleon, Idefcs) to stabilize training, and has since been adopted by pure language models like Gemma 2, DCLM, and OLMo2.
- **Effectiveness:** LayerNorms are "shockingly effective" for stability without significantly affecting performance.


### 3.4 📉 Logit Soft-Capping
- **Concept:** Applying a Tanh function to the logits before the softmax operation to cap them at a maximum value.
- **Purpose:** Prevents logits from becoming excessively large.
- **Usage:** Used by Gemma 2 and OLMo 2, though some evidence suggests it might have performance issues compared to QK norm.


## 4. Attention Head Variants & Inference Cost Optimization
While the core attention mechanism remains similar, variations address inference costs and context length.
![alt_text](/assets/images/llm-from-scratch/03/14.png "image_tooltip")

### 4.1 🔀 Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)
**Motivation:**
Standard multi-head attention's KV cache incurs significant memory access overhead, especially for long sequences and small batch sizes. This leads to poor arithmetic intensity during inference.
> "Arithmetic intensity is not good."

**MQA:**
- Instead of having distinct key and value heads for each query head, MQA uses **one shared set of key and value heads** across all query heads.
- **Benefit:** Dramatically reduces the size of the KV cache and memory movement during inference, improving arithmetic intensity and making longer sequence lengths viable.
> "We have much fewer items to move in and out of memory (KV Cache)."

![alt_text](/assets/images/llm-from-scratch/03/15.png "image_tooltip")


**Multi-Query Attention (MQA) Explained:**
MQA is an optimization to reduce inference costs, particularly memory movement and the KV (Key-Value) cache.

**The Problem with Standard Attention during Inference:**
- **Training vs. Inference:** Training has high "arithmetic intensity" (ratio of arithmetic ops to memory access). GPUs like this.
- **Incremental Generation:** Inference is autoregressive (one token at a time), which is not as parallelizable.
- **The KV Cache:** To avoid recomputing past keys/values, a KV cache is used. It grows with each new token.
- **Poor Arithmetic Intensity:** Repeatedly loading the growing KV cache for each new token lowers arithmetic intensity. The $n/d$ term (sequence length / model dimension) is a "core inference cost trade-off".

**MQA as a Solution:**
- **Key Idea:** MQA uses multiple query heads but only **one key and one value head**.
- **Reduced Memory Movement:** Sharing key/value projections significantly reduces the KV cache size and memory access.
- **Improved Arithmetic Intensity:** Sharing K and V dimensions improves arithmetic intensity, making longer sequences more viable.

Although MQA is conceptually straightforward (broadcasting K and V values), efficient implementations are hard to find. Kernels in FlashAttention or VLLM's PagedAttention expect pre-broadcasted KV-cache values.

**Performance:** Can sometimes incur a small perplexity hit but offers significant runtime savings.

```python
import torch
import math

# Batch size (N), number of heads (h), key/value dimension (d_k), current seq length (L)
N, h, d_k, L = 2, 4, 16, 5

# ----------------------------
# Cached K and V values across previous timesteps
# K and V are shared across all heads in MQA (no `h` dimension)
# Shape explanation:
#   K: [N, 1, L, d_k]  ->  batch_size, singleton head dim, seq_length, key_dim
#   V: [N, 1, L, d_k]  ->  same shape as K
# Example: batch=2, seq_len=5, key dim=16
K = torch.randn(N, 1, L, d_k)
V = torch.randn(N, 1, L, d_k)

# ----------------------------
# New Q, K, V for current incremental step (generating one token)
# Q has separate heads dimension h since queries differ per head
# K_incr, V_incr are shared for all heads (no h dim)
# Shapes:
#   Q_incr: [N, h, 1, d_k]  -> batch, heads, current token (1), key dim
#   K_incr: [N, 1, 1, d_k]  -> batch, singleton head dim, current token, key dim
#   V_incr: [N, 1, 1, d_k]  -> same as K_incr
Q_incr = torch.randn(N, h, 1, d_k)
K_incr = torch.randn(N, 1, 1, d_k)
V_incr = torch.randn(N, 1, 1, d_k)

# ----------------------------
# Update KV cache by appending new keys and values for current token
# Concatenate along the sequence length dimension (dim=2)
K = torch.cat([K, K_incr], dim=2)  # New K shape: [N, 1, L+1, d_k]
V = torch.cat([V, V_incr], dim=2)  # New V shape: [N, 1, L+1, d_k]

# ----------------------------
# Compute attention logits
# Q_incr shape: [N, h, 1, d_k]
# K shape: [N, 1, L+1, d_k]
# PyTorch broadcasts K across the heads dimension h for matmul
# Compute dot product between query and all keys:
logits = torch.matmul(Q_incr, K.transpose(-2, -1))
# Resulting logits shape: [N, h, 1, L+1]
# Each element logits[n,h,0,l] is dot product of Q at batch n, head h, token 0 with key at position l

# ----------------------------
# Apply scaled softmax along sequence length dimension (dim=-1)
softmax_out = torch.softmax(logits / math.sqrt(d_k), dim=-1)
# Shape: [N, h, 1, L+1]
# Softmax turns logits into attention weights over all cached tokens

# ----------------------------
# Compute weighted sum over values V
# softmax_out shape: [N, h, 1, L+1]
# V shape: [N, 1, L+1, d_k], broadcasted to match heads dimension h
attn_out = torch.matmul(softmax_out, V)
# Result shape: [N, h, 1, d_k]
# This is the attended representation per head for the current token

# ----------------------------
# Example summary:
# - Q_incr differs across heads (h dimension) allowing multiple perspectives
# - K and V are shared for all heads (no h dimension) saving memory and bandwidth
# - Broadcasting handles replication of K and V across heads during matmul
# - KV cache grows with sequence length as tokens are generated incrementally

```

**GQA: An Extension of MQA**
- **Grouped Query Attention (GQA)** is a recent extension that offers a knob to control the trade-off between inference efficiency and model expressiveness.
- Instead of going all the way to just one key/value dimension (as in MQA), GQA allows for a reduced number of key/value dimensions, grouped among the query heads.
- This provides a balance, as going from multi-head all the way to multi-query can sometimes be too aggressive.

**Performance Considerations**
- While MQA offers substantial compute wins during inference, it has been observed to sometimes incur a small perplexity (PPL) hit.
- However, GQA has shown low to no performance hit, making it a generally preferred approach.
- **Benefit:** Provides a knob to trade off between expressiveness (key-query ratio) and inference efficiency, often with low to no performance hit.


### 4.2 🪟 Sparse and Sliding Window Attention
**Motivation:** The quadratic computational cost of full self-attention with increasing sequence length is prohibitive for very long contexts.

- **Sparse Attention (OpenAI 2019):** Restricts the attention pattern to a subset of tokens (e.g., local windows, diagonal patterns) to reduce compute costs while maintaining effective receptive field through depth. Used by GPT-3.
- **Sliding Window Attention (Mistral):** Each attention head only attends to a fixed-size window around the current token. The effective context length is extended by depth.

**Sparse and Sliding Window Attention Explained:**
These are variations of the attention mechanism designed to reduce computational costs and enable models to handle longer context lengths.

- **The Problem:** Standard self-attention's computational cost scales quadratically with sequence length, which is expensive for long sequences.
- **Sparse Attention:**
    - **Concept:** Restrict connections instead of allowing every token to attend to every other token.
    - **Mechanism:** Might involve local windows or structured patterns like diagonals.
    - **Trade-off:** Trades some expressiveness for significant runtime reductions.
    - **Usage:** GPT-3 used these tricks for larger attention windows.
- **Sliding Window Attention:**
    - **Concept:** A token only attends to a small, fixed-size window around its position.
    - **Mechanism:** The effective receptive field grows with network depth, allowing information to propagate over longer distances.
    - **Purpose:** Controls computational resources for longer contexts.
    - **Usage:** Mistral models are noted for using this.

**Current Standard Trick (Hybrid Approach):**
![alt_text](/assets/images/llm-from-scratch/03/16.png "image_tooltip")
Recent models (Cohere Command A, LLaMA 4, Gemma) interleave "full" attention layers with "sparse" or "sliding window" attention layers.

- **Interleaving:** Every Nth layer (e.g., 4th) might use full self-attention with no position embedding for long-range information capture.
- **Local Attention with RoPE:** The other layers use sliding window attention with RoPE for local dependencies.
- **Benefits:** This strategy controls system-level costs and aids length extrapolation.

> "Every 4th layer is a full attention. Long-range info via NoPE, short-range info via RoPE + SWA." This combines the benefits of full attention for long-range dependencies with the efficiency of sparse attention for local context.