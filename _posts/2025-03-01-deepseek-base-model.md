---
layout: post
title: DeepSeek Base Models Series
subtitle: Base Models
categories: Large-Language-Model
tags: [Blog]
banner: "/assets/images/banners/yuanpang-wa-iceburg2.jpg"
---

# 🧠 DeepSeek Base Models Series

The blog is structured around two primary categories of DeepSeek’s work: **Base Models** and **Reasoning Models**.  
📌 In **Part One**, we focus on DeepSeek’s **Base Models**.

We’ll walk through the core ideas of four foundational DeepSeek papers, along with example PyTorch code to illustrate key components like **Multi-head Latent Attention (MLA)** and **Mixture-of-Experts (MoE)** architectures:

- [DeepSeek LLM: Scaling Open-Source Language Models with Longtermism](https://arxiv.org/pdf/2401.02954)  
- [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/pdf/2401.06066)  
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/pdf/2405.04434)  
- [DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437)

Much of the content is distilled from the key themes and innovations presented in this [YouTube discussion](https://www.youtube.com/watch?v=Hbmr1dJutnk&t=1423s), which highlights nine of DeepSeek’s most impactful papers. The video features insights from AI researcher **He Junxian** (HKUST), offering a deep technical dive into DeepSeek’s contributions to advancing **Artificial General Intelligence (AGI)**.



## 📄 [DeepSeek LLM Scaling Open-Source Language Models with Longtermism](https://arxiv.org/pdf/2401.02954)

### 🎯 Main Theme
This paper in January 2024 likely details DeepSeek's approach to scaling open-source LLMs with a long-term vision. While the specific innovations are not elaborated upon in the provided text, the title suggests a focus on sustainable and scalable development for open-source models, a crucial aspect for broader AI adoption and research.

"DeepSeek LLM Scaling Open-Source Language Models with Longtermism" is a **foundational paper** for DeepSeek, marking their initial achievement in developing large language models. It encapsulates DeepSeek's core operational philosophy, characterized by an open, honest, low-key, and rigorous scientific approach.

### 🔍 Key Aspects of the Paper

- **🔁 Replication and Refinement**  
  While the paper is not presented as having significant "special innovation" in its core model design, it is described as a high-quality "reproduction of Llama 2". DeepSeekLLM utilized the same model architecture and training methods as Llama 2, but distinguished itself through its data. It was developed as a Chinese-English model, incorporating higher quality data than Llama 2.

- **📊 Model Sizes and Performance**  
  The model was released in two sizes: **7 billion (7B)** and **67 billion (67B)** parameters, corresponding to Llama 2's offerings. Notably, DeepSeekLLM outperformed Llama 2 70B, especially in Chinese. Upon its release, the open-source 7B and 67B versions were recognized as **some of the strongest open-source models in China**.

- **📚 Rigorous Academic Approach**  
  The paper demonstrates a deep commitment to scientific rigor:

  - **📈 Learning Rate Scheduler**  
    Replacing cosine with a **multi-step scheduler** for dynamic training adjustment.

  - **📐 In-depth Scaling Laws Analysis**  
    DeepSeekLLM delved into scaling of **hyperparameters** and proposed a more accurate formula considering **attention overhead**.

  - **🔍 Data Quality Insight**  
    Showed that **data quality affects optimal scaling configuration**, emphasizing early awareness of data-centric AI.

### ⚖️ 1. Re-evaluation of Scaling Laws

- Past works by **Hoffmann et al.** and **Kaplan et al.** lacked consistent hyperparameter details.
- DeepSeek aimed to **clarify and improve** scaling law methodology for open-source LLMs.

### 📏 2. Scaling Laws for Hyperparameters

- **Focus**: Investigated batch size (B) and learning rate (η) scaling.
- **Power Law Relationship**:
  - η_opt = 0.3118 · C^-0.1250  
  - B_opt = 0.2920 · C^0.3271
- **🧩 Insight**: Optimal values fall in **broad ranges**, simplifying practical tuning.
- **⚠️ Limitation**: Further variables beyond compute budget (C) need exploration.

### 🔢 3. Estimating Optimal Model and Data Scaling

- Introduced **"non-embedding FLOPs/token (M)"** as a new scale representation.
- Critiqued older metrics (N1, N2) for misestimating compute costs.
- **IsoFLOP strategy** used to allocate between model (M) and data (D) scale.
- **Formulas**:
  - M_opt = 0.1715 · C^0.5243  
  - D_opt = 5.8316 · C^0.4757


### 🧮 4. Scaling Laws with Different Data Quality

- **💡 Discovery**: High-quality data shifts optimal scaling to **larger models**.
- **📉 Data exponent (b)** goes down, **📈 model exponent (a)** goes up.
- **📚 Speculation**: Cleaner, logically coherent data favors parameter growth.
- Reinforces DeepSeek's focus on **data curation** as a scientific foundation.


### 📏 Example: How Much Data and Compute to Train GPT-3–Scale Models?

Let’s use DeepSeek's scaling laws to estimate how much **data**, **compute**, and **GPU resources** are needed to train models like GPT-3 or smaller ones (e.g., 1B, 7B).  
We assume a compute budget of:  
$[
C = 1 \times 10^{23} \text{ FLOPs}
]$


#### 🔢 Step 1: Use DeepSeek's Formulas

From the DeepSeek paper, the optimal **model compute per token** (M) and **data size** (D) are:

$[
M_{\text{opt}} = 0.1715 \cdot C^{0.5243}
]$
$[
D_{\text{opt}} = 5.8316 \cdot C^{0.4757}
]$

Plug in $( C = 10^{23} )$:

- $( M_{\text{opt}} \approx 1.96 \times 10^{11} \text{ FLOPs/token} )$
- $( D_{\text{opt}} \approx 5.09 \times 10^{11} \text{ tokens} )$


#### 📊 Result: What Does This Mean?

To **train a model using 1e23 FLOPs** (GPT-3 level), the ideal setup is:

| Metric               | Value                                 |
|----------------------|----------------------------------------|
| Compute Budget $( C )$ | $( 1 \times 10^{23} )$ FLOPs             |
| Model Size $( M )$     | $( 1.96 \times 10^{11} )$ FLOPs/token    |
| Training Tokens $( D )$| $( 5.09 \times 10^{11} )$ tokens         |
| Equivalent to         | ~509B tokens of high-quality text      |

🧠 This tells you **how large the model’s internal compute** should be, and **how much data** it should be trained on to perform well.


#### 🚀 Step 2: GPU Estimate (A100s)

Let’s estimate how many **NVIDIA A100 GPUs** you’d need to train this in **30 days**.

- A100 throughput (realistic): **~1.25 × 10¹⁴ FLOPs/sec**
- Total training time for 1 A100:
  $[
  \frac{1 \times 10^{23}}{1.25 \times 10^{14}} \approx 8.0 \times 10^{8} \text{ seconds} ≈ 25.4 \text{ years}
  ]$
- To finish in **30 days**, you'd need:
  $[
  \frac{25.4 \times 365}{30} ≈ 3090 \text{ A100 GPUs}
  ]$


#### 📦 What If You're Training a Smaller Model?

Let’s scale down the compute and data requirements based on **model size** and assume proportional scaling:

| Model Size  | Approx. Compute (FLOPs) | Tokens Needed | A100s for 30d |
|-------------|--------------------------|----------------|----------------|
| 1B params   | $( ~2 \times 10^{21} )$     | ~12B tokens     | ~62 GPUs       |
| 7B params   | $( ~1.5 \times 10^{22} )$   | ~75B tokens     | ~470 GPUs      |
| 13B params  | $( ~3.0 \times 10^{22} )$   | ~150B tokens    | ~930 GPUs      |
| GPT-3 (175B)| $( 1 \times 10^{23} )$      | ~509B tokens    | ~3090 GPUs     |

⚠️ These are **order-of-magnitude estimates**, assuming DeepSeek’s scaling law holds.


#### ✅ TL;DR

To train a GPT-3–level model using **optimal scaling**:

- 📚 You need ~**509B tokens** of high-quality training data
- ⚙️ Each token requires ~**196B FLOPs**
- 🖥️ On **A100 GPUs**, you’d need ~**3090 GPUs for 30 days**

If you scale down the model, you save compute, data, and cost — but still benefit by **balancing model size and data size carefully**, just like DeepSeek's paper advises. 📐


### 🔍 Unprecedented Transparency Regarding Benchmarks

DeepSeek openly demonstrated **benchmark cheating** via their **C-Eval benchmark**:

- Training on test-related MCQs inflated scores (e.g., 47 → 71).
- DeepSeek **published true scores** and detailed manipulation methods.
- This transparency was **rare and commendable**, especially in a competitive market.

### 🧭 Reflection of DeepSeek's Culture

This paper set a **cultural precedent**:

- **No marketing fluff**  
- **Deep technical rigor**  
- **Genuine scientific exploration**  
- Acts like an **academic lab within industry**

DeepSeek LLM Scaling paper reflects a team committed to **long-term, open-source AGI advancement** with **honesty, humility, and high standards**.

---

## 🧠 [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/pdf/2401.06066)

**Main Theme**: This paper in January 2024 explores the **Mixture-of-Experts (MoE)** architecture — a highly efficient paradigm for large language models (LLMs). DeepSeek's contribution centers on achieving *ultimate expert specialization*, improving how different “experts” handle specific tasks or data types. This leads to improved performance and efficiency. MoE is a major trend in modern model design, allowing massive models to be trained and run more efficiently. ⚙️


### 📚 Background

The paper **"DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models"** marks DeepSeek's entry into MoE models, following their initial dense base model. Rather than being a final product, this paper is a *technical study* exploring MoE architectures, laying groundwork for future models like **DeepSeek-V2**.

### 🔍 Key Innovations in DeepSeekMoE

#### 🧱 Transition to MoE Architecture

- DeepSeekLLM (base model) used a **dense** architecture like LLaMA 2.
- DeepSeekMoE introduces **Mixture-of-Experts**:
  - The model is divided into multiple **experts**.
  - Instead of activating *all* parameters, MoE models activate a **small subset** — reducing inference cost 💡.
  - This aligns with rumors that large models like **GPT-4** use MoE to save compute.

![alt_text](/assets/images/deepseek/moe.png "image_tooltip")

#### 👨‍🔬 Expert Specialization at Scale

- **Challenge**: Past MoE models used only 8–16 experts.
- **DeepSeekMoE Innovation**: Increased expert count to **64 specialized experts** 🧠.
  - Promotes **clearer differentiation** and better **task specialization**.
  - Reduces overlapping knowledge between experts.
  - Supports **finer-grained specialization**.


#### 🧩 Shared Experts vs Proprietary Experts

- 🆕 **Concept**: DeepSeek introduced **shared experts**.
  - **Shared Experts**: Handle general knowledge (e.g., grammar, syntax).
  - **Proprietary Experts**: Specialize in domains (e.g., math, physics).
- 🎯 **Purpose**: Only activate specialized experts when needed — improving **efficiency**.
- 🔁 **Analogy**: Shared expert = common sense; proprietary expert = domain expert.


#### 💰 Efficiency & Cost Reduction

- MoE models activate **only part of the network** — drastically reducing inference cost.
- A **16B MoE model** matched performance of DeepSeekLLM 7B **dense model**, using only **40% of the inference compute** 🧮.
- This demonstrated that MoE models could achieve **high performance at lower cost**, making them practical for deployment.


#### 🧪 Experimental Scale & Roadmap

- Trained **2B and 16B MoE models** to validate ideas.
- **145B MoE** was **partially trained**, showing strong potential.
- Served as a *sandbox* to validate ideas before full scaling in **DeepSeek-V2** 🚀.


### 📝 Summary

DeepSeekMoE is a **critical research step** in DeepSeek’s journey toward efficient, scalable AI:

- ✅ Pioneered **64 expert specialization** in MoE.
- ✅ Introduced **shared experts** for general knowledge.
- ✅ Demonstrated **40% inference cost savings** vs. dense models.
- ✅ Built foundation for future scaling in DeepSeek-V2.

⚡ This paper solidified DeepSeek’s reputation for **efficiency-focused**, **technically rigorous** innovation in large model design.

---

## 🚀 [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/pdf/2405.04434)

**Main Theme**: This paper in May 2024 builds upon the MoE foundation, highlighting DeepSeek-V2 as a model that is not only "strong" in performance but also "economical and efficient." This suggests optimizations in terms of computational resources (both training and inference costs) while maintaining or improving quality. This aligns with the broader industry need for more practical and deployable large models.

The paper "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model" builds upon DeepSeek's previous work, DeepSeekLLM (a dense model) and DeepSeekMoE (an exploratory Mixture-of-Experts model), to introduce a large-scale, efficient, and cost-effective MoE language model. The core aim of DeepSeek-V2 is to develop a model that is "strong, economical, and efficient".

Here are the key aspects and innovations of DeepSeek-V2:

**Large-Scale Mixture-of-Experts (MoE) Architecture**:
- DeepSeek-V2 boasts a massive total parameter count of 236 billion, yet it achieves remarkable efficiency by activating only 21 billion parameters during inference. This sparse activation significantly reduces the computational cost compared to dense models of similar total size.
- Building on the premise from DeepSeekMoE that a higher number of experts leads to better specialization, DeepSeek-V2 dramatically scales up the expert count. It utilizes 2 shared experts and 160 proprietary (specialized) experts. This is a substantial increase compared to the 64 proprietary experts in DeepSeekMoE and the typical 8 or 16 experts used in other contemporary MoE models like Mixtral 8x22B. The large number of proprietary experts is designed to achieve finer-grained specialization and clearer differentiation among them, allowing each to learn distinct knowledge. The shared experts handle universal, foundational language understanding, ensuring common knowledge is readily accessible for any input.

**Multi-head Latent Attention (MLA) Mechanism**:
- A significant and original contribution by DeepSeek in V2 is the introduction of Multi-head Latent Attention (MLA). This mechanism aims to further reduce inference costs, particularly by addressing the memory consumption of the KV Cache.
- The KV Cache stores the Key (K) and Value (V) matrices of previously processed tokens during sequence generation, preventing redundant computation and speeding up inference. MLA compresses these K and V matrices into lower-dimensional "latent" representations. This innovation leads to a drastic reduction of over 90% in KV Cache memory usage.
- The reduction in KV Cache memory directly translates to a significant increase in generation throughput. DeepSeek-V2 achieves a 5.76 times faster generation throughput compared to the earlier DeepSeekLLM 67B model. This means the model can generate responses much faster and at a lower operational cost. Unlike other methods like Multi-Query Attention (MQA) or Grouped-Query Attention (GQA) that reduce KV Cache by sharing or grouping K/V heads, which often compromise performance, MLA achieves similar compression while maintaining or even improving performance.

### 🧠 MLA details

DeepSeek-V2 introduces a novel attention mechanism called Multi-head Latent Attention (MLA), which is a key innovation designed to enhance inference efficiency and reduce memory footprint, particularly for the Key-Value (KV) cache.

**The Problem MLA Addresses: KV Cache Bottleneck**

Traditional Transformer models typically use Multi-Head Attention (MHA). During the generative inference process, these models need to cache all previously computed Key (K) and Value (V) matrices for each token to accelerate subsequent computations. This accumulated KV cache becomes a significant bottleneck, consuming a large amount of memory and limiting the maximum batch size and sequence length that can be processed efficiently.

While other methods like Multi-Query Attention (MQA) and Grouped-Query Attention (GQA) were proposed to reduce the KV cache, they often compromise model performance. Ablation studies show that MHA generally outperforms GQA and MQA on hard benchmarks. MLA was developed to achieve "the best of both worlds," providing significant KV cache reduction without sacrificing performance, and even achieving superior results compared to MHA.


### 🛠 How Multi-head Latent Attention (MLA) Works

The core idea of **MLA (Multi-head Latent Attention)** is to drastically reduce the amount of memory needed during inference by **compressing the Key and Value matrices** into much smaller vectors — without losing important information.

This compression is achieved through **low-rank projection** — think of it like storing a “summary” of the Key/Value data instead of every detail.


#### 🔄 1. Standard MHA vs. MLA Compression — An Analogy

Let’s imagine a standard **Multi-Head Attention (MHA)** setup like a full-resolution photograph. Each token produces a high-resolution image (key `k_t`, value `v_t`) that we store in a big album (KV cache) to reuse during generation. Over time, the album gets huge.

In MLA, we store only a **compact sketch** of each image — like a 32x32 thumbnail — instead of the full resolution. Then, when needed, we **reconstruct an approximate version** of the original using a special decoder.

#### 🔍 Detailed Breakdown:

- **In Standard MHA**:
  - Each input token `h_t` is projected into:
    - `q_t` (query), `k_t` (key), and `v_t` (value)
  - These are divided into `n_h` attention heads.
  - During inference, for every token you've seen before, you **must cache all of `k_t` and `v_t`** — full-size matrices.
  - 💾 **Memory-intensive**: the cache grows linearly with sequence length × number of layers × key/value size.

- **In MLA**:
  - Instead of computing and storing full `k_t` and `v_t`, you first **compress** `h_t` into a **compressed latent vector `c_KV_t`** using a down-projection matrix `W_DKV`.
    - 📦 Think of `c_KV_t` as a small “summary blob” that contains enough information to regenerate `k` and `v` later.
  - When the model needs `k` and `v`, it **reconstructs** them on-the-fly by up-projecting `c_KV_t` using matrices `W_UK` and `W_UV`.

✅ **Only `c_KV_t` is cached** instead of full `k_t` and `v_t`. This reduces memory usage **by over 90%**.

#### ⚙️ Optimization Trick: No Real "Reconstruction" Needed

There’s a clever trick here: instead of actually reconstructing `k_t` and `v_t` every time, the up-projection matrices (`W_UK`, `W_UV`) are **absorbed** into the downstream matrices (`W_Q`, `W_O`).

💡 So the model **pretends** it has full keys/values, but under the hood, it’s operating on the compressed forms — saving both memory and compute!

#### 💡 2. Query Compression (for Training Efficiency)

MLA also compresses the **queries**, but for a different reason.

- During training, memory for activations (especially from queries) can be a bottleneck.
- So `q_t` is also down-projected into `c_Q_t` (using `W_DQ`) and then up-projected to `q_C_t` (using `W_UQ`) for use in attention.
- Unlike KV compression, this is mainly a **training optimization** to reduce peak memory usage.

#### 🧭 3. Decoupled Rotary Position Embedding (RoPE)

Rotary Position Embedding (RoPE) encodes position into `k` and `q`. But it causes problems when used with compressed keys.

Here’s why:

- Normally, RoPE adds position info **after** computing the keys — this works in standard MHA.
- But in MLA, if you apply RoPE before compression, the positional info gets **entangled** with compression matrices (`W_DKV`, `W_UK`), making it impossible to do the neat optimizations.

💡 **Solution**: **Decoupled RoPE** — carry positional info separately.

#### How It Works:
- For each token, MLA generates extra "RoPE vectors":
  - A **RoPE query** `q_R_t,i` and a **shared RoPE key** `k_R_t`
- These are kept **separate** from the compressed attention heads (`q_C_t`, `k_C_t`)
- Right before attention is calculated, the RoPE vectors and the compressed vectors are **concatenated** to form complete attention heads.

📦 This allows RoPE to be applied **without interfering** with MLA's compression tricks.

#### 📉 How Much Smaller is the KV Cache?

Let’s quantify this:

- Assume `d_h` is the hidden size per head.
- DeepSeek-V2 sets:
  - Compression size `d_c = 4 × d_h`
  - RoPE head size `d_R_h = d_h / 2`
- So total cache size per token per layer is **`d_c + d_R_h = 4.5 × d_h`**
- In contrast, standard MHA would cache `2 × d_h` per head for both key and value.

**Result**: MLA reduces per-token cache to just **~4%** of MHA for large MoE models (like DeepSeek-V2).


![alt_text](/assets/images/deepseek/MLA.png "image_tooltip")


### 📈 Benefits of MLA in DeepSeek-V2

MLA's design delivers significant improvements in efficiency:

**Massive KV Cache Reduction**: DeepSeek-V2 reduces the KV cache by an impressive 93.3% compared to DeepSeek 67B. This means the KV cache memory footprint is cut to less than one-tenth of its original size. Table 1 notes that MLA's KV cache size is roughly equivalent to GQA with only 2.25 groups, while still delivering stronger performance than MHA. For large MoE models (like DeepSeek-V2), MLA reduces KV cache per token to just 4% of what MHA would require, and 14% for smaller MoE models.

**Boosted Generation Throughput**: The reduction in KV cache directly translates to much faster inference. DeepSeek-V2 achieves a maximum generation throughput that is 5.76 times faster than DeepSeek 67B. This enables DeepSeek-V2 to serve a much larger batch size.

**Stronger Performance**: Despite the drastic reduction in KV cache, MLA achieves "superior performance compared with MHA". Evaluation results confirm that models equipped with MLA perform better on hard benchmarks compared to those using MHA, across both small and large MoE scales.

**Economical Deployment**: DeepSeek-V2 further optimizes deployment by converting its parameters to FP8 precision and applying KV cache quantization (compressing each element to 6 bits on average). These optimizations, combined with MLA, contribute to DeepSeek-V2's ability to achieve a generation throughput exceeding 50K tokens per second on a single node with 8 H800 GPUs.

MLA's effectiveness is also demonstrated in DeepSeek-V2-Lite, a smaller model (15.7B total parameters, 2.4B activated) that also incorporates MLA and DeepSeekMoE, showing overwhelming performance advantages over previous smaller base and chat models, especially in reasoning, coding, and math.

### 🏆 Exceptional Performance and Cost-Effectiveness

- DeepSeek-V2 demonstrates remarkable efficiency in training. It achieves superior performance compared to DeepSeekLLM 67B while saving 42.5% of the training computation.
- In terms of performance, DeepSeek-V2 significantly surpasses the Llama 3 400B base model in Chinese tasks, as well as in reasoning tasks like code and mathematics. It also performs comparably to Llama 3 400B in English tasks.
- Despite its large total parameter count, DeepSeek-V2's low activated parameter count (21B) makes its deployment cost even lower than Mixtral 8x22B (which has an activated parameter count of approximately 39B). This cost-effectiveness led to DeepSeek-V2's highly competitive pricing, which reportedly triggered a "price war" in China's large language model service market.
-The engineering optimizations, including the successful implementation of FP8 mixed precision training (as seen in DeepSeek-V3, which built on V2's architecture), contribute to DeepSeek's ability to train large models stably and at exceptionally low costs. For instance, DeepSeek-V3 trained 671 billion parameters on 14.8 trillion tokens without any loss spikes or rollbacks, costing only $5.57 million.

### 🧪 DeepSeek's R&D Philosophy

- From DeepSeekMoE onwards, DeepSeek has consistently focused on efficiency and cost-effectiveness in its research.
- The company exhibits a "courageous" R&D culture, willing to explore and take risks with novel architectures and techniques (like the high expert count and MLA), even when they are unproven at scale or deviate from mainstream practices. This approach of not blindly following others, but rather aiming to "reduce costs and make the whole thing more efficient," has allowed DeepSeek to develop unique advantages and distinguish itself in the industry.
- DeepSeek's commitment to transparent and academically rigorous paper writing, including detailed technical descriptions and open-sourcing its models, also stands out in the industry.

**In summary**, DeepSeek-V2 represents a significant advancement by DeepSeek, showcasing how innovative architectural designs like an extremely high number of specialized MoE experts combined with novel attention mechanisms like MLA can lead to powerful models that are also highly economical and efficient to train and deploy.

---

## Pytorch Example of MOE and MLA

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_kv_compress):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_kv_compress = d_kv_compress

        # Projections
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_DKV = nn.Linear(d_model, d_kv_compress)  # Down-project h_t → c_KV_t
        self.W_UK = nn.Linear(d_kv_compress, d_model)   # Up-project → key
        self.W_UV = nn.Linear(d_kv_compress, d_model)   # Up-project → value
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, h_t, cache_kv_latents=None):
        """
        h_t: [B, T, d_model] input hidden states
        cache_kv_latents: Optional, cached c_KV_t from past tokens (for autoregressive decoding)
        Returns: output of shape [B, T, d_model]
        """
        B, T, _ = h_t.size()

        # Step 1: Compute compressed KV latent vector
        c_KV = self.W_DKV(h_t)  # [B, T, d_kv_compress]

        # If caching, you can concatenate with past latents here
        if cache_kv_latents is not None:
            c_KV = torch.cat([cache_kv_latents, c_KV], dim=1)  # [B, T_total, d_kv_compress]

        # Step 2: Recover keys and values
        keys = self.W_UK(c_KV).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)   # [B, nH, T_total, d_head]
        values = self.W_UV(c_KV).view(B, -1, self.n_heads, self.d_head).transpose(1, 2) # [B, nH, T_total, d_head]

        # Step 3: Compute queries
        queries = self.W_Q(h_t).view(B, T, self.n_heads, self.d_head).transpose(1, 2)   # [B, nH, T, d_head]

        # Step 4: Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.d_head ** 0.5     # [B, nH, T, T_total]
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, values)  # [B, nH, T, d_head]

        # Step 5: Merge heads
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)         # [B, T, d_model]
        return self.out_proj(context), c_KV  # Return output and new cache

h_t = torch.randn(2, 10, 768)  # batch=2, seq_len=10, hidden_size=768
mla = MultiHeadLatentAttention(d_model=768, n_heads=12, d_kv_compress=192)
out, c_KV = mla(h_t)
print(out.shape)  # torch.Size([2, 10, 768])
print(c_KV.shape) # torch.Size([2, 10, 192])

```

```python
class MoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=4, top_k=1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Experts: each has its own FFN
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])

        # Gating network
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x):
        """
        x: [B, T, d_model]
        Returns: [B, T, d_model]
        """
        B, T, D = x.shape
        x_flat = x.view(B * T, D)  # Flatten to [B*T, D]

        gate_logits = self.gate(x_flat)               # [B*T, num_experts]
        topk_weights, topk_indices = torch.topk(gate_logits, self.top_k, dim=-1)  # [B*T, top_k]

        topk_probs = F.softmax(topk_weights, dim=-1)  # [B*T, top_k]
        expert_outputs = torch.zeros_like(x_flat)

        for k in range(self.top_k):
            expert_idx = topk_indices[:, k]  # [B*T]
            mask = torch.zeros_like(expert_outputs)
            for i in range(self.num_experts):
                selected = (expert_idx == i)
                if selected.any():
                    input_i = x_flat[selected]
                    output_i = self.experts[i](input_i)
                    expert_outputs[selected] += output_i * topk_probs[selected, k].unsqueeze(-1)

        return expert_outputs.view(B, T, D)

moe = MoE(d_model=768, d_ff=2048, num_experts=4, top_k=1)
x = torch.randn(2, 10, 768)
y = moe(x)
print(y.shape)  # torch.Size([2, 10, 768])

```

**Combining MLA + MoE in a Transformer Layer (Sketch)**

```python
class TransformerMLAMoELayer(nn.Module):
    def __init__(self, d_model, n_heads, d_kv_compress, d_ff, num_experts):
        super().__init__()
        self.attn = MultiHeadLatentAttention(d_model, n_heads, d_kv_compress)
        self.moe = MoE(d_model, d_ff, num_experts)

    def forward(self, x, cache=None):
        attn_out, new_cache = self.attn(x, cache)
        moe_out = self.moe(attn_out)
        return moe_out, new_cache

```

---

## [DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437)

**Main Theme**: As a technical report in Feburary 2025, DeepSeek-V3 likely presents further advancements and refinements to their base model architecture. While specific details are not provided, it signifies continuous progress in their core LLM capabilities, possibly incorporating new techniques for scaling, training, or performance enhancement.

The DeepSeek-V3 Technical Report details DeepSeek's advanced large-scale Mixture-of-Experts (MoE) language model, focusing on being "strong, economical, and efficient". DeepSeek-V3 serves as the base model for DeepSeek-R1.

Here's a summary of its key aspects and innovations:

- **Scale and Efficiency**:
-- **Massive Parameter Count**: DeepSeek-V3 has a total of 671 billion parameters, making it substantially larger than its predecessor, DeepSeek-V2 (236 billion parameters).  
-- **Sparse Activation**: Despite its colossal total size, DeepSeek-V3 maintains efficiency by activating only approximately 30 billion parameters during inference.  
-- **Extensive Training Data**: The model was trained on 14.8 trillion tokens, a notable increase from DeepSeek-V2's 8 trillion tokens.  
-- **Cost-Effective Training**: DeepSeek-V3 was trained on 2,000 H800 GPUs at an estimated cost of just $5.57 million USD, compared to models like Llama 3 400B which reportedly cost over $30 million.  
-- **Stable Training Process**: The training was highly stable, with no loss spikes or rollbacks.

- **Architectural and Technical Innovations**:
-- **Mixture-of-Experts (MoE) Architecture**: DeepSeek-V3 scales up the expert count to 256 proprietary experts + 1 shared expert.  
-- **Multi-head Latent Attention (MLA)**: Fully integrated, reducing KV cache memory by over 90% and boosting generation throughput.  


### Auxiliary-Loss-Free Load Balancing

In MoE models, a key challenge is **balanced expert utilization**. Imbalances can cause routing collapse and inefficiency.

**Traditional Auxiliary Loss Issue**:
- Typically, an auxiliary loss is used to penalize unbalanced usage.
- However, this can harm performance by distorting routing for balance’s sake.

**DeepSeek-V3’s Strategy**:
1. **Bias Term ($b_i$)**: Each expert gets a trainable bias.
2. **Routing with Bias**: The routing score becomes $s_{i,t} + b_i$ (used for selecting top-K experts).
3. **Gating without Bias**: Final output uses the original $s_{i,t}$ for calculating $g_{i,t}$.
4. **Dynamic Bias Adjustment**:  
   - After each batch, increase $b_i$ if underused, decrease if overused.
   - Controlled by hyperparameter $\gamma$.

**Complementary Loss**:
- A minor **sequence-wise auxiliary loss** with $\alpha = 0.0001$ is added.
- This prevents extreme imbalance in any single sequence.

**Batch-wise vs. Sequence-wise**:
- Balancing is done across batches, not per-sequence, allowing more specialization.
- Ablation studies confirm performance improvements.

**No Token Dropping**:
- Unlike other MoE systems, no tokens are dropped during training or inference.

### Multi-Token Prediction (MTP)

- Trains the model to predict **multiple tokens simultaneously** (e.g., 3 tokens).
- Encourages planning and enables techniques like **speculative decoding**.
- Considered a **bold innovation**, as it had not been used at scale before.


### Infrastructure: Training Framework and FP8 Mixed Precision

**Framework**:  
- Uses custom **HAI-LLM** with a combination of:
  - **Pipeline Parallelism (PP)**: 16-way  
  - **Expert Parallelism (EP)**: 64-way across 8 nodes  
  - **Data Parallelism (DP)**: ZeRO-1  


#### 1. DualPipe for Pipeline Parallelism

**Problem**:  
- Expert parallelism across nodes causes high communication overhead.

**Solution - DualPipe**:  
- Splits each block into: attention, all-to-all dispatch, MLP, all-to-all combine.  
- For backward: splits attention and MLP into "backward for input" & "backward for weights".  
- **Bidirectional scheduling**: Feeds micro-batches from both ends of pipeline.  
- **Overlaps compute and communication** for near-zero communication cost.  
- Requires 2x parameter memory but manages it efficiently.


#### 2. Efficient All-to-All Communication Kernels

- Uses **NVLink (160 GB/s)** intra-node and **InfiniBand (50 GB/s)** inter-node.  
- Limits token dispatch to 4 nodes to reduce IB load.  
- Tokens routed to same-index GPU across nodes (via IB) then to final GPU (via NVLink).  
- Achieves near-optimal performance with **only 20 SMs**.  
- Scales up to 13 activated experts per token in theory.


#### 3. Extreme Memory Efficiency

- **RMSNorm & MLA up-projections** are recomputed instead of cached.  
- **EMA** stored and updated on CPU.  
- **Shared embedding/output head** reduces duplication.  


#### 4. FP8 Mixed Precision Training

**Key Innovations**:
- **Most GEMMs** (Fprop, Dgrad, Wgrad) in FP8 for speed/memory efficiency.  
- **Sensitive ops** (norms, embeddings, MoE gating) stay in BF16/FP32.  
- **Weights/optimizer states** stored in higher precision.  
- **Quantization**:  
  - Activations: tile-wise (1x128)  
  - Weights: block-wise (128x128)  
- **Precision Improvements**:  
  - FP8 accumulation on **CUDA Cores** every 128 elements.  
  - Consistently uses **E4M3** for FP8 tensors.  
- **Online Quantization**:  
  - Max value per tile/block calculated on-the-fly.  
  - Simplifies framework; avoids delayed quantization.  

**Storage and Communication**:
- Activations cached in FP8.  
- Inputs to Linear (post-attn) stored in E5M6.  
- MoE inputs/output gradients stored/communicated in FP8 with power-of-2 scaling.  
- MoE forward/backward combine remains in BF16.


### Efficient Reinforcement Learning (GRPO)

- Uses **GRPO**, a simpler alternative to PPO (no separate value model).
- For code/math tasks, uses **rule-based rewards** (e.g., correctness check).
- For open-ended tasks, still uses **reward models**.


### Performance

- Matches or outperforms **Llama 3 400B** in English.
- **Beats Llama 3 400B** in Chinese, code, and math.
- Lower activation count = **lower deployment cost** than Mixtral 8x22B.


### DeepSeek’s R&D Philosophy

- Maintains a strong focus on **cost-efficiency** and **scalable innovation**.
- Willing to **experiment** with bold ideas (e.g., MLA, MTP, FP8).
- Commits to **transparency and open-sourcing** (MIT license for V3).

