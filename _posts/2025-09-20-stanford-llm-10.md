---
layout: post
title: Scaling Laws Details with Examples 💻
subtitle: Language Modeling from Scratch Lecture 10
categories: Large-Language-Model
tags: [Stanford-LLM-From-Scratch-2025]
banner: "/assets/images/banners/yuanpang-wa-iceburg2.jpg"
---

# ⚖️ Scaling — Case Study and Details  

This lecture, **“Scaling – Case Study and Details,”** dives into **best practices for scaling and hyperparameter tuning** in large language models (LLMs). It revisits whether the **Chinchilla-derived scaling methodologies** still hold in modern model development and explores **recent case studies** (CerebrasGPT, MiniCPM, DeepSeek) alongside the math behind **stable training across scales**.

[Course link](https://stanford-cs336.github.io/spring2025/)


## 🎯 I. Motivation and Overview  

The lecture addresses critical questions for modern LM builders:
- ✅ Does the **Chinchilla scaling approach** still hold?  
- 💰 Can we **save compute** during scaling analysis?  
- 🧠 Which **architectures or parameterizations** scale predictably?  

After the post-ChatGPT wave, **detailed scaling data** became secretive. Thus, this lecture draws insights from **publicly transparent scaling studies** — notably **CerebrasGPT**, **MiniCPM**, and **DeepSeek LLM** — now considered the *gold standard* for scaling law methodology.


## 🧪 II. Scaling in Practice — Model Case Studies  

### ⚙️ Cerebras-GPT  
**Scaling Range:** 0.1B → 13B parameters, following the **Chinchilla recipe**.  

![alt_text](/assets/images/llm-from-scratch/10/1.png "image_tooltip")

- **🔑 Core Finding:** Stability via **Maximal Update Parametrization (muP)**.  
  - Standard parameterization (SP) → “big oscillations” around the predicted scaling line.  
  - muP → smooth, predictable scaling curves.  

- **🧭 Hyperparameter Strategy:**  
  - SP required LR retuning as model size grew.  
  - muP produced **stable learning curves** across scales.  

- **🧰 Implementation:**  
  - Conducted hyperparameter searches on **tiny models (40M)**.  
  - Used muP to reliably **scale those hyperparameters up** to 13B.  
  - muP sets **per-layer LRs and initialization variances** differently from SP.

![alt_text](/assets/images/llm-from-scratch/10/2.png "image_tooltip")
![alt_text](/assets/images/llm-from-scratch/10/3.png "image_tooltip")
![alt_text](/assets/images/llm-from-scratch/10/4.png "image_tooltip")

---

### ⚙️ MiniCPM (2024)  
**Scale:** 1.2B → 2.4B parameters, performing on par with many 7B models.  

- **🧩 muP for Stability:** Like Cerebras, MiniCPM used muP to simplify and stabilize scaling, saving ~5× compute from smallest to largest runs.  
- **📦 Optimal Batch Size:**  
  - Confirmed **log-log linear trend** between terminal loss and batch size.  
  - As loss decreases, batch size must **polynomially increase**.  
- **⚡ Optimal Learning Rate:**  
  - With muP, **minimum optimal LR remains constant** across scales → no complex LR tuning.  
- **📉 WSD Learning Rate Schedule:**  
  - Developed **Warm-up Stable Decay (WSD)** to make Chinchilla-style data analysis cheaper.  
  - Unlike cosine LR (depends on run length), WSD’s *flat “stable phase”* allows **reuse of runs** for multiple data checkpoints.  
- **📈 Chinchilla Analysis:**  
  - Used Method 1 (lower envelope) + Method 3 (joint fit).  
  - Found **192 tokens per parameter**, much higher than Chinchilla’s 20:1 → aligns with **LLaMA 3’s high ratio**.

![alt_text](/assets/images/llm-from-scratch/10/6.png "image_tooltip")

### ⚡ Warm-up Stable Decay (WSD) Learning Rate Schedule

![alt_text](/assets/images/llm-from-scratch/10/5.png "image_tooltip")

The **Warm-up Stable Decay (WSD)** learning rate schedule introduces a more flexible and cost-efficient way to manage training dynamics compared to traditional cosine learning rate schedules. It plays a key role in modern scaling law analysis, especially in models like **MiniCPM** and **DeepSeek**.

### 🧩 Structure of the WSD Learning Rate Schedule

The **WSD** schedule is a **piecewise linear (trapezoid-shaped)** learning rate curve, divided into **three distinct phases**:

1. **🔥 Warm-up Phase**  
   Rapidly increases the learning rate from zero to its maximum value — similar to a cosine warm-up.  
   *Goal:* Stabilize gradients and prevent early training instability.

2. **🟩 Stable Phase**  
   Keeps the learning rate constant for the majority of training.  
   *Goal:* Enable consistent learning and predictable loss decay.

3. **🧊 Decay Phase**  
   Rapidly cools down the learning rate to its minimum (or zero).  
   *Goal:* Refine the model and reach the terminal loss efficiently.  

> 🧠 **DeepSeek’s variant** used a fast warm-up followed by **two decay steps of 10% each** — striking a balance between speed and stability.


#### 🎯 Motivation — Fixing the Cosine Schedule Problem

The **main motivation** for WSD comes from limitations in **cosine learning rate schedules** when performing **Chinchilla-style scaling law analysis**.

- **The Cosine Problem:**  
  The cosine schedule’s shape depends on the **target termination point** (i.e., number of tokens).  
  - Small runs → faster cool-down.  
  - Large runs → slower cool-down.  

- **The Consequence:**  
  Because these decay curves differ, **early checkpoints** from a long run **cannot represent** the scaling behavior of shorter runs.  
  To get accurate scaling fits, researchers would have to train **N² separate runs** (for N target lengths) — extremely expensive!


#### 💡 Advantage — Making Chinchilla Analysis Cheaper

WSD offers a **computationally efficient workaround** to this problem:

1. **Train once** through the full **stable phase** (flat middle region).  
2. **Reuse checkpoints** from earlier in the stable phase to simulate shorter training runs.  
3. **Apply new decay phases** from those checkpoints to reach different target endpoints.

This means researchers can perform **Chinchilla-style scaling analysis** using **almost one run**, rather than N² runs — dramatically reducing compute requirements while keeping scaling curves consistent.


#### 📊 Empirical Performance

**In practice:**

- 🧠 **MiniCPM** popularized the WSD schedule.  
  - While its loss curves look *less smooth* than cosine,  
    it often **matches or beats** the cosine minimum at every token count.

- 🚀 **DeepSeek** reported similar success — their WSD-style schedule  
  performs **on par with cosine**, maintaining stability across scales.

**Trade-off:**  
The **decay timing** is crucial.  
- The **stable phase** allows the model to explore far from initialization.  
- The **decay phase** is essential to “anneal” the loss to a lower final value.

> 🧩 In short: WSD = Warm-up → Stable learning → Controlled Decay = ⚙️ Efficient scaling, 💰 cheaper compute, and 📈 competitive results.

![alt_text](/assets/images/llm-from-scratch/10/7.png "image_tooltip")


---

### ⚙️ DeepSeek LLM (V1, 2024)  

The **DeepSeek LLM (V1, 2024)** stands out as a strong example of a **carefully engineered, scientifically grounded** large-scale model.  
It features models with **7B** and **67B** parameters — both delivering performance comparable to **LLaMA 2** of similar sizes and rivaling **Mistral** models.  
What makes DeepSeek notable is its **“serious science”** approach to scaling: methodical, data-driven, and computation-efficient.


#### ⚙️ Scaling Strategy — Direct Estimation (No muP)

![alt_text](/assets/images/llm-from-scratch/10/8.png "image_tooltip")

A defining characteristic of **DeepSeek V1** is its **decision *not* to use Maximal Update Parametrization (muP)**, unlike models such as CerebrasGPT and MiniCPM.

Instead, DeepSeek employed a **direct estimation approach** to achieve stable hyperparameters:

1. **🧩 Assumption** — Most Transformer hyperparameters are *invariant to scale*.  
2. **📈 Scaling Analysis** — They identified only two **non-invariant hyperparameters**:  
   - Optimal **batch size**  
   - Optimal **global learning rate**  
3. **📊 Extrapolation Process** —  
   - Conducted small-scale experiments and collected models within **0.25% of minimum loss**.  
   - Fitted **scaling laws** for optimal batch size and learning rate as functions of **compute**.  
   - Extrapolated these fits to estimate hyperparameters for full-scale (7B and 67B) models.  

> ⚠️ The resulting **global learning rate scaling fit** was described as “*somewhat suspicious looking*,” implying that the relationship may not follow a perfect power law — but it remained empirically effective.


#### 🔬 Chinchilla Analysis and Optimal Sizing

DeepSeek performed a **Chinchilla-style IsoFLOPS analysis (Method 2)** — a key benchmark for determining optimal trade-offs between **model size** and **data size** under fixed compute budgets.

![alt_text](/assets/images/llm-from-scratch/10/9.png "image_tooltip")

- **📘 IsoFLOPS Method:**  
  Sweep parameter counts at fixed FLOP budgets and identify the **minimum-loss curve**.  
- **🔮 Predictive Power:**  
  The resulting scaling fits **accurately predicted** the final loss outcomes of both the **7B** and **67B** models.  
  - They successfully extrapolated from **10²⁰ FLOPs** small-scale runs to **10²⁴ FLOPs** large-scale training.  
  - This validated their **scaling law methodology** as both practical and predictive.


#### 📉 Learning Rate Schedule — WSD-Style

To complement their scaling strategy, DeepSeek adopted a **Warm-up Stable Decay (WSD)-style** learning rate schedule, inspired by MiniCPM.

- **🧱 Structure:**  
  - Rapid **warm-up** phase  
  - Long **stable** plateau  
  - Two **decay steps** of **10% each**  

- **⚡ Performance:**  
  - Matched or slightly exceeded **cosine learning rate** performance.  
  - Enabled **Chinchilla-style scaling analysis** at a **fraction of the compute cost**.  

- **💰 Efficiency Gain:**  
  By reusing the stable phase for different cool-down checkpoints, DeepSeek drastically reduced the number of full training runs needed for scaling curve estimation.


#### 🧾 Summary — The DeepSeek Recipe

| Component | Approach | Key Insight |
|------------|-----------|--------------|
| **Parametrization** | No muP | Relied on empirical direct fitting of batch size & LR |
| **Scaling Analysis** | Direct extrapolation from small runs | Used near-optimal small models to fit scaling laws |
| **Compute Allocation** | Chinchilla IsoFLOPS (Method 2) | Achieved accurate prediction across 4 orders of magnitude in FLOPs |
| **Learning Rate Schedule** | WSD-style (fast warm-up, stable plateau, 2-step decay) | Matched cosine performance, cheaper scaling analysis |

> 🧠 **In essence:**  
> DeepSeek V1 achieved strong results through **empirical scaling discipline**, not parameterization tricks — proving that a rigorous, data-driven approach can rival even the most sophisticated optimization methods.

---

### 🌍 Other Recent Scaling Insights  

| 🧠 Model | ⚙️ Key Method | 📊 Scaling Result | 📚 Year |
|-----------|---------------|------------------|---------|
| **LLaMA 3** | IsoFLOPS-style | Optimal 39:1 tokens-to-parameter ratio; fitted sigmoid between NLL & benchmark accuracy | 2024 |
| **Hunyuan-1** | IsoFLOPS for MoE | Extended scaling to expert layers | 2024 |
| **Minimax-01** | Chinchilla Method 1 | Compared Lightning Attention (linear time) vs Softmax Attention; similar scaling performance | 2025 |

![alt_text](/assets/images/llm-from-scratch/10/10.png "image_tooltip")
![alt_text](/assets/images/llm-from-scratch/10/11.png "image_tooltip")
![alt_text](/assets/images/llm-from-scratch/10/12.png "image_tooltip")
![alt_text](/assets/images/llm-from-scratch/10/13.png "image_tooltip")

---

## 🧮 III. Maximal Update Parametrization (muP) — In-Depth  

muP ensures **scale-invariant hyperparameters**, allowing stable transfer of learning rates and initialization across model sizes.

### 🧠 Conceptual Basis — *Spectral Conditions*  
muP applies two constraints when scaling network width $( n_l )$:

1. **A1: Activation Stability**  
   - Activations per coordinate remain Θ(1).  
   - Initialization variance $( σ )$ must satisfy:  
     $[
     σ^2 = Θ\left(\frac{1}{n_{l-1}} \min(1, \frac{n_l}{n_{l-1}})\right)
     ]$
   - Prevents activation explosion/vanishing.

2. **A2: Update Stability**  
   - Change in activation per gradient step must remain Θ(1).  
   - Assuming ΔL = O(1):  
     - For **SGD:** $( η_l = Θ(n_l / n_{l-1}) )$  
     - For **Adam:** $( η_l = Θ(1 / \text{fan-in}) )$

---

### ⚖️ muP vs Standard Parameterization (SP)  

| 🔍 Aspect | 🧩 SP | 🚀 muP |
|------------|-------|--------|
| **Initialization** | $( 1/n_{l-1} )$ | Same base form |
| **Learning Rate** | Global constant (Θ(1)) | Per-layer, $( 1/\text{fan-in} )$ |
| **Stability** | Scale-sensitive | Scale-invariant |
| **Practical Benefit** | Requires LR tuning | Transfers LR across scales |

---

### ✅ Empirical Validation  

- **Transferability:** Optimal LR scales reliably (width 128 → 2048).  
- **Robustness:** Works with  
  - SwiGLU / Squared ReLU  
  - Varying batch sizes  
  - Zero-attention inits  

- **⚠️ Known Failures:**  
  1. Learnable gains in **RMSNorm**  
  2. Exotic optimizers (e.g., **Lion**)  
  3. Strong **weight decay (≥0.1)**  

Despite these caveats, muP remains **a powerful and practical tool** for stable scaling.

## 🧭 IV. Recap — Scaling in the Wild  

Modern scaling efforts face three core challenges:

1. **🏗️ Architectural Hyperparameters:** Choosing width, depth, and shape.  
2. **⚙️ Optimizer Hyperparameters:** Learning rate, batch size, and stability.  
3. **💰 Compute Cost:** Chinchilla-style sweeps are expensive.

### 🧩 Strategies by Frontier Labs  

| 🧪 Goal | 🔧 Strategy | 🏁 Example |
|----------|--------------|------------|
| **Hyperparameter Search** | Assume muP stability or fit scaling laws from small runs | DeepSeek (fit laws), MiniCPM & CerebrasGPT (muP) |
| **Reducing Sweep Cost** | Use WSD-like schedules to reuse runs | MiniCPM, DeepSeek |
| **Model Sizing** | Replicate IsoFLOPS (Method 2) to find optimal token-to-parameter ratio | All major LMs |


### 💡 Key Takeaway  
Scaling laws remain **the foundation of efficient model development** — but modern practice refines them with **muP**, **WSD**, and **IsoFLOPS** to handle today’s trillion-parameter regimes.  

> “Scaling smartly is no longer about bigger models — it’s about **predictable, stable, and efficient** growth.” 🚀


## IV 🧮 How to Actually Run a Scaling Law Experiment — Step-by-Step

Scaling laws let researchers predict large model performance and hyperparameters from **small, cheap experiments**.  
This is the blueprint followed by DeepMind, OpenAI, Anthropic, and MiniCPM teams.

---

### 🔧 Step 1: Tuning the Learning Rate (LR) and Batch Size (Fixed Dataset)

The first phase focuses on **optimizer stabilization** — finding the optimal **learning rate (LR)** and **critical batch size (B₍crit₎)** for small models trained on a *fixed dataset*.

#### 🧩 Why Fix the Dataset?

At this stage, we’re isolating **optimization dynamics** — not data–model tradeoffs.  
We want to know how LR and batch scale with model size under identical data and compute settings.

| Setting | Typical Choice | Explanation |
|----------|----------------|-------------|
| Dataset | Fixed 10B–20B tokens | Keeps experiments consistent |
| Model sizes | 125M → 1B parameters | Cheap but informative |
| Compute | Same order of magnitude | Fair comparison across scales |

#### ⚗️ Step-by-Step

1. **Train small models** (e.g., 125M, 350M, 1B params) on the same dataset.  
2. **Sweep peak learning rates** (e.g., 1e-5 → 3e-3) with a standard schedule (cosine, linear, or short WSD).  
3. **Measure validation loss** and fit a parabola to find the LR minimizing loss.  
4. **Identify critical batch size (B₍crit₎)** — the largest batch before performance plateaus.

| Model Size | Optimal LR | Critical Batch Size |
|-------------|-------------|---------------------|
| 125M | 3e-4 | 2K |
| 350M | 2.5e-4 | 4K |
| 1B | 2e-4 | 8K |

#### 🧮 Fit Scaling Laws

You can now fit smooth relations between model size \(N\) and optimal hyperparameters:

$[
\eta^*(N) \propto N^{-\alpha}, \quad B_{crit}(N) \propto N^{\beta}
]$

These equations tell you how LR and batch scale with model width or depth —  
forming the *foundation* for all later scaling analysis.

> 💡 Some groups (e.g., MiniCPM) use a short **Warmup-Stable-Decay (WSD)** schedule here  
> to separate warmup, flat, and decay phases cleanly — but this is optional in Phase 1.

---

### ⚙️ Step 2: Handle Learning Rate Scaling — Two Paths

Once you know how LR behaves with scale, you can **choose one of two strategies**:

| Path | Method | Description | Example |
|------|---------|--------------|----------|
| **A. muP (Maximal Update Parametrization)** | Scale-invariant LR | Modify initialization and per-layer LR to make a single LR work across all model widths. | MiniCPM, CerebrasGPT |
| **B. Empirical Fitting (No muP)** | Fit LR scaling law | Directly fit a power-law relation between model size and optimal LR, e.g. \(LR \propto m^{-0.25}\). | DeepSeek |

- **If using μP:** You only tune LR once on the smallest model — it transfers automatically.  
- **If not using μP:** Fit the LR vs. size curve empirically and extrapolate to large models.

---

### 📈 Step 3: Fit Critical Batch Size Scaling

Batch size scaling is usually **log-linear** with loss or compute:

$[
\log(B_{crit}) = a + b \cdot \log(L_{target})
]$
or
$[
B_{crit} \propto \text{Compute}^{\beta}
]$

Interpretation:
- Better models (lower loss) → larger $(B_{crit})$
- Fit a straight line in log–log space → extrapolate for large models.

> **Output:** Predicted $(B_{crit})$ for your final large-scale run.

---

### 📊 Step 4: Run the Chinchilla IsoFLOP Analysis (WSD Introduced Here ✅)

Now comes the **core scaling law experiment** — the **Chinchilla analysis**.

This phase determines the **optimal ratio between model parameters (M)** and **training tokens (N)**  
for a fixed total compute budget.

#### 🧮 Compute Budget

$[
\text{Total FLOPs} \approx 6 \times M \times N
]$

#### 🧠 Why WSD (Warmup-Stable-Decay)?

Training from scratch for every target data length is too expensive.  
WSD enables checkpoint reuse:

1. Train once through the **stable phase** (flat LR in step1).  
2. **Rewind** checkpoints to simulate shorter runs.  
3. Apply a new **decay** phase for each simulated endpoint.

This lets you explore multiple token budgets from a *single run* —  
cutting Chinchilla sweep compute by 3–5×.

#### Procedure

1. **Choose FLOP budgets:** e.g., $(10^{20})$–$(10^{24})$ FLOPs.  
2. **Sweep tradeoffs:** For each budget, vary $(M)$ and $(N)$ to keep FLOPs constant.  
3. **Plot validation loss vs. model size** → the curve is convex.  
4. **Find minima:** The loss minimum gives the optimal $(N:M)$ ratio.

> **Result:** Optimal tokens-to-parameter ratio, e.g.  
> - Chinchilla: 20:1  
> - LLaMA 3: 39:1  
> - MiniCPM: 192:1  

---

### 🚀 Step 5: Scale Up and Train the Final Model

After Phases 1–4, you now have:

- ✅ Stable learning rate scaling (μP or empirical fit)  
- ✅ Predicted critical batch size $(B_{crit})$  
- ✅ Optimal data–model ratio (from IsoFLOP/Chinchilla analysis)

#### Apply these to your final large model:

| Parameter | Source | Example |
|------------|---------|----------|
| LR | From μP or power-law fit | $(2.0 \times 10^{-4})$ |
| Batch size | From $(B_{crit})$ scaling | 8K |
| Tokens | From optimal N:M ratio | 70B model → 2.7T tokens |

---

### 📊 Example Summary Table

| 🧠 Phase | Goal | Outputs | Example |
|-----------|-------|----------|----------|
| **Phase 1** | Stabilize optimizer, tune LR & batch | μP or empirical LR scaling, $(B_{crit})$ | LR ≈ 3e-4, $(B_{crit})$ ≈ 2K |
| **Phase 2** | Joint data–model scaling (IsoFLOP) | Optimal N:M ratio | 39 tokens/param |
| **Phase 3** | Final large run | Tokens, compute, and hyperparams | 70B model → 2.7T tokens |

---

### 📉 Typical Plots and Results

- **Log–log plot:** Loss vs. compute (linear trend until saturation).  
- **Chinchilla curve:** Convex loss vs. model size under fixed FLOPs; minimum gives optimal tradeoff.  
- **Batch scaling curve:** Log–linear relation defining \(B_{crit}\).  

Scaling laws demonstrate that the relationship between **resources (data, params, compute)**  
and **performance (loss)** is *remarkably linear on a log–log plot*.  

> Think of it like using a telescope:  
> studying small, cheap models lets you predict the behavior of massive frontier systems —  
> with near-astronomical accuracy.
