---
layout: post
title: Scaling laws 💻
subtitle: Language Modeling from Scratch Lecture 9
categories: Large-Language-Model
tags: [Stanford-LLM-From-Scratch-2025]
banner: "/assets/images/banners/yuanpang-wa-iceburg2.jpg"
---

# ⚙️ The Predictable World of Scaling Laws in Language Models

Scaling laws provide **simple, predictive rules** 📈 that govern the performance of Language Models (LMs), offering a pathway to **optimize large-scale model design** without relying on expensive, full-scale experimentation.  
They enable developers to **tune hyperparameters on small models** and confidently extrapolate to production-scale systems 🚀.

[Course link](https://stanford-cs336.github.io/spring2025/)


## 🧠 Part 1: Data Scaling – The Log-Log Linear Relationship

The most fundamental scaling observation is the relationship between **dataset size ($n$)** and **error**.  
Empirically, plotting **test loss vs. dataset size** on a log-log scale yields a **linear relationship**, indicative of **power law (scale-free) scaling** ⚖️.

![alt_text](/assets/images/llm-from-scratch/09/1.png "image_tooltip")


### 🔬 Theoretical Basis

The polynomial decay of estimation error underlies this behavior:
- For simple models (e.g., mean estimation), error decays as $1/n$ → slope of **-1**.
- For neural networks (flexible, nonparametric models), the decay is **much slower** ⏳.

![alt_text](/assets/images/llm-from-scratch/09/2.png "image_tooltip")

#### 🧩 Intrinsic Dimensionality
Observed shallow slopes (e.g., LMs: $\alpha \approx 0.095$) are linked to the **intrinsic dimensionality ($d$)** of the data manifold.  
In nonparametric settings, error scales as $n^{-1/d}$ → slope = **−1/d**.  
This means the scaling exponent reveals the **inherent difficulty of the learning task** 🎯.

### 🔍 Key Data Scaling Applications

* 📚 **Data Composition:**  
  Affects the **offset**, not the **slope**, of the scaling curve.  
  → Enables optimal data mixtures using small-scale models.

* 🔁 **Data Repetition:**  
  When data is reused, diminishing returns occur.  
  Scaling laws track this via **effective data (unique tokens)**, helping balance between **high-quality repeats** and **new, lower-quality data**.

## 🧮 Understanding Log-Linear Scaling Between Data and Error

One of the most fundamental — yet often misunderstood — ideas in scaling laws is the **log-linear relationship** between **data size** and **model error (or loss)**.

---

### 📊 What Does "Log-Linear" Mean?

When you train a model and record:
- **n** → amount of training data (tokens or samples)
- **E(n)** → resulting error or loss

you’ll find that if you plot **log(error)** vs. **log(data size)**, the points form a **straight line** 📈.  
That’s what “log-linear” means.

Mathematically, this corresponds to a **power law relationship**:

$[
E(n) = A \cdot n^{-\alpha} + C
]$

where  
- $( A )$ is a scaling constant,  
- $( \alpha )$ (alpha) is the **scaling exponent**,  
- $( C )$ is the irreducible error floor.

Taking logs gives:

$[
\log E = \log A - \alpha \log n
]$

which is a **linear equation in log-space**:
> Slope = −α → constant rate of improvement per data doubling.

### 💡 Intuitive Example

| Data Size (n) | Error (E) | log(n) | log(E) |
|---------------|-----------|--------|--------|
| 1M | 1.0 | 6 | 0.00 |
| 10M | 0.7 | 7 | -0.15 |
| 100M | 0.5 | 8 | -0.30 |
| 1B | 0.35 | 9 | -0.46 |

Plotting these on a **log-log scale** yields a straight line with slope ≈ −0.15 → the **log-linear** relationship.

### 🧠 Why It Matters

This relationship means **model improvement with more data is predictable**:
- Each data doubling reduces error by a constant multiplicative factor.
- You can **train small models** to estimate this slope and **extrapolate** performance for larger data sizes — saving huge compute costs.

In large language models (Kaplan et al., 2020; Chinchilla, 2022),  
the scaling exponent \( \alpha \) ≈ **0.095**, indicating a **shallow but consistent improvement**.

## ⚖️ Data Scaling vs. Joint Data–Model Scaling

Now comes the subtle — but crucial — point:  
Does log-linear scaling hold if we *only* increase data? Or do we need to grow the model too? 🤔


### 🧩 Pure Data Scaling (Fixed Model)

If the model size **stays fixed** and you **increase data**:

$[
E(n) = A \cdot n^{-\alpha} + C
]$

✅ You’ll initially see log-linear improvement.  
But after a point, the model **saturates** — it can’t absorb more information.

Think of the model as a **bucket** 🪣 and data as **water** 💧:
- A small bucket can only hold so much.
- Beyond that, pouring more data doesn’t help — it just spills over.
- Similarly, after the saturation point, the curve **flattens** and scaling **breaks**.

### 🧠 The Joint Data–Model Scaling Law

To **stay in the linear regime**, both data size and model size must **scale together**.

The general form is:

$[
E(n, m) = n^{-\alpha} + m^{-\beta} + C
]$

where $( m )$ = model size (parameters).

Empirically, optimal performance follows a near-linear relationship:

$[
n_{\text{optimal}} \propto m
]$

This means:
> 📏 **The amount of data should increase proportionally to model size**  
> to maintain predictable power-law improvement.

### 🧮 The Chinchilla Rule of Thumb

From Hoffmann et al. (2022, *Chinchilla*):
- For compute-optimal training, the best ratio is roughly  
  **20 tokens per parameter**.

This is the **“Chinchilla ratio”**, balancing data and model size so the model learns efficiently.

Modern LLMs (like **Llama 3**) deliberately use **more data (≈215 tokens/param)** —  
this “**overtraining**” trades higher training cost for **lower inference cost** later on 💰.

### 🧭 Practical Summary

| Scenario | Model | Data | Effect |
|-----------|--------|------|--------|
| 🟢 **Increase data, fixed model** | Constant | ↑ | Initial log-linear gain → then saturation |
| 🟠 **Increase model, fixed data** | ↑ | Constant | Improves until overfitting or wasted capacity |
| 🔵 **Increase both proportionally** | ↑ | ↑ | Sustains clean power-law scaling → optimal efficiency |

### 🧩 TL;DR

| Concept | Meaning |
|----------|----------|
| **Log-linear (data scaling)** | On a log-log plot, error decreases linearly with data size (power law) |
| **Saturation** | When the model is too small to use more data effectively |
| **Joint scaling law** | Data and model size must grow together for efficient scaling |
| **Optimal ratio** | ≈ 20 tokens per parameter (Chinchilla) |
| **Modern trend** | Overtraining (e.g., Llama 3 uses ~215 tokens/param) for inference efficiency |

> 💬 *In short:*  
> “Scaling laws are linear in log-space — but only if you scale data and model together.  
> Stop pouring water into a small bucket; build a bigger one.”


## 🧩 Part 2: Model Engineering via Extrapolation

Scaling laws offer a roadmap 🗺️ for selecting **architectures**, **optimizers**, and **aspect ratios (depth/width)** using **small-scale training**.

![alt_text](/assets/images/llm-from-scratch/09/3.png "image_tooltip")

| ⚙️ Hyperparameter | 💡 Scaling Law Insight |
| :--- | :--- |
| **Architecture** | Transformers show a constant factor compute advantage over LSTMs. Newer designs (GLU, MoE) may outperform across compute budgets. |
| **Optimizer** | Choice (e.g., ADAM vs. SGD) yields predictable constant factor efficiency gaps. |
| **Depth/Width** | Beyond 2 layers → diminishing returns 📉. Embedding layers behave differently and should be analyzed separately. |
| **Batch Size ($B_{crit}$)** | Strong diminishing returns past the **Critical Batch Size**. Larger models require proportionally larger batches to reach lower loss. |
| **Learning Rate (LR)** | Optimal LR is **scale-dependent**. 🧮  
  **muP (Maximal Update Parametrization)** makes LR stable across model sizes → easy small-to-large transfer. |

⚠️ **Caution:**  
Training loss (e.g., perplexity) scales predictably, but **downstream performance** may not follow the same clean log-linear pattern.

---

### 🧮 Batch Size and Diminishing Returns

The fundamental observation regarding batch size is that it exhibits **strong diminishing returns** past a certain point. This behavior defines two distinct regimes of scaling efficiency:

1. **Efficient Scaling Regime**  
   When the batch size is **smaller than the noise scale**, increasing the batch size is almost equivalent to taking more gradient steps.  
   → This is the ideal scenario, allowing practitioners to leverage **data parallelism** while maintaining the optimization efficiency of more gradient updates.

2. **Ineffective Scaling Regime**  
   Once the batch size becomes **comparable to the noise scale**, additional samples in the batch no longer reduce useful noise.  
   → Optimization progress becomes dominated by the **curvature of the loss landscape (bias term)** rather than noise reduction, leading to **strong diminishing returns**.

![alt_text](/assets/images/llm-from-scratch/09/4.png "image_tooltip")

#### 🧠 The Critical Batch Size (\(B_{\text{crit}}\))

The **Critical Batch Size** ($(B_{\text{crit}})$) marks the **boundary** between these two regimes.  
It is defined as the threshold where scaling transitions from **efficient** to **inefficient**.

![alt_text](/assets/images/llm-from-scratch/09/5.png "image_tooltip")

Formally:

$[
B_{\text{crit}} = \frac{\text{minimum number of examples for target loss}}{\text{minimum number of steps for target loss}}
]$

In theoretical analyses, $(B_{\text{crit}})$ relates to the **gradient noise** expected from random sampling within the batch — serving as a useful analytical quantity to understand training dynamics.


#### 📉 Scaling of $(B_{\text{crit}})$ with Loss Target

A key insight from scaling analysis is that the **critical batch size itself scales predictably** as a function of the **target loss**.

Empirically:

> The smaller the target loss, the larger the batch size that can be effectively utilized.

This means that as models train and their loss decreases (i.e., they become better), the **optimal batch size grows**.

**Intuition:**

- When aiming for a very **low loss**, optimization becomes more sensitive.  
- Gradients need to be **more precise (de-noised)**.  
- A **larger batch size** provides this de-noising — similar to how the **learning rate is reduced** later in training.


#### 🧩 Practical Implications

Understanding $(B_{\text{crit}})$ is crucial for **resource allocation** and **scaling efficiency** in large-scale training.

When increasing both compute and model size, engineers face a trade-off:

- Use **larger batches** with fewer steps  
  vs.  
- Use **smaller batches** with more steps.

Scaling analyses suggest that as compute increases, **reasonable parallelism** can be achieved:

> The **number of total training steps** can stay roughly constant while the **batch size grows**.

This efficient trade-off — where resource investment enables larger batches without a proportional increase in steps — represents **good news for data-parallel training** at scale.


---

### 🚀 Learning Rate Scaling and Maximal Update Parametrization (muP)

The sources provide detailed context on the importance of the **learning rate (LR)** in scaling large language models (LMs) and introduce **Maximal Update Parametrization (muP)** as a novel solution to stabilize the optimal learning rate across different model sizes.

![alt_text](/assets/images/llm-from-scratch/09/6.png "image_tooltip")

#### ⚠️ The Challenge of Learning Rate Scaling

When training large language models — particularly standard Transformer architectures — the **optimal learning rate** is **not stable** across scales. It depends on **model width and depth**, making it **scale-dependent**.

#### 🧩 Standard Practice

In traditional parameterization (the default used in most deep learning frameworks):

- When you **scale up a model** (making it wider or deeper), the **optimal learning rate** changes with size.  
- Specifically, as the **model width** increases (e.g., larger MLP dimensions), the **optimal LR decreases** — often following a rough inverse relationship with width (∝ 1/width).  
- Conversely, **smaller models** tolerate **larger learning rates**.  
- Practitioners sometimes fit an **empirical scaling law** for learning rate vs. model size to predict good hyperparameters.

This dependency implies that a learning rate tuned on a small model **cannot be directly transferred** to a much larger one.  
That breaks one of the core goals of scaling laws — to **tune hyperparameters efficiently on small models** and then **extrapolate to large models**.

#### 🧠 Maximal Update Parametrization (muP)

To overcome this instability, researchers proposed **Maximal Update Parametrization (muP)** — a principled method for **scaling-aware initialization and learning rate design**.

#### 🎯 Goal of muP

muP aims to **reparameterize** the model such that the **optimal learning rate remains constant** across different model widths.

In other words:

> Tune the learning rate once on a small model → use the *same* LR for a much larger model without retuning.

#### ⚙️ Mechanism of muP

Under the muP framework, several scaling rules are applied depending on **model width**:

1. **Scaling Initialization**  
   The variance of parameter initialization is adjusted as a function of width.  
   (Ensures gradients and activations stay in a stable range as models widen.)

2. **Scaling Learning Rates**  
   Learning rates of parameters in different layers (e.g., weights vs. biases) are scaled differently to maintain consistent update magnitudes.

3. **Scaling Forward Paths**  
   The output of certain layers is multiplied by width-dependent scale factors so that signal propagation remains balanced across scales.

If these adjustments are implemented correctly, the model’s parameterization becomes **scale-aware** — meaning gradient magnitudes, signal propagation, and update sizes all behave consistently across model sizes.


#### 🧩 Practical Benefits of muP

The **main benefit** of muP is **stability and transferability** of hyperparameters:

- You can tune the learning rate once on a **small prototype model**.
- The same learning rate will remain **optimal** (or near-optimal) as you scale to **larger models**.
- This saves huge amounts of compute and simplifies large-scale experiments.

While in theory muP makes the LR *exactly* scale-invariant, in practice it achieves **highly predictable scaling** — a major step toward reproducible and efficient LM training.


#### 🔬 Extensions and Future Directions

Since muP’s introduction, similar ideas have appeared, such as **Meta’s “meta-p”** parameterization (reportedly used in *Llama 4*), suggesting an active research focus across labs on **stabilizing learning rates during scaling**.


#### ✅ Summary

| Concept | Description |
|----------|--------------|
| **Problem** | Learning rate changes unpredictably with model size (larger → smaller LR) |
| **Solution (muP)** | Reparameterize model and scale initialization, learning rates, and activations by width |
| **Goal** | Make the optimal learning rate *stable* across model sizes |
| **Benefit** | Tune once on small models → reuse LR at large scale |
| **Trend** | Research continues (e.g., Meta’s “meta-p”) to improve cross-scale training stability |



## ⚖️ Part 3: Joint Scaling and Optimal Compute Tradeoffs

A key question:  
> 🧮 Should we prioritize **more data ($n$)** or **bigger models ($m$)?**

![alt_text](/assets/images/llm-from-scratch/09/7.png "image_tooltip")

### 📈 Joint Scaling Laws
Model error can be expressed as:  
$$E = n^{-\alpha} + m^{-\beta} + C$$  
This joint formulation allows optimal allocation of **compute** between **data** and **parameters**.


### 🧮 The Chinchilla Principle (Hoffman et al. 2022)

The **Chinchilla paper (Hoffmann et al., 2022)** is a pivotal work in the study of **scaling laws**, focusing on how to determine the **optimal tradeoff between model size (m)** and **data size (n)** for a **fixed compute budget**.

![alt_text](/assets/images/llm-from-scratch/09/8.png "image_tooltip")

#### 🎯 Core Motivation

The central question Chinchilla addressed:

> For a fixed amount of compute (total FLOPs), should we train a **larger model for fewer steps**, or a **smaller model for more steps**?

Earlier large models like GPT-3 were **undertrained for their size** — they had too many parameters and too little data.  
Chinchilla set out to find the **compute-optimal configuration** that yields the **best model for a given training budget**.

#### ⚔️ Challenging Earlier Scaling Estimates

Before Chinchilla, earlier works such as **Rosenfeld (2020)** and **Kaplan (2020)** proposed empirical **joint scaling laws** between data, model, and error. However, **Chinchilla argued those fits were inaccurate** due to methodological flaws.

#### 🧩 Key Issue: Learning Rate Schedules

One major source of error was the **improper handling of learning rate schedules**:

- Many models in earlier studies were trained with **cosine decay schedules** but **not run to completion**.  
- Truncating such training early produces **undertrained models**, which are **not equivalent** to models trained fully at that compute level.
- This led to **incorrect scaling ratios** between data and model size.

By running all models **to completion**, Chinchilla corrected this and derived much more accurate scaling constants.

#### ⚖️ The Chinchilla Optimal Ratio

The Chinchilla analysis produced the now-famous **optimal ratio**:

> **20 tokens per parameter**

This ratio reflects the **optimal balance** between **model size (parameters)** and **data size (tokens)** for a fixed compute budget.

In practical terms:
- A model with **1B parameters** should ideally be trained on **20B tokens**.
- This ensures **no undertraining (too little data)** and **no overtraining (too much data)** for the available compute.

#### 🧮 Methods for Fitting Scaling Laws

The Chinchilla authors described **three main methods** for empirically deriving these scaling relationships.  
They found **methods 1 and 2** produced consistent, reliable results, while **method 3** required later correction.


#### **Method 1 – Minimum over Runs**

- Train many models of varying sizes across multiple total FLOP budgets.  
- For each configuration, track the **minimum validation loss** achieved.  
- The **lower envelope** (minimum over all runs) forms a clean **power-law curve** with compute.  
- The parameter sizes corresponding to these optimal points follow predictable scaling relationships.


#### **Method 2 – IsoFLOPs (Canonical Method)**

![alt_text](/assets/images/llm-from-scratch/09/9.png "image_tooltip")

This is the **canonical Chinchilla method** and is conceptually simple:

1. **Fix a total compute budget (FLOPs)**.  
2. **Sweep across model sizes** (m), adjusting data size (n) so that total compute remains constant.  
3. For each compute level, this produces a **convex loss curve** as a function of model size.  
4. The **minimum of this curve** (found via fitting) gives the **optimal tradeoff** between data and parameters.  
5. Repeating this across multiple compute budgets yields a smooth scaling law that reveals the **optimal tokens-per-parameter ratio**.

✅ Methods 1 and 2 produced **consistent results** confirming the 20:1 ratio.


#### **Method 3 – Joint Fits**

This method fits a **joint scaling law** of the form:

$[
E = n^{-\alpha} + m^{-\beta} + C
]$

- Involves training a grid of models across multiple \( (m, n) \) combinations.  
- Fit the above function via **least squares regression** to derive exponents and constants.  
- Initially flawed due to residual errors in the fitting process.  
- Later **replications corrected the errors**, confirming that this method also matched results from Methods 1 and 2.


#### 🧩 Train-Optimal vs. Inference-Optimal Models

A key distinction from Chinchilla is between:

| Type | Optimization Focus | Typical Use |
|------|--------------------|--------------|
| **Train-Optimal** | Best performance for a fixed *training* compute | Research training curves, scaling analysis |
| **Inference-Optimal** | Best *deployment* efficiency for given inference cost | Production LLMs (e.g., Llama, Claude, Gemini) |

While Chinchilla determined the **train-optimal** configuration,  
modern LLMs are increasingly designed for **inference-optimal** efficiency.

💡 That means —  
Companies **overtrain** models (use far more tokens than 20× parameters) to make them **smaller, faster, and cheaper** at inference time.


#### 📊 Examples of Token-to-Parameter Ratios

| Model | Tokens / Parameter Ratio |
|--------|---------------------------|
| GPT-3 | 2 : 1 |
| Chinchilla (optimal) | 20 : 1 |
| Llama 3 70B | 215 : 1 |

Modern models often operate far above the Chinchilla ratio, trading **extra training compute** for **massive inference efficiency gains**.

![alt_text](/assets/images/llm-from-scratch/09/10.png "image_tooltip")


#### 💰 The Economic View of Scaling Laws

Scaling laws — especially those revealed by Chinchilla — function like **economic optimization models** for AI training:

> Instead of blindly throwing compute at bigger models, they tell us how to **invest compute most efficiently** — balancing model size and data size to maximize performance under a strict budget.

This concept has become a cornerstone of **efficient foundation model training** strategies across all major AI labs.



> 💬 *"Scaling laws turn deep learning from trial-and-error into engineering."*  
> — paraphrased from the lecture’s closing message
