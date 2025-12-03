---
layout: post
title: LLM Alignment - Reinforcement Learning
subtitle: Language Modeling from Scratch Lecture 16
categories: Large-Language-Model Reinforcement-Learning
tags: [Stanford-LLM-From-Scratch-2025]
banner: "/assets/images/banners/yuanpang-wa-iceburg2.jpg"
---

# LLM Alignment - Reinforcement Learning

This post continues the exploration of **Reinforcement Learning (RL)** techniques for aligning **Large Language Models (LLMs)** — tracing the evolution from **Direct Preference Optimization (DPO)** to **Proximal Policy Optimization (PPO)** and **Generalized Reinforcement Learning with Policy Optimization (GRPO)**, culminating in practical **Reinforcement Learning from Verifiable Rewards (RLVR)** case studies like **Deepseek R1**, **Kimi K1.5**, and **Qwen 3**.

[Course link](https://stanford-cs336.github.io/spring2025/)



## 🧩 I. Recap: RLHF and Direct Preference Optimization (DPO)

### 🔍 Direct Policy Optimization (DPO): Simplifying RLHF “Without Tears”

**Direct Policy Optimization (DPO)** is an algorithm designed to simplify the complex training procedures associated with **Reinforcement Learning from Human Feedback (RLHF)** — achieving the same objectives *“without tears.”*

Instead of relying on reinforcement loops, DPO reframes RLHF as a **supervised learning problem** on an alternatively parameterized objective.


#### 🧩 Detailed Explanation of DPO

At its core, DPO optimizes an **underlying reward** using **pairwise preference data** — i.e., observing which of two model responses ($ y^{+} $ vs. $ y^{-} $) is preferred by human (or LM) judges.

DPO Simplifies RLHF by:
1. **Removing the Reward Model:**  
   It eliminates the need for a separate learned reward function.
2. **Dropping On-Policy Rollouts:**  
   It avoids complex procedures like collecting new rollouts or maintaining outer training loops, which are required in PPO-based RLHF.
3. **Using Simple Supervised Gradients:**  
   Instead of reinforcement signals, DPO directly applies:
   - **Positive gradients** on preferred responses (“good stuff”).  
   - **Negative gradients** on rejected responses (“bad stuff”).  
   Each update is **weighted appropriately** based on preference confidence or the implied reward gap.

#### 💡 Why DPO Became Dominant

DPO quickly gained traction in the open-source ecosystem because of its **simplicity** and **comparable performance** to traditional PPO-based RLHF:

- **Ease of Implementation:**  
  Works with standard supervised learning pipelines — no separate critic, no policy rollout.
- **Stable Training:**  
  Deterministic loss and gradients (no stochastic rollouts).
- **Empirical Parity:**  
  Controlled experiments show that DPO achieves *similar results to PPO* in simulation benchmarks — without the heavy engineering burden.

> 🏁 In short:  
> **DPO ≈ PPO-level alignment** with **SFT-level simplicity** — making it the go-to approach for many open RLHF systems.

### 🧮 Deep Dive: Understanding the DPO Formula (Compared to the Stiennon RLHF Objective)

Let’s go step by step — starting from the **Stiennon et al. (2020)** RLHF formulation (used in InstructGPT), and then showing how **Direct Policy Optimization (DPO)** mathematically simplifies it while keeping the same intent.


#### 1) The Stiennon RLHF Objective (Original PPO-based Formulation)

In the traditional RLHF setup (from *Stiennon et al., 2020, “Learning to summarize with human feedback”*), the model (policy) $\pi_\theta(y|x)$ is trained to **maximize the expected reward** assigned by a **learned reward model** $r_\phi(x, y)$, **while staying close to the supervised fine-tuned (SFT) reference policy** $\pi_{\text{ref}}(y|x)$.

The objective is:

$$
\max_\theta \; \mathbb{E}_{x \sim D,\, y \sim \pi_\theta(\cdot|x)} 
\Big[ \, r_\phi(x, y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} \, \Big]
$$

where:

- $r_\phi(x, y)$ → reward from human preference model  
- $\beta$ → scaling factor controlling tradeoff between reward and staying close to reference  
- The KL term $-\log \frac{\pi_\theta}{\pi_{\text{ref}}}$ prevents the model from drifting too far from $\pi_{\text{ref}}$

This was optimized via **PPO** — an on-policy reinforcement learning algorithm.  
However, PPO is **expensive** and **unstable**: you need rollouts, value networks, and advantage estimation.


#### 2) DPO: Eliminating the Reward Model

DPO starts from the same **goal** — maximize a hidden reward while staying close to the reference —  
but removes the **explicit reward model** and **RL machinery**.

Instead, DPO uses the **pairwise preference data** directly:  
for a given prompt $x$, and two model outputs $y^{+}$ (preferred) and $y^{-}$ (dispreferred),  
we assume there exists an **implied latent reward** $r(x, y)$ satisfying the **Bradley–Terry preference model**:

$$
P(y^{+} \succ y^{-} \mid x)
\;=\;
\sigma \!\Big( r(x, y^{+}) - r(x, y^{-}) \Big)
\;=\;
\frac{1}{1 + \exp\!\big[-(r(x, y^{+}) - r(x, y^{-}))\big]}
$$

where $\sigma(\cdot)$ is the logistic (sigmoid) function.


#### 3) Expressing the Reward in Terms of the Policy

This is the **key step** in understanding how DPO works.  
We’ll start with the standard RLHF objective, then show **how we can algebraically “solve for” the reward** in terms of the policy itself — removing the need for a reward model.


##### 1️⃣ Start from the RLHF Objective

In Reinforcement Learning from Human Feedback (as used in PPO training),  
we want the model (policy) $\pi_\theta(y|x)$ to generate outputs that both:

1. Have **high reward** $r_\phi(x, y)$ (as judged by a reward model), and  
2. **Stay close** to the reference model $\pi_{\text{ref}}(y|x)$ (so it doesn’t go off-distribution).

This trade-off is expressed mathematically as:

$$
J(\pi_\theta)
= 
\mathbb{E}_{y \sim \pi_\theta(\cdot|x)}
\Big[ \, r_\phi(x, y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} \, \Big]
$$

Here $\beta$ controls how much the model is allowed to deviate from the reference.


##### 2️⃣ Think: What Is the *Optimal* Policy for This Objective?

If we imagine that the reward function $r_\phi$ is fixed,  
the best (optimal) policy $\pi^*$ that maximizes this objective should put **more probability mass** on high-reward outputs — but also stay close to $\pi_{\text{ref}}$.

We can find this optimal policy **analytically** by applying a bit of calculus of variations.

Let’s take the derivative of the objective w.r.t. $\pi(y|x)$ and set it to zero  
(while ensuring $\sum_y \pi(y|x) = 1$ to keep it a valid probability distribution).

After rearranging, we get:

$$
\pi^*(y|x)
\; \propto \;
\pi_{\text{ref}}(y|x) \,
\exp\!\left( \tfrac{1}{\beta} r_\phi(x, y) \right)
$$

This is a **Boltzmann (softmax) distribution** over rewards —  
the higher the reward, the more likely the model is to sample that response.

##### 3️⃣ Rewriting This Equation — Solve for the Reward

The above relationship means that **the optimal policy and the reward function are linked**.

If we take the logarithm of both sides, we can express the reward directly in terms of the policy:

$$
\log \pi^*(y|x)
=
\log \pi_{\text{ref}}(y|x)
+ \tfrac{1}{\beta} r_\phi(x, y)
+ C(x)
$$

where $C(x)$ is a normalization constant that ensures probabilities sum to 1.

Rearranging gives:

$$
r_\phi(x, y)
=
\beta \Big[ \log \pi^*(y|x) - \log \pi_{\text{ref}}(y|x) \Big] + \text{const.}
$$

##### 4️⃣ Intuitive Meaning

This is the **DPO key insight**:

> The “implied reward” of an action $y$ can be inferred directly from how much the optimal policy $\pi^*$ prefers it over the reference policy $\pi_{\text{ref}}$.

- If $\pi^*$ assigns **more probability** to $y$ than $\pi_{\text{ref}}$,  
  → that means $r_\phi(x, y)$ must be **positive** (it’s a good answer).  
- If $\pi^*$ assigns **less probability** to $y$ than $\pi_{\text{ref}}$,  
  → that means $r_\phi(x, y)$ must be **negative** (a bad answer).

In other words:
$$
\text{Reward} \; \propto \; \text{Log-probability difference between the learned model and the reference.}
$$

##### 5️⃣ How DPO Uses This

DPO says:  
“Let’s not learn $r_\phi$ at all. Let’s **replace $r_\phi$ in the loss** using the formula above, and train $\pi_\theta$ directly.”

So when you see this expression inside the DPO loss:

$$
(\log \pi_\theta(y^{+}|x) - \log \pi_{\text{ref}}(y^{+}|x))
-
(\log \pi_\theta(y^{-}|x) - \log \pi_{\text{ref}}(y^{-}|x))
$$

it’s literally the **reward difference** between the preferred and rejected responses,  
**computed from the model’s log-probs** instead of a learned reward model.

##### 6️⃣ Analogy: A Shortcut Around the Reward Model

Imagine PPO is like this:

> 👩‍🏫 “First, train a separate *reward model* to tell you what’s good.  
> Then, use PPO to adjust your main model to get more of that reward.”

Now DPO says:

> 🧮 “Actually, we can skip the reward model —  
> the ratio between our model and the reference already *implies* what the reward must have been.”

Formally, this substitution:
$$
r(x, y) = \beta \big[\log \pi_\theta(y|x) - \log \pi_{\text{ref}}(y|x)\big]
$$
lets us train **purely from preference pairs** using a simple logistic loss — no rollout, no critic, no value function.

> 🧠 **Intuition**:  
> DPO realizes that if you know how much better your model should rate an answer than your old model (reference), you already know the *reward difference*.  
> You don’t need a separate reward model — the policy itself *is* the reward estimator.


#### 4) Plugging the Reward Back Into the Preference Likelihood

Substituting the above expression for $r(x, y)$ into the Bradley–Terry likelihood:

$$
P(y^{+} \succ y^{-} \mid x)
=
\sigma \!\Big(
\beta \Big[
(\log \pi_\theta(y^{+}|x) - \log \pi_{\text{ref}}(y^{+}|x))
-
(\log \pi_\theta(y^{-}|x) - \log \pi_{\text{ref}}(y^{-}|x))
\Big]
\Big)
$$

This is exactly the DPO likelihood!  
We can now **train directly on this logistic loss** without ever learning $r_\phi$.

#### 5) The DPO Loss Function

We **maximize** the log-likelihood of the human preferences,  
or equivalently **minimize** the negative log-likelihood loss:

$$
\mathcal{L}_{\text{DPO}}
\;=\;
-\mathbb{E}_{(x, y^{+}, y^{-}) \sim D}
\Big[
\log \sigma \!\Big(
\beta \big[
(\log \pi_\theta(y^{+}|x) - \log \pi_{\text{ref}}(y^{+}|x))
-
(\log \pi_\theta(y^{-}|x) - \log \pi_{\text{ref}}(y^{-}|x))
\big]
\Big)
\Big]
$$

where:

- $ \pi_\theta $ = the policy we’re training  
- $ \pi_{\text{ref}} $ = the fixed SFT model  
- $ \beta $ = a temperature-like scaling hyperparameter  

Intuitively:
- If the model already ranks $y^{+}$ higher than $y^{-}$ → low loss.  
- If not → large gradient pushing $ \pi_\theta(y^{+}|x) $ up and $ \pi_\theta(y^{-}|x) $ down relative to the reference.

#### 6) DPO vs. PPO — Conceptual Equivalence

| Aspect | PPO (Stiennon RLHF) | DPO |
|--------|---------------------|------|
| Reward | Learned model $r_\phi(x,y)$ | Implied via policy log-probs |
| Optimization | On-policy, requires rollouts | Off-policy, purely supervised |
| KL Regularization | Explicit in objective | Implicit via $\pi_{\text{ref}}$ |
| Implementation | Complex (value fn, GAE, clipping) | Simple (logistic loss on pairs) |
| Data | Online rollouts | Static preference dataset |

So, **DPO is mathematically equivalent to optimizing the same Stiennon RLHF objective**,  
but under the closed-form substitution of $r(x, y)$ derived from the optimal policy.

- DPO says: “Let’s skip the middleman (the reward model).  
  We can infer what the reward *would have been* from how much we should deviate from the reference policy.”
- The $\beta$ term controls **how much deviation** we allow —  
  small $\beta$ → stay close to SFT; large $\beta$ → chase reward harder.

> 💬 In essence:
> $$
> \textbf{DPO = PPO objective with the reward model algebraically integrated out.}
> $$

That’s why it’s called **RLHF without tears** —  
you get nearly the same effect as PPO-trained RLHF,  
but with a single, stable **logistic regression loss** on preference pairs.

---

### 🧪 DPO in Practice: Step-by-Step Training (Inputs, Outputs, Loss, Pseudocode)

This subsection gives a **concrete, engineering-ready** view of **Direct Policy Optimization (DPO)**: what goes in, what comes out, and how each training step runs.


#### 1) Problem Setup & Notation

- **Policy (trainable):** $ \pi_\theta(y \mid x) $
- **Reference policy (frozen):** $ \pi_{\text{ref}}(y \mid x) $ (e.g., the SFT checkpoint)
- **Pairwise preference data:** tuples $ (x, y^{+}, y^{-}) $ where annotators (or an auto-rater) prefer $ y^{+} $ over $ y^{-} $ for prompt $ x $
- **Inverse-temperature (scale):** $ \beta > 0 $
- **Optional length normalization:** divide sequence log-probs by token count to reduce length bias

We work with **sequence log-likelihoods**, i.e., sums of per-token log-probs under teacher forcing:

$$
\log \pi_\theta(y \mid x) \;=\; \sum_{t=1}^{T(y)} \log \pi_\theta\!\big(y_t \,\big|\, x, y_{<t}\big)
$$

(Optionally use **length-normalized** $ \frac{1}{T(y)} \log \pi_\theta(y \mid x) $.)


#### 2) Core DPO Objective (Sequence-Level Preference Loss)

Define the **relative log-odds** between the trainable policy and the reference:

$$
\Delta_\theta(x, y^{+}, y^{-}) \;=\; 
\Big[\log \pi_\theta(y^{+} \mid x) - \log \pi_\theta(y^{-} \mid x)\Big]
\;-\;
\Big[\log \pi_{\text{ref}}(y^{+} \mid x) - \log \pi_{\text{ref}}(y^{-} \mid x)\Big]
$$

The **DPO loss** for one pair is a logistic loss:

$$
\mathcal{L}_{\text{DPO}}(x, y^{+}, y^{-})
\;=\;
- \log \sigma\!\Big(\beta \cdot \Delta_\theta(x, y^{+}, y^{-})\Big)
\quad \text{where} \quad 
\sigma(z)=\frac{1}{1+e^{-z}}
$$

Minimizing $ \mathcal{L}_{\text{DPO}} $ **increases** $ \log \pi_\theta(y^{+} \mid x) $ and **decreases** $ \log \pi_\theta(y^{-} \mid x) $, **relative** to the frozen reference.

> Intuition: keep $ \pi_\theta $ close to $ \pi_{\text{ref}} $ unless the pairwise signal **pushes** it to favor $ y^{+} $ over $ y^{-} $.

#### 3) I/O Specs (Shapes, Batching)

- **Inputs (per batch):**
  - Prompts: token ids $ X \in \mathbb{N}^{B \times L_x} $
  - Chosen responses: $ Y^{+} \in \mathbb{N}^{B \times L_{+}} $
  - Rejected responses: $ Y^{-} \in \mathbb{N}^{B \times L_{-}} $
  - Attention masks for each (causal + padding)
  - (Optional) precomputed $ \log \pi_{\text{ref}}(Y^{\pm} \mid X) $
- **Outputs:**
  - Scalar loss $ \mathcal{L}_{\text{DPO}} $ (averaged over the batch)
  - Gradients w.r.t. $ \theta $ only (reference has no grads)
- **Common sizes:**
  - Batch size $ B \in \{32, 64, 128\ldots\} $ (effective batch via grad-accum)
  - Context window $ L_x + L_{\pm} \leq $ model max length
  - Mixed precision (fp16/bf16) recommended

#### 4) Step-by-Step Training Loop

**Per training step:**

1. **Sample a mini-batch** of $ (x, y^{+}, y^{-}) $ pairs of size $ B $.
2. **Tokenize & pack** the sequences: concatenate $ [x \oplus y^{+}] $ and $ [x \oplus y^{-}] $.
3. **Forward (policy):**
   - Compute per-token log-probs under $ \pi_\theta $ with teacher forcing.
   - Reduce to sequence log-probs:
     $$
     \ell^{+}_\theta \leftarrow \log \pi_\theta(y^{+} \mid x),
     \qquad
     \ell^{-}_\theta \leftarrow \log \pi_\theta(y^{-} \mid x)
     $$
     (Optionally normalize by length.)
4. **Forward (reference, frozen):**
   - Either compute on-the-fly or load cached values:
     $$
     \ell^{+}_{\text{ref}} \leftarrow \log \pi_{\text{ref}}(y^{+} \mid x),
     \qquad
     \ell^{-}_{\text{ref}} \leftarrow \log \pi_{\text{ref}}(y^{-} \mid x)
     $$
5. **Compute margin:**
   $$
   \Delta_\theta \leftarrow (\ell^{+}_\theta - \ell^{-}_\theta) - (\ell^{+}_{\text{ref}} - \ell^{-}_{\text{ref}})
   $$
6. **Compute loss:**
   $$
   \mathcal{L}_{\text{DPO}} \leftarrow - \frac{1}{B} \sum_{i=1}^{B} \log \sigma\!\big(\beta \cdot \Delta_{\theta}^{(i)}\big)
   $$
7. **Backward + optimizer step** on $ \theta $; **do not** update $ \pi_{\text{ref}} $.
8. **Log metrics:** loss, accuracy surrogate $ \mathbb{1}[\Delta_\theta > 0] $, KL to reference, lengths, etc.
9. **Repeat** until convergence (or curriculum phase change).


#### 5) Practical Tweaks & Regularization

- **Length normalization:**  
  Use $ \frac{1}{T(y)} \log \pi(\cdot) $ to avoid rewarding verbosity.  
- **KL monitor (or penalty):**  
  Track $ \mathrm{KL}\!\left(\pi_\theta \;\|\; \pi_{\text{ref}}\right) $ to prevent drift; optionally add a soft penalty:
  $$
  \mathcal{L} \;=\; \mathcal{L}_{\text{DPO}} \;+\; \lambda_{\text{KL}} \cdot \mathrm{KL}
  $$
- **Temperature/scale $ (\beta) $:**  
  Larger $ \beta $ sharpens the logistic, making updates more confident but potentially unstable.
- **Pair curation:**  
  Filter low-signal pairs (near ties), balance topics and lengths, and dedupe prompts.
- **Caching $ \pi_{\text{ref}} $:**  
  Precompute/cage reference log-probs to halve forward cost at train time.
- **Gradient checkpointing / Flash-Attention:**  
  Reduce memory footprint on long contexts.


#### 6) Multi-Preference & N-Best Generalization

If you have $ K $ candidates $ \{y^{(k)}\}_{k=1}^{K} $ with a **total order** or **pairwise graph** of preferences, train on all induced pairs $ (y^{(i)}, y^{(j)}) $ where $ y^{(i)} \succ y^{(j)} $.  
Use **pair weighting** (e.g., by score gap or judge confidence) to focus on high-signal comparisons.


#### 7) Diagnostics to Watch

- **Win-rate vs. reference:** $ \mathbb{P}[\Delta_\theta > 0] $ on a held-out set  
- **Reward hacking signs:** sudden KL spikes, length explosion, template mimicry  
- **Entropy / calibration:** sampling diversity, temperature sensitivity  
- **Generalization:** eval on out-of-distribution prompts

#### 8) Minimal Pseudocode (Batch)

```python
# Given: policy pi_theta (trainable), reference pi_ref (frozen), beta
batch = sample_batch(pairs)  # [(x, y_pos, y_neg)] * B

# Forward under policy (teacher forcing)
logp_pos = seq_logprob(pi_theta, x, y_pos, length_normalize=True)
logp_neg = seq_logprob(pi_theta, x, y_neg, length_normalize=True)

# Forward under reference (cached or compute)
logp_pos_ref = seq_logprob(pi_ref, x, y_pos, length_normalize=True, no_grad=True)
logp_neg_ref = seq_logprob(pi_ref, x, y_neg, length_normalize=True, no_grad=True)

# Margin
delta = (logp_pos - logp_neg) - (logp_pos_ref - logp_neg_ref)

# DPO loss
loss = - torch.mean(torch.logsigmoid(beta * delta))

# Backprop
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### 📊 Results and Variants

DPO achieves performance comparable to PPO but is easier to implement. Key variants include:

- **SimPO (Simple Preference Optimization):** Removes the reference policy $ \pi_{\text{ref}} $ and normalizes by response length.  
- **Length-normalized DPO:** Keeps $ \pi_{\text{ref}} $ but normalizes by response length.

### ⚠️ Pitfalls in RLHF

1. **Overoptimization / Reward Overfitting:**  
   Optimizing for the proxy reward diverges from true human intent.  
   Overfitting appears in both human and LM-generated rewards.  

2. **Mode Collapse / Entropy Loss:**  
   RLHF models become overconfident and less probabilistic than SFT-trained ones.


## 🧠 II. Expanding the Scope: PPO and GRPO

The move toward PPO and GRPO (and the broader domain of Reinforcement Learning from Verifiable Rewards, or RLVR) is motivated by the desire to work in domains where the true reward can be optimized exactly, quickly, and efficiently at scale—similar to successes in traditional RL like AlphaGo.
DPO is not well suited for tasks like math questions where the reward is based on correctness (verifiable answers), because the data is typically not inherently pairwise (in the form of Bradley-Terry comparisons).
PPO and GRPO are policy gradient methods designed to optimize rewards in actual RL tasks. They are applied in the RLVR setting where rewards are dense and verifiable (e.g., whether a math solution is correct).

This section introduces **PPO** and its simplified cousin **GRPO**. For more detailed explanations on PPO and GRPO, you can refer my previous [blogs](https://gozhiyuan.github.io/large-language-model/reinforcement-learning/2025/10/10/llm-rl-01.html)

### 🔁 PPO Recap

**Proximal Policy Optimization (PPO)** builds on policy gradient and TRPO foundations:

- **Policy Gradients:** High variance and slow (purely on-policy).  
- **TRPO:** Adds KL constraints for stable off-policy updates.  
- **PPO:** Simplifies TRPO by **clipping** the probability ratio:
  $$
  L_{\text{PPO}} = \mathbb{E} \left[ \min \left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
  $$

### 🧱 PPO in Practice

Implementing PPO for LLMs is complex ("37 details problem"). Core components include:

- **Outer Loop:** Collect rollouts, compute losses, backprop.  
- **Value Function:** A second model estimating expected reward, doubling GPU memory.  
- **Reward Shaping:** Dense reward at the final token + per-token KL regularization.  
- **Generalized Advantage Estimation (GAE):**  
  When $ \gamma = \lambda = 1 $, GAE reduces to a **baselined policy gradient**:
  $$
  A_t = R_t - V(s_t)
  $$

### ⚙️ Why GRPO?

PPO is heavy and memory-intensive, while DPO struggles with **non-pairwise** data like verifiable math problems.

**GRPO (Generalized Reinforcement Learning with Policy Optimization)** simplifies PPO by **removing the value function**.

#### 🧾 Advantage Computation in GRPO

For a group of samples per question:

$$
A_i = \frac{R_i - \mu_{\text{group}}}{\sigma_{\text{group}}}
$$

where:

- $ R_i $ = reward for the $i$-th response  
- $ \mu_{\text{group}} $ = mean reward across group  
- $ \sigma_{\text{group}} $ = standard deviation within the group  

This normalizes each sample’s advantage (a **z-score**), effectively removing the need for a value model.

#### ⚠️ Known Issues with GRPO

1. **Invalid Baseline:**  
   $ \sigma_{\text{group}} $ scaling breaks unbiasedness and upweights low-variance (too easy/hard) samples.  
2. **Length Bias:**  
   Models may exploit reward division by length — producing overly long wrong answers or ultra-short correct ones.


## 🚀 III. Case Studies in RLVR

The lecture concludes with **three case studies** illustrating practical RLVR recipes.


### 🧩 1. DeepSeek R1

The **Deepseek R1** project stands as one of the most influential and successful open implementations of **Reinforcement Learning from Verifiable Rewards (RLVR)**, demonstrating that **simple, outcome-based RL** can achieve state-of-the-art reasoning performance — surpassing even OpenAI’s O1.


#### 1) Significance and Algorithm

**Deepseek R1** is remarkable because it achieved **O1-level (or better) reasoning performance** with an *open, transparent, and relatively simple RL pipeline*.

- **Base Algorithm:**  
  R1 builds upon **GRPO** (*Generalized Reinforcement Learning with Policy Optimization*), first proposed in **DeepseekMath**.
- **Why GRPO?**  
  GRPO was selected because it’s **much simpler and more efficient** than PPO — it eliminates the **Value model**, which halves GPU memory requirements and simplifies training.
- **Reward Type:**  
  R1 only uses **outcome-level rewards** (e.g., correct/incorrect answer), **not process supervision** (no intermediate step-level rewards).
- **Negative Results:**  
  R1’s success **disproved the necessity** of complex systems like **MCTS (Monte Carlo Tree Search)** or **PRMs (Process Reward Models)** — these methods were tested but found **ineffective** for reasoning performance.


#### 2) R1-zero — The Controlled Setting

**R1-zero** is the simplest and most “controlled” version of the R1 experiment — a pure RL phase trained on top of the **Deepseek-V3 base model**.

##### 🎯 Rewards Used
1. **Accuracy Reward:**  
   A binary reward (1 for correct, 0 for incorrect) for solving math problems.
2. **Format Reward:**  
   Encourages the model to wrap its reasoning in explicit tags such as  
   `<think>` … `</think>`.  
   The lecture notes highlight that this format reward was *“a pretty critical part”* of getting reasoning RL to work.

##### 📈 Results
Using just these two reward signals, **R1-zero achieved performance approaching OpenAI O1**, showing that effective reasoning can emerge even from a simple, verifiable reward structure.


#### 3) Phenomena and Re-Analysis

The R1 paper reported several **emergent behaviors** observed during R1-zero training:

##### 🌱 Observed Phenomena
1. **Longer Chains of Thought (CoT):**  
   CoTs grew progressively longer as training continued — interpreted as the model “thinking harder” on harder problems.
2. **Emergent Backtracking (“Aha” Moments):**  
   The model occasionally revised or corrected its own reasoning mid-output, resembling human “aha” behavior.

##### 🔍 Re-Analysis
Subsequent analyses suggested that these effects may not be as “magical” as they first appeared:
- The **increasing CoT length** could be an **artifact of GRPO’s biased objective**, which tends to reward longer incorrect responses and shorter correct ones due to normalization.
- The **“aha moments”** may already exist in the **base Deepseek-V3** model, rather than being newly learned behaviors.

#### 4) The Full R1 Pipeline

The **full R1 training pipeline** adds multiple stages and refinements to R1-zero, forming a more comprehensive **reasoning-RLVR → RLHF hybrid** process.

| Stage | Description |
|-------|--------------|
| 1 | **Deepseek-V3 Reasoning SFT (CoT)** |
| 2 | **Reinforcement Learning (GRPO)** |
| 3 | **SFT/RLHF Post-Training** |

##### 🧠 Step 1: SFT Initialization

- The model begins with **Supervised Fine-Tuning (SFT)** on **long CoT datasets**.  
- **Goal:** Improve interpretability and stabilize reasoning before RL training.
- Even **a small amount of SFT data** proved sufficient to “bootstrap” reasoning — indicating that the base model already had latent reasoning skills that SFT helped surface.

##### ⚙️ Step 2: RL (GRPO Phase)

- The RL step uses the **same GRPO structure** as R1-zero, but adds one crucial component:
  - **Language Consistency Reward:**  
    Added to prevent **language mixing** (e.g., switching between English and Chinese) that emerged when aggressively training with RL.
- This stage teaches the model to perform *clean reasoning* while maintaining coherent language use.

##### 🔄 Step 3: SFT / RLHF Post-Training

After the reasoning-RLVR phase, R1 adds a **post-training step** combining traditional SFT and RLHF:

1. **SFT:**  
   - Merges two data sources:  
     • **Reasoning data:** For unverifiable tasks (e.g., “write a proof”), using **Deepseek-V3** as a judge.  
     • **Non-reasoning data:** From the general **Deepseek-V3 SFT dataset**.
2. **RLHF:**  
   - Uses the **same GRPO-based** reasoning RLHF pipeline (like R1-zero) for preference fine-tuning.


#### 5) Distillation and Key Observations

##### 💧 Distillation
- The R1 model was used to generate **~800k Chains of Thought (CoT) traces**, which served as training data for **smaller models** (e.g., Qwen 2.5).
- These distilled models achieved **large performance boosts** — e.g., **+25%** improvement on the **Amy benchmark** for a 32B model.
- This shows that **reasoning ability can be transferred** downward efficiently.

##### 🔬 Negative Scientific Results

R1 also reported **important negative findings**, helping clarify what *doesn’t* work:

1. **Process Reward Models (PRMs):**  
   - Intermediate-step reward systems didn’t outperform outcome-based rewards.  
   - Earlier DeepseekMath results hinted at PRM benefits, but R1 found them less effective overall.
2. **MCTS (Monte Carlo Tree Search):**  
   - Search-based reasoning methods failed to replicate or surpass O1-level performance.  
   - Outcome-based GRPO remained more efficient and stable.


#### 🧭 Summary Takeaways

| Concept | R1’s Approach |
|----------|----------------|
| Algorithm | GRPO (no value function, outcome-only reward) |
| Reward Type | Verifiable outcome (correct/incorrect) |
| Architecture | 3-stage: SFT → GRPO RL → SFT/RLHF |
| Key Additions | Language consistency reward, CoT formatting tags |
| Results | Surpassed O1 reasoning, reproducible with open data |
| Insights | Simpler RL works; MCTS & PRMs unnecessary |
| Impact | Enabled distillation to smaller models (e.g., Qwen 2.5) |


> 💬 **In essence:**  
> Deepseek R1 proved that *“less is more”* in reasoning RL.  
> With just outcome rewards, GRPO, and lightweight SFT priming, R1 matched or exceeded O1 — no PRM, no MCTS, no magic.


### 🔢 2. Kimi K1.5

The **Kimi K1.5** model is another landmark in **Reinforcement Learning from Verifiable Rewards (RLVR)**, released contemporaneously with **Deepseek R1**.  
While R1 demonstrated that simple GRPO could achieve O1-level reasoning, Kimi K1.5 provided a **complementary approach** — emphasizing data curation, efficient RL design, and explicit **Chain of Thought (CoT) length control**.


#### 1) Overall Strategy and Performance

**Kimi K1.5** adopts a three-step pipeline for developing reasoning LLMs:

1. **Dataset Construction**  
2. **Supervised Fine-Tuning (SFT)** for long CoTs  
3. **Reinforcement Learning (RL)** for verifiable reasoning

- Achieved or exceeded **OpenAI O1**-level performance across diverse reasoning benchmarks.
- Released **concurrently** with Deepseek R1, enabling a direct comparison between two open, high-performing RLVR frameworks.


#### 2) Data Curation and Supervised Fine-Tuning (SFT)

Kimi’s data preparation strategy is one of its most distinctive features — later influencing models like **Qwen 3**.

1. **Balancing Topics:**  
   Curated math-style datasets to maintain **discipline balance** (e.g., algebra, geometry, probability).

2. **Exclusions:**  
   Removed **multiple-choice** and **true/false** questions — arguing these tasks are too easy to “game” and can yield **false positives**.  
   Focused only on **verifiable problems** (e.g., numeric or symbolic answers checkable via regex or LM-based judges).

3. **Difficulty Filtering (Critical Step):**  
   Kimi used the model’s **own SFT performance** to detect hard examples:
   - Sample 8 completions per problem (best-of-8).  
   - Keep only those where the **SFT model failed all 8 attempts**.  
   → Ensures RL focuses on *unsolved, challenging tasks* rather than trivial data.

4. **SFT for Long CoTs:**  
   Performed standard **Supervised Fine-Tuning** on **long reasoning traces**, priming the model to “think step-by-step” before RL.  
   (Exact SFT dataset and prompts were not publicly described.)


#### 3) The Kimi RL Algorithm

Kimi’s RL formulation is inspired by the **DPO-style derivation** — where an equality between the **policy ratio** and **reward** is derived —  
but adapted for **verifiable rewards** (rather than preference pairs).


The optimal policy relationship is:

$$
\pi^*(y|x) \propto \pi_{\text{ref}}(y|x) \exp\!\left(\tfrac{1}{\beta} r(x, y)\right)
$$

Instead of inserting this into the **Bradley–Terry** preference likelihood (as in DPO),  
Kimi defines a **squared loss** to enforce this equality directly:

$$
\mathcal{L}_{\text{Kimi}} 
= 
\mathbb{E}\Big[ 
\big( 
\log \pi_\theta(y|x) - \log \pi_{\text{ref}}(y|x) - \tfrac{1}{\beta} r(x, y)
\big)^2
\Big]
$$

This makes training smoother, as it avoids the logistic term and instead encourages the policy to match the target log-prob difference implied by the reward.

The final update rule behaves like a **baselined policy gradient** with an added **regularization term**, encouraging stability during RL training.


| Feature | GRPO | Kimi K1.5 |
|----------|------|------------|
| Baseline | Group mean of rewards | Explicit average (no std. division) |
| Variance Normalization | Divides by σ (z-score) | No division by σ |
| Regularization | Implicit via clipping | Explicit penalty on policy update |
| Reward Type | Outcome-level (verifiable) | Outcome-level (verifiable) |
| Complexity | Simpler | Slightly more structured |

> 🧠 Kimi’s RL can be viewed as a “**convergent evolution**” of RL algorithms — combining GRPO’s simplicity with DPO’s theoretical grounding and PPO’s stability cues.


#### 4) Length Control and Curriculum Learning

One of Kimi K1.5’s most innovative contributions lies in its **explicit CoT length management** —  
a deliberate contrast to Deepseek R1, whose CoTs tended to grow excessively long during RL.

To control inference cost and encourage concise reasoning:

- A **length reward (λ)** is added to the reward function:
  - Range: roughly **−0.5 (too long)** to **+0.5 (ideal short)**.
- For **correct answers** → positive incentive for shorter CoTs.  
- For **incorrect answers** → mild penalty for overly long reasoning chains.

The effective reward per example becomes:

$$
R'(x, y) = R_{\text{accuracy}}(x, y) + \lambda \cdot f(\text{length}(y))
$$

where $ f(\text{length}) $ maps sequence length to a normalized range.

##### 🧪 Training Schedule

- The **length reward** is **disabled early** in training.  
  Enabling it too soon caused convergence to **degenerate solutions** (short, incorrect answers).  
- Activated **later** once the model had developed basic reasoning ability.


##### 📚 Curriculum Strategy

Kimi employed a **curriculum learning** approach to make RL more stable and data-efficient.

1. **Difficulty Labels:**  
   Each example was tagged (manually or via an LLM) with a difficulty level.

2. **Training Order:**  
   Model trained **from easy → hard**, gradually improving reasoning complexity.

3. **Sampling Strategy:**  
   To prioritize learning from unsolved problems, sampling probability was proportional to:

   $$
   P_{\text{sample}} \propto (1 - \text{success\_rate})
   $$

   Meaning that once a problem was consistently solved, it was **dropped** from sampling.


##### ⚖️ Equivalence Checks and Reward Verification

For math reasoning, Kimi used a **reward model trained on 800K samples** to verify whether the LLM’s output matched the ground-truth answer.  
This system effectively performed **semantic equivalence checking** using regex and LLM-based matching —  
providing **robust, automated verification** for rewards.


#### 5) RL Infrastructure and System Design

Kimi K1.5’s technical report also discussed **RL systems engineering** — a topic rarely covered in research papers.

##### ⚙️ Infrastructure Challenges

- **On-Policy Inefficiency:**  
  RL requires frequent **rollouts**, making training slower and less parallelizable than pretraining.
- **System Complexity:**  
  Continuous weight updates between **training workers** and **rollout workers** add coordination overhead.
- **Batch Imbalance:**  
  Long CoTs lead to **uneven batch durations**, reducing GPU utilization.

##### 🧱 Practical Setup

- **Dedicated RL Workers:** for policy optimization.  
- **Dedicated Inference Workers (VLM-based):** for performing rollouts and collecting verifiable outcomes.  
- Communication between workers via **message passing** to synchronize policy weights and results.


#### 🎯 Analogy and Perspective

If **Deepseek R1** focused on giving the “student” (the model) the **simplest and most general learning tool (GRPO)**,  
then **Kimi K1.5** focused on giving the student a **refined and disciplined study plan** —  
using smart **data filtering**, **difficulty-aware curricula**, and **length constraints** to ensure efficient learning and practical inference costs.


| Aspect | Description |
|--------|-------------|
| RL Framework | DPO-inspired squared loss with explicit regularization |
| Reward Type | Verifiable (math correctness, equivalence checking) |
| Data Strategy | Difficulty-based filtering and topic balancing |
| Curriculum | Progressive (easy → hard) with sampling ∝ (1 - success_rate) |
| Length Control | Explicit length reward, enabled mid-training |
| Infrastructure | Hybrid worker setup with message passing |
| Comparison to R1 | Shorter CoTs, more structured RL, similar performance |

> 💬 **In summary:**  
> Kimi K1.5 proved that **structured RLVR** — with careful data curation, progressive training, and explicit length regulation —  
> can achieve the same high reasoning performance as Deepseek R1, but with **greater inference efficiency** and **cleaner reasoning traces**.


### 🧮 3. Qwen 3

**Qwen 3** represents the most recent and advanced open reasoning model discussed in the lecture.  
It successfully integrates lessons and techniques from earlier systems like **Deepseek R1** and **Kimi K1.5**, achieving **performance that surpasses both OpenAI O1 and Deepseek R1**.


#### 1) Training Pipeline and Core Strategy

Qwen 3 follows the **standard reasoning model playbook**, combining **Supervised Fine-Tuning (SFT)** and **Reinforcement Learning (RL)** on verifiable rewards.

1. **SFT for Long CoT:**  
   Trains the model to produce detailed, interpretable **Chains of Thought (CoTs)** before any RL fine-tuning.

2. **Reasoning RL (GRPO):**  
   Applies **Generalized Reinforcement Learning with Policy Optimization (GRPO)** on verifiable, outcome-level rewards to strengthen reasoning.

3. **Thinking Mode Fusion:**  
   Introduces a unique **inference-time control mechanism** that allows toggling between long reasoning and short direct answers — controlling **CoT length** and **inference cost**.

4. **General RLHF:**  
   A final alignment stage using broader preference data to improve general instruction-following and dialogue capabilities.

#### 2) Data Curation and Efficiency

Qwen 3 inherits many of its **data curation insights** from R1 and Kimi K1.5 but refines them further for **data efficiency** and **quality control**.

- **Filtering for Difficulty:**  
  Like Kimi K1.5, Qwen used **best-of-n sampling** (e.g., best-of-8) to identify **difficult problems** that the base SFT model failed to solve.  
  → Ensures RL focuses on reasoning-intensive, non-trivial cases.

- **Decontamination and Manual Filtering:**  
  Removed near-duplicate or overly similar examples to evaluation data.  
  Manually filtered early SFT CoTs to distinguish **genuine reasoning** from **lucky guesses** or **pattern memorization**.

A striking finding from Qwen 3 is its **extreme data efficiency** during the RL stage:

- Qwen achieved strong reasoning gains using **only ~3,995 examples** for GRPO-based RL.  
- This suggests that **verifiable reward RL** can extract reasoning skills **very efficiently**, similar to **instruction-tuning** or **CoT distillation** in smaller data regimes.

> 💡 **Key takeaway:** Reasoning capability scales more with *reward quality* than *dataset size*.


#### 3) Qwen 3 Innovation: Thinking Mode Fusion

The **main innovation** introduced by Qwen 3 is **Thinking Mode Fusion**, a mechanism that provides **explicit control over reasoning length and cost** during inference.


During fine-tuning, the model learns **two modes of reasoning**, each marked by a special tag:

1. **Thinking Mode:**  
   - Data includes `<think>` tags followed by full **Chain of Thought (CoT)** reasoning.  
   - Encourages deep, step-by-step analytical reasoning before answering.

2. **Non-thinking Mode:**  
   - Data includes `<no_think>` tags that prompt the model to **skip CoT generation** and produce **immediate answers**.  
   - Useful for quick inference or time-limited applications.

By combining both in training, the same model can **switch dynamically** between these reasoning styles at inference.


##### ⏹️ Early Stopping and Graceful Degradation

One of the elegant features of Thinking Mode Fusion is **early stopping**, enabling the model to adaptively terminate its reasoning process.

- If the user or system imposes a **time constraint**, a **termination string** (e.g.,  
  “Given limited time, I’ll provide a direct answer now.”) signals the model to switch from `<think>` to `<no_think>` mode.
- This allows the model to **end its reasoning gracefully**, still providing a reasonable answer without completing a full CoT.

**Benefit:**  
- Controlled **inference latency**  
- Predictable **compute cost**  
- **Smooth degradation** in performance rather than abrupt failure


#### 4) Tradeoffs and Alignment

Qwen 3 performed careful **ablation studies** to analyze how its different stages interact, revealing nuanced tradeoffs.

- **General vs. Reasoning Performance:**  
  - Reasoning RL + Thinking Mode Fusion improved both general and reasoning tasks.  
  - However, the **final general-purpose RLHF stage** (alignment training) slightly **reduced math/STEM accuracy**.
  
- **Interpretation:**  
  There appears to be a **tradeoff** between **general-purpose alignment** (helpfulness, safety) and **specialized reasoning ability**.  
  Aggressive alignment may slightly “dull” the sharp reasoning capabilities gained from focused RLVR.


#### 🧭 Summary of Qwen 3 Insights

| Aspect | Qwen 3 Approach |
|--------|------------------|
| RL Framework | GRPO on verifiable rewards |
| Key Innovation | Thinking Mode Fusion (dual reasoning modes) |
| Data Strategy | Difficulty-filtered, decontaminated CoT data |
| Data Scale | ~3.9k samples for RL (highly efficient) |
| Alignment | Final RLHF stage for general capabilities |
| Tradeoff | Slight math/STEM loss post-alignment |
| Inference Control | Early stopping & mode switching for cost management |


> 💬 **In summary:**  
> Qwen 3 unified the best elements of R1 (simple GRPO) and Kimi K1.5 (structured data and length control),  
> then added **Thinking Mode Fusion** — a clever way to make reasoning models both **smart and practical**.  
> It shows that with a small, well-curated dataset and verifiable RL, reasoning models can be both **efficient and deployable** at scale.


---

## 🧭 Conclusion

While RLHF remains prone to overoptimization on noisy, subjective feedback, **RLVR** offers a scalable path forward — leveraging **verifiable rewards** to train reasoning models effectively.

**GRPO** serves as the core lightweight engine of this transition, powering models like **DeepSeek R1**, **Kimi K1.5**, and **Qwen 3**, which demonstrate that **simple, verifiable RL** can achieve or even exceed the performance of heavily engineered closed systems.

> 💬 “Less human noise, more verifiable truth — that’s the philosophy of RLVR.”