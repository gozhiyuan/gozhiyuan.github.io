---
layout: post
title: LLM Alignment - SFT/RLHF
subtitle: Language Modeling from Scratch Lecture 15
categories: Large-Language-Model Reinforcement-Learning
tags: [Stanford-LLM-From-Scratch-2025]
banner: "/assets/images/banners/yuanpang-wa-iceburg2.jpg"
---

# 🎓 RLHF & Alignment: Making LLMs Useful and Safe

This lecture, **CS336 Lecture 15**, dives into **Reinforcement Learning from Human Feedback (RLHF)** and **alignment** — the crucial *post-training* step that makes large pre-trained models like GPT-3 become helpful and safe assistants (like InstructGPT and ChatGPT).  
It follows the classic **three-step process** introduced in the *InstructGPT* paper:

> **Pretraining → Supervised Fine-Tuning (SFT) → Reinforcement Learning (RLHF)**

[Course link](https://stanford-cs336.github.io/spring2025/)



## 🧭 1. The Class Thus Far and Lecture Goals

The lecture begins by situating RLHF within the broader training pipeline.

### From GPT-3 to ChatGPT
Pretraining yields powerful but uncontrolled models (e.g., GPT-3) — great at language generation but poor at following instructions or maintaining safety. RLHF bridges this gap by teaching LMs *how* to behave.

**Key goals and motivations:**

- **Instruction Following & Control**  
  Modern models can follow nested, multi-step instructions and compose skills (e.g., code + reasoning).
- **Safety & Moderation**  
  Large-scale deployment introduces risks — scams, toxic content, and misuse — requiring moderation.
- **The Core Objective**  
  Collect data on *desired* behaviors, then train the LM to emulate them.
- **The Standard Recipe**  
  1. **Imitate** experts via SFT.  
  2. **Reinforce** desirable outputs via RLHF.


## 🧑‍🏫 2. Supervised Fine-Tuning (SFT)

SFT teaches the model to **imitate** expert demonstrations — it’s essentially behavioral cloning.

### 🧩 Ingredients of SFT Data

Two main ingredients define SFT:
1. **Data quality and diversity**
2. **Training methodology**

The lecture reviews three hallmark datasets:

| Dataset | Source & Construction | Style / Traits |
|----------|----------------------|----------------|
| **FLAN** | Aggregates many NLP datasets (Natural Instructions, T0 SFT, etc.) | Task-oriented, large scale, sometimes “unnatural” for chat |
| **Alpaca** | AI-generated instructions using LLaMA prompts | Natural, conversational, long-form responses |
| **OpenAssistant (Oasst)** | Community-written | Detailed, citation-rich, complex reasoning |


### ✨ Data Quality and Style Effects

- **Style Matters**  
  Human evaluators (and AI judges) prefer longer, list-formatted responses (~60–70% preference rate).
- **Knowledge & Hallucination**  
  If data contains niche or factual content unseen in pretraining, models may “hallucinate” citations or facts to fit expected structures.
- **Safety Tuning**  
  Small, well-chosen safety datasets (~500 examples) can drastically improve safety — but risk *over-refusal* if not balanced.

![alt_text](/assets/images/llm-from-scratch/15/1.png "image_tooltip")
![alt_text](/assets/images/llm-from-scratch/15/2.png "image_tooltip")


### ⚙️ SFT at Scale

Modern labs scale SFT in ways closer to pretraining:

1. **Pretrain on web corpus**  
2. **Mix in instruction data during stable LR phase** (mid-training)  
3. **Finish with short, high-quality SFT**

This “two-phase” or **mid-training** approach prevents catastrophic forgetting and yields stronger generalization.

![alt_text](/assets/images/llm-from-scratch/15/3.png "image_tooltip")


## 🧠 Bonus: Relationship Between Instruction Tuning (SFT) and Knowledge Acquisition

The relationship between **instruction tuning (Supervised Fine-Tuning, or SFT)** and **teaching knowledge** is complex and nuanced. It depends heavily on **scale**, **training strategy**, and **integration** within the broader model development pipeline. Importantly, SFT behaves very differently from both **pre-training** and **Reinforcement Learning from Human Feedback (RLHF)**.


### I. Instruction Tuning and Knowledge Acquisition

**Instruction tuning (SFT)** primarily aims to teach *desired behaviors* — such as style, formatting, and instruction following — rather than new factual knowledge. However, as scaling techniques evolve, the boundary between “behavioral tuning” and “knowledge teaching” has started to blur.

#### ⚠️ Limitations of SFT for Teaching Knowledge

- **Extraction vs. Acquisition**  
  SFT works best when extracting or refining pre-trained behaviors, *not* when introducing new information. Because instruction tuning datasets are small compared to pre-training corpora, they lack the diversity and coverage required for reliable factual learning.

- **Hallucination Risk (Tail Knowledge)**  
  Fine-tuning on “tail knowledge” — information outside the model’s prior — can cause **hallucinations**:  
  - When the model is forced to reproduce a fact or citation it doesn’t know, it may fabricate details to match the *expected structure* (e.g., inventing sources).  
  - This can lead to the paradox where adding factually correct examples *hurts* performance if the model isn’t ready for that knowledge depth.

- **Style Over Factuality**  
  Instruction tuning reliably instills stylistic traits — long answers, bullet lists, politeness — which human raters tend to favor. However, this *style bias* can overshadow factual correctness, especially when crowd workers focus more on formatting than truthfulness.


#### 🔄 The Blurring Line: Midtraining / Two-Phase Training

In modern LLM development, the line between pre-training and instruction tuning is fading due to **midtraining**, also called **two-phase training**:

1. Mix **instruction-tuning data** into the **late stage** of pre-training (during learning-rate decay).  
2. Finish with a short, high-quality **SFT round**.

When scaled up sufficiently, this hybrid approach allows instruction data to **instill new knowledge**, merging behavior alignment and factual enrichment — all while avoiding catastrophic forgetting.

![alt_text](/assets/images/llm-from-scratch/15/4.png "image_tooltip")


### II. Comparison: Pre-training vs. Instruction Tuning vs. RLHF

| **Feature** | **Pre-training** | **Instruction Tuning (SFT)** | **Reinforcement Learning (RLHF)** |
|--------------|------------------|-------------------------------|------------------------------------|
| **Primary Goal** | Build broad capabilities (reasoning, QA) using massive text data; next-token prediction. | Extract capabilities and align behavior (instruction following, formatting, safety) via imitation. | Optimize a reward function *R(y, x)* to align outputs with human preferences. |
| **Knowledge Transfer** | Core mechanism for world knowledge; massive and diverse. | Risky and limited for new (“tail”) knowledge unless scaled up (midtraining). | Indirectly reinforces correctness and abstention; messy but promising for factual alignment. |
| **Data Type** | Massive, unstructured corpora (Common Crawl, code, books). | Human-written instruction–response pairs. | Pairwise preference data (A vs. B) for reward modeling. |
| **Cost Profile** | Extremely high compute, low annotation cost. | High annotation cost — expert-written, long-form data. | Relatively cheaper — scalar feedback easier to collect than full demonstrations. |


#### 🔍 Key Differences in Detail

1. **SFT vs. Pre-training (on Knowledge)**  
   - Pre-training injects the majority of factual and conceptual knowledge.  
   - SFT guides *how* that knowledge is used (style, tone, task framing).  
   - When scaled and mixed during pre-training, SFT begins acting like pre-training itself, embedding new knowledge while maintaining alignment.

2. **SFT vs. RLHF (on Correction)**  
   - **SFT** imitates expert responses — even if it must hallucinate to match structure (e.g., fake citations).  
   - **RLHF**, by contrast, optimizes for *human-preferred correctness*: rewarding factual responses, discouraging confident but wrong outputs, and training the model to gracefully abstain when uncertain.

Together, these three phases form a pipeline:
> **Pre-training → SFT → RLHF**  
> Capabilities → Imitation & Style → Optimization & Alignment

### 🍳 Analogy: Training an Elite Chef

| Phase | Analogy | Purpose |
|--------|----------|----------|
| **Pre-training** | The chef studies a massive cookbook (the Internet), mastering all basic techniques. | Acquires raw skills and knowledge. |
| **Instruction Tuning (SFT)** | The chef is given expert recipes and presentation standards. | Learns style, instruction following, and consistent behavior. |
| **Reinforcement Learning (RLHF)** | The chef cooks for a panel of judges who give preference feedback (rewards). | Learns to optimize for human satisfaction, not just imitation. |

If you ask this chef to recreate a dish they’ve *never seen* (tail knowledge), they might confidently **hallucinate** ingredients just to match the expected plating — just like an LM forced to mimic structure without understanding content.  
Through RLHF, however, they learn what *tastes right*, refining behavior beyond imitation.

### 🧩 Summary Takeaway

- **Pre-training** = Knowledge foundation  
- **SFT** = Instruction and style alignment  
- **RLHF** = Reward optimization and behavioral refinement  

Together, these steps transform a raw language model into a **useful, truthful, and human-aligned assistant**.


## 🧠 3. Reinforcement Learning from Human Feedback (RLHF)

Once imitation reaches its limits, we move from **fitting data** to **optimizing behavior**.

### 🎯 From Imitation to Optimization

| Method | Goal | Data Needed |
|---------|------|-------------|
| **SFT** | Fit expert demonstrations (policy imitation) | Instruction–response pairs |
| **RLHF** | Optimize expected reward | Pairwise preferences (A vs. B) |

Instead of minimizing cross-entropy, RLHF maximizes  
$[
\mathbb{E}_{p_\theta(y|x)} [R(y, x)]
]$

### Why RLHF?

1. **Cost Efficiency** – Scalar feedback is cheaper than full expert responses.  
2. **Generator–Validator Gap** – People don’t always write the outputs they *prefer*. Evaluating is easier than generating.  


### 🧾 RLHF Data Collection

- **Pairwise Feedback:** Annotators compare outputs (A vs. B), creating preference pairs.  
- **Reward Model:** Learns to assign a scalar reward to any completion.  
- **Crowd Biases:** Formatting and verbosity often outweigh factuality in human ratings.  
- **AI Feedback:** GPT-4 acts as a scalable annotator — high agreement with human raters.  
- **Length Effects:** “Better” answers often just mean *longer* answers.


In RLHF, the model generates multiple outputs (known as **rollouts**).  
Annotators — either **humans** or another **language model (LM)** — then compare two outputs (**A vs. B**) and select the better one.  
These comparisons form the **pairwise dataset** used to train the Reward Model.


### The Role of the Reward Model (RM)

1. **Objective — The Bradley–Terry Model of Preferences**  
   Each possible LM output is assumed to have an underlying *scalar reward* $( R )$.  
   The probability that one output is preferred over another follows a logistic function of their reward difference:
   $[
   P(A > B) = \frac{e^{R(A)}}{e^{R(A)} + e^{R(B)}}
   ]$

2. **Training the RM**  
   The RM learns to assign scalar reward values that best match human (or AI) preferences on these pairs.  
   Its goal is to minimize the divergence between predicted preferences and observed human rankings.

3. **Application in RL**  
   Once trained, the RM becomes the **reward function** $( R(y, x) )$ that the RL policy seeks to maximize — guiding the model toward outputs humans prefer.


### Pairwise Feedback Criteria and Content

Annotators typically follow three high-level **evaluation pillars** when deciding which output is better:

| **Criterion** | **Description** |
|----------------|----------------|
| **Helpfulness** | Answers the prompt clearly, with detail and global sensitivity (e.g., clarifying “football”). |
| **Truthfulness** | Ensures factual correctness and avoids hallucination. |
| **Harmlessness** | Avoids toxicity, bias, and impoliteness. |

These pillars guide consistent scoring across tasks.


### Challenges and Biases in Data Collection

While RLHF pairwise data is cheaper than supervised data, collecting *high-quality* comparisons is still difficult:

- **Correctness Checking Difficulty**  
  Annotators often have <1 minute per task, leaving little time to verify factual accuracy or math correctness.
- **Length Bias**  
  Both humans and AI judges exhibit a strong preference (~60–70%) for *longer* outputs, leading RLHF-tuned models to generate overly verbose responses.
- **Annotator Demographics**  
  Cultural and linguistic biases influence preferences, sometimes prioritizing formatting or tone over factual truth.
- **AI Feedback (AI Feedback Loops)**  
  Due to cost and scale, labs now use **AI feedback** — typically GPT-4 — as a proxy rater.  
  GPT-4’s pairwise rankings show near-human agreement but also a **self-preference bias** for its own style.

### How RL Algorithms Use Pairwise Data

The RL agent (the fine-tuned LM) aims to maximize the **expected reward** under the reward function $( R(y, x) )$:
$[
\max_{\pi_\theta} \, \mathbb{E}_{x \sim D, y \sim \pi_\theta(\cdot|x)} [ R(y, x) - \beta \log \frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)} ]
]$

- $( \pi_\theta )$: The current policy (fine-tuned model)  
- $( \pi_\text{ref} )$: The reference model (e.g., the SFT model)  
- $( \beta )$: The **KL penalty** coefficient that keeps the policy close to the reference, preventing catastrophic drift or over-optimization


## 4. Optimization Algorithms: PPO and DPO

### 🌀 PPO — *Proximal Policy Optimization*

PPO, introduced in *InstructGPT*, is the canonical but complex RLHF algorithm.

- **Policy Gradient Core:**  
  PPO adjusts probabilities to upweight high-reward outputs and downweight poor ones.
- **Clipping Mechanism:**  
  It restricts policy updates via a clipping ratio (ε), ensuring that new probabilities stay within a bounded region relative to the reference model.
- **Goal:**  
  Encourage improvement *without* deviating too far — maintaining stability and avoiding collapse.

> PPO = “Carefully controlled reward climbing.”


### ⚡ DPO — *Direct Preference Optimization*

DPO simplifies RLHF by avoiding explicit reinforcement learning loops.

- **No Reward Model Needed**  
  Instead of training a separate RM, DPO directly optimizes the model using preference pairs.
- **Mechanism:**  
  Takes **positive gradient steps** on preferred outputs and **negative steps** on dispreferred ones:
  $[
  L_\text{DPO} = -\log \sigma\big(\beta (\log \pi_\theta(y_\text{preferred}|x) - \log \pi_\theta(y_\text{dispreferred}|x))\big)
  ]$
- **Interpretation:**  
  Turns the reward learning problem into a **maximum likelihood** task — simpler, faster, and often equally effective.
- **Advantage:**  
  Removes need for on-policy rollouts and reward tuning loops.

> DPO = “RLHF without the RL.”


### Optimization Pitfalls

Regardless of method, optimizing for reward introduces risks:

- **Overoptimization / Overfitting**  
  The model may exploit the reward function, producing high-scoring but low-quality answers.
- **Mode Collapse**  
  The LM can lose its probabilistic nature, yielding repetitive, deterministic responses.
- **Calibration Drift**  
  Excessive reward pressure distorts likelihood calibration — hurting performance on non-RL tasks.
