---
layout: post
title: Mixture of Experts 🤖
subtitle: Language Modeling from Scratch Lecture 4
categories: Large-Language-Model
tags: [Stanford-LLM-From-Scratch-2025]
banner: "/assets/images/banners/yuanpang-wa-iceburg2.jpg"
---

# Mixture of Experts 🤖

Mixture of Experts (MoE) architectures have rapidly become a cornerstone in developing high-performance, large-scale language models (LLMs). Once a "bonus lecture" topic, MoEs are now fundamental to state-of-the-art systems.

- **🚀 High Performance:** MoEs offer significant advantages over traditional dense models in terms of performance per computational FLOP.
- **🧠 Core Idea:** Instead of one giant feed-forward network (FFN), MoEs use many specialized "expert" FFNs and a "router" that selects which experts to use for each input token.
- **💡 Key Innovation:** This sparse activation allows models to have a massive number of parameters without a proportional increase in computation.

For more detailed explanations of the MoE models, you can refer to my previous blog on [Deepseek model series](https://gozhiyuan.github.io/large-language-model/2025/03/01/deepseek-base-model.html)

[Course link](https://stanford-cs336.github.io/spring2025/)


## 1. What is a Mixture of Experts (MoE)?
A MoE is a specialized neural network architecture where a standard component, typically the feed-forward network (FFN), is replaced by many "expert" FFNs and a selector layer (the "router").

- **Core Concept:** "Replace big feedforward with (many) big feedforward networks and a selector layer."
- **Sparsity:** Unlike dense models where every parameter is activated for every input, MoEs activate only a small subset of parameters. This allows for an increase in the total number of parameters without a proportional increase in FLOPs.
> "You can increase the # experts without affecting FLOPs."
- **Focus on MLPs:** While MoEs could be applied to attention heads, the classic approach is to split up the densely connected FFNs. This is where "all the action is."

## 2. Why are MoEs Gaining Popularity?
MoEs offer several compelling advantages that have driven their widespread adoption:

- **🚀 Same FLOP, More Parameters, Better Performance:** This is the primary driver. MoEs allow models to have more parameters without affecting FLOPs. Empirical evidence consistently shows that at the same FLOP count, you get better performance from an MoE than a dense model.
- **🚄 Faster Training:** MoEs are generally faster to train. The OlMoE paper shows that the training loss for a dense model goes down much more slowly than for an MoE.
- **🏆 Highly Competitive:** MoEs consistently achieve highly competitive results versus their dense equivalents, often outperforming them on benchmarks.
- **🌐 Parallelization Benefits:** MoEs are parallelizable to many devices. Each expert can fit on a separate device, enabling "expert parallelism." This makes MoEs very popular for training very large models.
- **👑 Dominance in High-Performance Models:** Most of the highest-performance open models are MoEs, including Grok, Mixtral, DBRX, Qwen, DeepSeek, and Llama 4.

## 3. Challenges and Why MoEs Haven't Been More Popular (Historically)
Despite their advantages, MoEs historically faced significant hurdles:

- **🏗️ Infrastructure Complexity:** The biggest advantages of MoEs are realized in multi-node training environments, which are complex to set up and optimize.
- **🤔 Heuristic & Unstable Training:** The discrete nature of routing decisions makes them non-differentiable, posing a difficult optimization problem. This has led to "significant systems complexities."
- ** overfitting on Smaller Data:** Sparse MoEs can overfit during fine-tuning on smaller datasets due to their gigantic parameter count, leading to a large gap between training and validation loss.

## 4. MoE Architecture and Key Variations
### 4.1. General Structure
- Typically, MoEs "replace MLP with MoE layer."
- While less common, some work has explored "MoE for attention heads."

### 4.2. Key Variances in MoE Design:
- **Routing Function:** How tokens are assigned to experts.
- **Expert Sizes:** The number of experts and their individual dimensions.
- **Training Objectives:** How the routing and expert utilization are optimized.

## 5. Routing Functions
### 5.1. Overview
Many routing algorithms "boil down to ‘choose top k’." There are three main types:

1.  **Token Chooses Expert (Most Common):** Each token has a preference for different experts, and the top-k experts are chosen for each token.
2.  **Expert Chooses Token:** Each expert has a preference over tokens, and the top-k tokens are chosen for each expert. This has the benefit of balanced expert utilization.
3.  **Global Routing via Optimization:** Solving a complex optimization problem to ensure a balanced mapping between experts and tokens.

### 5.2. Common Routing Variants:
- **Top-k Routing:** "Almost all the MoEs do a standard ‘token choice topk’ routing."
    - A gating mechanism (e.g., a linear regressor) scores experts for each token, and the top-k are selected.
    - **Examples of k:** Switch Transformer (k=1), Gshard (k=2), Mixtral (k=2), DBRX (4), DeepSeek (7).
    - **Gating Mechanism:** The final output is a weighted sum of the expert outputs: `sum(G_i * Expert_i(x))`.
- **Hashing:** A "common baseline" where a hashing function maps inputs to experts.
> "Even if you're doing hashing so no semantic information at all you will still get gains from a hashing based MoE which is pretty wild."

### 5.3. Other Routing Methods (Less Common Now):
- **Reinforcement Learning (RL):** Used in early work to learn discrete routing policies.
> "RL is the ‘right solution’ but gradient variances and complexity means it’s not widely used."
- **Linear Assignment/Matching Problems:** Elegant, but the "cost of doing this is much higher than the benefits."

## 6. Expert Configurations
### 6.1. Shared and Fine-Grained Experts:
A key innovation, particularly from Chinese LLM groups like DeepSeek, is the use of "Smaller, larger number of experts + a few shared experts that are always on."

- **Fine-grained Experts:** "Slicing" the FFNs into smaller matrices, allowing for a larger number of experts while controlling the parameter count. This approach is "really really useful."
- **Shared Experts:** Some experts are "always on," processing all tokens, to "capture shared structure." Results are mixed on the benefits of shared experts.

### 6.2. Common Configurations:
- **Early Google Models (GShard, Switch Transformer):** Very large numbers of routed experts (e.g., 2048, 64).
- **Transition (Mixtral, DBRX, Grok):** Typically 8-16 total experts, with 2 or 4 active.
- **DeepSeek/Qwen/Llama 4 Era:** Significant increase in total experts (e.g., 60-256), often incorporating fine-grained and shared experts.

## 7. Training MoEs
The non-differentiable nature of sparse gating is a major training challenge.

### 7.1. Solutions for Non-Differentiability:
- **Reinforcement Learning (RL):** Conceptually "the most principle thing," but "not widely used" due to cost and stability issues.
- **Stochastic Approximations:** Adding noise to routing decisions to encourage exploration. This approach was "generally abandoned."
- **Heuristic Balancing Losses (Most Common):** The goal is to ensure experts are utilized evenly.
    - A loss term penalizes uneven expert utilization.
    - Without balancing, "early on in training the model just picks like one or two experts and all the other experts are dead."
- **DeepSeek Variations:**
    - **Per-expert balancing:** Ensures experts get an even number of tokens per batch.
    - **Per-device balancing:** Balances load across devices.
    - **Auxiliary-loss-free balancing (DeepSeek v3):** Introduces a learnable per-expert bias to encourage utilization.

### 7.2. Systems Considerations in Training:
- **Parallelization:** MoEs "parallelize nicely" through "expert parallelism," where each FFN can reside on a separate device.
- **Sparse Matrix Multiplications (MMs):** Modern libraries like MegaBlocks use "smarter sparse MMs" for efficient computation.

### 7.3. Stability Issues:
- **Router Stability:** Using Float32 for the expert router and adding an auxiliary Z-loss helps stabilize training.
- **Fine-tuning Overfitting:** "Sparse MoEs can overfit on smaller fine-tuning data."
    - **Solutions:** Fine-tune only non-MoE MLPs, or "use lots of data."

### 7.4. Upcycling:
- **"Can we use a pre-trained LM to initialize a MoE?" Yes.**
- Upcycling involves taking a dense pre-trained model, copying its MLP layers to form MoE experts, and then training the router from scratch.
- This is a "very cost-effective way of getting MoE."

## 8. DeepSeek MoE Evolution: A Case Study
DeepSeek's MoE architectures illustrate the progression in MoE design:

- **DeepSeek v1 (16B params, 2.8B active):**
    - Standard top-k routing.
    - 2 shared + 64 fine-grained experts (1/4 size), 6 active.
    - Standard auxiliary loss balancing.
- **DeepSeek v2 (236B params, 21B active):**
    - Same top-k selector.
    - 2 shared + 160 fine-grained experts (1/10 size), 6 active.
    - **Key Innovations:** Top-M Device Routing and Communication Balancing Loss.
- **DeepSeek v3 (671B params, 37B active):**
    - 1 shared + 258 fine-grained experts, 8 active.
    - **Key Innovations:** Aux-loss-free balancing, sequence-wise auxiliary loss, and Sigmoid+Softmax TopK + TopM.

## 9. Non-MoE Innovations in DeepSeek v3
DeepSeek v3 also incorporates other architectural optimizations:

- **🧠 Multi-head Latent Attention (MLA):** A KV-caching optimization where Q, K, V are expressed as functions of a lower-dimensional, "latent" activation. This saves KV cache memory.
- **⏩ Multi-Token Prediction (MTP):** Uses small, lightweight models to predict multiple steps ahead. DeepSeek v3 currently only predicts "one token ahead."