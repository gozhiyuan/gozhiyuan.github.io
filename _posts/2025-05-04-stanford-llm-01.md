---
layout: post
title: Language Modeling from Scratch Overview and Tokenization
subtitle: Language Modeling from Scratch Lecture 1
categories: Stanford-LLM-From-Scratch-2025
tags: [llm]
banner: "/assets/images/banners/yuanpang-wa-iceburg2.jpg"
---

# Language Modeling from Scratch Overview and Tokenization

[Course link](https://stanford-cs336.github.io/spring2025/)


## 1. Course Philosophy and Motivation: The "Build It from Scratch" Ethos

- **Philosophy**: "To understand it, you have to build it."
- **Problem**: Increasing abstraction in AI research disconnects researchers from the underlying systems.
- **Goal**: Combat abstraction crisis by re-engaging with foundational tech and rethinking the full ML stack.

### 🔑 Key Takeaways
- **Deep Understanding Through Implementation**: Not just using models—building them from scratch.
- **Abstraction Crisis**: High-level APIs obscure critical behaviors.
- **Enabling Fundamental Research**: Students will learn to redesign and co-optimize data, systems, and models.


## 2. The Challenges of Industrialization and Scale

- **Industrial-Scale Models**: GPT-4 rumored at 1.8T params, $100M+ training cost.
- **Private Clusters**: Companies like XAI with 200,000 H100s.
- **Lack of Transparency**: Competitive pressures limit public documentation.

### ⚠️ Challenges
- **Prohibitive Costs**: Academic replication is infeasible.
- **Black Box Models**: No insight into how closed models are built.
- **Scale Mismatch**: Small-scale behaviors may mislead; emergent phenomena only visible at scale.

## 3. Three Types of Knowledge: Mechanics, Mindset, Intuitions

- **Mechanics**: How components work (fully teachable).
- **Mindset**: Efficiency-obsessed attitude (culture pioneered by OpenAI).
- **Intuitions**: Knowing what decisions matter—partially teachable.

### 🎯 Key Point
- Course emphasizes **mechanics** and **scaling mindset**, but acknowledges intuitions require experience at scale.

## 4. The "Bitter Lesson" Reinterpreted: Algorithms at Scale Matter

- **Clarification**: Not "scale over algorithms," but "algorithms at scale."
- **Efficiency is Everything**: Improvements > Moore’s law (e.g., 44x ImageNet training efficiency).

### 💡 Core Principle
> "What is the best model one can build given a certain compute and data budget?"

## 5. Historical Context and Current Landscape of Language Models

### 🕰️ Historical Milestones
- **Shannon**: Language modeling roots in information theory.
- **2000s**: N-gram models trained on trillions of tokens.
- **2010s**: Deep learning stack matures (Bengio, seq2seq, Adam, Transformer).
- **2018–Now**: Foundation Models (Elmo, BERT, T5, GPT series).

Today's frontier models   
- OpenAI's o3: https://openai.com/index/openai-o3-mini/
- Anthropic's Claude Sonnet 3.7: https://www.anthropic.com/news/claude-3-7-sonnet
- xAI's Grok 3: https://x.ai/news/grok-3
- Google's Gemini 2.5: https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/
- Meta's Llama 3.3: https://ai.meta.com/blog/meta-llama-3/
- DeepSeek's r1: https://arxiv.org/pdf/2501.12948
- Alibaba's Qwen 2.5 Max: https://qwenlm.github.io/blog/qwen2.5-max/
- Tencent's Hunyuan-T1: https://tencent.github.io/llm.hunyuan.T1/README_EN.html

### 🔍 Open vs. Closed Models
- **Closed**: API-only (e.g., GPT-4).
- **Open Weight**: Weights + partial info (e.g., LLaMA).
- **Open Source**: Full transparency (Meta, DeepSeek, AI2, etc.).


## 6. Course Logistics and Expectations

- **Format**: 5-unit intensive course.
- **Audience**: For those with a "need to understand to the atoms."
- **Assignments**: No scaffolding, full implementations from scratch.
- **Cluster**: Use H100 GPUs (Together AI); start early to avoid congestion.
- **AI Tools**: Use at your own risk.


## 7. Course Structure: The Five Pillars of Efficiency

![alt_text](/assets/images/llm-from-scratch/01/1.png "image_tooltip")

### 🧱 1. Basics
- **Tokenizer**: Byte Pair Encoding (BPE), some tokenizer-free exploration.
- **Model**: Transformer, with variants (SwiGLU, RoPE, Mixture of Experts).
  - Activation functions: ReLU, SwiGLU
  - Positional encodings: sinusoidal, RoPE
  - Normalization: LayerNorm, RMSNorm
  - Placement of normalization: pre-norm versus post-norm 
  - MLP: dense, mixture of experts 
  - Attention: full, sliding window, linear
  - Lower-dimensional attention: group-query attention (GQA), multi-head latent attention (MLA)
  - State-space models: Hyena

- **Training**: AdamW, learning rate schedules, batch size, regularization. 
  - Optimizer (e.g., AdamW, Muon, SOAP)  
  - Learning rate schedule (e.g., cosine, WSD)
  - Batch size (e..g, critical batch size)
  - Regularization (e.g., dropout, weight decay)
  - Hyperparameters (number of heads, hidden dimension): grid search

🧪 **Assignment 1**:  
Implement tokenizer, Transformer, cross-entropy loss, AdamW. Leaderboard: perplexity on OpenWebText.


### ⚙️ 2. Systems
- **Kernels**: GPU internals, Triton for tiling and fusion.
- **Parallelism**: Data, tensor, model parallelism (FSDP).
- **Inference**: Optimize prefill vs. decode. Inference now dominates compute cost.

🧪 **Assignment 2**:  
Implement Triton kernel, parallelism; benchmark and profile constantly.


### 📈 3. Scaling Laws
- **Chinchilla Optimal**: Given flops, there's an optimal model size and data quantity.
- **Linear Fit**: Scaling laws plot reveals nearly linear relationships.

🧪 **Assignment 3**:  
Fit a scaling law via API queries; minimize loss within a flops budget. Leaderboard-based.

### 🧹 4. Data
- **Core Idea**: "The model does what the data tells it to."
- **Curation**: Crawl → Parse → Clean → Filter → Deduplicate → Select.
- **Evaluation**: Perplexity, MMLU, instruction-following, agent evaluations.

🧪 **Assignment 4**:  
Process Common Crawl data; deduplicate, classify, filter. Leaderboard: minimize perplexity under token budget.


### 🤖 5. Alignment
- **Goal**: Make model useful, safe, instruction-following.
- **Techniques**:
  - SFT (e.g., 1000 examples can bootstrap instruction-following)
  - DPO (Direct Preference Optimization)
  - GRPO (Group Relative Preference Optimization)

🧪 **Assignment 5**:  
Implement SFT, DPO, GRPO; evaluate alignment effectiveness.

## 8. Efficiency as the Overarching Principle

### 💡 Why Efficiency?
- **Tokenization**: Avoid byte-level inefficiency.
- **Architecture**: All tweaks aim to reduce compute-to-accuracy ratio.
- **Training**: One-epoch = maximize useful exposure per token.
- **Data**: Curated to avoid wasting GPU time.
- **Alignment**: Well-aligned small models can outperform unaligned large ones.

### 🔮 Forward Looking
- Shift from "compute constrained" → "data constrained" frontier labs.
- Re-thinking multi-epoch training and architecture design as data becomes scarcer than flops.

## 9. Tokenization in Language Modeling

Tokenization is a fundamental process in [language modeling](w), serving as the initial step to convert raw text into a format suitable for computational models.

### 🧾 Definition and Purpose

- **Tokenization** is the process of converting raw text (Unicode strings) into a sequence of integers, where each integer represents a *token*.
- It includes:
  - Encoding: string → tokens.
  - Decoding: tokens → string.
- The **vocabulary size** is the number of unique token integers.
- The goal is to segment the string and map each part to an integer, forming a fixed-length input to the model.
- Ideally, tokenization is **reversible**, allowing perfect reconstruction of the original string.


### 🧪 Tokenization Approaches and Their Limitations

#### A. **Character-Based Tokenization**
- Each Unicode character → integer code point.
- **Problems**:
  - Huge vocabulary (e.g., 😄 has code point 127,757).
  - Inefficient vocabulary usage.
  - **Compression ratio**: ~1.5 bytes/token (not optimal).

#### B. **Byte-Based Tokenization**
- Converts Unicode strings into byte sequences (e.g., via UTF-8), then maps bytes (0–255) to tokens.
- **Pros**: Very small vocabulary (256).
- **Cons**:
  - Sequence lengths explode.
  - Terrible for model efficiency due to quadratic scaling in attention layers.

#### C. **Word-Based Tokenization**
- Splits text by word boundaries (e.g., via regex).
- **Problems**:
  - Vocabulary can be unbounded.
  - Struggles with **"UNK" tokens** (unseen words).
  - Difficulties in computing **perplexity** accurately.


### 🔗 Byte Pair Encoding (BPE)

#### 🧠 Background
- Invented by Philip Gage in 1994 for data compression.
- Introduced to NLP in neural machine translation.
- Adopted by GPT-2for efficient and robust tokenization.

#### 💡 Core Idea
- Learns merges from data instead of hard-coded token boundaries.
- Common substrings become single tokens; rare substrings split into smaller units.

#### ⚙️ BPE Algorithm Steps
1. Convert input string into a byte sequence.
2. Repeatedly merge the **most common** adjacent token pair in the training corpus.
3. Add new tokens to the vocabulary with each merge.
4. The sequence shortens, improving the **compression ratio**.

- **Example**: 
  - Input: "cat and hat"
  - Common pair: `('t', 'h')` (116, 104) → new token: 256
  - Now, "th" → 256, saving tokens.

#### 🔁 Encoding / Decoding
- **Encoding**: Convert to bytes, then apply learned merges in order.
- **Decoding**: Reverse the merges to reconstruct the original string.

#### ⚠️ Efficiency Notes
- The naïve BPE encoding process can be inefficient.
- Optimizations involve restricting merges to relevant tokens during encoding.

### 🔍 Practical Observations and Design Notes

- **Space Handling**: 
  - Spaces often prefix tokens.
  - `"hello"` ≠ `" hello"` → treated as distinct tokens.
  
- **Numbers**: 
  - Often split into non-semantic tokens: e.g., `"123"` → `"1", "2", "3"` or `"12", "3"`.

- **Compression Ratio**: 
  - Metric of efficiency: bytes per token.
  - GPT-2 BPE achieves ~**1.6 bytes/token** (vs. 1.0 for byte-based).

- **Tokenizer-Free Methods**:
  - Approaches starting from raw bytes exist but haven’t scaled to frontier models yet.


### 📊 Comparison of Tokenization Methods

| Tokenization Method      | Description                                                                 | Pros                                                                 | Cons                                                                                           | Typical Vocabulary Size | Compression Ratio (bytes/token) | Sequence Length Efficiency |
|--------------------------|-----------------------------------------------------------------------------|----------------------------------------------------------------------|------------------------------------------------------------------------------------------------|--------------------------|-------------------------------|-----------------------------|
| **Character-based**      | Maps each Unicode character to a token                                      | - Simple<br>- Reversible                                             | - Huge vocab due to rare Unicode chars<br>- Inefficient use of vocab<br>- Long sequences       | Very large (~100k+)      | ~1.5                          | Poor                        |
| **Byte-based**           | Converts text into bytes (e.g., UTF-8), maps each byte (0–255) to a token   | - Small fixed vocab (256)<br>- Language-agnostic<br>- Fully reversible | - Very long sequences<br>- Poor model efficiency (quadratic attention cost)                    | 256                      | 1.0                           | Very poor                   |
| **Word-based**           | Splits text into words using regex or spaces                               | - Easy to interpret<br>- Short sequences for natural languages       | - Unbounded vocab<br>- “UNK” tokens for out-of-vocab words<br>- Not robust across languages    | Huge (can be >1M)        | ~1.3–2.0                     | Good (if vocab fits well)   |
| **Subword-based (BPE)**  | Learns frequent char pairs from data and merges them recursively            | - Balance of vocab size and sequence length<br>- No “UNK” tokens<br>- Language-agnostic | - Encoding can be slow<br>- Doesn’t understand meaning<br>- Token boundaries not semantic       | Medium (30k–50k typical) | ~1.5–1.6                    | Good                        |
| **Unigram LM**           | Learns a probabilistic model over subwords and selects best segmentation    | - Can model multiple segmentations<br>- Often better for morphologically rich languages | - More complex training<br>- Similar tokenization time to BPE                                  | Medium (~30k)            | ~1.5–1.6                    | Good                        |
| **Byte-level BPE**       | Applies BPE on raw bytes instead of Unicode chars                           | - No Unicode normalization needed<br>- Compact<br>- Robust           | - Long sequences<br>- Compression not as good as char-level BPE                                | Medium (~50k)            | ~1.2–1.3                    | Moderate                    |
| **Tokenizer-free (raw)** | Operates directly on raw bytes or characters (no explicit tokenizer)        | - Simplifies pipeline<br>- Fully reversible<br>- Great for multi-modal | - Still experimental<br>- Not efficient with current models<br>- Long sequences                | N/A                      | 1.0                           | Poor (currently)            |

### Notes:
- **Compression Ratio** = average number of bytes per token (lower is better).
- **Sequence Length Efficiency** = shorter sequences generally lead to lower memory and compute costs (especially in Transformers).
- **BPE** = Byte Pair Encoding widely used in models like GPT-2 and GPT-3.
- **Unigram LM** is used in SentencePiece.
- **Tokenizer-free methods** are explored in some state-space models and remain an open research area.


### 📚 Relevance to Stanford CS336

- **Unit 1 (Basics)** includes implementing a **BPE tokenizer**.
- **Assignment 1**: Students must build the tokenizer from scratch.
- Emphasizes **efficiency-driven design**:
  - Tokenization helps avoid wasting compute on poor-quality data.
  - Aggressive data filtering and tokenization both serve efficiency.

> ⚠️ The instructors note: “Hopefully in the future, we'll get rid of this lecture entirely when models can just learn directly from bytes.”

