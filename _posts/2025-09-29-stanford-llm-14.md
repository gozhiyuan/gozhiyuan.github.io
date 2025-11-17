---
layout: post
title: Filtering and Deduplication Algorithms for LLM Data Processing
subtitle: Language Modeling from Scratch Lecture 6
categories: Large-Language-Model Reinforcement-Learning
tags: [Stanford-LLM-From-Scratch-2025]
banner: "/assets/images/banners/yuanpang-wa-iceburg2.jpg"
---

# Filtering and Deduplication Algorithms for LLM Data Processing

This lecture dives deeply into **how raw web data is transformed into clean, usable training data** for large language models (LLMs), focusing specifically on **filtering** and **deduplication** algorithms.  
Raw data moves from live services to dumps or crawls and must undergo **HTML-to-text conversion, language/quality/toxicity filtering, and deduplication** to become viable training material.


[Course link](https://stanford-cs336.github.io/spring2025/)

## 1️⃣ Filtering Algorithms

Filtering aims to extract a **high-quality subset (T′)** from massive raw data (R), guided by a smaller **target dataset (T)**.  
A good filtering algorithm must generalize beyond T and scale efficiently to terabytes of data.

### 1.1 🧮 N-gram Models (KenLM)

N-gram models estimate the likelihood of word sequences based on local co-occurrence statistics.

- **Estimation (MLE):**  
  $[
  P(w_n | w_{n-1},...,w_1) = \frac{count(w_1,...,w_n)}{count(w_1,...,w_{n-1})}
  ]$
- **Challenge:** Sparse counts increase with larger n.
- **Solution:** Use **Kneser–Ney smoothing** to handle unseen n-grams by backing off to lower orders.
- **Implementation:** **KenLM** – an efficient open-source LM originally for machine translation.
- **Use Case (CCNet/LLaMA):**  
  Text paragraphs are scored by **perplexity** (lower = higher quality).  
  Top 1/3 of documents are kept.  
  ➜ Fast but coarse filtering method.


### 1.2 ⚡ FastText Classifier

A lightweight, high-speed linear classifier ideal for large-scale text classification.

- **Architecture:** Uses a small hidden dimension (H) instead of full vocabulary size (V), reducing parameters to H×(V+K).  
- **Speed:** Comparable accuracy to neural nets but **orders of magnitude faster**.  
- **N-gram Handling:** Uses a **hashing trick**—hashing n-grams into fixed bins (e.g., 10M bins).  
- **Filtering Use:** Typically binary classification (**good vs. bad**).  
  While more complex models (e.g., BERT, LLaMA) could work, **FastText is used for scalability** across massive R.


### 1.3 🧠 Importance Resampling (DSIR)

**Data Selection for Language Models via Importance Resampling (DSIR)** takes a distribution-matching approach.

- **Goal:** Select samples so their distribution matches a desired **Target (P)** rather than the raw **Proposal (Q)**.  
- **Mechanism:** Assign each sample from Q a **weight (W ∝ P/Q)** and resample proportionally.  
- **Implementation:** Use **hashed n-grams** to estimate both P and Q distributions.  
- **Advantage:** Better captures **diversity**, while classifiers like FastText only predict membership.


### ⚙️ General Filtering Framework

Across all methods (KenLM, FastText, DSIR):

1. **Train a scoring model** on T (target) and R (raw).  
2. **Filter R** by keeping samples above a score threshold.


## 2️⃣ Filtering Applications

Filtering techniques are applied to **language identification**, **quality filtering**, and **toxicity filtering**.


### 2.1 🌍 Language Identification

Identify text in a target language (e.g., English) to maximize effective compute.

- **Motivation:** Models trained on mixed-language data waste resources.  
  Example: BLOOM trained with only 30% English.  
- **Implementation:** Use **FastText’s 176-language classifier**, trained on multilingual corpora.  
- **Example:** **DOLMA** retains pages with **P(English) ≥ 0.5**.  
- **Caveats:** Struggles with:
  - Short or low-resource languages  
  - Similar languages (e.g., Malay vs. Indonesian)  
  - Code-switching


### 2.2 💎 Quality Filtering

Aim: Maximize the informational and educational value of data.

#### 🧰 Model-Based Filtering
- **GPT-3:** Classifier trained with *Wikipedia/WebText2/Books* as positives and *Common Crawl* as negatives.  
- **LLaMA/RedPajama:** Used *pages cited by Wikipedia* as positives.

#### 🤖 Synthetic Data Driven Filtering (phi-1)
- **Approach:** Use a strong LM (e.g., GPT-4) to rate or label data by **educational value**.  
- **Example (phi-1 model):**
  - GPT-4 classified 100k Python documents from *The Stack*.  
  - Positives → target dataset (T).  
  - Trained a **Random Forest** classifier to scale filtering cheaply.  
  - Result: Achieved **17.68% HumanEval accuracy (vs. 12.19%)** despite fewer training steps.


### 2.3 🚫 Toxicity Filtering

Ensures removal of harmful, obscene, or threatening text.

- **Dataset:** **Jigsaw Toxic Comments** (Wikipedia talk pages, manually labeled).  
- **Implementation:** **Two FastText classifiers**—one for “hate,” one for “NSFW.”  
- **Usage:** Adopted in **DOLMA** pipeline to enforce safety in training data.


## 3️⃣ Deduplication

Deduplication removes both **exact** and **near duplicates**, crucial for:
- Reducing redundant compute  
- Preventing memorization  
- Avoiding copyright/privacy issues

### 🧩 Examples
- Mirror sites  
- Repeated boilerplate (e.g., ToS, templates)  
- Slight formatting or token differences  

Naïve pairwise comparison is **O(N²)**—impossible at scale—so efficient **hash-based** methods are used.


### 3.1 🔑 Hashing and Exact Deduplication

Hashing compresses text (sentence/document) into small signatures.

- **Preferred:** Fast, non-cryptographic hashes (e.g., **MurmurHash**).  
- **Exact Match:** Group identical hashes → keep one representative.  
- **C4 Example:** Used **exact match on 3-sentence spans** for deduplication.  
  ➜ High precision, zero recall for near duplicates.


### 3.2 🌸 Bloom Filters

A **probabilistic structure** for efficient membership testing.

- **Logic:**  
  - “No” → definitely not in set.  
  - “Yes” → probably in set (small false positive chance).  
- **Structure:** Memory-efficient **bit array**; multiple hash functions (K).  
- **False Positive Rate (f):**  
  Lowered exponentially by tuning **bins (M)** and **hash count (K)**.  
- **Example:** **DOLMA** used Bloom filters for **paragraph-level deduplication** with **f ≈ 10⁻¹⁵**.


### 3.3 🌀 Approximate Deduplication (MinHash & LSH)

Removes *near duplicates* using similarity-based hashing.

#### 🧮 Jaccard Similarity
$[
J(A,B) = \frac{|A ∩ B|}{|A ∪ B|}
]$
Near duplicates typically have **J ≥ 0.9**.

#### 🔢 MinHash
- Converts a set into a compact hash signature.  
- **Property:**  
  $( P[h(A)=h(B)] = J(A,B) )$  
- Compute by hashing all elements and taking the **minimum hash**.

#### 🧭 Locality Sensitive Hashing (LSH)
Improves MinHash by controlling similarity thresholds.

- **Construction:** Divide N hash functions into **B bands** × **R rows** each (N = B×R).  
- **Collision Rule:** Two docs collide if *any band’s R hashes* match exactly.  
- **Tuning:**
  - ↑R → stricter (higher similarity required)  
  - ↑B → looser (more matches)

#### 🔮 Future Directions
- Replace n-grams with **embedding-based similarity** for semantic deduplication.  
- Tradeoff: **Higher compute cost**, **more robust near-duplicate detection**.


## ✅ Summary

Filtering and deduplication are **core to LLM data quality**:
- Filtering ensures **linguistic, topical, and ethical quality**.
- Deduplication ensures **efficiency and privacy**.
- Together, they transform chaotic web data into **structured, scalable training corpora**—the invisible backbone of modern LLMs.
