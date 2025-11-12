---
layout: post
title: The Crucial Role of Data in Training Language Models 💻
subtitle: Language Modeling from Scratch Lecture 13
categories: Large-Language-Model
tags: [Stanford-LLM-From-Scratch-2025]
banner: "/assets/images/banners/yuanpang-wa-iceburg2.jpg"
---

# The Crucial Role of Data in Training Language Models

This lecture highlights the **central role of data** in the development of language models, following previous discussions about architectures and training strategies. It dissects the **data pipeline**, explores **historical datasets**, and addresses **legal and ethical issues** surrounding data use.


[Course link](https://stanford-cs336.github.io/spring2025/)


## 🏗️ Introduction

While earlier lectures focused on *how* to train models (architecture, optimization, scaling laws), this one shifts attention to *what* data we train on.

### 🔥 Hot Take
- **Data is the most important ingredient** for building effective language models.
- Companies are often transparent about architectures but **secretive about datasets**, as seen with LLaMA 3’s vague reference to “a variety of sources up to end of 2023.”

### 🕵️ Reasons for Secrecy
- **Competitive advantage**
- **Legal liability**, especially regarding copyright

### ⚙️ Nature of Data Work
Even though foundation models rely less on annotation than classical supervised tasks, **curation and cleaning** remain labor-intensive.  
Data is a **long-tail problem**—scaling with human effort rather than compute.

### 🧬 Stages of Training
1. **Pre-training** – Massive raw text data (e.g., web crawls)  
2. **Mid-training** – Smaller curated datasets for skills (e.g., math, coding)  
3. **Post-training** – Instruction-following fine-tuning and safety alignment

A **base model** = pre-training + mid-training  
An **instruct/chat model** = post-training added on top

### 🧩 Data Pipeline Framework

Live Service (e.g., Reddit)
↓
Raw Snapshot (crawl/dump)
↓
Processed Text (filtering, cleaning)
↓
Aggregated Dataset (e.g., Dolma, The Pile)


## 📚 Pre-training Data Deep Dive

The lecture surveys key datasets used across model generations.

| Model & Year | Dataset | Description |
|---------------|----------|--------------|
| **BERT (2018)** | BooksCorpus, Wikipedia | 7k free ebooks (later removed) + Wikipedia articles. Highlighted document-level training. Vulnerable to data poisoning before dumps. |
| **GPT-2 (2019)** | WebText | Reddit links (>3 karma) → 8M pages, 40 GB text. Not released; OpenWebText is open replication. |
| **Common Crawl** | Web Crawl | Nonprofit monthly crawl since 2007. Produces WARC/WET files; HTML-to-text conversion (e.g., Trafilatura) crucial for quality. Mostly copyrighted. |
| **CCNet (Meta, 2019)** | Filtered Common Crawl | Dedup + language ID + Wikipedia-likeness filter. |
| **C4 / T5 (Google, 2019)** | Colossal Clean Crawled Corpus | Heuristics: remove profanity, short pages, or code. |
| **GPT-3 (2020)** | Common Crawl + Books + Wikipedia + WebText2 | 400 B tokens, filtered via quality classifier trained on high-quality sources. |
| **The Pile (EleutherAI, 2021)** | 22 domains | Open, curated mix: Common Crawl (jusText), PubMed, arXiv, Enron, Gutenberg, Books3 (later removed). |
| **Gopher (DeepMind, 2021)** | MassiveText | Manual filtering rules + SafeSearch toxicity filter. |
| **LLaMA (2022)** | CCNet + GitHub + C4 | Classifier trained on Wikipedia citations → 1.2 T tokens (RedPajama v1 replication). |
| **RefinedWeb (2023)** | Common Crawl | Advocated “web data is all you need” if filtered well; 15 T tokens. |
| **DOLMA (AI2, 2024)** | Common Crawl + Stack + S2 + Reddit | 3 T tokens; heuristic + toxicity filtering. |
| **DataComp-LM (2024)** | Filtered Common Crawl | 240 T token pool; 3.8 T baseline. Used GPT-4-style instruction data for quality classifier—return to model-based filtering. |
| **Nemotron-CC (NVIDIA, 2024)** | Common Crawl | 6.3 T tokens. Used LM scoring for educational value + synthetic rewriting for data enhancement. |


## ⚖️ Copyright and Legal Issues

Most internet data is **copyrighted**. Understanding the law is essential.

### 📜 Copyright Basics
- Protects *expression*, not *ideas*
- Duration: ~75 years
- Registration required to **sue**, not to **protect**

### 🪪 Legal Use of Copyrighted Work
1. **License** – e.g., explicit contracts, Creative Commons  
2. **Fair Use (Section 107)** – depends on:
   - **Purpose**: educational/transformative favored  
   - **Nature**: factual favored over creative  
   - **Amount**: small portion favored  
   - **Market Effect**: less harm to original creator favored  

### 🧩 Foundation Models & Copyright
- **Training itself may technically copy data**, violating copyright.
- **Transformative defense**: learning ideas, not expression.
- **Market effect**: LMs still threaten creators’ income.

### 🧾 Terms of Service (TOS)
Platforms (e.g., YouTube) can forbid scraping even if fair use applies.


## 🧮 Mid-training & Post-training

These stages refine specific capabilities and behaviors.

### 📖 Long Context
- Efficient to add during **mid-training** (not pre-training).  
- Uses **long documents**: PG-19, Proof-Pile.

### 🧠 Task Standardization
Efforts to unify NLP tasks into instruction-following templates:
- **Super-Natural Instructions (2022)** – 1,600+ tasks  
- **FLAN (2022)** – 1,800+ tasks, enabling zero/few-shot transfer

### 💬 Instruction Following & Chat
- **Synthetic Data Generation**
  - *Self-Instruct (Alpaca)* – use GPT-4 to create instruction pairs  
  - *Vicuna* – ShareGPT conversations  
  - *Baize* – self-chat loops  
- **Quality Improvements**
  - *Evol-Instruct* – harder questions  
  - *MAmmoTH2* – mined quiz QA pairs from web  
- **Human-Annotated Data**
  - *LLaMA 2 Chat* – 27k expert examples outperforming large open sets  
- **Distillation**
  - Proprietary LMs (e.g., GPT-4) often used despite TOS limits  
  - Open models (e.g., Mixtral, DeepSeek) now preferred for legality


## 🧾 Summary

### Key Takeaways
- **Data requires extensive work** — it doesn’t fall from the sky.
- The **pipeline** from raw to trainable text involves:
  - Crawling → Cleaning → Deduplication → Quality Filtering
- **Data quality** is the key differentiator among models.
- Major **challenges** remain in copyright, privacy, and transparency.
- Huge **opportunities** exist to make data curation *scientific* rather than heuristic.
