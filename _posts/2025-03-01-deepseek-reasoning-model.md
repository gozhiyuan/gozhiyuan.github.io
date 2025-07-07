---
layout: post
title: DeepSeek Reasoning Models Series
subtitle: Reasoning Models
categories: blog
tags: [llm]
banner: "/assets/images/banners/yuanpang-wa-iceburg2.jpg"
---

# DeepSeek Reasoning Models Series

📌 In **Part Two**, we focus on DeepSeek’s **Reasoning Models**.
- [DeepSeek-Coder: When the Large Language Model Meets Programming - The Rise of Code Intelligence](https://arxiv.org/pdf/2401.14196)
- [DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence](https://arxiv.org/pdf/2406.11931)
- [MATH-SHEPHERD: VERIFY AND REINFORCE LLMS STEP-BY-STEP WITHOUT HUMAN ANNOTATIONS](https://arxiv.org/pdf/2312.08935)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/pdf/2402.03300)
- [DeepSeek-Prover: Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data](https://arxiv.org/pdf/2405.14333) and  
- [DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search](https://arxiv.org/pdf/2408.08152)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/pdf/2501.12948)


## 💻 [DeepSeek-Coder: When the Large Language Model Meets Programming - The Rise of Code Intelligence](https://arxiv.org/pdf/2401.14196)

**Main Theme:**  
DeepSeek Coder v1 is one of DeepSeek's early models, specifically designed for **code intelligence**. It marked the beginning of DeepSeek's focus on the *reasoning* aspect of large language models.

### 🔍 Breakdown of DeepSeek Coder v1:

- **Model Type:**  
  DeepSeek Coder v1 is a **dense model**, unlike later MoE-based DeepSeek models.

- **Base and Training Data:**  
  Its architecture closely mirrors DeepSeek LLM (a Llama 2 reproduction), but it is **primarily trained on code data** instead of general text.

- **Sizes Available:**  
  Released in sizes ranging from **1.3B to 33B** parameters.

- **Open-Source:**  
  All versions of DeepSeek Coder v1 were **open-source** ✅

### 🌟 Significance and Impact:

- DeepSeek Coder v1 helped build DeepSeek’s **early international reputation**.  
- While their English LLMs faced stiff competition, the **code models stood out** — especially in the **1.3B–30B range** — and became popular with developers due to solid performance.
- Code models like this are important for **real-world productivity** and **practical applications**.

- **Relation to DeepSeek Coder v1.5:**  
  DeepSeek Coder v1.5 was developed by **continued pre-training** of v1, further specializing the model on code.

🧩 Overall, DeepSeek Coder v1 set the stage for a broader trend of developing **code-focused LLMs** across the industry.


## 🧠 [DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence](https://arxiv.org/pdf/2406.11931)

**Main Theme:**  
DeepSeek Coder v2 represents a significant evolution, built specifically for **code intelligence** using a **Mixture-of-Experts (MoE)** design.

### 🔍 Breakdown of DeepSeek Coder v2:

- **Model Type and Architecture:**  
  - Transitions from **dense (v1)** to **MoE architecture**.
  - Built on **DeepSeek V2** (general MoE model).
  - Trained with an **additional 6 trillion tokens of code data**.

- **Training + Reward Models:**  
  - At the time, DeepSeek still employed **reward models** for reinforcement learning (RL) in code tasks.  
  - This contrasts with later approaches (e.g., DeepSeek R1), which moved to **rule-based feedback**.
  - Despite having access to ground-truth unit test feedback (0/1), they opted for reward models due to **noisy or incomplete test coverage**.
  - Experiments showed **reward model–based RL outperformed** rule-based or no-RL approaches at that time.

### 🔁 Relation to DeepSeek Coder v1:

- **Architecture:**  
  - v1: Dense (like Llama 2)  
  - v2: MoE (based on DeepSeek V2)

- **Training Data:**  
  - v1: Trained primarily on code from scratch  
  - v2: **Continued pre-training** from a general MoE base (DeepSeek V2) using more code data

- **Impact:**  
  - v1 built DeepSeek’s early code-model reputation, especially among developers.  
  - v2 continued the focus and evolved the architecture.

### 🌐 Industry Context:

DeepSeek’s work on Coder v2 fits into the broader LLM trend of releasing **code-specialized models** — a space where LLMs can **directly improve developer productivity**.

## 🧮 [MATH-SHEPHERD: VERIFY AND REINFORCE LLMS STEP-BY-STEP WITHOUT HUMAN ANNOTATIONS](https://arxiv.org/pdf/2312.08935)

**DeepSeek Math Shepherd**, often referred to by the speaker as "M Shepherd" or "SF," represents a significant early contribution from DeepSeek in the realm of **code and math intelligence**, specifically focusing on **reasoning**. It is distinct from the later **"DeepSeek Math"** model, which introduced the **GRPO** algorithm.

### 🎯 Purpose and Model Type

- **Goal:**  
  DeepSeek Math Shepherd was designed to develop a **process-based reward model** for **multi-step reasoning problems**, especially in mathematics.

- **Process-Based Judgement:**  
  Unlike models that only evaluate the **final answer**, this approach evaluates the **correctness of each intermediate step** in a multi-step solution.

### 🧠 Key Innovation: Automatic Label Construction

- A major challenge in process-based models is obtaining **step-by-step correctness labels**.

- While OpenAI’s *“Let’s Verify Step by Step”* paper (e.g., PRM800K with 800k human-labeled math solutions) used **human annotation**, DeepSeek innovated by generating these labels **automatically** without human help.

- **How It Works:**  
  If a specific step, when fixed, consistently leads to **correct final answers** across multiple completions, it is considered **correct**.  
  If it leads to **incorrect answers**, it’s deemed **incorrect**.

### 🧪 Improving Accuracy Through Sampling and Selection

- During inference, instead of generating one solution, the model samples **multiple outputs** (e.g., 64 solutions per math problem) 🔁

- The **reward model** then evaluates each solution to determine their step-by-step or outcome correctness.

- It selects the **best** solution based on reward scores — this **“best-of-N” strategy** boosts final accuracy ✅

- A graph shows that the **process-based model ("SF")** outperforms consistency checks and outcome-based models as the number of sampled solutions increases — a form of **early test-time scaling**.

![alt_text](/assets/images/deepseek/shephard.png "image_tooltip")

- This strategy — increasing inference-time compute to improve results — is now common in models like **O1** and **R1**.

### 🌍 Impact and Significance in the Community

- DeepSeek showed that **automatically generated labels** perform comparably to human-annotated ones.

- Their **process-based approach** outperformed consistency-based and outcome-based models in solution evaluation.

- **First Open-Source Process-Based Model:**  
  DeepSeek Math Shepherd was one of the **earliest** and **only open-source** process-based reward models in the field.

- As a result, many researchers used **Math Shepherd** directly in their own research.

### 🔄 Context and Evolution

- At its release (early 2024), the AI community still leaned toward **reward models** for reinforcement learning — even for tasks like code, despite available rule-based feedback.

- DeepSeek Coder V2 also used reward models, before later models (like DeepSeek R1) shifted to **rule-based feedback**.

- **Reward models** can suffer from **generalization issues** (e.g., trained on elementary math, but fails on complex math).  
  In contrast, **rule-based feedback** is **more robust but sparse**.

- DeepSeek Math Shepherd helped **refine understanding** of when and how to use reward models in reasoning tasks.

- Its techniques were later reused in models like **DeepSeek Math**, which adopted **Shepherd’s process-based reward model** for **GRPO training**.

## 🧮 [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/pdf/2402.03300)

**DeepSeek Math**, often referred to as "DeepSeek Math" or "dpsm" in the sources, is a significant specialized model from DeepSeek that focuses on **mathematical reasoning**. It is distinct from the earlier **"DeepSeek Math Shepherd" (M Shepherd)**, although it builds upon some of its concepts, especially regarding **process-based supervision for reward models**.

### 🎯 Purpose and Specialization

- DeepSeek Math is a **specialized model for mathematics**, designed to excel in **multi-step reasoning** problems in this domain.
- The `dpsm7b` model was considered the **best open-source math base model at its scale** for a long time.

### 💡 Key Innovation: GRPO Algorithm

- **GRPO** stands for **Generalized Reinforcement Learning with Policy Optimization**.
- It was introduced to replace **PPO (Proximal Policy Optimization)** due to PPO’s high cost and its requirement for a **separate value model**.
- GRPO eliminates the value model by:
  - Sampling multiple responses for each input
  - Assigning rewards to each
  - Using the **average reward as a baseline** to compute policy updates
- GRPO is **more efficient**, requiring less compute and memory.
- It became the **standard RL algorithm** for DeepSeek models like **V2, V3, and R1**, and was widely adopted across open-source RL frameworks.

![alt_text](/assets/images/deepseek/grpo.png "image_tooltip")

### 🔁 Online and Offline Training Strategies

#### 1️⃣ Offline Sampling/Training

- Data is generated once from a **fixed supervised fine-tuned (SFT)** model.
- Examples: **Rejection Sampling Fine-tuning (RFT)**, **Direct Preference Optimization (DPO)**
- Pros: Simple and resource-efficient  
- Cons: Data doesn't adapt to the model's evolving behavior

![alt_text](/assets/images/deepseek/rft.png "image_tooltip")

#### 2️⃣ Online Sampling/Training

- Data is generated in real-time from the **currently training model** (policy evolves)
- Examples: **PPO**, **GRPO**
- Pros: Adapts to recent model behavior, can lead to better performance  
- Cons: More resource-intensive, complex to stabilize, higher GPU requirements

DeepSeek acknowledged that **online training**, though expensive, leads to better generalization and performance — as seen in **Math Shepherd** and **DeepSeek Math**.

### 🧱 Base Model and Training Strategy

- DeepSeek Math (dpsm7b) is a **7B parameter model**
- Built via **continued pretraining** from `deepseek-coder-base-v1.5-7b`
- Trained with **120 billion math-specific tokens**
- Model is open-source, but **training data is not**

### 🔁 Evolution of Reward Modeling

- DeepSeek Math continues the **process-based reward modeling** approach from Math Shepherd
- Focus: Evaluate **each step** of a solution, not just the final answer
- Used this reward model for **GRPO training**
- Later, DeepSeek found that **rule-based feedback** (e.g., correct final answer or unit tests) is more robust for verifiable tasks like math and code
- This led to models like **DeepSeek R1** abandoning reward models for rule-based methods


### 🤔 Why RL Works (and Sometimes Doesn’t)

DeepSeek Math contains a very insightful section analyzing **why reinforcement learning works — or doesn't** — for reasoning tasks.

#### 🔍 Key Observations:

- **Pass@1 vs Pass@K (PK@1 vs PK@K):**
  - RL improves **PK@1** (most likely answer is correct)
  - But when sampling multiple answers (**K=8 or 16**), the improvement **disappears or reverses**
- **Conclusion:**  
  RL improves **ranking of known correct answers**, not necessarily **generation of new ones**
- This shows RL might **not fundamentally improve reasoning**, but helps pick better among what it already knows

#### 🧠 Scientific Rigor:

- DeepSeek was **honest and self-critical**, acknowledging that RL gains were **overstated**
- Suggested **reward model generalization** was a bottleneck
- This led to DeepSeek R1's shift toward **rule-based feedback** for verifiable tasks


### 🚀 Performance and Inference Strategy

- **DeepSeek Math 7B** outperforms **Llama 34B** models on many math benchmarks
- At inference:
  - Model samples multiple solutions (e.g., 64)
  - A reward model (like Math Shepherd) ranks them
  - The **best** solution is selected → improves accuracy
- This is **test-time scaling**: more compute = better results ✅

## 🧠 [DeepSeek-Prover: Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data](https://arxiv.org/pdf/2405.14333) and  
## [DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search](https://arxiv.org/pdf/2408.08152)

**DeepSeek Prover** is a specialized research effort by DeepSeek focused on **theorem proving** in large language models (LLMs). This task falls under the broader domain of **reasoning**, a major area of DeepSeek’s innovation beyond base models. While it may not have received the same attention as DeepSeek R1 or foundational models, DeepSeek Prover has played a vital role in shaping DeepSeek’s understanding of **reinforcement learning (RL)** and **reward models** for **verifiable reasoning** tasks.

### 🎯 Purpose

- DeepSeek Prover aims to **advance formal theorem proving** capabilities in LLMs.
- Theorem proving differs from general math problems by requiring proofs to be written in a **formal, verifiable language** (e.g., Lean).

### ✅ Key Characteristic — Verifiability

- The core feature of theorem proving (similar to coding and math) is that **correctness can be objectively verified** using external tools.
- For DeepSeek Prover, the **Lean proof verifier** provides this **binary (0/1) feedback**.
- This contrasts with **open-ended tasks** (e.g., writing or summarization) where such rule-based feedback is hard to define.

### 🔄 Methodology — Self-Improvement and Iteration

- The initial DeepSeek Prover used an **iterative self-refinement** strategy:
  - Generate proofs → check with Lean verifier → keep only correct ones → retrain
- This is similar to **distillation** and **self-improvement**.
- The success of this method contributed to DeepSeek’s realization that **online, adaptive training** (where data evolves with the model) is superior to static, offline datasets — a theme echoed in **DeepSeek Math**.

### 🚀 Evolution to DeepSeek Prover V1.5 — Incorporating RL

- **Released in August 2024**, DeepSeek Prover V1.5 marked the shift to **direct reinforcement learning**.
- Used **GRPO (Group Relative Policy Optimization)** — DeepSeek’s efficient online RL algorithm.
- Importantly:
  - **No learned reward model was used**
  - Instead, used **direct rule-based feedback** (from Lean verifier) as the reward signal
- This was a **strategic pivot** — acknowledging that in verifiable tasks, **rule-based feedback** is more robust and scalable than neural reward models.

### 🧭 Exploration of Complex Decoding (MCTS)

- DeepSeek Prover explored **Monte-Carlo Tree Search (MCTS)** to enhance inference:
  - MCTS = simulate multiple proof paths → search for best one
  - Though powerful, it added significant complexity
- Later models like **DeepSeek R1** would **abandon such complexity**, adopting the philosophy of **"大道至简"** ("great simplicity").


### 🧩 Significance in DeepSeek’s Research Journey

- Alongside **DeepSeek Coder**, DeepSeek Prover reflects DeepSeek’s **early commitment to reasoning** capabilities in LLMs.
- Key milestone in the evolution from:
  - Reliance on **learned reward models** (e.g., in DeepSeek Coder V2, DeepSeek Math)  
  ➡️ To  
  - Realization of the **strength of rule-based feedback** for verifiable tasks.
- DeepSeek Prover’s insights directly informed the **design of DeepSeek R1**, which **uses only rule-based rewards** for training on math and code.
- Though smaller in scope than base model work, DeepSeek Prover represents **critical academic rigor** and **foundational insight** into **effective RL for reasoning**.

## 🧠 [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/pdf/2501.12948)

**DeepSeek R1** is a flagship initiative by DeepSeek that aims to enhance the **reasoning capabilities** of large language models (LLMs) through **reinforcement learning (RL)**, following what is often called the **"O1 route"**. It marks a major step in DeepSeek's evolution toward effective RL-based training for tasks such as **math** and **code generation**.

### 🎯 DeepSeek R1: Incentivizing Reasoning Capability

**Philosophy:** DeepSeek R1 follows the principle of **"大道至简"** ("great simplicity") in applying RL, especially in **reward mechanisms**.

- **✅ Rule-Based Rewards:**  
  - For **verifiable tasks** like math and coding, DeepSeek R1 **does not use a learned reward model**.  
  - Instead, it uses **simple rule-based correctness checks**:
    - ✅ Math: check if the final boxed answer is correct  
    - ✅ Coding: check if the code passes unit tests using a compiler  

- **📉 Departure from Learned Rewards:**  
  - A major shift from earlier methods used in **DeepSeek Coder V2** and **Math Shepherd**, which relied on **process-based supervision** and neural reward models.  
  - DeepSeek found that:
    - Neural reward models were prone to **reward hacking**  
    - They increased training complexity and cost  
    - Rule-based feedback proved more **robust and scalable**

- **⚙️ Training Method - GRPO:**  
  - DeepSeek R1 uses **GRPO (Group Relative Policy Optimization)**, a lightweight, **online RL algorithm** first introduced in **DeepSeek Math**.  
  - GRPO eliminates the need for a separate value model, reducing cost and complexity.  
  - R1's base is **DeepSeek V3**, a large **Mixture-of-Experts (MoE)** model with **671 billion parameters**.

- **🧬 Multi-Stage Training with Cold Start:**  
  - Stage 1: Collect long **Chain-of-Thought (CoT)** examples to fine-tune DeepSeek-V3-Base  
  - Stage 2: RL training focused on reasoning  
  - Stage 3: Use **rejection sampling** to collect new SFT data  
  - Stage 4: Final RL using prompts from all task domains

- **📈 Performance:**  
  - **Math:** 79.8% Pass@1 on AIME 2024, 97.3% on MATH-500  
  - **Coding:** 96.3% percentile on Codeforces, 2029 Elo  
  - Strong on **MMLU**, **GPQA Diamond**, significantly beating DeepSeek V3

### 🧪 DeepSeek R1-Zero: Pure RL Without SFT

A foundational variant to test **pure reinforcement learning**, with **no supervised fine-tuning**.

- **🔁 Self-Evolution:**  
  - RL is applied **directly** to the DeepSeek-V3-Base  
  - Emerges with:
    - 🔍 **Self-verification**
    - 💭 **Reflection**
    - 🔗 **Long Chains of Thought**

- **💡 "Aha Moment":**  
  - Model learns to **spend more time thinking**, even adopting **anthropomorphic language**  
  - Demonstrates LLMs can self-learn **problem-solving strategies** via RL

- **📊 Performance Gains:**  
  - AIME 2024 Pass@1 jumped from **15.6% → 71.0%**, comparable to **OpenAI-o1-0912**

- **🚫 Drawbacks:**  
  - Poor readability  
  - Language mixing in reasoning steps  
  - These led to development of full **DeepSeek R1** with cold-start data

### 🔗 OpenAI O1 and Long Chain-of-Thought (CoT)

OpenAI’s **O1 models** pioneered the idea of **scaling reasoning by increasing CoT length**.

- **📊 Comparable Performance:**  
  - DeepSeek R1 matches **OpenAI-o1-1217** on many reasoning benchmarks

- **🔗 CoT Importance:**  
  - DeepSeek R1-Zero naturally generates **long reasoning chains**  
  - DeepSeek R1’s "cold-start" stage includes specially curated **long CoT examples**  
  - Emphasizes longer thinking = better reasoning

- **⚙️ Inference Simplicity:**  
  - Earlier models (e.g., DeepSeek Prover V1.5) used **complex decoding** like MCTS  
  - DeepSeek R1 chooses **simpler, more efficient decoding**, following the **"大道至简"** philosophy

![alt_text](/assets/images/deepseek/r1.png "image_tooltip")


### 🧠 Summary

**DeepSeek R1** and **R1-Zero** reflect DeepSeek’s commitment to advancing RL-based LLMs for reasoning by:

- Prioritizing **rule-based feedback** for verifiable tasks  
- Using efficient RL via **GRPO**  
- Encouraging long, emergent **Chains of Thought**  
- Avoiding unnecessary complexity in both training and inference

These models mark a significant step toward **scalable**, **interpretable**, and **powerful** reasoning in open LLMs.
