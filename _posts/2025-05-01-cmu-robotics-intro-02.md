---
layout: post
title: Robot Learning Overview
subtitle: Robot Learning Lecture 2
categories: CMU-Robot-Learning-2024
tags: [robot]
banner: "/assets/images/banners/yuanpang-wa-iceburg2.jpg"
---

# 🧠 Robot Learning Overview

This lecture provides a **comprehensive introduction to Robot Learning**, a field focused on achieving **embodied intelligence in the physical world**. The key challenge lies in the integration of **algorithms, data, computation, and hardware** to allow robots to perform tasks that are easy for humans but hard for machines — a phenomenon known as **Moravec’s Paradox**.

[Course link](https://16-831-s24.github.io/)

The course emphasizes that although there has been significant progress (e.g., DRL in simulation, imitation learning), **unifying perception and control** remains a major challenge. Topics include:
- Machine/Deep Learning foundations
- Reinforcement Learning (Model-Free, Model-Based, Offline)
- Imitation Learning
- Sim2Real, Safe RL, Multi-Task Learning
- Use of **LLMs** in robotics

---

# 🔑 Key Themes and Concepts

## 2.1 🤖 Defining Robot Learning and Its Goal

- **Embodied Intelligence**: Intelligence that is situated in a body and interacts with the physical world.
- **Interdisciplinary Nature**: Combines *Algorithm, Data, Computation, and Hardware*.

## 2.2 🌀 Moravec's Paradox and Its Implications

- **Paradox**: "The hard problems are easy, and the easy problems are hard."
- **Not Just Hardware**: Even cheap robots can succeed with human control. The real challenge is smarter algorithms.
- **Tightly Coupled Components**: Real-world robotics requires closed-loop *Perception–Planning–Control–Actuation*.
- **High Dimensionality**: Both inputs (vision, sensors) and outputs (motor control) are high-dimensional.
- **Current State**: Many failures still exist in tasks that are easy for humans (e.g., DARPA 2015, Mobile ALOHA).

## 2.3 🔍 Current State of Robot Learning

Robot learning methods span from simulated training to real-world deployment, from trial-and-error learning to data-efficient offline strategies. Below is a structured comparison of each major method and example systems illustrating their current progress and challenges.


### 🧪 2.3.1. Simulation-Based Learning

Simulation provides a scalable and safe platform for robot learning without the cost and risk of real-world experiments.

#### ✅ Strengths:
- Fast, parallelizable training.
- Full observability and perfect resets.
- Ideal for prototyping and reward shaping.

#### ⚠️ Limitations:
- Sim-to-real transfer gap.
- Incomplete physics fidelity (especially contacts, sensors).

#### 🔍 Notable Examples:

- **GT Sophy**  
  - **Method**: Distributed Deep RL + engineered rewards.  
  - **Task**: Overtakes human drivers in *Gran Turismo*.  
  - **Innovation**: Scalable platform with real-time racing policies.  
  - **Key Insight**: Emergent tactical behavior purely from trial-and-error.  
  - 📄 [Sony AI - GT Sophy](https://ai.sony/blog/peeking-under-the-hood-of-gt-sophy)

- **Eureka**  
  - **Method**: Uses LLMs (e.g., GPT-4) to auto-generate reward functions.  
  - **Task**: Pen spinning via robot hand in sim.  
  - **Key Insight**: LLM-generated rewards outperformed manual ones on 83% of tasks.  
  - 📄 [Eureka: LLM-based Reward Design](https://arxiv.org/abs/2310.12931)

---

### 🔀 2.3.2. Bridging Simulation and Reality

#### 🧱 A. Sim2Real
Train in simulation and deploy in real-world directly.

- **DeepMind 1-1 Soccer**  
  - Simulated training of agents that exhibit teamwork and locomotion.  
  - Shows promise of sim-trained policies in real-world environments.

#### 🔁 B. Sim2Real + Adaptation
Adapt policies post-deployment to handle sim-to-real discrepancy.

- **DATT (Deep Adaptive Trajectory Tracking)**  
  - **Approach**: DRL + L1 adaptive control.  
  - **Task**: Quadrotor trajectory tracking in wind.  
  - **Result**: Stable adaptation to real-world disturbances.  
  - 📄 [DATT - CoRL 2023](https://proceedings.mlr.press/v229/huang23a/huang23a.pdf)

#### 🔄 C. Real2Sim2Real
Learn simulators from real data → train DRL policies → deploy.

- **Status**: An emerging technique. Key for accurate sim construction in contact-rich tasks.  
- **Use case**: Tasks like rope manipulation where real dynamics are hard to model manually.

### 📚 2.3.3. Learning from Data

#### 🤖 A. Imitation Learning
Learn from expert demonstrations (often teleoperation).

- **Mobile ALOHA**  
  - **Method**: Diffusion Policy trained from teleop demos.  
  - **Task**: Multi-task manipulation.  
  - **Limitation**: Still fails on "simple" tasks due to perception–control integration.  
  - 📄 [Diffusion Policy](https://arxiv.org/abs/2303.04137)

#### 🧠 B. Model-Based Reinforcement Learning
Learn a model of the environment, then plan/control within it.

- **Whipping a Rope**  
  - Learned delta-dynamics model to control complex deformable object.  
- **Meta-learned dynamics**  
  - Adaptive control across tasks/environments.
- **RoboCook (CoRL 2023)**  
  - **Method**: GNN-based structured world models.  
  - **Task**: Cook with tools and deformable objects.  
  - **Achievement**: CoRL Best Systems Paper Award.  
  - 📄 [RoboCook - Shi et al.](https://arxiv.org/abs/2306.14447)

#### 💾 C. Offline RL
Learn policies from pre-collected datasets (non-interactive).

- **Difference from Imitation**: Can use non-expert and mixed-quality data.  
- **Speaker**: Aviral Kumar (CMU, expert in Offline RL).  
- **Use case**: Robotics with limited data budgets or safety constraints.  
  - 📄 [Offline RL Survey](https://arxiv.org/abs/2005.01643)

#### 🔄 D. Model-Free RL
Direct trial-and-error in environment (e.g., Q-learning, PPO, SAC).

- **Techniques**:  
  - Policy Gradient (REINFORCE)  
  - Actor-Critic (A2C, DDPG, SAC)  
  - Value-based (DQN, Q-learning)  
- **Advantage**: Simplicity, no model learning required.  
- **Drawback**: Data-hungry, unstable in real-world due to safety concerns.


### 🧠 2.3.4. Specialized Topics

#### 🔐 A. Safe RL
Ensure policies avoid collisions and remain stable under uncertainty.

- Applications: Surgical robotics, autonomous driving, assistive robots.

#### 🌐 B. Multi-Task / Adaptive / Transferable RL
Learn generalizable policies across tasks or robot morphologies.

- Methods: Goal-conditioned RL, meta-RL, domain adaptation.

#### 📦 C. LLMs / VLMs for Robotics
Leverage foundation models to boost high-level planning and low-level control.

- **VoxPoser**: Uses LLMs for task planning from human instructions.  
- **Eureka**: GPT-4 generates reward functions for RL.  
- **Speaker**: Yafei Hu (CMU) on Foundation Models in Robotics.  
  - 📄 [Survey & Meta-Analysis](https://arxiv.org/abs/2311.05869)

### 📊 Method Comparison Summary

| Method                  | Real-World Use | Data Need  | Sim Dependence | Safety | Generalization |
|------------------------|----------------|------------|----------------|--------|----------------|
| Simulation DRL         | ❌             | 🔋 Low     | ✅ High         | ✅     | ❌             |
| Sim2Real + Adaptation  | ✅ Partial     | 🔋 Medium  | ✅ High         | ⚠️     | ⚠️             |
| Real2Sim2Real          | ✅ (in progress) | 🔋 High    | 🔁 Built        | ✅     | ✅             |
| Imitation Learning     | ✅             | 💽 Demo-heavy | ⚠️ Medium     | ✅     | ⚠️             |
| Model-Based RL         | ✅             | 🔋 Efficient | ⚠️ Medium     | ✅     | ✅             |
| Offline RL             | ✅             | 💽 Pre-collected | ❌          | ✅     | ⚠️             |
| Model-Free RL (Online) | ✅ (rare)      | 🔋 Huge    | ❌             | ⚠️     | ⚠️             |
| LLMs/VLMs in Robotics  | ✅ Emerging    | 📦 Pretrained | ❌            | ✅     | ✅             |


Robot learning today is a diverse and rapidly evolving field, striving to unify perception and control in dynamic environments through learning-based approaches. Each method contributes unique strengths—and when combined, they edge us closer to general-purpose embodied intelligence.


## 2.4 📚 Course Structure and Core Methodologies

- **ML/DL Refresher**: CNN, RNN, Transformers, optimization, uncertainty.
- **Imitation Learning**: Learn from expert behavior.
- **Model-Free RL**: Q-learning, Policy Gradient, Actor-Critic.
- **Model-Based RL**: Learn dynamics models + use LQR, iLQR, MPC.
- **Offline RL**: Avoid unsafe exploration; includes Inverse RL.
- **Bandits & Exploration**: Simpler exploration-exploitation setups.
- **Special Topics**:
  - 🛡️ Safe Robot Learning
  - 🔄 Multi-task, Adaptive Learning
  - 🕹️ Simulation and Sim2Real
  - 🤖 LLMs/VLMs for high/low-level robot control (e.g., VoxPoser, Eureka)

## 2.5 🌐 Broader Applications of Sequential Decision-Making

- Finance & Trading
- Cluster Management
- Plasma Control (Nuclear Fusion)
- RLHF for LLMs

