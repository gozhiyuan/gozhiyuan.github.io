---
layout: post
title: What is Robot Learning
subtitle: Robot Learning Lecture 1
categories: CMU-Robot-Learning-2024
tags: [robot]
banner: "/assets/images/banners/yuanpang-wa-iceburg2.jpg"
---

# 🤖 *Introduction to Robot Learning (CMU 16-831)*

## I. 📘 Course Overview and Core Concepts

The **"16-831: Introduction to Robot Learning"** course, taught by **Professor Guanya Shi** at Carnegie Mellon University (CMU), focuses on the fundamental principles and applications of robot learning.

> **Theme**: *"Learning to make sequential decisions in the physical world"*

[Course link](https://16-831-s24.github.io/)

This concept is broken down into three components:

### 🔍 Learning
- Emphasizes **data-driven** approaches and **continuous improvement** through data.
- Contrasts with traditional methods like:
  - "Search & planning"
  - "Classic control"
  - "Optimal control"
  - which operate **"W/o learning & data"**

### 🔁 Sequential Decisions
- "The current action/decision influences the next state and the next action."
- Distinguishes from non-sequential problems like:
  - "Bandits"
  - "Standard supervised learning"

### 🌍 Physical World
- Requires **interaction in the closed-loop** — also called *embodied intelligence*.
- Contrasts with virtual domains (e.g., "RL for games", "LLMs") that **don’t involve physical interaction**.

---

## II. ⚠️ Uniqueness and Challenges of Robot Learning

Robot learning presents **unique challenges**, especially compared to **LLMs/GPTs** and **DRL in games** like AlphaGo.

### A. 🧠 Contrasting with LLMs/GPTs

LLMs rely on:
- **Architecture**: Transformer  
- **Data**: Web text, books, wikis  
- **Loss**: Next-token prediction  
- **Optimization**: SGD  
- **Generation**: Autoregressive

➡️ **Challenges in Robot Learning**:
- ❓ *Where is the data from?*  
- 📦 *How to use physical-world data?*  
- Data collection is **costly, slow, and task-specific**.

### B. 🎮 Contrasting with DRL in Games (e.g., AlphaGo, DQN)

| Aspect                      | Games 🕹️                       | Robotics 🤖                           |
|----------------------------|--------------------------------|--------------------------------------|
| Environment Dynamics       | Known & static                 | Unknown & dynamic                    |
| Task Scope                 | One specific task              | Many diverse tasks                   |
| Goal Definition            | Clear (reward function)        | Often unclear or implicit            |
| Learning Mode              | Offline suffices               | Requires online adaptation           |
| Action Speed               | Less constrained               | Real-time (e.g., 50Hz)               |
| Failure Tolerance          | Allow failures                 | Physics doesn’t forgive 💥           |

---

## III. 🎯 Goal and Current State of Robot Learning

### A. 🧠 Ultimate Goal: General-Purpose Embodied Intelligence

- *"Build general-purpose embodied intelligence by learning to make sequential decisions in the physical world."*
- Vision: Robots that can do **"thousands of tasks in thousands of environments"**
- Requires synergy in:
  - 📊 Algorithm & Data
  - 🧮 Computation
  - 🦾 Hardware

### B. 📉 Current Progress and Gaps

- Progress in **domain-specific intelligence**
- But **"still far from general-purpose embodied intelligence!"**

### C. ⚙️ Role of Learning vs. Non-Learning Methods

#### Examples of Non-Learning Success:
- 🚀 Apollo 11: *Optimal + Robust Control*
- 🦿 Boston Dynamics: *Trajectory Optimization + MPC*
- 🚜 Offroad Autonomy: *Sampling-based MPC*

#### Why Learning is Needed:
- 📉 Modeling is hard
- 🔁 Tasks/environments change
- 🧠 Policy space may be limited
- ❌ Optimizer could be wrong
- 🤯 Assumptions may not hold

Learning = **Tightly integrated and adaptive**, while traditional = **Modular and brittle**

---

## IV. 📚 Course Structure and Topics

![alt_text](/assets/images/robot-learning/01/1.png "image_tooltip")

### 📌 Topics Overview:
1. **Intro to Robot Learning** (Lectures 1–2)
2. **Machine/Deep Learning Refresher** (Lectures 3–4)
3. **Imitation Learning** (Lectures 5–6)
4. **Model-Free RL** (Lectures 7–12)
   - Q-Learning, Policy Gradient, etc.
5. **Model-Based RL** (Lectures 13–16)
6. **Bandits & Exploration** (Lectures 17–18)
7. **Offline RL** (Lecture 19)
8. **Special Topics**:
   - Inverse RL
   - Sim2Real
   - Safe RL
   - Multi-task & Adaptive RL (Lectures 20, 22–24)
9. **Challenges & Opportunities** (Lecture 25)

