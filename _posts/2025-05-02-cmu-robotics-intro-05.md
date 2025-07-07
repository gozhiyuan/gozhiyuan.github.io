---
layout: post
title: Markov Decision Processes (MDP) Basics and Imitation Learning
subtitle: Robot Learning Lecture 5
categories: CMU-Robot-Learning-2024
tags: [robot]
banner: "/assets/images/banners/yuanpang-wa-iceburg2.jpg"
---

# 🧠 Markov Decision Processes (MDP) Basics and Imitation Learning

This lecture provides a review of key themes and concepts related to Imitation Learning (IL) in the context of Robot Learning

[Course link](https://16-831-s24.github.io/)


## 🔑 Key Themes and Concepts

### 1. Markov Decision Processes (MDPs) and Partially Observed MDPs (POMDPs)

Sequential decision-making in robotics is formalized using MDPs and POMDPs:

- **State Space (S)**:  
  $S$ is the state space. $s_t \in S$ is the state at time step $t$.

- **Action Space (A)**:  
  $A$ is the action space. $a_t \in A$ is the action at time step $t$.

- **Observation Space (O)**:  
  $O$ is the observation space. $o_t \in O$ is the observation at time $t$.

- **Transition Probability (p)**:  
  $s_{t+1} \sim p(\cdot \mid s_t, a_t)$ — how the environment evolves given current state and action.

- **Observation Model (h)**:  
  $o_t \sim h(\cdot \mid s_t)$ — how observations are generated from states.

- **Reward Function (r)**:  
  $r: S \times A \rightarrow \mathbb{R}$ — evaluates the desirability of state-action pairs.

#### MDP vs. POMDP

- **MDP**: Tuple $(S, A, p, r)$.  
  Learn policy $\pi(a_t \mid s_t)$ using full state access.

- **POMDP**: Tuple $(S, A, O, p, h, r)$.  
  Learn policy $\pi(a_t \mid o_t)$ based only on observations.

- **Markov Property**:  
  Once current state is known, history can be discarded: future is conditionally independent of the past given the present.

- **Goal of RL/Control**:  
  Maximize cumulative reward: $\sum_t \gamma^t r(s_t, a_t)$ with $0 < \gamma \le 1$.  
  Can be over finite or infinite time horizons.

---

### 2. Introduction to Imitation Learning (IL)

- **Main Idea**:
  - Collect expert data: $(s_t, a_t)$ or $(o_t, a_t)$.
  - Train a function to map observations/states to actions.

- **Historical Context**:  
  Deep IL dates back to 1989 (CMU).  
  Modern examples: *Mobile ALOHA*, *Diffusion Policy*.

---

### 3. Imitation Learning vs. Supervised Learning (Behavior Cloning Challenge)

- **Behavior Cloning (BC)**:
  - Policy $\hat{a} = \pi(s_t)$ trained on expert data $(s_t, a_t)$.
  - Similar to supervised learning, e.g., image → steering direction.

- **Problem**:  
  “It is NOT standard supervised learning!”

![alt_text](/assets/images/robot-learning/05/1.png "image_tooltip")

#### Key Issues:

- **I.I.D. Assumption Violation**:
  - Test-time inputs come from the learned policy, not the expert.
  - Causes distribution mismatch and cascading errors.

- **Cascading Failures**:
  - Policy makes one mistake → enters unseen state → makes more mistakes.
  - Domain shift is *dynamic* and *fatal* in sequential tasks.

- **“Cliff Walking” Problem**:
  - Small mistake → off-distribution → unrecoverable.
  - Test error grows **quadratically** with time steps.
  - Paradox: IL works better with mistake-filled data!

---

### 4. Addressing Behavior Cloning Issues

#### Data Collection and Augmentation

- **Add Mistakes and Corrections**:
  - Train on examples that teach recovery behavior.
  - E.g., intentionally perturb expert actions, then correct.

- **Synthetic Data Augmentation**:
  - Fake recovery data (e.g., perturb MPC expert trajectories).

#### Algorithmic Fix: **DAgger (Dataset Aggregation)**

- **Core Idea**: Collect new expert labels where the learned policy fails.

- **Iterative Steps**:
  1. Train initial policy on expert data.
  2. Run policy to generate trajectories.
  3. Ask expert to relabel visited states.
  4. Aggregate new data with old.
  5. Train new policy and repeat.

- **Effectiveness**:
  - Learns from the **distribution induced by the learned policy**.
  - Reduces compounding errors.
  - Developed at CMU (2011), has **regret guarantees**.

- **Main Limitation**:
  - Requires frequent querying of the expert — can be expensive.

---

### 5. Future Directions (Teased in Lecture)

- **Use powerful models**: Reduce errors inherently.
- **Multi-task learning**: Shared structure across tasks improves generalization (Lecture 23).

---

### 6. Summary: Imitation Learning vs. Continuous Supervised Learning vs. Reinforcement Learning


#### 📘 Imitation Learning (IL)

Imitation Learning is a technique in sequential decision-making where the goal is to **mimic expert behavior** from demonstrations (e.g., \((s_t, a_t)\) pairs). The most common form is **Behavior Cloning (BC)**, which simply treats IL as a supervised learning problem.

#### 🔄 Is Imitation Learning Just Continuous Supervised Learning?

**Not quite. Here's the distinction:**

| Aspect | Continuous Supervised Learning | Imitation Learning |
|--------|-------------------------------|--------------------|
| **Training data** | Static or streaming i.i.d. samples | Sequences of expert trajectories |
| **Prediction mode** | Predict each output independently | Predict actions that affect future states |
| **Ambiguity** | Same input might map to multiple outputs; model may output an average | Averaging can be catastrophic in control (e.g., averaging left and right = crash) |
| **Test time behavior** | Model generalizes due to i.i.d. assumption | **Cascading errors** due to distribution shift |
| **Sequential impact** | Often not considered | **Critical** — each decision changes future states |


#### ⚠️ Why "Averaging" Fails in IL

In supervised learning, if the same input maps to different outputs during training, the model may learn to **output an average** (especially in regression).

#### 🧠 Example: Steering a Car

- Suppose the same camera image occurs in two contexts: expert turns left and expert turns right.
- A supervised model may learn to **steer straight** — the average of left and right — which could drive the car **off the road**.

> **Key Insight**: In IL, **only one** of the expert actions will result in a good outcome. Averaging can lead to **unsafe** or **unrecoverable** states.


#### 🧱 Sequential Nature & Cascading Errors

- In IL, actions influence **future states**.
- A small prediction error at time \( t \) can lead the agent into a state **never seen** in the training data (since the expert never made that mistake).
- This causes a **distribution shift** and leads to **cascading errors**.

#### 🧠 Example: Cliff Walking

- The agent walks along a cliff.
- A tiny mistake (1 pixel drift) pushes it off-path.
- Now it's in a region it was never trained on.
- **More errors** accumulate, and it quickly fails catastrophically.

> Test-time error can grow **quadratically** with the number of timesteps.


#### 🤝 Is Imitation Learning a Type of Reinforcement Learning?

**Not exactly**, but they are closely related and often used together.

| Aspect | Imitation Learning | Reinforcement Learning |
|--------|---------------------|-------------------------|
| **Learning signal** | Expert demonstrations (states + actions) | Rewards from interacting with environment |
| **Objective** | Mimic the expert policy $(\pi(a \mid s))$ | Maximize expected cumulative reward |
| **Exploration** | None or limited | Essential |
| **Data source** | Pre-collected demos | Generated through trial and error |
| **Training method** | Supervised learning on trajectories | Policy gradient, Q-learning, actor-critic, etc. |
| **Main challenge** | Distribution shift, cascading failures | Credit assignment, sparse/delayed rewards |


#### 🔧 Techniques That Bridge IL and RL

1. **DAgger (Dataset Aggregation)**:
   - Trains an initial policy on expert data.
   - Runs the policy, collects states where it makes mistakes.
   - Queries the expert for corrections and adds those to the dataset.
   - Retrains on the aggregated dataset.
   - **Fixes distribution shift** by continuously updating the training set to include the learner’s mistakes.

2. **Inverse Reinforcement Learning (IRL)**:
   - Learns a **reward function** from expert demonstrations.
   - Solves an RL problem using the inferred reward.

3. **Pretraining with IL + Fine-tuning with RL**:
   - Learn a reasonable initial policy with BC.
   - Fine-tune with RL to improve performance or achieve goals beyond the expert.


#### 🧠 Summary

- Imitation Learning is **not** standard supervised learning because actions have **sequential consequences**.
- Imitation Learning is **not** full RL because it lacks **reward-based learning** and **exploration**, but both operate in **sequential decision-making** settings.
- Simple behavior cloning **suffers from distribution shift** and cascading errors.
- Techniques like **DAgger** mitigate this by incorporating interaction and expert feedback.
- Powerful models and hybrid IL+RL approaches are increasingly used to overcome IL limitations.
