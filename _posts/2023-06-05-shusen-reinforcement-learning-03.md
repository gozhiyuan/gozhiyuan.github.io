---
layout: post
title: Policy-Based Reinforcement Learning
subtitle:
categories: Reinforcement-Learning
tags: [YouTube]
banner: "/assets/images/banners/yuanpang-wa-valley.jpg"
---

# 🧭 Policy-Based Reinforcement Learning — Directly Learning to Act

**Policy-Based Reinforcement Learning (RL)**, also known as **Policy Learning**, focuses on directly modeling and optimizing the agent’s **policy** $( \pi )$, i.e., the agent’s *behavior function*.  
This contrasts with **Value-Based RL** (like DQN), which indirectly learns the policy by estimating the optimal value function $( Q^*(s, a) )$.

This blog explains the foundations, mathematics, and algorithms of **Policy-Based RL**, including **REINFORCE** and **Actor–Critic** methods.


## 1️⃣ Policy Function Approximation

### 🎯 What Is a Policy?

The **policy function** $( \pi(a|s) )$ defines how an agent chooses actions based on the current state.  
It is a **probability distribution** over possible actions.

Example:
$[
\pi(\text{left}|s) = 0.2, \quad \pi(\text{right}|s) = 0.1, \quad \pi(\text{up}|s) = 0.7
]$
When the agent is in state $( s )$, it randomly samples an action $( A \sim \pi(\cdot|s) )$, meaning “up” is most likely to be chosen.


### ⚙️ Why We Need a Policy Network

1. **Simple Cases:**  
   If there are few states and actions, the policy can be stored in a lookup table.

   Example:  
   | State | Left | Right | Up |
   |--------|-------|-------|----|
   | s₁ | 0.3 | 0.4 | 0.3 |
   | s₂ | 0.1 | 0.2 | 0.7 |

   But in real-world problems (like robotics or video games), states are high-dimensional and continuous — impossible to store in a table.

2. **Scalability Challenge:**  
   When the number of states or actions grows large (or infinite), tabular methods break down.

3. **Neural Network Solution:**  
   To handle large spaces, we use a **Policy Network** $( \pi(a|s; \theta) )$ parameterized by trainable weights $( \theta )$.  
   The network directly outputs the probabilities of all possible actions given state $( s )$.


### 🧠 Policy Network Architecture

- **Input:** State $( s )$ (e.g., image, sensor readings, position vector).  
- **Hidden Layers:** Convolutional or dense layers to extract features.  
- **Output:** Action probabilities via a **softmax layer**, ensuring:
  $[
  \sum_{a \in A} \pi(a|s; \theta) = 1
  ]$

Example:  
If the input is a game screenshot, the network might output:
| Action | Probability |
|---------|--------------|
| Left | 0.15 |
| Right | 0.10 |
| Jump | 0.75 |

The agent then randomly samples an action from this distribution (favoring “Jump”).


## 2️⃣ Policy Objective and Policy Gradient

### 🎯 Objective Function

The goal is to find parameters $( \theta )$ that **maximize the expected performance** of the policy.  
This is expressed as:
$[
J(\theta) = E[V(S; \theta)]
]$

Here, $( V(s; \theta) )$ is the **expected value** of being in state $( s )$ under policy $( \pi(a|s; \theta) )$:
$[
V(s; \theta) = \sum_a \pi(a|s; \theta) \, Q_\pi(s, a)
]$


### 🧮 Policy Gradient Ascent

To maximize $( J(\theta) )$, we update parameters via **gradient ascent**:
$[
\theta \leftarrow \theta + \beta \frac{\partial V(s; \theta)}{\partial \theta}
]$
where $( \beta )$ is the learning rate (step size).

The **policy gradient** — the derivative of the expected return with respect to $( \theta )$ — is given by:
$[
\frac{\partial V(s; \theta)}{\partial \theta} = E_{A \sim \pi(\cdot|s; \theta)} \left[ \frac{\partial \log \pi(A|s; \theta)}{\partial \theta} \, Q_\pi(s, A) \right]
]$

This formula says:
- Sample actions $( A )$ from the current policy.
- Weight their gradients by their corresponding **Q-values** (how good that action was).


### 🔍 Intuition

The term $( \frac{\partial \log \pi(A|s; \theta)}{\partial \theta} )$ acts as a **directional signal** — telling the network how to adjust its parameters to make *good actions* more probable and *bad actions* less probable.

Example:  
If “Jump” yields high future reward, the network will increase $( \pi(\text{Jump}|s; \theta) )$.  
If “Duck” leads to losing points, the probability of “Duck” will decrease.


## 3️⃣ The Policy Gradient Algorithm

Because computing the exact expectation $( E[\cdot] )$ is often infeasible, Policy-Based RL uses **sampling** to estimate gradients — this leads to **stochastic policy gradient** algorithms.


### 🔄 Single-Step Update Process

At each timestep $( t )$:

1. **Observe State:** $( s_t )$
2. **Sample Action:** $( a_t \sim \pi(\cdot|s_t; \theta_t) )$
3. **Estimate $( Q_\pi )$:** Compute an estimate $( q_t \approx Q_\pi(s_t, a_t) )$
4. **Compute Log-Gradient:**  
   $[
   d_{\text{log},t} = \frac{\partial \log \pi(a_t|s_t; \theta)}{\partial \theta}
   ]$
5. **Compute Gradient Estimate:**  
   $[
   g(a_t, \theta_t) = q_t \cdot d_{\text{log},t}
   ]$
   This is an **unbiased estimate** of the true policy gradient.
6. **Update Parameters:**  
   $[
   \theta_{t+1} = \theta_t + \beta \, g(a_t, \theta_t)
   ]$

This process updates the policy so that actions with higher estimated rewards become **more likely** in the future.


## 4️⃣ Methods for Estimating $( Q_\pi(s, a) )$

The way we approximate $( Q_\pi(s, a) )$ defines different **policy gradient algorithms**:

### 🧩 Option 1: REINFORCE (Monte Carlo Method)

- The agent runs a full episode (e.g., from start to game over).  
- It collects the full trajectory:
  $[
  (s_0, a_0, r_0, s_1, a_1, r_1, …, r_H)
  ]$
- The **observed discounted return** $( u_t )$ serves as the Q-value estimate:
  $[
  q_t = u_t = \sum_{k=t}^{H} \gamma^{k-t} r_k
  ]$
- Update rule:
  $[
  \theta_{t+1} = \theta_t + \beta \, u_t \, \frac{\partial \log \pi(a_t|s_t; \theta)}{\partial \theta}
  ]$

🧠 *Intuition:* If an action led to a high final score, the network increases its probability.  
If it led to failure, it decreases it.


### ⚖️ Option 2: Actor–Critic Method

Instead of waiting until the end of the episode, the **Actor–Critic** framework uses two neural networks:

| Component | Role | Function |
|------------|------|-----------|
| 🎭 **Actor** | Policy Network | Selects actions via $( \pi(a|s; \theta) )$ |
| 🧮 **Critic** | Value Network | Estimates $( Q_\pi(s, a) )$ or $( V_\pi(s) )$ |

The Critic provides **instant feedback** to the Actor, stabilizing and accelerating training.

The update rule becomes:
$[
\theta_{t+1} = \theta_t + \beta \, (r_t + \gamma V(s_{t+1}) - V(s_t)) \, \frac{\partial \log \pi(a_t|s_t; \theta)}{\partial \theta}
]$

This term $( (r_t + \gamma V(s_{t+1}) - V(s_t)) )$ is the **Temporal Difference (TD) error**, indicating how much better or worse the outcome was than expected.


## 🧠 Summary and Insights

| Concept | Description | Formula / Idea |
|----------|--------------|----------------|
| **Policy Function** | Defines how actions are chosen | $\pi(a \mid s; \theta)$ |
| **Objective Function** | Expected value of the policy | $J(\theta) = E[V(S; \theta)]$ |
| **Policy Gradient** | Direction to improve policy | $E[\nabla_\theta \log \pi(A \mid S; \theta) \, Q_\pi(S, A)]$ |
| **REINFORCE** | Monte Carlo method using full-episode returns | $q_t = u_t$ |
| **Actor\text{–}Critic** | Uses separate networks for policy and value | TD-based updates |



### 💡 Key Takeaways

- **Policy-Based RL** directly optimizes behavior, rather than inferring it from value estimates.  
- It’s particularly powerful for:
  - Continuous action spaces (e.g., robotics control).
  - Stochastic environments.
  - Learning diverse behaviors via exploration.
- Algorithms like **REINFORCE** and **Actor–Critic** form the foundation of modern methods like **A2C**, **PPO**, and **SAC**.

> In essence, Policy-Based RL lets the agent **learn how to act directly**, discovering strategies that maximize long-term rewards — one gradient step at a time.
