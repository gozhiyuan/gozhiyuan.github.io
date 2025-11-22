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


## 1️. Policy Function Approximation

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
   | **State** | **Left** | **Right** | **Up** |
   |------------|-----------|-----------|---------|
   | $s_1$ | 0.3 | 0.4 | 0.3 |
   | $s_2$ | 0.1 | 0.2 | 0.7 |

   But in real-world problems (like robotics or video games), states are high-dimensional and continuous — impossible to store in a table.

2. **Scalability Challenge:**  
   When the number of states or actions grows large (or infinite), tabular methods break down.

3. **Neural Network Solution:**  
   To handle large spaces, we use a **Policy Network** $\pi(a \mid s; \theta)$ parameterized by trainable weights $\theta$.  
   The network directly outputs the probabilities of all possible actions given state $s$.


### 🧠 Policy Network Architecture

- **Input:** State $s$ (e.g., image, sensor readings, position vector).  
- **Hidden Layers:** Convolutional or dense layers to extract features.  
- **Output:** Action probabilities via a **softmax layer**, ensuring:  

  $$
  \sum_{a \in A} \pi(a \mid s; \theta) = 1
  $$

Example:  
If the input is a game screenshot, the network might output:

| **Action** | **Probability** |
|-------------|-----------------|
| Left        | 0.15 |
| Right       | 0.10 |
| Jump        | 0.75 |

The agent then randomly samples an action from this distribution (favoring “Jump”).



## 2️. Policy Objective and Policy Gradient

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

## 3. Understanding the Policy Gradient Objective and Approximate State-Value Function

### 1️⃣ The Approximate State-Value Function $V(s; \theta)$

In **Policy-Based Reinforcement Learning**, we use a **policy network** $\pi(a|s; \theta)$ to represent the probability of taking an action $a$ given a state $s$, where $\theta$ are the trainable parameters.

The **approximate state-value function** is defined as:
$$
V(s; \theta) = \sum_a \pi(a|s; \theta) \, Q_\pi(s, a)
$$

This represents the **expected return** starting from state $s$, following the stochastic policy $\pi(\cdot|s; \theta)$ thereafter.

- The term $\pi(a|s; \theta)$ gives the **probability** of each possible action.
- The term $Q_\pi(s, a)$ gives the **expected reward** from taking that action.
- Their product and summation capture the **expected value** over all possible actions.

Hence, $V(s; \theta)$ measures how good the current policy (parameterized by $\theta$) is when starting from state $s$.


### 2️⃣ The Objective Function $J(\theta)$ and Its Expectation

The **objective function** $J(\theta)$ is defined as the **expectation** of the state-value function:
$$
J(\theta) = \mathbb{E}_{S \sim p_\pi(S)}[V(S; \theta)]
$$

Here:
- $S$ represents states sampled from the **state distribution** under the current policy.
- The expectation $\mathbb{E}[V(S; \theta)]$ measures the **average performance** of the policy over all states it encounters.

So, maximizing $J(\theta)$ means **maximizing the expected long-term return** of the policy.

Because it’s an expectation, we can only compute a **sampled estimate** during training — this is what makes it a **stochastic gradient**.


### 3️⃣ Why We Take the Gradient of $J(\theta)$

To improve the policy, we perform **gradient ascent** on $J(\theta)$:
$$
\theta \leftarrow \theta + \beta \, \nabla_\theta J(\theta)
$$

where:
- $\beta$ is the learning rate,
- $\nabla_\theta J(\theta)$ is the **policy gradient** — the direction in parameter space that most increases the expected return.

This is analogous to climbing a hill — each update nudges $\theta$ uphill toward higher rewards.


### 4️⃣ Deriving the Policy Gradient from $V(s; \theta)$

Starting with:
$$
V(s; \theta) = \sum_a \pi(a|s; \theta) \, Q_\pi(s, a)
$$

Taking the derivative with respect to $\theta$ gives:
$$
\nabla_\theta V(s; \theta) = \sum_a \nabla_\theta \pi(a|s; \theta) \, Q_\pi(s, a)
$$

- The term $\nabla_\theta \pi(a|s; \theta)$ measures how the policy probabilities change when $\theta$ changes.
- The term $Q_\pi(s, a)$ acts as a **weight** — actions with high Q-values will push the gradient stronger in their direction.

Using the **logarithmic trick**:
$$
\nabla_\theta \pi(a|s; \theta) = \pi(a|s; \theta) \, \nabla_\theta \log \pi(a|s; \theta)
$$

we can rewrite the gradient as:
$$
\nabla_\theta V(s; \theta) = \mathbb{E}_{A \sim \pi(\cdot|s; \theta)} [\nabla_\theta \log \pi(A|s; \theta) \, Q_\pi(s, A)]
$$

This is the **Policy Gradient Theorem**, and it forms the mathematical foundation for algorithms like **REINFORCE** and **Actor-Critic**.


### 5️⃣ Stochastic Policy Gradient Estimate (Practical Form)

In practice, we approximate this expectation using **samples**:
$$
g(a_t, \theta_t) = \nabla_\theta \log \pi(a_t|s_t; \theta_t) \, Q_\pi(s_t, a_t)
$$

This gives an **unbiased stochastic estimate** of the true gradient.  
The policy parameters are then updated as:
$$
\theta_{t+1} = \theta_t + \beta \, g(a_t, \theta_t)
$$

- In **REINFORCE**, $Q_\pi(s_t, a_t)$ is replaced by the **observed discounted return** $u_t$.
- In **Actor–Critic**, $Q_\pi(s_t, a_t)$ is approximated by a **critic network** that learns via TD learning.


### 6️⃣ Why $Q_\pi(s,a)$ Appears in the Policy Gradient

The Q-function serves as a **score multiplier**:
- If $Q_\pi(s,a)$ is large → increase the probability $\pi(a|s)$.
- If $Q_\pi(s,a)$ is small or negative → decrease $\pi(a|s)$.

Thus, $Q_\pi(s,a)$ directly determines how strongly the policy should favor or avoid certain actions.

This ensures that the policy gradually shifts probability mass toward **actions that yield higher expected returns**, leading to improved behavior over time.

### 🧭 Summary

| **Concept** | **Meaning** | **Formula** |
|--------------|-------------|--------------|
| **Policy Network** | Defines action probabilities | $\pi(a \mid s; \theta)$ |
| **Approx. State Value** | Expected return under $\pi$ | $V(s; \theta) = \sum_a \pi(a \mid s; \theta) \, Q_\pi(s, a)$ |
| **Objective Function** | Expected performance | $J(\theta) = \mathbb{E}[V(S; \theta)]$ |
| **Policy Gradient** | Direction to improve policy | $\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi(A \mid S; \theta) \, Q_\pi(S, A)]$ |
| **Stochastic Estimate** | Sample-based update | $g(a_t, \theta_t) = \nabla_\theta \log \pi(a_t \mid s_t; \theta_t) \, Q_\pi(s_t, a_t)$ |



### 🧮 Two Forms of the Policy Gradient — From Summation to Expectation

The **policy gradient** is the derivative of the approximate state-value function $V(s; \theta)$ with respect to the policy parameters $\theta$.  
It is fundamental because Policy-Based Reinforcement Learning aims to maximize the **objective function** $J(\theta) = \mathbb{E}[V(S; \theta)]$ using **gradient ascent**.

There are **two key mathematical forms** of the policy gradient, derived from the same principle but applied differently depending on whether the **action space** is **discrete** or **continuous**.


#### 🧩 Form 1: The Summation Form (Derivative of $V(s; \theta)$)

This is the **original, exact formulation** — it explicitly sums over all possible actions in the discrete action set $\mathcal{A}$.

$$
V(s; \theta) = \sum_{a \in \mathcal{A}} \pi(a|s; \theta) \, Q_\pi(s, a)
$$

Taking the derivative with respect to $\theta$:
$$
\nabla_\theta V(s; \theta) = \sum_{a \in \mathcal{A}} \nabla_\theta \pi(a|s; \theta) \, Q_\pi(s, a)
$$

This shows:
- The **gradient** depends on how the policy probability $\pi(a|s; \theta)$ changes when $\theta$ changes.
- The **Q-function** acts as a **weight** — if an action yields a higher $Q_\pi(s, a)$, its probability gets reinforced more strongly.

🧠 **Interpretation**:
Each action’s contribution to the gradient is proportional to both how much the policy changes and how valuable the action is.

⚠️ **Limitation**:
- The summation form is **only practical for discrete action spaces**, where enumerating all actions is feasible.
- In continuous control (e.g., steering angles, torques), the action set $\mathcal{A}$ is infinite, so direct summation is impossible.

#### Form 2: The Expectation Form (Policy Gradient Theorem)

To generalize for continuous actions and enable **sample-based learning**, we use the **log-derivative trick**:
$$
\nabla_\theta \pi(a|s; \theta) = \pi(a|s; \theta) \, \nabla_\theta \log \pi(a|s; \theta)
$$

Substituting into the previous formula:
$$
\nabla_\theta V(s; \theta) 
= \sum_{a \in \mathcal{A}} \pi(a|s; \theta) \, \nabla_\theta \log \pi(a|s; \theta) \, Q_\pi(s, a)
$$

This can be expressed compactly as an **expectation**:
$$
\nabla_\theta V(s; \theta) 
= \mathbb{E}_{A \sim \pi(\cdot|s; \theta)} \big[ \nabla_\theta \log \pi(A|s; \theta) \, Q_\pi(s, A) \big]
$$

🎯 **Advantages of the Expectation Form**:
- Works for **continuous** and **discrete** actions.
- Allows **sampling-based estimation** — no need to enumerate all actions.
- Enables **stochastic gradient ascent** (used in algorithms like REINFORCE and Actor–Critic).


#### Practical Stochastic Estimate (Sample-Based Update)

In practice, we sample **one action** $a_t$ from the policy distribution $\pi(\cdot|s_t; \theta_t)$ and compute:

$$
g(a_t, \theta_t) = \nabla_\theta \log \pi(a_t | s_t; \theta_t) \, Q_\pi(s_t, a_t)
$$

This $g(a_t, \theta_t)$ is an **unbiased stochastic estimate** of the true gradient.

Then we perform a **gradient ascent update**:
$$
\theta_{t+1} = \theta_t + \beta \, g(a_t, \theta_t)
$$

Here:
- $\beta$ is the learning rate.
- $Q_\pi(s_t, a_t)$ can be approximated by:
  - The **discounted return** $u_t$ (Monte Carlo / REINFORCE), or
  - The **critic’s estimate** (in Actor–Critic methods).


#### 🔍 Summary Comparison

| **Form** | **Expression** | **Works For** | **Description** |
|-----------|----------------|----------------|------------------|
| **Summation Form** | $\nabla_\theta V(s; \theta) = \sum_a \nabla_\theta \pi(a \mid s; \theta) \, Q_\pi(s, a)$ | Discrete actions | Exact but computationally expensive for large action spaces |
| **Expectation Form** | $\nabla_\theta V(s; \theta) = \mathbb{E}_{A \sim \pi(\cdot \mid s; \theta)} [\, \nabla_\theta \log \pi(A \mid s; \theta) \, Q_\pi(s, A) \,]$ | Discrete + Continuous | Enables stochastic sampling and gradient ascent |
| **Sampled Estimate** | $g(a_t, \theta_t) = \nabla_\theta \log \pi(a_t \mid s_t; \theta_t) \, Q_\pi(s_t, a_t)$ | Both | Used in REINFORCE and Actor–Critic updates |


## 4. The Policy Gradient Algorithm

Because computing the exact expectation $( E[\cdot] )$ is often infeasible, Policy-Based RL uses **sampling** to estimate gradients — this leads to **stochastic policy gradient** algorithms. The **Expectation Form** of the policy gradient is a **theoretical identity**, not an algorithm by itself.  
It expresses the gradient of the expected return as an expectation over actions sampled from the policy:

$$
\nabla_\theta V(s; \theta) 
= \mathbb{E}_{A \sim \pi(\cdot|s; \theta)} 
\big[ \nabla_\theta \log \pi(A|s; \theta) \, Q_\pi(s, A) \big]
$$

- The formula tells us *what direction* to move in parameter space to improve the policy.
- It assumes we could perfectly evaluate $Q_\pi(s, a)$ for every state and action — which we cannot do directly.
- Therefore, in practice, we **approximate this expectation** by *sampling trajectories* and *estimating returns*.

The **Monte Carlo (MC) process** is one **way to approximate** the expectation above.


### 🔄 Single-Step Update Process

At each timestep $t$:

1. **Observe State:** $s_t$  
2. **Sample Action:** $a_t \sim \pi(\cdot \mid s_t; \theta_t)$  
3. **Estimate $Q_\pi$:** Compute an estimate $q_t \approx Q_\pi(s_t, a_t)$  
4. **Compute Log-Gradient:**  
   $$
   d_{\text{log}, t} = \frac{\partial \log \pi(a_t \mid s_t; \theta)}{\partial \theta}
   $$
5. **Compute Gradient Estimate:**  
   $$
   g(a_t, \theta_t) = q_t \cdot d_{\text{log}, t}
   $$
   This is an **unbiased estimate** of the true policy gradient.  
6. **Update Parameters:**  
   $$
   \theta_{t+1} = \theta_t + \beta \, g(a_t, \theta_t)
   $$

This process updates the policy so that actions with higher estimated rewards become **more likely** in the future.





## 45. Methods for Estimating $Q_\pi(s, a)$

The way we approximate $Q_\pi(s, a)$ defines different **policy gradient algorithms**:


### 🧩 Option 1: REINFORCE (Monte Carlo Method)

Since $Q_\pi(s, a)$ is unknown, the REINFORCE algorithm replaces it with the **observed discounted return** from a sampled trajectory:
$$
u_t = \sum_{k=t}^H \gamma^{k-t} r_k
$$

Then we estimate the gradient using a single sample:
$$
g(a_t, \theta_t) = \nabla_\theta \log \pi(a_t|s_t; \theta_t) \, u_t
$$

and perform the parameter update:
$$
\theta_{t+1} = \theta_t + \beta \, g(a_t, \theta_t)
$$

So, REINFORCE is a **Monte Carlo implementation** of the **Policy Gradient Theorem** —  
it computes the gradient empirically by playing full episodes and using their observed returns to approximate the true expectation.

Thus:
- The **update itself** still happens per timestep.  
- But the **information it uses** (the discounted return $u_t$) comes from the **entire sampled trajectory**. So you still need the **entire trajectory** to compute the reward for that single action.  
- This is why we say **REINFORCE is Monte Carlo** — because it waits until the episode ends to compute returns.

🧠 **In short:**  
> Sampling one action at a time is part of the update rule,  
> but collecting the **reward signal** from a full trajectory makes the method *Monte Carlo*.


### ⚖️ Option 2: Actor–Critic Method

Instead of waiting until the end of the episode, the **Actor–Critic** framework uses two neural networks:

| **Component** | **Role** | **Function** |
|----------------|-----------|--------------|
| 🎭 **Actor** | Policy Network | Selects actions via $\pi(a \mid s; \theta)$ |
| 🧮 **Critic** | Value Network | Estimates $Q_\pi(s, a)$ or $V_\pi(s)$ |

The Critic provides **instant feedback** to the Actor, stabilizing and accelerating training. So we no longer need to wait until the episode finishes — we can update **every timestep**. This makes Actor–Critic *faster* and less variable than Monte Carlo.

The update rule becomes:  
$$
\theta_{t+1} = \theta_t + \beta \, (r_t + \gamma V(s_{t+1}) - V(s_t)) \, \frac{\partial \log \pi(a_t \mid s_t; \theta)}{\partial \theta}
$$

This term $(r_t + \gamma V(s_{t+1}) - V(s_t))$ is the **Temporal Difference (TD) error**, indicating how much better or worse the outcome was than expected.


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
