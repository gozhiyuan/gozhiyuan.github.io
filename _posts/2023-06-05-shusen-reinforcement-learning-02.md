---
layout: post
title: Value-Based Reinforcement Learning Foundations
subtitle:
categories: Reinforcement-Learning
tags: [YouTube]
banner: "/assets/images/banners/yuanpang-wa-valley.jpg"
---

# Value-Based Reinforcement Learning Foundations

Value-based reinforcement learning (RL) focuses on estimating how *valuable* it is to take a particular action in a given state — quantified as the **expected discounted future reward**.  
This approach underpins algorithms like **Q-learning** and **Deep Q-Networks (DQN)**.

## 1️⃣ Value-Based Reinforcement Learning Foundations

### 🧩 The Action-Value Function $( Q_\pi(s, a) )$

The **Action-Value Function** for a policy $( \pi )$, denoted $( Q_\pi(s_t, a_t) )$, is the **expected discounted return** given that the agent starts from state $( s_t )$, takes action $( a_t )$, and then follows policy $( \pi )$ thereafter.

$[
Q_\pi(s_t, a_t) = E[U_t \mid S_t = s_t, A_t = a_t]
]$

The **discounted return** $( U_t )$ is defined as:
$[
U_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \dots
]$
where $( \gamma )$ (gamma) is the *discount factor* that determines how much the agent values future rewards.

The expectation $( E[\cdot] )$ accounts for:
- **Future actions** $( A_{t+1}, A_{t+2}, … )$ drawn from the policy $( \pi )$.
- **Future states** $( S_{t+1}, S_{t+2}, … )$ drawn from the environment’s transition probability $( p(s'|s, a) )$.

#### 🧠 Example:
Imagine an agent playing *Super Mario*.  
If Mario is at position $( s_t )$ and chooses the action “jump,” then $( Q_\pi(s_t, \text{jump}) )$ is the *expected total score* (coins, enemies defeated, etc.) Mario can still accumulate if he continues playing according to policy $( \pi )$.


### 🌟 The Optimal Action-Value Function $( Q^*(s, a) )$

The **goal of value-based RL** is to find the **optimal** action-value function, $( Q^*(s, a) )$.

$[
Q^*(s_t, a_t) = \max_\pi Q_\pi(s_t, a_t)
]$

- **Definition:** $( Q^*(s, a) )$ represents the *maximum expected discounted return* achievable by taking action $( a )$ in state $( s )$, and then behaving optimally thereafter.
- **Significance:** No policy can outperform $( Q^*(s, a) )$; it represents the *best possible outcome*.
- **Optimal Action Selection:**
  $[
  a^* = \arg\max_a Q^*(s, a)
  ]$
  The optimal policy $( \pi^* )$ always chooses the action with the highest $( Q^*(s, a) )$.
- **Challenge:** $( Q^*(s, a) )$ is **unknown**, and directly estimating it for complex environments is computationally infeasible.

#### 🎮 Example:
If Mario’s $( Q^*(s, a) )$ values for the current frame are:
| Action | Q*(s,a) |
|--------|----------|
| Jump   | 2500     |
| Left   | 1000     |
| Right  | 3000     |
Then the optimal policy will choose **“Right”**, since it yields the maximum expected reward.


## 2️⃣ Deep Q-Network (DQN)

The **Deep Q-Network (DQN)** is a breakthrough solution that uses a **neural network** to *approximate* $( Q^*(s, a) )$.  
Introduced by DeepMind (2015), it allowed agents to play Atari games directly from pixels — achieving human-level performance.

### 🧮 Network Structure

- **Function Approximation:**  
  DQN uses a neural network $( Q(s, a; w) )$, parameterized by weights $( w )$, to approximate $( Q^*(s, a) )$.
- **Input:** The **state** $( s )$ (e.g., the current game screen).  
- **Output:** A vector of Q-values for all possible actions $( a \in A )$.

#### Example:
If the action space is `{left, right, jump}`, the network might output:
| Action | Q(s,a;w) |
|--------|-----------|
| Left   | 2000 |
| Right  | 1000 |
| Jump   | 3000 |

Here, the agent selects:
$[
a_t = \arg\max_a Q(s_t, a; w)
]$
→ **“Jump”**, since it gives the highest predicted value.

### 🧭 Exploration vs. Exploitation
To balance **exploration** (trying new actions) and **exploitation** (choosing the best-known action), DQN often uses an **ε-greedy policy**:
- With probability **ε**, choose a random action (explore).
- With probability **1 - ε**, choose $( \arg\max_a Q(s, a; w) )$ (exploit).


## 3️⃣ Temporal Difference (TD) Learning

**Temporal Difference (TD) Learning** is the key algorithm used to train the DQN.  
It updates the Q-function *incrementally*, based on new experience, without waiting for the episode to finish.

### 🔁 Recursive Identity

The return $( U_t )$ can be defined recursively:
$[
U_t = R_t + \gamma U_{t+1}
]$

Since $( Q(s_t, a_t; w) )$ approximates $( U_t )$, we have:
$[
Q(s_t, a_t; w) \approx R_t + \gamma Q(s_{t+1}, a_{t+1}; w)
]$

This forms the foundation for **bootstrapping** — updating current estimates using future estimates.

### 🎯 TD Target (Bootstrapping)

The **TD target** provides a more accurate, immediate estimate of the true return:
$[
y_t = r_t + \gamma \max_a Q(s_{t+1}, a; w)
]$

- $( y_t )$: the **target** (what the model should predict)
- $( Q(s_t, a_t; w) )$: the **current prediction**
- The term $( \max_a Q(s_{t+1}, a; w) )$ looks ahead one step to find the *best possible next action* — this makes it an **off-policy** update (Q-learning).

#### 🧠 Intuitive Example:
Suppose an agent in a maze gets:
- $( r_t = +10 )$ (found a key)
- $( \gamma = 0.9 )$
- Next state’s best action gives $( Q(s_{t+1}, a'; w) = 100 )$

Then:
$[
y_t = 10 + 0.9 \times 100 = 100
]$
If the current prediction $( Q(s_t, a_t; w) = 80 )$,  
the **TD error** is $( (80 - 100) = -20 )$,  
meaning the agent underestimated the value of this action.


### ⚙️ Training Algorithm

Each training step updates the neural network parameters $( w )$ using gradient descent on the **TD loss**:

1. **Observe & Predict:**  
   Predict $( q_t = Q(s_t, a_t; w_t) )$
2. **Environment Response:**  
   Observe reward $( r_t )$ and next state $( s_{t+1} )$
3. **Compute TD Target:**  
   $( y_t = r_t + \gamma \max_a Q(s_{t+1}, a; w_t) )$
4. **Compute Loss:**  
   $[
   L_t = \frac{1}{2}(q_t - y_t)^2
   ]$
5. **Update Weights:**  
   $[
   w_{t+1} = w_t - \alpha (q_t - y_t) \nabla_w Q(s_t, a_t; w_t)
   ]$
   where $( \alpha )$ is the learning rate.

This process repeats as the agent interacts with the environment, **gradually improving** its estimation of $( Q^*(s, a) )$.


## 🧭 Summary

| Concept | Description | Formula |
|----------|--------------|----------|
| **Q-function** | Expected discounted return for a state–action pair under policy $\pi$ | $Q_\pi(s, a) = E[U_t \mid S_t = s, A_t = a]$ |
| **Optimal Q-function** | Maximum achievable expected return | $Q^*(s, a) = \max_\pi Q_\pi(s, a)$ |
| **TD Target** | Bootstrapped update target for learning | $y_t = r_t + \gamma \max_a Q(s_{t+1}, a; w)$ |
| **DQN Update Rule** | Gradient descent step | $w_{t+1} = w_t - \alpha \, (Q(s_t, a_t; w_t) - y_t) \, \nabla_w Q(s_t, a_t; w_t)$ |


### 💡 Intuition Recap

- **Value-based RL**: Learn *how good* each action is.
- **Q\***: The theoretical best possible score.
- **DQN**: Neural approximation of Q\* for complex environments.
- **TD Learning**: Bootstrapped training using short-term feedback.

Together, these components form the backbone of modern RL systems — from **Atari-playing agents** to **autonomous robots** learning through experience.
