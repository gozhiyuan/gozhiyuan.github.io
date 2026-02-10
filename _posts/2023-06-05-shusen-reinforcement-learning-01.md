---
layout: post
title: Reinforcement Learning Basics
subtitle:
categories: Reinforcement-Learning
tags: [YouTube]
banner: "/assets/images/banners/yuanpang-wa-valley.jpg"
---

# Reinforcement Learning Basics

Follow this awesome [tutorial](https://www.youtube.com/watch?v=vmkRMvhCW5c&list=PLvOO0btloRnsiqM72G4Uid0UWljikENlU) by 
Shusen Wang, which provides a foundational overview of reinforcement learning (RL) — starting from probability theory and building up to key RL concepts such as states, actions, policies, and value functions.


## 1. 🎲 A Little Bit of Probability Theory

The lecture begins with essential probability concepts required for RL understanding:

### 🔹 Random Variable (X)
- A **random variable** is an unknown value dependent on random events.
- Convention: uppercase `X` for a random variable, lowercase `x` for an observed value.  
  *Example:* Flipping a coin four times → observed outcomes  
  $( x_1 = 1, x_2 = 1, x_3 = 0, x_4 = 1 )$

### 🔹 Probability Density Function (PDF)
- Describes the **relative likelihood** that a continuous random variable takes a specific value.
- Example: the **Gaussian (normal) distribution** is a continuous distribution with a defined PDF.

### 🔹 Probability Mass Function (PMF)
- Defines probabilities for **discrete** random variables.  
  *Example:* For $( X \in \{1,3,7\} )$,  
  $( p(1)=0.2, p(3)=0.5, p(7)=0.3 )$.

### 🔹 Properties of PDF/PMF
- **Domain:** the set of possible outcomes of $( X )$.
- For continuous distributions:  
  $( \int_X p(x)dx = 1 )$
- For discrete distributions:  
  $( \sum_{x \in X} p(x) = 1 )$

### 🔹 Expectation (E)
- The expected value of a function $( f(X) )$:

  - Continuous: $( E[f(X)] = \int_X p(x)f(x)dx )$
  - Discrete: $( E[f(X)] = \sum_{x \in X} p(x)f(x) )$

### 🔹 Random Sampling
Example: Drawing balls from a bin.  
If $( P(\text{red})=0.2, P(\text{green})=0.5, P(\text{blue})=0.3 )$,  
then each draw is a random sample from this probability distribution.


## 2. 🤖 Reinforcement Learning Terminologies

Key components of the RL framework:

### 🔹 State (s) and Action (a)
- **State (s):** current situation or environment snapshot.
- **Action (a):** possible move (e.g., `{left, right, up}`).

### 🔹 Policy (π)
- Defines the agent’s **behavioral rule**.  
  $( \pi(a|s) = P(A=a | S=s) )$
- A **stochastic policy** assigns probabilities to actions, e.g.  
  $( \pi(\text{left}|s)=0.2, \pi(\text{right}|s)=0.1, \pi(\text{up}|s)=0.7 )$
- Policies can be **deterministic** or **randomized**.

### 🔹 Reward (R)
- **Numerical feedback** from the environment.  
  Examples:
  - $( R=+1 )$: collect a coin  
  - $( R=+10000 )$: win the game  
  - $( R=-10000 )$: game over  
  - $( R=0 )$: nothing happens  

### 🔹 State Transition
- Moving from old state $( s )$ to new state $( s' )$ after action $( a )$.  
  $( p(s'|s,a) = P(S'=s'|S=s,A=a) )$
- The environment adds **randomness** to transitions.


## 3. 🔄 Agent–Environment Interaction and Randomness

RL involves continuous loops of interaction, with two key randomness sources:

### 🎲 Two Sources of Randomness
1. **Actions:** $A \sim \pi(\cdot \mid s)$ — from the agent’s policy.  
2. **States:** $S' \sim p(\cdot \mid s, a)$ — from the environment.

### 🧩 The RL Interaction Loop
1. Agent observes state $s_t$.  
2. Chooses action $a_t \sim \pi(\cdot \mid s_t)$.  
3. Executes $a_t$, receives new state $s_{t+1}$ and reward $r_t$.


### 🧵 Trajectory / Episode
A full trajectory:  
$( (s_0, a_0, r_0, s_1, a_1, r_1, \dots, s_H, a_H, r_H) )$

An **episode** lasts until termination (e.g., game win/loss).

### 🎯 Policy Goal
Find a **policy π** that **maximizes total (cumulative) reward**.


## 4. 💰 Rewards and Returns

### 🔹 Return (Cumulative Future Reward)
The **return** $( U_t )$ is the sum of all future rewards:
$[
U_t = R_t + R_{t+1} + R_{t+2} + \dots
]$

### 🔹 Discounted Return
Future rewards are **discounted** by factor $( \gamma \in [0,1) )$:
$[
U_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \dots + \gamma^{H-t} R_H
]$

This reflects that future rewards are **less valuable** than immediate ones.

### 🔹 Randomness of Return
Since $( R_t, R_{t+1}, …, R_H )$ depend on stochastic transitions and actions,  
$( U_t )$ itself is a **random variable**.  
A specific realization (observed value) is denoted $( u_t )$.

## 5. 📈 Value Functions: Qπ(s, a) and Vπ(s)

Value functions are central to reinforcement learning — they quantify *how good* a state or action is under a given policy $( \pi )$.  
There are two main types: the **Action-Value Function (Q-function)** and the **State-Value Function (V-function)**.


### 1️⃣ Action-Value Function $( Q_\pi(s, a) )$

The **Action-Value Function** (often called the **Q-function**) measures the expected quality of taking a specific action $( a )$ in a specific state $( s )$, assuming the agent follows policy $( \pi )$ thereafter.

#### 🧮 Definition and Formula

Formally:
$[
Q_\pi(s_t, a_t) = E[U_t \mid S_t = s_t, A_t = a_t]
]$

Where the **discounted return** $( U_t )$ is:
$[
U_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \dots + \gamma^{H-t} R_H
]$

Here:
- $( R_t )$: reward at time *t*  
- $( \gamma )$: discount factor $( (0 \le \gamma \le 1) )$  
- $( H )$: horizon or end of episode  

#### 🔍 Dependencies and Randomness

When computing $( Q_\pi(s_t, a_t) )$:

1. The **initial state** $( s_t )$ and **action** $( a_t )$ are **known (observed)**.  
2. All **future states** $( S_{t+1}, \dots, S_H )$ and **future actions** $( A_{t+1}, \dots, A_H )$ are **random variables**.  
3. The expectation $E[\cdot]$ averages over this randomness, determined by:  
   - The **policy** $\pi$, governing actions: $A_{t+1} \sim \pi(\cdot \mid S_{t+1})$  
   - The **environment transition** $p$, governing states: $S_{t+1} \sim p(\cdot \mid s_t, a_t)$
4. Therefore, $( Q_\pi(s_t, a_t) )$ depends on:
   - The **current state** $( s_t )$
   - The **current action** $( a_t )$
   - The **policy** $( \pi )$
   - The **state-transition function** $( p )$

#### 💡 Interpretation

$( Q_\pi(s, a) )$ evaluates how **advantageous** it is for an agent to take action $( a )$ in state $( s )$, assuming it continues to act according to policy $( \pi )$.  
It reflects both **immediate reward** and **expected future rewards**.


### 2️⃣ State-Value Function $( V_\pi(s) )$

The **State-Value Function** measures the expected return of being in a state $( s )$, assuming the agent follows policy $( \pi )$.  
Unlike $( Q_\pi )$, it does not commit to a specific action but averages over all possible actions under the policy.

#### 🧮 Definition and Formula

The general expression:
$[
V_\pi(s_t) = E_A [Q_\pi(s_t, A)]
]$
where $( A \sim \pi(\cdot | s_t) )$.

- **For Discrete Actions:**
  $[
  V_\pi(s_t) = \sum_a \pi(a|s_t) \, Q_\pi(s_t, a)
  ]$

- **For Continuous Actions:**
  $[
  V_\pi(s_t) = \int \pi(a|s_t) \, Q_\pi(s_t, a) \, da
  ]$

#### 💡 Interpretation

For a fixed policy $( \pi )$:
- $( V_\pi(s) )$ measures how *good* it is to be in state $( s )$.  
- Taking the expectation over all possible states:
  $[
  E_S[V_\pi(S)]
  ]$
  provides a measure of **how good the policy $( \pi )$** is overall.


| Function | Definition | Input | Output | Interpretation |
|-----------|-------------|--------|----------|----------------|
| $( Q_\pi(s, a) )$ | Expected discounted return given $( s, a )$ | State + Action | Scalar value | “How good is it to take action *a* in state *s*?” |
| $( V_\pi(s) )$ | Expected return over all actions from $( s )$ | State | Scalar value | “How good is it to be in state *s*?” |

Both functions form the backbone of RL algorithms like **Q-learning**, **SARSA**, and **Actor–Critic**, which estimate and optimize these quantities to learn effective policies.


### 🌟 The Optimal Action-Value Function $( Q^\*(s, a) )$

The **Optimal Action-Value Function**, denoted as $( Q^\*(s, a) )$, represents the **best possible performance** an agent can achieve from any given state–action pair under an **optimal policy** $( \pi^\* )$.

#### 🧩 Definition

Formally, $( Q^\*(s, a) )$ is defined as the **maximum expected discounted return** achievable by:
1. Taking action $( a )$ in state $( s )$, and then  
2. Following the **optimal policy** $( \pi^\* )$ thereafter.

$[
Q^\*(s, a) = \max_\pi \, Q_\pi(s, a)
]$

In words:
> $( Q^\*(s, a) )$ tells us the **highest possible long-term return** (expected cumulative discounted reward) the agent can get if it behaves optimally after taking action $( a )$ in state $( s )$.

#### 🔁 Relation to $( Q_\pi(s, a) )$

- $( Q_\pi(s, a) )$: Expected return **under a specific policy** $( \pi )$.  
- $( Q^\*(s, a) )$: Expected return **under the best possible policy** $( \pi^\* )$.

Thus, $( Q^\*(s, a) )$ is conceptually derived from all possible $( Q_\pi )$ functions by finding the **maximum** value across all policies.

$[
Q^\*(s, a) = E[U_t \mid S_t = s, A_t = a, \pi = \pi^\*]
]$

Where the discounted return is:
$[
U_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \dots
]$

#### ⚙️ Bellman Optimality Equation

The optimal $( Q^\* )$ function satisfies the **Bellman Optimality Equation**:

$[
Q^\*(s, a) = E_{s'} \left[ R(s, a) + \gamma \max_{a'} Q^\*(s', a') \right]
]$

This recursive relation means:
- The optimal value of $( Q^\*(s, a) )$ equals the **immediate reward** plus the **discounted optimal future return**, assuming the agent always takes the best next action.


#### 🚀 Intuition and Interpretation

- $( Q^\*(s, a) )$ provides a **ground truth measure of optimality** — what is the best you can do from here, if you act perfectly afterward.
- Once $( Q^\*(s, a) )$ is known, the **optimal policy** can be derived directly by always picking the action that maximizes $( Q^\*(s, a) )$:
  $[
  \pi^\*(s) = \arg\max_a Q^\*(s, a)
  ]$
- This is the foundation of many RL algorithms such as:
  - **Q-learning**
  - **Deep Q-Networks (DQN)**
  - **Double Q-learning**


#### 💡 Summary

| Concept | Definition | Meaning |
|----------|-------------|----------|
| $( Q_\pi(s, a) )$ | Expected discounted return following policy $( \pi )$ | “How good is it to take action *a* in state *s* under π?” |
| $( Q^\*(s, a) )$ | Maximum expected return across all possible policies | “What is the *best possible* return achievable from *(s, a)*?” |
| $( \pi^\*(s) )$ | Policy that maximizes $( Q^\*(s, a) )$ for every state | “Always choose the best action according to $( Q^\* )$.” |

In short:  
> **$( Q^\*(s, a) )$** defines the gold standard of decision-making in RL — the best long-term outcome achievable through optimal behavior.


## 6. 🎲 Randomness vs. Determinism in Reinforcement Learning

The concepts of **“random” (stochastic)** and **“deterministic”** are fundamental to reinforcement learning, particularly in describing the behavior of the **agent** and the **environment**.  
Randomness is introduced in two main areas: the **Policy** (agent’s action selection) and the **State Transition** (environment’s response).

### 🧠 Randomness in the Agent’s Policy

The **policy (π)** defines the agent’s behavior — how it chooses an action given a state. Policies can be **stochastic** or **deterministic**.

#### • Random (Stochastic) Policy
- The policy $\pi(a \mid s)$ represents the **probability** of taking action $A = a$ given the current state $S = s$.
- Upon observing $( S=s )$, the agent’s action $( A )$ is **random**, drawn from the policy distribution:  
  $( A \sim \pi(\cdot|s) )$.
- **Example:**  
  If an agent is in state $( s )$, its policy might specify:  
  $( \pi(\text{left}|s)=0.2, \pi(\text{right}|s)=0.1, \pi(\text{up}|s)=0.7 )$.  
  The agent samples its next action based on these probabilities.

#### • Deterministic Policy
- A deterministic policy always selects **the same action** for a given state, with probability 1.  
- Formally:  
  $( \pi(a|s) = 1 )$ for a specific action $( a )$, and $( 0 )$ for all others.  
- The action selection is **fully predictable** and contains **no randomness**.

### 🌎 Randomness in the Environment’s State Transition

The **environment’s response** to an agent’s action can also be random or deterministic.

#### • Random (Stochastic) State Transition
- The transition from an old state $( s )$ to a new state $( s' )$ after action $( a )$ can be **random**, due to environmental uncertainty.  
- The state transition probability is defined as:  
  $( p(s'|s,a) = P(S' = s' | S = s, A = a) )$
- **Example:**  
  If the agent takes action “up” in state $( s )$:
  - It might reach new state $( s'_1 )$ with probability 0.8,  
  - Or another state $( s'_2 )$ with probability 0.2.  
  Hence, $( S' \sim p(\cdot|s,a) )$.

#### • Deterministic State Transition
- In a deterministic environment, taking action $( a )$ in state $( s )$ always results in **the same next state $( s' )$** with probability 1.  
  $( p(s'|s,a) = 1 )$

### 🔄 Overall Randomness and Return

Because **both** the agent’s policy $( \pi )$ and the environment’s transition function $( p )$ can be random, the entire **trajectory** of future states and rewards is inherently **stochastic**.

- The **return** $( U_t )$ (sum of future discounted rewards) is a **random variable** at time $( t )$.  
- Each future reward $R_n$ is random because it depends on:  
  - $A_n$, drawn from the **policy** $\pi(\cdot \mid S_n)$, and  
  - $S_n$, drawn from the **transition function** $p(\cdot \mid S_{n-1}, A_{n-1})$.

Thus, the **Action-Value Function** $( Q_\pi(s,a) )$ and the **State-Value Function** $( V_\pi(s) )$ are defined as **expectations** — they compute the **expected discounted return**, averaging over all possible random trajectories under policy $( \pi )$.

### 🧩 Summary

Randomness is intrinsic to the RL framework:
- **Agent side:** stochastic policies introduce randomness in action selection.  
- **Environment side:** stochastic transitions introduce randomness in next states.  
- Consequently, **returns and value functions** must handle expectations over these random outcomes.

> In essence, reinforcement learning models uncertainty — both in **decision-making** (the agent) and in **consequence** (the environment).


## 7. 🧪 Evaluating Reinforcement Learning

### 🔹 OpenAI Gym
A standard **toolkit** for developing and testing RL algorithms.

### 🔹 Example Environments
- **Classical control:** CartPole, Pendulum  
- **Atari games:** Pong, Breakout  
- **Continuous control:** MuJoCo (Ant, Humanoid)

### 🧰 Example — CartPole
In Python:
```python
import gym
env = gym.make("CartPole-v1")
state = env.reset()
done = False
while not done:
    action = env.action_space.sample()  # random action
    next_state, reward, done, info = env.step(action)
```

The environment returns new state, reward, and a done flag (done=1 when the episode ends).


## 8. 🧾 Summary of Key Reinforcement Learning Terminologies

### I. 🎲 Foundational Probability Concepts

These concepts describe the **nature of uncertainty** and how random variables are mathematically modeled in reinforcement learning.

| **Terminology** | **Notation** | **Definition** |
|------------------|--------------|----------------|
| **Random Variable** | $( X )$ (Uppercase) | An unknown value whose outcome depends on random events. Examples include future **states (S)**, **actions (A)**, **rewards (R)**, and the **return (U_t)**. |
| **Observed Value** | $( x )$ (Lowercase) | The specific, known value of a random variable after a random event occurs. Example: the observed coin flip sequence $( x_1=1, x_2=1 )$, or an observed discounted return $( u_t )$. |
| **Probability Density Function (PDF)** | $( p(x) )$ | Provides the **relative likelihood** that a continuous random variable takes a particular value. |
| **Probability Mass Function (PMF)** | $( p(x) )$ | Defines the **probability** that a discrete random variable equals a particular value. |
| **Expectation** | $( E[f(X)] )$ | The **average value** of a function $( f(X) )$. Computed as $( \sum_{x \in X} p(x) f(x) )$ for discrete distributions or $( \int_X p(x) f(x) dx )$ for continuous ones. |


### II. 🤖 Core Reinforcement Learning Components

These terms define the **agent**, the **environment**, and how they **interact** over time.

| **Terminology** | **Notation** | **Definition** |
|------------------|--------------|----------------|
| **State** | $s$ | Represents the **current situation** or frame of the environment. |
| **Action** | $a$ | A **choice** the agent can make, e.g. `{left, right, up}`. The full set of possible actions is denoted $A$. |
| **Reward** | $R$ or $r$ | **Numerical feedback** from the environment. Example: +1 for collecting a coin, −10000 for hitting a Goomba, 0 when nothing happens. Rewards guide policy learning. |
| **Policy** | $\pi$ | The **agent’s behavioral rule**, defining the probability of taking an action given a state: $\pi(a \mid s) = P(A = a \mid S = s)$. |
| **State Transition** | $p(s' \mid s, a)$ | The **probability distribution** describing how the environment moves from one state $s$ to the next $s'$ after taking action $a$. |
| **Trajectory** | $(s_0, a_0, r_0, \ldots)$ | The **sequence of interactions** between the agent and environment: $(s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_H, a_H, r_H)$. |
| **Episode** | N/A | One **complete run** from start to end (e.g., a game of Mario — from start until win or loss). |


### III. 💰 Return and Value Functions

These terms describe how reinforcement learning **quantifies long-term performance** and evaluates states or actions.

| **Terminology** | **Notation** | **Definition** |
|------------------|--------------|----------------|
| **Return** | $U_t$ | The **cumulative future reward** received from time $t$ onward. |
| **Discount Factor** | $\gamma$ | A **tuning parameter** ($0 \le \gamma \le 1$) that reduces the weight of future rewards, modeling time preference. |
| **Discounted Return** | $U_t$ | The **cumulative discounted future reward**:<br> $U_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \ldots + \gamma^{H-t} R_H$. <br> Since rewards are random, $U_t$ is also a **random variable**. |
| **Observed Discounted Return** | $u_t$ | The **realized value** of the discounted return computed from observed rewards $r_t, r_{t+1}, \ldots$. |
| **Action-Value Function** | $Q_\pi(s, a)$ | The **expected discounted return** given the agent is in state $s$ and takes action $a$, assuming it continues following policy $\pi$. Measures *“how good it is to take action a in state s.”* |
| **State-Value Function** | $V_\pi(s)$ | The **expected value** of being in state $s$, computed as the expectation of $Q_\pi(s, A)$ where $A \sim \pi(\cdot \mid s)$:<br> $V_\pi(s) = E_A[Q_\pi(s, A)]$. |


### 🧠 Summary

- **Probability** defines randomness in environment and policy.  
- **Core RL concepts** define how the agent interacts and learns.  
- **Value functions** quantify the quality of actions and states for decision-making.  

Together, these form the mathematical backbone of modern **Reinforcement Learning**.
