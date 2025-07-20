---
layout: post
title: Actor-Critic Algorithms in Reinforcement Learning
subtitle: Reinforcement Learning Lecture 6
categories: Reinforcement-Learning
tags: [UCB-Deep-Reinforcement-Learning-2023]
banner: "/assets/images/banners/yuanpang-wa-valley.jpg"
---

# Actor-Critic Algorithms in Reinforcement Learning

This lecture focuses on **Actor-Critic algorithms** in Deep Reinforcement Learning. It covers the evolution from basic policy gradients, the role of value functions, various policy evaluation techniques, practical implementation considerations, and advanced variance reduction methods.

[Course Link](https://rail.eecs.berkeley.edu/deeprlcourse/)


## 1. Introduction to Actor-Critic Algorithms

Actor-Critic algorithms augment the basic policy gradient framework with **learned value functions** (or Q-functions) to improve performance.  
The core idea is to combine the strengths of policy-based and value-based methods:

- **Actor**: decides actions
- **Critic**: evaluates actions to guide policy updates

### Anatomy of an RL Algorithm:

1. **Generate Samples** (🟧): Run the current policy in the environment to collect trajectories.
2. **Estimate Return / Fit Value Function** (🟩): Fit a neural network to estimate value functions.
3. **Improve Policy** (🟦): Use estimated values or advantages to update the policy.


## 2. Improving Policy Gradients with Value Functions

Policy gradient methods like **REINFORCE** suffer from **high variance** because the gradient estimate relies on sampled returns, which can be noisy due to the stochastic nature of both the policy and the environment.

![alt_text](/assets/images/reinforcement-learning/06/1.png "image_tooltip")

In vanilla policy gradient:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot R_t \right]
$$

- $R_t$ is the **reward-to-go**, estimated from a single trajectory.
- This introduces **high variance**, especially in environments with sparse or delayed rewards.

To reduce variance, we introduce a **critic** that estimates value functions. This gives us better targets for the policy gradient, trading variance for some bias.

### 🎭 The Actor

- The policy $\pi_\theta(a_t | s_t)$, responsible for selecting actions.
- Learns via gradient ascent on an improved policy gradient estimate.

### 🧮 The Critic

Estimates **value functions** to provide low-variance learning signals for the actor.


### 🔹 Q-function: $Q^\pi(s_t, a_t)$

> Expected total reward after taking action $a_t$ in state $s_t$ and following policy $\pi$:
$$
Q^\pi(s_t, a_t) = \mathbb{E} \left[ \sum_{t'} \gamma^{t'-t} r_{t'} \mid s_t, a_t \right]
$$

- In policy gradient: replaces $R_t$ with $Q^\pi(s_t, a_t)$ for lower variance.


### 🔹 Value Function: $V^\pi(s_t)$

> Expected total reward starting from state $s_t$ and following $\pi$:
$$
V^\pi(s_t) = \mathbb{E}_{a_t \sim \pi} \left[ Q^\pi(s_t, a_t) \right]
$$

### 🔹 Advantage Function: $A^\pi(s_t, a_t)$

> Measures how much better an action is compared to the average action at $s_t$:
$$
A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t)
$$

- **Crucial for variance reduction.**
- Encourages actions that are better-than-average:
$$
\nabla_\theta J(\theta) = \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot A^\pi(s_t, a_t) \right]
$$


### 🛠️ Why It Works

- **High variance** in REINFORCE comes from using full returns $R_t$ from a single sample.
- The **critic** uses function approximation to estimate $V^\pi$ and $Q^\pi$, smoothing out the randomness over many experiences.
- This introduces **some bias** but **massively reduces variance**, improving learning stability.


### ✅ Summary

| Component        | Role                                   | Benefit                         |
|------------------|----------------------------------------|----------------------------------|
| Actor            | Learns the policy                      | Selects optimal actions         |
| Critic           | Estimates value functions              | Reduces variance of gradients   |
| $Q^\pi(s, a)$    | Expected return of action              | More accurate feedback          |
| $V^\pi(s)$       | Baseline value                         | Helps define advantages         |
| $A^\pi(s, a)$    | Action quality vs average              | Sharpens learning signal        |


The **Advantage Actor-Critic (A2C)** method uses $A^\pi(s_t, a_t)$ instead of raw returns, striking a powerful balance between **bias and variance**, and accelerating training by **focusing updates** on actions that exceed expectations.


## 3. Policy Evaluation: Fitting Value Functions

![alt_text](/assets/images/reinforcement-learning/06/2.png "image_tooltip")

### 3.1. Monte Carlo Evaluation with Function Approximation

#### 🔍 Core Idea
- Estimate value of state $s_t$ by **rolling out** full episodes and computing:
  
  $$
  y_{i,t} = \sum_{t'=t}^{T} r(s_{i,t'}, a_{i,t'})
  $$

- This is called **reward-to-go**.
- It’s **unbiased**, but has **high variance** due to randomness in trajectories.

#### 🤖 With Function Approximation
- Train a neural network $\hat{V}^\pi_\phi(s)$ to fit:

  $$
  \min_\phi \sum_i \left( \hat{V}^\pi_\phi(s_{i,t}) - y_{i,t} \right)^2
  $$

- Inputs: states $s_{i,t}$  
- Targets: Monte Carlo returns $y_{i,t}$ from sample rollouts

#### ✅ Benefits
- Learns to **generalize**: nearby or similar states share value estimates.
- **Reduces variance** by averaging over noisy returns.

#### ⚠️ Downside
- May introduce **bias** if value function is inaccurate.
- Trade-off: **Slight bias is often worth the large variance reduction.**

#### 💡 Example
Suppose two trajectories start from the same state $s_t$:
- Traj 1: reward-to-go = 100
- Traj 2: reward-to-go = 20  
Neural net learns that $s_t$ likely has a value ≈ 60.


### 3.2. Bootstrap Estimates

#### 🔍 Core Idea
- Use a **1-step TD estimate** instead of full return:

  $$
  y_t = r_t + \gamma \hat{V}^\pi_\phi(s_{t+1})
  $$

- Only look **one step ahead**, then use the critic's prediction.

#### ✅ Benefits
- **Much lower variance** than Monte Carlo
- Doesn’t require waiting until episode ends

#### ⚠️ Downside
- Introduces **bias** because $\hat{V}$ is imperfect.
- Error at $s_{t+1}$ affects $s_t$ → error propagation

#### 💡 Example
From $s_t$:
- Reward $r_t = 1$
- $\hat{V}(s_{t+1}) = 5$
- Target: $1 + 0.99 * 5 = 5.95$


### 3.3. Bias-Variance Trade-Off

| Method              | Bias     | Variance | Notes                                     |
|---------------------|----------|----------|-------------------------------------------|
| Monte Carlo         | ❌ None  | 🔺 High  | True expectation, but noisy               |
| Bootstrap (TD)      | ✅ Yes   | 🔻 Low   | Fast updates, but relies on learned $\hat{V}$ |
| Function Approx MC  | ✅ Small | 🔻 Lower | Reduced variance from generalization      |

🎯 **Goal**: Find a balance between bias and variance  
➡ Techniques like **N-step returns** or **Generalized Advantage Estimation (GAE)** achieve this.


Let's walk through both methods with a short trajectory:

Assume a trajectory from time steps $t = 0, 1, 2$ with:

- States: $s_0, s_1, s_2$
- Rewards: $r_0 = 1$, $r_1 = 2$, $r_2 = 3$
- Discount: $\gamma = 0.9$

**🎲 Monte Carlo (MC):**
- At $s_0$, reward-to-go = $1 + 0.9 \cdot 2 + 0.9^2 \cdot 3 = 1 + 1.8 + 2.43 = 5.23$
- At $s_1$, reward-to-go = $2 + 0.9 \cdot 3 = 2 + 2.7 = 4.7$
- At $s_2$, reward-to-go = $3$

So training data is:
(s_0, 5.23), (s_1, 4.7), (s_2, 3.0)
Train value network to predict those returns from the input states.

**⚡ Bootstrap (TD(0)):**
- At $s_2$, there’s no $s_3$, so maybe:
$y_2 = r_2 = 3$
- At $s_1$, bootstrap target:
$y_1 = 2 + 0.9 \cdot \hat{V}(s_2)$
- At $s_0$, bootstrap target:
$y_0 = 1 + 0.9 \cdot \hat{V}(s_1)$

So you use the network’s own prediction to estimate value targets.

### 3.4. Discount Factor ($\gamma$)

#### 🧠 Why use $\gamma \in (0,1)$?
- Prefers **sooner** rewards (time preference)
- Prevents **infinite** returns in infinite-horizon settings
- Can be viewed as **"probability of death"** each step

#### 📉 Variance Reduction
- Distant rewards contribute **less** due to $\gamma^k$
- Helps stabilize learning

#### 💡 Example
Compare full returns with and without discounting:

Un-discounted:
$$
G = 1 + 1 + 1 + \cdots = \infty
$$

Discounted ($\gamma = 0.9$):
$$
G = 1 + 0.9 + 0.9^2 + \cdots = \frac{1}{1 - 0.9} = 10
$$

#### 🧠 Summary

- **Monte Carlo (MC)** methods estimate true returns but are noisy.
- **Bootstrap** methods are biased but efficient and low variance.
- **Function approximation** helps generalize and reduce variance.
- **Discount factors** stabilize training and make distant rewards less influential.
- All these tools aim to produce a **more reliable critic** to improve policy gradients.


## 4. Actor-Critic Algorithm Structure

![alt_text](/assets/images/reinforcement-learning/06/3.png "image_tooltip")

1. **Generate Samples**
2. **Fit Value Function**
3. **Evaluate Advantage**:  
   $( \hat{A}^\pi(s, a) = r + \gamma \hat{V}^\pi(s') - \hat{V}^\pi(s) )$
4. **Construct Policy Gradient**
5. **Gradient Ascent**

![alt_text](/assets/images/reinforcement-learning/06/4.png "image_tooltip")

### Batch Actor-Critic Algorithm  
A typical batch actor-critic algorithm involves these steps:

1. **Generate samples**: Run the current policy to collect a batch of $N$ trajectories.

2. **Fit the approximate value function**: Train the neural network $\hat{V}^\pi_\phi(s)$ using the collected samples. This can be done with Monte Carlo estimates or bootstrap estimates.

3. **Evaluate the approximate advantage**:  
   For each state-action tuple $(s_i, a_i)$ in the sampled data, calculate the advantage as:  
   $$\hat{A}^\pi(s_i, a_i) = r(s_i, a_i) + \gamma \hat{V}^\pi_\phi(s'_i) - \hat{V}^\pi_\phi(s_i)$$  
   This is a common way to use the critic as a state-dependent baseline.  
   The term $(Q - V)$ is called the advantage function, representing how much better an action $a_t$ is compared to the average expected performance in state $s_t$.

4. **Construct a policy gradient estimator**:  
   Use these advantage values to compute the policy gradient, typically by multiplying the gradient of the log-probability of the action by the approximate advantage:  
   $$\nabla_\theta J(\theta) \approx \nabla_\theta \log \pi_\theta(a|s) \hat{A}^\pi(s, a)$$

5. **Take a gradient ascent step**:  
   Update the policy parameters $\theta$ using the calculated policy gradient:  
   $$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$$

### Online Actor-Critic Algorithm  
Online actor-critic algorithms update the policy and value function at every single time step, rather than collecting a full batch of trajectories first. A basic online algorithm looks like this:

1. **Take an action** $a$ from the policy $\pi_\theta(a|s)$ and observe a transition $(s, a, s', r)$.

2. **Update the value function**:  
   Update $\hat{V}^\pi_\phi(s)$ using a bootstrap target:  
   $$y \approx r + \gamma \hat{V}^\pi_\phi(s')$$  
   This update only requires the immediate next state $s'$, not the entire future trajectory.

3. **Evaluate the advantage**:  
   Calculate the advantage for the current step as:  
   $$\hat{A}^\pi(s, a) = r + \gamma \hat{V}^\pi_\phi(s') - \hat{V}^\pi_\phi(s)$$

4. **Construct and apply policy gradient**:  
   Estimate the policy gradient as:  
   $$\nabla_\theta J(\theta) \approx \nabla_\theta \log \pi_\theta(a|s) \hat{A}^\pi(s, a)$$  
   and update the policy parameters.

5. **Repeat** this process at every single time step.



For deep reinforcement learning, online updates with a single sample can lead to high variance and unstable training for neural networks. To address this, batching is often used in practice, even in online settings, through:

![alt_text](/assets/images/reinforcement-learning/06/5.png "image_tooltip")

- **Synchronized Parallel Actor-Critic**:  
  Multiple "workers" (simulators) run in parallel, collecting transitions. The collected data is then aggregated into a batch for synchronous updates to the value function and policy.

- **Asynchronous Parallel Actor-Critic (A3C)**:  
  Workers run asynchronously. When a worker collects enough transitions, it makes an update using its local data and the latest global parameters, without waiting for other workers.  
  This can be faster but introduces a slight bias because transitions might have been generated by slightly older actor parameters. However, this bias is often outweighed by the performance gains.


## 5. Design Decisions & Practical Implementations

### 5.1 Neural Network Architectures

- **Separate Networks**:
  - ✅ Simpler, stable
  - ❌ No shared features

- **Shared Network**:
  - ✅ Efficient, shared learning
  - ❌ Harder to train

### 5.2 Batch Sizes and Parallelism

- **Synchronous Parallel**:
  - ✅ Stable batches
  - ❌ More compute

- **Asynchronous Parallel**:
  - ✅ Fast
  - ❌ Slight bias


### 5.3 Off-Policy Actor-Critic

Online and batch actor-critic methods discussed above are generally **on-policy**, meaning they use data collected by the **current policy** to update that same policy.  
**Off-policy** actor-critic algorithms aim to leverage **transitions generated by older policies**, typically stored in a **replay buffer**. This can significantly improve **sample efficiency**.

![alt_text](/assets/images/reinforcement-learning/06/6.png "image_tooltip")

#### Challenges of Off-Policy Learning

Using old transitions directly introduces challenges:

- The action $a_i$ from the replay buffer did **not come from** the latest policy $\pi_\theta$.
- Therefore, the **policy gradient** cannot be directly computed using these old actions, as it needs to be an **expectation under the current policy**.

To fix this, off-policy methods introduce several adjustments:

#### 1. Learn a Q-function Instead of a V-function

- A Q-function $Q^\pi(s, a)$ represents the expected return starting from state $s$, taking action $a$, and then following policy $\pi$.
- The **Q-function is valid for any action $a$**, not just those taken by the current policy.
- This allows the critic to be trained on actions **from the replay buffer**, even if they were produced by older policies.

#### 2. Compute Targets for Q-function

The Q-function target is:
$$
y_i = r_i + \gamma \hat{Q}^\pi_\phi(s'_i, a'_i)
$$
- $a'_i$ is the action the **current policy** $\pi_\theta$ would take in state $s'_i$.
- $a'_i$ can be **sampled from the policy network** without interacting with the simulator.

#### 3. Policy Update for Off-Policy

The policy gradient is estimated as:
$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_i \nabla_\theta \log \pi_\theta(a^\pi_i | s_i) \hat{Q}^\pi_\phi(s_i, a^\pi_i)
$$
- $a^\pi_i$ is sampled from the **current policy** $\pi_\theta$ at a **replay buffer state** $s_i$.
- Unlike on-policy methods that rely on advantage estimates, this off-policy update uses the **Q-values directly**.

> While using Q-values instead of advantages may increase **variance**, it’s manageable because:
> - Many actions can be sampled from the policy network
> - No need to interact with the environment to get new data

#### Remaining Bias

A source of bias remains:  
- The states $s_i$ in the replay buffer may **not reflect** the distribution induced by the current policy.
- This is generally **accepted**, as the replay buffer provides **broad state coverage**, which helps generalization.


#### Example: Soft Actor-Critic (SAC)

A notable example of an off-policy actor-critic method is **Soft Actor-Critic (SAC)**, which:
- Uses entropy-regularized objectives
- Maintains sample efficiency through off-policy learning
- Improves exploration and stability with stochastic policies


## 6. Critics as Baselines (Unbiased Variance Reduction)

### 6.1 State-Dependent Baselines

Unbiased if the baseline **only depends on state**:

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i,t} \nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t}) (Q - V)
$$

- ✅ Unbiased  
- ✅ Lower variance than constant  
- ❌ Still high variance vs actor-critic


### 6.2 Control Variates: Action-Dependent Baselines

Baseline $( b(s,a) )$ introduces bias, but can correct it:

$$
\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi(a|s) (Q - b(s,a))] + \mathbb{E}[\nabla_\theta \mathbb{E}[b(s,a)]]
$$

- ✅ Very low variance with good $( b(s,a) )$


### 6.3 Eligibility Traces & N-Step Returns

#### 🎯 Motivation for N-Step Returns

Policy gradient methods face a **bias-variance trade-off**:

- **Monte Carlo (MC)** estimates (full trajectory return) are **unbiased** but have **high variance**.
- **Actor-Critic (1-step)** estimates use a learned value function to **reduce variance**, but introduce **bias** due to function approximation errors.

**N-Step Returns** provide a middle ground by using a mix of real rewards and estimated values.


#### 🔁 N-Step Return: Concept

The **n-step return** is defined as:

$$
\hat{A}^{(n)}_t = \sum_{t'=t}^{t+n-1} \gamma^{t'-t} r(s_{t'}, a_{t'}) + \gamma^n \hat{V}_\phi(s_{t+n}) - \hat{V}_\phi(s_t)
$$

- First term: **Actual discounted rewards** over $n$ steps (adds variance)
- Second term: **Estimated value** at $s_{t+n}$ (adds bias if $\hat{V}_\phi$ is inaccurate)
- Third term: **Baseline subtraction** (for variance reduction)


#### ⚖️ Bias-Variance Trade-Off

| Parameter | Effect |
|----------|--------|
| **Large n** | ✅ Lower bias (less reliance on $\hat{V}$)<br>❌ Higher variance (more sample noise) |
| **Small n** | ✅ Lower variance (less noisy estimate)<br>❌ Higher bias (more reliance on possibly inaccurate $\hat{V}$) |

> **Optimal n** is typically intermediate — not 1 (pure TD), not $\infty$ (pure MC).


#### 🧠 Intuition

- Early rewards are **less noisy** → estimate them using **real rewards**.
- Far-future rewards are **more uncertain** → use **value function estimate**.

This strategy **reduces variance** while keeping **bias manageable**.


### 6.4 Generalized Advantage Estimation (GAE)

Instead of choosing a fixed $n$ for n-step return, **GAE** computes a **weighted sum** of all possible n-step estimators, providing a **smoother bias-variance trade-off**.

#### 📐 GAE Definition

GAE advantage estimator:

$$
\hat{A}^{GAE}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

Where:

$$
\delta_t = r(s_t, a_t) + \gamma \hat{V}_\phi(s_{t+1}) - \hat{V}_\phi(s_t)
$$

- $\delta_t$ is the **TD error**
- $\lambda \in [0, 1]$ controls the **bias-variance trade-off**


#### ⚙️ Lambda ($\lambda$) in GAE

| Lambda ($\lambda$) | Behavior |
|------------------|----------|
| $\lambda \approx 0$ | → **1-step TD**<br>→ **Low variance**, **high bias** |
| $\lambda \approx 1$ | → **Full MC return**<br>→ **Low bias**, **high variance** |

> GAE allows smooth interpolation between **TD(0)** and **MC** using $\lambda$.


#### 🎛 Gamma ($\gamma$) for Discounting

- Smaller $\gamma$:
  - Reduces weight on far-future rewards → **less variance**, **more bias**
  - Often interpreted as **shorter planning horizon** or **risk aversion**


#### 🧪 Practical Use

- GAE is used in **modern actor-critic methods** like **PPO** and **TRPO**.
- Introduced in:  
  📝 *High-Dimensional Continuous Control with Generalized Advantage Estimation*, Schulman et al., 2016


#### 🔚 Summary

| Technique | Purpose | Trade-Off |
|----------|---------|-----------|
| N-step returns | Mix real rewards + value estimate | Adjust $n$ for bias-variance |
| GAE | Blend all $n$ with weighted sum | Tune $\lambda$ for smooth trade-off |


## 7. Summary & Examples

Actor-Critic combines **actor** and **critic** for reduced variance and efficient learning.

### Key Takeaways:

- **Policy Evaluation**: Fit $( V^\pi )$, $( Q^\pi )$, or $( A^\pi )$
- **Discounting**: Enables infinite horizon, lowers variance
- **Architecture**: Shared vs separate networks
- **Batching**: Sync/async, replay buffers
- **Baselines**: Reduce variance (state-dependent = no bias)
- **GAE / Traces**: Bias-variance tuning


### Examples in Literature:

- **TD-Gammon (1992)**, **AlphaGo (2016)**
- **GAE**: Continuous control
- **A3C**: Asynchronous, online
- **SAC**: Off-policy with entropy regularization
- **Q-Prop**: Control variates for sample-efficient learning
