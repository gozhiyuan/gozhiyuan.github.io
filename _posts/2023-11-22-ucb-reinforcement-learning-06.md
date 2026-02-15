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
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot R_t \right]
$$

- $R_t$ is the **reward-to-go**, estimated from a single trajectory.
- This introduces **high variance**, especially in environments with sparse or delayed rewards.

To reduce variance, we introduce a **critic** that estimates value functions. This gives us better targets for the policy gradient, trading variance for some bias.

### 🎭 The Actor

- The policy $\pi_\theta(a_t \mid s_t)$, responsible for selecting actions.
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
\nabla_\theta J(\theta) = \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot A^\pi(s_t, a_t) \right]
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


## 4. Actor-Critic Training Regimes (Online, Batch, Offline)

![alt_text](/assets/images/reinforcement-learning/06/3.png "image_tooltip")

Many terms are overloaded. A clean way is to separate **three axes**:

1. **Data source**: current policy data vs replay data vs fixed offline dataset
2. **Update schedule**: per-step update vs mini-batch update vs large-batch/epoch update
3. **Target type**: Monte Carlo vs bootstrap (or in-between n-step / GAE)

### 4.1 Batch Actor-Critic (On-Policy)

![alt_text](/assets/images/reinforcement-learning/06/4.png "image_tooltip")

Typical loop:

1. Roll out current policy `pi_theta` for `N` trajectories.
2. Fit critic `V_phi` (or a Q-critic) on this fresh batch.
3. Compute advantages (often GAE / n-step).
4. Update actor with policy gradient.
5. Discard old batch (or use only a few epochs, then recollect data).

Policy gradient form:
$$
\nabla_\theta J(\theta) \approx \frac{1}{N}\sum_{i,t}\nabla_\theta \log \pi_\theta(a_{i,t}\mid s_{i,t})\hat{A}_{i,t}
$$
Plain text fallback: `grad J(theta) ~= (1/N) * sum_{i,t} grad log pi_theta(a_{i,t}|s_{i,t}) * Ahat_{i,t}`.

Properties:
- ✅ Stable updates, easy to reason about
- ✅ Strong with trust-region style methods (TRPO/PPO)
- ❌ Lower sample efficiency (data is used briefly)

### 4.2 Online Actor-Critic (Streaming, Step-Wise)

Online AC updates immediately after each transition:

1. Observe transition `(s_t, a_t, r_t, s_{t+1})`
2. TD target:
$$
y_t = r_t + \gamma \hat{V}_\phi(s_{t+1})
$$
Plain text fallback: `y_t = r_t + gamma * V_phi(s_{t+1})`.
3. Advantage:
$$
\hat{A}_t = y_t - \hat{V}_\phi(s_t)
$$
Plain text fallback: `Ahat_t = y_t - V_phi(s_t)`.
4. Update actor and critic right away

Properties:
- ✅ Fast reaction to new experience
- ✅ Low memory requirement
- ❌ High gradient noise with deep nets if truly one-sample updates

### 4.3 "Online-Batch" Actor-Critic (Most Practical Deep RL)

This phrase means: **data is collected online**, but **training is done in batches**.

Common pipeline (A2C/PPO-style):
1. Run $K$ environments in parallel for $T$ steps
2. Build one batch of size $K \times T$
3. Compute returns/advantages (often GAE)
4. Train with SGD mini-batches

So it is "online" in data collection, but "batch" in optimization.

### 4.4 Monte Carlo vs Bootstrap Inside Actor-Critic

These are **target estimators**, not separate algorithm families.

Monte Carlo target:
$$
y_t^{MC}=\sum_{l=0}^{T-t}\gamma^l r_{t+l}
$$

Bootstrap (TD(0)) target:
$$
y_t^{TD}=r_t+\gamma\hat{V}_\phi(s_{t+1})
$$

n-step target:
$$
y_t^{(n)}=\sum_{l=0}^{n-1}\gamma^l r_{t+l}+\gamma^n \hat{V}_\phi(s_{t+n})
$$

Comparison:

| Target | Bias | Variance | Typical use |
|---|---|---|---|
| Monte Carlo | Low | High | Short episodes, low noise tasks |
| TD(0) bootstrap | Higher | Low | Online AC, fast continual updates |
| n-step / GAE | Medium | Medium | PPO/A2C default in practice |

### 4.5 Off-Policy vs Offline Actor-Critic

These two are related but not identical:

- **Off-policy actor-critic (online)**:
  - Still interacts with environment
  - Also reuses replay data from older behavior policies
  - Examples: DDPG, TD3, SAC

- **Offline actor-critic**:
  - No environment interaction during training
  - Learns only from a fixed dataset
  - Must handle out-of-distribution action errors very carefully
  - Examples: CQL, IQL, AWAC-style pipelines

So: offline RL is a stricter setting than off-policy RL.

## 5. Design Decisions & Practical Implementations

### 5.1 Neural Network Architectures

| Design | Pros | Cons | Typical choice |
|---|---|---|---|
| Separate actor/critic networks | Stable optimization, less gradient interference | More parameters, no shared representation | Default when training is unstable |
| Shared backbone + two heads | Better compute efficiency, shared features | Actor and critic gradients can conflict | Common in on-policy vision/state encoders |
| Partially shared (early shared, late separate) | Balance efficiency and stability | More tuning complexity | Good compromise in larger models |

Rule of thumb:
- If critic loss dominates and policy collapses, move toward **more separation**.
- If compute is tight and features overlap strongly, use **shared trunk + loss balancing**.

### 5.2 Batch Sizes and Parallelism

![alt_text](/assets/images/reinforcement-learning/06/5.png "image_tooltip")

#### Synchronous Parallel (A2C-style)
- Workers collect data, then all wait for joint update.
- ✅ Consistent policy version per batch (cleaner gradients)
- ✅ Better reproducibility and debugging
- ❌ Idle waiting on slow workers, higher wall-clock cost

#### Asynchronous Parallel (A3C-style)
- Workers update without global synchronization.
- ✅ High throughput
- ✅ Better hardware utilization
- ❌ Policy-lag / stale-gradient bias (worker data may be slightly old)

Practical guidance:
- Prefer synchronous if you care about stability and ablations.
- Prefer asynchronous when throughput is the top constraint.

### 5.3 Replay Buffer System Design (Core for Off-Policy / Offline AC)

![alt_text](/assets/images/reinforcement-learning/06/6.png "image_tooltip")

Store per transition:
$$
(s_t, a_t, r_t, s_{t+1}, d_t)
$$
Optional: behavior log-prob, episode id, n-step cumulative reward.

Key design choices:

1. **Capacity**
   - Small buffer: fresher but less diverse
   - Large buffer: diverse but may contain stale behavior

2. **Sampling**
   - Uniform sampling: simple and stable baseline
   - Prioritized replay: faster learning on high-TD-error samples, but needs importance correction

3. **Update-to-data ratio (UTD)**
   - Number of gradient steps per env step
   - Too high UTD can overfit stale data and destabilize critic

4. **Target stabilization**
   - Use target networks and soft update ($\tau$) to reduce bootstrapping instability

5. **Offline-specific safeguards**
   - Penalize OOD actions (conservative Q-learning ideas)
   - Constrain actor toward dataset support (behavior regularization / expectile-style methods)

#### Off-Policy Actor-Critic Objective (High-Level)

Critic target:
$$
y_i = r_i + \gamma \hat{Q}_{\bar{\phi}}(s'_i, a'_i),\quad a'_i \sim \pi_\theta(\cdot\mid s'_i)
$$

Actor objective (maximize Q, optionally with entropy regularization):
$$
J_{\text{actor}}(\theta)=\mathbb{E}_{s_i\sim \mathcal{D}}\!\left[\hat{Q}_\phi\!\left(s_i,\pi_\theta(s_i)\right)\right]+\alpha\,\mathbb{E}_{s_i\sim \mathcal{D}}\!\left[\mathcal{H}\!\left(\pi_\theta(\cdot\mid s_i)\right)\right]
$$


## 6. Critics as Baselines (Unbiased Variance Reduction)

### 6.1 State-Dependent Baselines

Unbiased if the baseline **only depends on state**:

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i,t} \nabla_\theta \log \pi_\theta(a_{i,t} \mid s_{i,t}) (Q - V)
$$

- ✅ Unbiased  
- ✅ Lower variance than constant  
- ❌ Still high variance vs actor-critic


### 6.2 Control Variates: Action-Dependent Baselines

Baseline $( b(s,a) )$ introduces bias, but can correct it:

$$
\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi(a \mid s) (Q - b(s,a))] + \mathbb{E}[\nabla_\theta \mathbb{E}[b(s,a)]]
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
- **Training Regimes**: Distinguish online, batch-on-policy, online-batch, and offline
- **Discounting**: Enables infinite horizon, lowers variance
- **Architecture**: Shared vs separate networks
- **Batching + Systems**: Sync/async workers, replay buffer design, UTD ratio
- **Baselines**: Reduce variance (state-dependent = no bias)
- **GAE / Traces**: Bias-variance tuning


### Examples in Literature:

- **TD-Gammon (1992)**, **AlphaGo (2016)**
- **GAE**: Continuous control
- **A3C**: Asynchronous, online
- **SAC**: Off-policy with entropy regularization
- **Q-Prop**: Control variates for sample-efficient learning


## 8. Worked Summary Example (One Trajectory, All Update Styles)

This final section uses one shared trajectory and shows how each training style updates differently.

### 8.1 Shared Setup

- Discount: `gamma = 0.9`
- Trajectory:
  - `t=0`: `(s0, a0, r1=1, s1)`
  - `t=1`: `(s1, a1, r2=2, s2)`
  - `t=2`: `(s2, a2, r3=3, terminal)`
- Critic snapshot before updates:
  - `V(s0)=4.0`, `V(s1)=3.0`, `V(s2)=1.0`, `V(terminal)=0`

1-step TD targets and advantages:
$$
y_0 = 1 + 0.9\cdot 3.0 = 3.7,\quad \hat{A}_0 = 3.7 - 4.0 = -0.3
$$
$$
y_1 = 2 + 0.9\cdot 1.0 = 2.9,\quad \hat{A}_1 = 2.9 - 3.0 = -0.1
$$
$$
y_2 = 3 + 0.9\cdot 0 = 3.0,\quad \hat{A}_2 = 3.0 - 1.0 = 2.0
$$

Monte Carlo returns:
$$
G_2=3,\quad G_1=2+0.9\cdot 3=4.7,\quad G_0=1+0.9\cdot 2+0.9^2\cdot 3=5.23
$$

---

### 8.2 On-Policy Batch (single update cycle)

Assume one training iteration `k` starts from parameters `(theta_k, phi_k)`.

Detailed workflow on this exact trajectory:

1. **Collect fresh data with current policy**
   - Run `pi_{theta_k}` in the environment.
   - We get `(s0,a0,r1,s1), (s1,a1,r2,s2), (s2,a2,r3,terminal)`.
   - This data is on-policy because it was generated by `theta_k`.

2. **Build learning targets from this same batch**
   - TD(0) targets from Section 8.1:
     - `y0=3.7, y1=2.9, y2=3.0`
   - Advantages:
     - `Ahat=[-0.3, -0.1, +2.0]`
   - If using n-step or GAE, only this computation changes; the pipeline is the same.

3. **Critic update on this batch**
   - Fit `V_phi` by minimizing:
$$
L_V(\phi)=\frac{1}{3}\sum_{t=0}^2\left(V_\phi(s_t)-y_t\right)^2
$$
   - Intuition with our numbers:
     - `V(s2)=1.0` is far below target `3.0`, so critic strongly increases value around `s2`.
     - `V(s0)=4.0` is above `3.7`, and `V(s1)=3.0` is above `2.9`, so those tend to move slightly down.

4. **Actor update using same transitions**
   - Policy loss (gradient ascent form):
$$
L_\pi(\theta)=-\frac{1}{3}\sum_{t=0}^2 \log \pi_\theta(a_t\mid s_t)\,\hat{A}_t
$$
   - Effect of signs:
     - `Ahat_0<0`, `Ahat_1<0`: decrease probability of `(a0|s0)` and `(a1|s1)`.
     - `Ahat_2>0`: increase probability of `(a2|s2)`.

5. **End of cycle: discard rollout and recollect**
   - After updates, parameters become `(theta_{k+1}, phi_{k+1})`.
   - Old rollout is no longer strictly on-policy for `theta_{k+1}`, so we recollect new data.
   - This is why on-policy batch methods are stable but less sample-efficient.

### 8.3 Synchronous Parallel (A2C-style)

Assume `K=2` workers, each collects `T=3` steps with the same parameter snapshot `theta_k`.

Detailed workflow with shared example:

1. **Parameter broadcast**
   - Global learner sends the same `(theta_k, phi_k)` to Worker 1 and Worker 2.

2. **Parallel rollout**
   - Worker 1 collects our shared trajectory (3 steps).
   - Worker 2 collects another 3-step trajectory from its own environment instance.
   - No one updates model weights during collection.

3. **Barrier synchronization**
   - Both workers stop and wait.
   - Combined batch size is `K*T=6` transitions.

4. **Compute targets/advantages on aggregated batch**
   - Worker 1 samples contribute `Ahat=[-0.3,-0.1,+2.0]`.
   - Worker 2 contributes its own advantages.
   - Learner averages losses across all 6 transitions.

5. **Single synchronized update**
   - One optimizer step produces `(theta_{k+1}, phi_{k+1})`.
   - This gradient is consistent with one parameter snapshot, so variance is lower and training is easier to reason about.

6. **Broadcast new weights and repeat**
   - All workers start the next rollout with the same fresh parameters.

Key difference vs 8.2:
- Same on-policy logic, but larger, more stable batches from parallel environments.
- Cost: faster workers can idle at the barrier waiting for slow workers.

### 8.4 Asynchronous Parallel (A3C-style)

Detailed timeline on the same trajectory:

1. **Start from global params**
   - Worker A pulls `theta_k` and begins collecting our shared trajectory.
   - Worker B also pulls `theta_k` but in a different environment instance.

2. **No barrier; workers update independently**
   - Worker B finishes first, computes gradients, and applies them to global weights.
   - Global model is now closer to `theta_{k+1}`.

3. **Worker A is now stale**
   - Worker A is still finishing rollout produced under old `theta_k`.
   - It computes advantages from that rollout (including `[-0.3,-0.1,+2.0]` for our shared path).

4. **Stale gradient applied to newer global model**
   - Worker A pushes gradient to global parameters that already changed.
   - So gradient was computed at old policy but applied at newer policy.

5. **Pull-latest and continue**
   - Worker A refreshes local weights and starts next rollout.

Why it behaves differently:
- Throughput is high because no waiting.
- Bias appears from policy lag (stale gradients), which can hurt stability if lag is large.

### 8.5 Online-Batch (PPO/A2C practical loop)

This is the common deep RL production pattern: online data collection, but SGD in mini-batches.

Why it can look similar to 8.3:
- Both often collect `K*T` on-policy transitions from parallel environments.
- Both can pause collection, compute advantages, then update parameters.

Core distinction:
- **8.3 (sync parallel)** describes a **systems topology**: workers are synchronized by a barrier and produce one aligned batch snapshot.
- **8.5 (online-batch)** describes an **optimization protocol**: what you do with a collected rollout buffer (mini-batches, number of epochs, clipping, normalization).

When they are effectively the same:
- If you do synchronized collection and then only one full-batch update (`1` epoch, no mini-batch reuse), 8.5 almost collapses to 8.3 behavior.

When they differ in practice (most PPO code):
- 8.3-style collection is used, but 8.5 optimization performs multiple SGD epochs over the same rollout.
- So one rollout can drive many optimizer steps before recollection.
- This improves hardware efficiency and learning speed, but pushes data farther from perfectly on-policy as epochs increase.

Detailed workflow:

1. **Rollout phase (online)**
   - Run `K` environments with current `theta_k`.
   - Store transitions in a short-lived rollout buffer until size `K*T`.
   - Our shared 3-step trajectory is one slice inside this rollout buffer.

2. **Freeze rollout and compute training signals**
   - Compute `y_t`, `Ahat_t`, returns (often with GAE).
   - Optionally normalize advantages across the whole rollout buffer.

3. **Optimization phase (batch SGD)**
   - Shuffle rollout buffer into mini-batches.
   - Run several epochs over the same collected data (for PPO, usually clipped objective).
   - Our shared trajectory can be reused multiple optimizer passes in this phase.

4. **Clear rollout buffer and recollect**
   - After fixed epochs, discard this rollout data.
   - Continue stepping env with updated policy and build the next on-policy batch.

Why it works well:
- More stable than pure step-wise online updates.
- Better hardware utilization via matrix mini-batches.
- Still near on-policy because data lifetime is short.

Concrete contrast on our shared trajectory:
- In 8.3, Worker 1's `[-0.3,-0.1,+2.0]` usually contributes once in one synchronized update.
- In 8.5 (PPO-style), the same three samples may be revisited across multiple mini-batches/epochs before being dropped.

### 8.6 Off-Policy with Replay Buffer (SAC/TD3-style)

Now behavior policy and update policy are different:

- Behavior policy that generated stored data: `mu`
- Current policy being optimized: `pi_theta`
- Replay stores: `(s,a,r,s',d)` and optionally behavior log-prob `log mu(a|s)`

Per update:

1. **Interact and store**
   - Environment step is generated by behavior policy `mu` (could be old actor + exploration noise).
   - Push transition tuple to replay.
   - Our shared trajectory transitions become replay rows that may be sampled many times later.

2. **Sample random mini-batch from replay**
   - Batch can contain very old and very new transitions.
   - This breaks temporal correlation and increases sample reuse.

3. **Critic target uses next action from current/target policy**, not necessarily stored action:
$$
y = r + \gamma Q_{\text{target}}(s', a'),\quad a' \sim \pi_{\text{target}}(\cdot\mid s')
$$
   - For our shared transition `(s1,a1,r2,s2)`, stored `a1` came from old `mu`, but target uses `a'` drawn from *current* target policy at `s2`.

4. **Critic update**
   - Regress `Q_\phi(s,a)` toward `y` on sampled replay actions `a`.
   - This learns from historical behavior while tracking current policy value.

5. **Actor update uses current policy actions on sampled states**
$$
J_{\text{actor}}(\theta)=\mathbb{E}_{s\sim D}\left[Q_\phi\!\left(s,\pi_\theta(s)\right)\right]
$$
   - States come from replay; actions are regenerated by current actor.
   - This avoids direct dependence on old behavior action probabilities in many implementations.

6. **Target-network update**
   - Soft update target critics (and target actor in TD3/DDPG) for stability.

How policy change affects target/action:
- Stored action `a` was from old `mu`; target action `a'` is from new/target policy.
- So target value tracks *current* policy improvement, while still reusing old states/transitions.

Action-space note:
- Discrete stochastic: expectation/sampling over action probabilities.
- Continuous stochastic (e.g., SAC): sample `a'` from Gaussian policy.
- Continuous deterministic (DDPG/TD3): `a' = pi_target(s')`.

If you explicitly reuse behavior action in policy gradient form, you need correction ratios like:
$$
\rho_t = \frac{\pi_\theta(a_t \mid s_t)}{\mu(a_t \mid s_t)}
$$
but many modern off-policy actor-critic methods avoid direct importance-weighted log-prob gradients by using the Q-maximization actor objective above.

### 8.7 Offline Actor-Critic (fixed dataset)

Detailed workflow:

1. **Freeze dataset**
   - No simulator interaction during training.
   - Dataset `D_offline` is fixed before optimization starts.
   - Our shared trajectory appears as static rows in this dataset.

2. **Offline mini-batch sampling**
   - Every gradient step samples only from `D_offline`.
   - Same rows can be replayed many times across epochs.

3. **Critic training (off-policy)**
   - Similar Bellman-style targets as replay methods.
   - But because we cannot collect corrective data, extrapolation error is dangerous.

4. **Conservative / behavior-regularized actor training**
   - Penalize actions far from dataset support, or regularize toward behavior policy.
   - Goal: prevent actor from exploiting critic errors on unseen actions.

5. **Iterate until convergence; deploy carefully**
   - Validation is typically done with held-out offline metrics and cautious online testing.

How shared example is used:
- It keeps reappearing during training because data is fixed.
- Main failure mode is OOD action overestimation: critic assigns high values to actions not supported by dataset.

---

### 8.8 Control Variates, N-step, and GAE on This Same Trajectory

#### A) Control Variates (Baselines)

Without baseline (REINFORCE-style weights): use `G_t` directly:
- weights: `[5.23, 4.7, 3.0]`

With state baseline `V(s_t)`, use advantage `G_t - V(s_t)`:
- weights: `[1.23, 1.7, 2.0]`

Interpretation:
- Mean shift removed, variance reduced, gradient direction kept unbiased for state-only baseline.

#### B) N-step Returns from `t=0`

1-step advantage:
$$
\hat{A}^{(1)}_0 = r_1 + \gamma V(s_1) - V(s_0) = 1 + 0.9\cdot 3 - 4 = -0.3
$$

2-step advantage:
$$
\hat{A}^{(2)}_0 = r_1 + \gamma r_2 + \gamma^2 V(s_2) - V(s_0)
= 1 + 0.9\cdot 2 + 0.9^2\cdot 1 - 4 = -0.39
$$

3-step (MC) advantage:
$$
\hat{A}^{(3)}_0 = G_0 - V(s_0) = 5.23 - 4 = 1.23
$$

Interpretation:
- Small `n`: more bootstrap bias, lower variance.
- Large `n`: less bootstrap bias, more sample variance.

#### C) GAE on the same deltas

TD deltas:
$$
\delta_0=-0.3,\quad \delta_1=-0.1,\quad \delta_2=2.0
$$

GAE at `t=0`:
$$
\hat{A}^{GAE}_0(\lambda)=\delta_0+\gamma\lambda\delta_1+(\gamma\lambda)^2\delta_2
$$

Examples:
- `lambda = 0`: $\hat{A}^{GAE}_0 = -0.3$ (pure TD(0)-like)
- `lambda = 0.95`: $\hat{A}^{GAE}_0 \approx 1.0766$
- `lambda = 1`: $\hat{A}^{GAE}_0 = -0.3 + 0.9(-0.1) + 0.9^2(2.0) = 1.23$ (MC-like)

Interpretation:
- `lambda` smoothly moves from low-variance TD to high-fidelity Monte Carlo.

### 8.9 Final Comparison Table

| Method | Uses fresh policy data only? | Reuses old data? | Main practical trade-off |
|---|---|---|---|
| On-policy batch | Yes | No (or very limited) | Stable but sample-hungry |
| Sync parallel | Yes (per synchronized batch) | Limited | Stable, but worker idle time |
| Async parallel | Mostly, but can be stale | Limited | Higher throughput, policy-lag bias |
| Off-policy replay | No | Yes (high reuse) | Efficient, but target/replay tuning critical |
| Offline actor-critic | No interaction | Yes (fixed only) | OOD extrapolation risk |
