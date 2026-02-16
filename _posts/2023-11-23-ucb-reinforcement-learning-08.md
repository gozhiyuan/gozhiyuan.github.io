---
layout: post
title: Q-Functions in Reinforcement Learning
subtitle: Reinforcement Learning Lecture 8
categories: Reinforcement-Learning
tags: [UCB-Deep-Reinforcement-Learning-2023]
banner: "/assets/images/banners/yuanpang-wa-valley.jpg"
---

# Q-Functions in Reinforcement Learning

Lecture 8 is about making Q-learning work **in practice**.

Lecture 7 gave the value-function theory.  
Lecture 8 focuses on practical stability tricks used in deep RL systems:

- replay buffers
- target networks
- double Q-learning
- multi-step targets
- handling continuous actions

[Course Link](https://rail.eecs.berkeley.edu/deeprlcourse/)

## 1. Recap: Fitted Q-Iteration and Online Q-Learning

![alt_text](/assets/images/reinforcement-learning/08/1.png "image_tooltip")

Fitted Q-Iteration (FQI) uses data tuples:

$$
(s_i,a_i,r_i,s'_i)
$$

and Bellman targets:

$$
y_i = r_i + \gamma \max_{a'} Q_\phi(s'_i,a')
$$

Then it fits:

$$
\min_\phi \frac{1}{2}\sum_i\left(Q_\phi(s_i,a_i)-y_i\right)^2
$$

Online Q-learning is the one-transition-at-a-time special case.

### Fitted Q-Iteration vs Online Q-Learning (Direct Comparison)

They use the same Bellman target idea, but the **update schedule** is different.

| Aspect | Fitted Q-Iteration (FQI) | Online Q-Learning |
|---|---|---|
| Data usage | Uses a batch dataset (often replay buffer) | Uses the newest transition immediately |
| Update frequency | Usually many SGD steps per collected chunk | Typically one SGD step per new step |
| Sample correlation | Lower (random batch sampling) | Higher (sequential trajectory data) |
| Stability | Usually more stable | Usually less stable without tricks |
| Typical view | Batch/off-policy regression loop | Streaming/off-policy TD loop |

FQI-style loop:

1. Collect many transitions.
2. Build targets for a sampled batch.
3. Run one or multiple regression updates.
4. Repeat.

Online Q-learning loop:

1. Take one action, observe one transition.
2. Build one target from that transition.
3. Run one update.
4. Repeat immediately.

#### Step-by-step example: FQI loop (batch style)

Assume:

- replay dataset currently has 3 transitions:
  - $(s_1,a_1,r=1,s'_1)$
  - $(s_2,a_2,r=0,s'_2)$
  - $(s_3,a_1,r=2,s'_3)$
- discount $\gamma=0.9$
- current target-network estimates:
  - $\max_{a'}Q(s'_1,a')=4.0$
  - $\max_{a'}Q(s'_2,a')=1.5$
  - $\max_{a'}Q(s'_3,a')=3.0$

Step 1 (collect): use these 3 transitions as one mini-batch.
Step 2 (targets):

$$
y_1=1+0.9\cdot4.0=4.6,\quad
y_2=0+0.9\cdot1.5=1.35,\quad
y_3=2+0.9\cdot3.0=4.7
$$

Step 3 (regression): run SGD on

$$
\sum_{i=1}^3\left(Q_\phi(s_i,a_i)-y_i\right)^2
$$

What is $Q_\phi(s_i,a_i)$ here?

- It is the network's current prediction for the **current state-action pair** from data.
- Define:
  - prediction $q_i = Q_\phi(s_i,a_i)$
  - target (target-network version):

$$
y_i = r_i + \gamma \max_{a_{\text{next}}} Q_{\phi_{\text{tgt}}}(s_{\text{next},i}, a_{\text{next}})
$$

  - target (no target network):

$$
y_i = r_i + \gamma \max_{a_{\text{next}}} Q_{\phi}(s_{\text{next},i}, a_{\text{next}})
$$
- The update minimizes $(q_i-y_i)^2$, so the TD-style error is:

$$
\delta_i = q_i - y_i
$$

So, in one sample:

- Current pair $(s_i,a_i)$ is used for the fitted prediction term.
- Next state $s_{\text{next},i}$ is used inside the bootstrap target term.
- In vanilla Q-learning/FQI, we do **max over next actions** $a_{\text{next}}$, not the actually sampled next action.
- $(q_i-y_i)$ is **not** usually 0 during training; if it were exactly 0 for all samples, learning would stop.

Step 4: optionally do multiple SGD passes on this batch (or other sampled batches) before collecting more data.

#### Step-by-step example: Online Q-learning loop (streaming style)

Assume:

- one new transition arrives: $(s,a,r=1,s')$
- $\gamma=0.9$
- current estimates at $s'$:
  - $Q(s',a_1)=2.0$
  - $Q(s',a_2)=3.0$
- current prediction for sampled pair: $Q(s,a)=2.5$
- learning rate $\alpha=0.5$

Step 1 (observe one transition): got $(s,a,1,s')$.
Step 2 (target):

$$
y=1+0.9\cdot\max(2.0,3.0)=3.7
$$

Step 3 (one update):

$$
Q_{\text{new}}(s,a)=Q(s,a)+\alpha\left(y-Q(s,a)\right)
=2.5+0.5(3.7-2.5)=3.1
$$

Step 4: immediately move to the next environment step and repeat with the next single transition.

In short: **Online Q-learning is the fast, streaming limit of FQI**.  
Same formula, different optimization regime.

## 2. Why Naive Online Q-Learning Is Unstable

Slides highlight two core issues.
![alt_text](/assets/images/reinforcement-learning/08/2.png "image_tooltip")

### 2.1 It is not full gradient descent

The target depends on the same parameters:

$$
y_i=r_i+\gamma\max_{a'}Q_\phi(s'_i,a')
$$

If we write the loss for one sample as

$$
L_i(\phi)=\frac{1}{2}\left(Q_\phi(s_i,a_i)-y_i(\phi)\right)^2,
$$

then both terms depend on $\phi$:

- prediction term: $Q_\phi(s_i,a_i)$
- target term: $y_i(\phi)=r_i+\gamma\max_{a'}Q_\phi(s'_i,a')$

In **full** gradient descent, chain rule would differentiate through both dependencies.
Standard Q-learning does **not** backprop through the target branch; it treats $y_i$ as fixed inside each update.

So practical Q-learning uses a **semi-gradient**:

$$
\nabla_\phi L_i \approx \left(Q_\phi(s_i,a_i)-y_i\right)\nabla_\phi Q_\phi(s_i,a_i)
$$

This approximation often works well, but it is not true gradient descent on a stationary objective.

#### What \"semi-gradient\" means in practice

In implementation, each update usually does:

1. Compute target with current (or target-network) parameters:
   $$
   y_i = r_i + \gamma\max_{a'}Q(s'_i,a')
   $$
2. **Freeze $y_i$ as a number for this update step**.
3. Update $\phi$ using gradient through $Q_\phi(s_i,a_i)$ only.
4. Recompute targets again in the next step/iteration.

Important distinction:

- **Semi-gradient** = optimization detail (do we differentiate through the target?).
- **Off-policy** = data-collection detail (can data come from a different behavior policy?).

Q-learning is both:

- off-policy (can learn from replay/older behavior policies), and
- semi-gradient (ignores gradient through target during each update).

#### Quick numeric intuition

Suppose for one sample:

- $Q_\phi(s_i,a_i)=2.0$
- $y_i=3.0$

Then error is $-1.0$, so update pushes $Q_\phi(s_i,a_i)$ upward.
But after parameters change, $y_i$ also changes (because it uses $Q_\phi$ at next state), so the target itself moves while you optimize.

### 2.2 Samples are highly correlated

Without replay, updates use consecutive transitions from one trajectory:

$$
(s_t,a_t,r_t,s_{t+1}),\ (s_{t+1},a_{t+1},r_{t+1},s_{t+2}),\ldots
$$

These samples are very similar (same episode, nearby states), so they are not i.i.d.
That breaks the usual SGD assumption that each mini-batch is a roughly independent draw from the data distribution.

Why this hurts:

- gradients point in very similar directions for many steps
- network can overfit to the current local region of state space
- when trajectory context changes, value estimates can swing back

#### Concrete intuition

Imagine a driving agent on one highway segment for 300 steps:

- many transitions are almost identical
- updates repeatedly emphasize that narrow region
- values for other road types (intersections, ramps) are not trained enough

Result: training looks unstable and generalization is weak.
Replay buffers fix this by mixing transitions from many times/episodes before each update.

### 2.3 Moving target effect

As soon as $\phi$ changes, many targets change too, because targets are built from Q-values.

For one sample:

$$
y_i = r_i + \gamma\max_{a'}Q_\phi(s'_i,a')
$$

If you update $\phi$ to fit sample A, then $Q_\phi$ at many next states changes.
That immediately changes target values for samples B, C, D, ... even before training on them.

So the learner is not just fitting a noisy label; it is fitting a label that keeps moving.

This is closely related to 2.1:

- 2.1 is the mathematical view (semi-gradient vs full gradient).
- 2.3 is the optimization view (labels drift during training).

#### Step-by-step instability example

Fix $r=1,\gamma=0.9$ for one transition, but next-state estimate changes during training:

- Iteration A: max next-Q = 3.0, target $y=3.7$
- Iteration B: max next-Q = 4.0, target $y=4.6$
- Iteration C: max next-Q = 2.5, target $y=3.25$

Same sample, different targets over time. This is why training can oscillate.

#### Two-sample example (why one update perturbs another sample's target)

Suppose we have two transitions:

- sample A: $(s_A,a_A,r_A=1,s'_A)$
- sample B: $(s_B,a_B,r_B=0,s'_B)$

Current next-state estimates:

- $\max_{a'}Q_\phi(s'_A,a')=3.0 \Rightarrow y_A=1+0.9\cdot3.0=3.7$
- $\max_{a'}Q_\phi(s'_B,a')=2.0 \Rightarrow y_B=0+0.9\cdot2.0=1.8$

Now do one SGD update mostly driven by sample A.
After that update, suppose estimates become:

- $\max_{a'}Q_\phi(s'_A,a')=3.4 \Rightarrow y_A=4.06$
- $\max_{a'}Q_\phi(s'_B,a')=2.6 \Rightarrow y_B=2.34$

Notice: we did not directly train on sample B in this step, but B's target still moved from 1.8 to 2.34.

This is the \"chasing a moving target\" problem.
Target networks reduce this by using slower-changing parameters in the target term.


### 2.4 Parallelization Solution
Parallelization is another way to reduce correlation in online Q-learning.
The core idea is simple: collect data from many environments/workers at once, so updates are not dominated by one trajectory.

![alt_text](/assets/images/reinforcement-learning/08/3.png "image_tooltip")

#### A) Synchronized Parallel Q-Learning

All workers collect transitions in parallel, then synchronize for one joint update.

How it works:

1. Each of $W$ workers gathers one transition (or a short rollout).
2. Combine all worker data into one batch.
3. Compute one gradient update from that aggregated batch.
4. Broadcast updated parameters back to workers.

Why it helps:

- Data from different workers is less correlated than a single trajectory.
- Batch gradient has lower variance.

Mini-example:

- 16 workers each collect one transition per cycle.
- Instead of updating from 1 sequential sample, you update from a 16-sample cross-worker batch.
- Correlation drops because samples come from different states/episodes/workers.

#### B) Asynchronous Parallel Q-Learning

Workers do not wait for each other.

How it works:

1. Worker pulls latest parameters from a shared parameter server (or shared model).
2. Worker collects data and computes gradients locally.
3. Worker pushes update back immediately.
4. Other workers continue in parallel with possibly slightly stale parameters.

Why this can still work for Q-learning:

- Q-learning is off-policy, so data from slightly older behavior policies is still usable.
- Throughput is higher because no global barrier.

Mini-example:

- Worker A updates model at time $t$.
- Worker B is still using parameters from $t-2$ while collecting data.
- That staleness is usually acceptable in off-policy Q-learning, though too much lag can hurt stability.

#### Practical note

Parallelization and replay buffers are often complementary:

- Parallelism increases data diversity and throughput.
- Replay buffer further decorrelates and reuses that data.

In many modern implementations, replay + target networks are still the default stability backbone, with parallel rollout workers added for scale.

## 3. Replay Buffers: Decorrelate and Reuse Data

![alt_text](/assets/images/reinforcement-learning/08/4.png "image_tooltip")

Replay buffer is a dataset $\mathcal{B}$ that stores transitions:

$$
(s_t,a_t,r_t,s_{t+1})
$$

Instead of learning only from the newest transition, Q-learning repeatedly trains on random mini-batches from $\mathcal{B}$.

### 3.1 Why replay helps

Replay improves training in three ways:

- **Decorrelates samples**: random sampling breaks strong temporal correlation.
- **Reduces gradient variance**: batch average is smoother than single-sample updates.
- **Reuses experience**: each transition can contribute to many updates.

### 3.2 Buffer lifecycle (what happens in practice)

1. **Collect** transition with behavior policy (often epsilon-greedy).
2. **Append** transition to buffer.
3. **Sample** random mini-batch from buffer.
4. **Compute targets** for sampled transitions.
5. **Update** network parameters with SGD.
6. **Repeat**, while old entries are evicted when capacity is full.

Common implementation details:

- warm-up period (fill buffer before learning starts)
- finite capacity with FIFO/ring eviction
- update ratio $K$: do $K$ gradient steps per new env step

### 3.3 Step-by-step workflow (numeric)

Assume:

- buffer capacity = 100000
- current size = 20000
- mini-batch size = 256
- update ratio $K=4$
- discount $\gamma=0.99$

Per environment step:

1. collect one new transition and append to buffer
2. repeat 4 times:
   - sample 256 random transitions
   - for each sampled item compute
     $$
     y_i = r_i + \gamma \max_{a'}Q_{\phi_{\text{tgt}}}(s'_i,a')
     $$
   - optimize
     $$
     \frac{1}{256}\sum_i\left(Q_\phi(s_i,a_i)-y_i\right)^2
     $$

So each new transition can influence learning many times (directly now, indirectly later when resampled).

### 3.4 Correlation example (why random sampling matters)

Without replay, consecutive updates might use:

$$
(s_t,a_t,r_t,s_{t+1}),\ (s_{t+1},a_{t+1},r_{t+1},s_{t+2}),\ (s_{t+2},a_{t+2},r_{t+2},s_{t+3})
$$

These are very similar and push gradients in a narrow direction.

With replay, one mini-batch may include transitions from:

- different episodes
- different parts of episodes
- older and newer policies

This makes updates more i.i.d.-like and usually much more stable.

### 3.5 Practical tuning tips

- Use a large enough buffer (too small behaves almost online).
- Start updates after a warm-up threshold.
- Tune update ratio $K$ carefully:
  - higher $K$ improves data efficiency
  - too high can overfit stale replay data
- Keep target networks with replay; they work best together.

### 3.6 Connection to Fitted Q-Iteration (FQI)

Replay-buffer Q-learning is very close to FQI in spirit.

FQI-style view:

1. collect a dataset
2. run batch Bellman-regression updates
3. collect more data and repeat

Replay-buffer Q-learning view:

1. continuously append new transitions to buffer
2. repeatedly sample random mini-batches from buffer
3. run Bellman-regression SGD updates

So you can think of replay-buffer Q-learning as an **incremental/streaming approximation of FQI**:

- same Bellman target structure
- same regression objective
- different scheduling (continuous refresh vs fixed dataset phases)

## 4. Target Networks: Freeze the Regression Target

Use two Q-networks:

- online network $Q_\phi$ (updated every gradient step)
- target network $Q_{\phi'}$ (updated more slowly)

Target uses the target network:

$$
y_i = r_i + \gamma\max_{a'}Q_{\phi'}(s'_i,a')
$$

This makes the inner regression problem much more stationary.

![alt_text](/assets/images/reinforcement-learning/08/5.png "image_tooltip")

### 4.1 Why target networks matter (more detail)

Without a target network, each update changes the same network used to define the target:

$$
y_i = r_i + \gamma \max_{a'} Q_\phi(s'_i,a')
$$

So both prediction and labels move together, creating the "chasing your own tail" behavior from Section 2.3.

With target networks, we decouple them:

- prediction uses online network $Q_\phi$
- label uses slower network $Q_{\phi'}$

$$
y_i = r_i + \gamma \max_{a'} Q_{\phi'}(s'_i,a')
$$

This makes the inner optimization look more like stable supervised regression.

### 4.2 Hard vs soft target updates (practical tradeoff)

![alt_text](/assets/images/reinforcement-learning/08/6.png "image_tooltip")

Hard update:

- every $N$ steps, copy $\phi' \leftarrow \phi$
- simple and common in classic DQN
- downside: lag is uneven (fresh right after copy, stale just before next copy)

Soft/Polyak update:

$$
\phi' \leftarrow \tau\phi' + (1-\tau)\phi
$$

- smoother lag at every step
- often easier to tune in continuous-control style algorithms
- common $\tau$ is close to 1 (e.g., 0.995 to 0.999 depending on convention)

Why this can work even though it is \"just parameter interpolation\":

- The target network is **not** the model you deploy for acting; it is a slowly moving teacher used only to compute stable Bellman targets.
- Exponential averaging acts like a **low-pass filter** on parameter noise from SGD, so target values change smoothly instead of jumping each step.
- Stability matters more than exact instant optimality for the target network; a slightly stale but smooth target is often better than a fresh but highly noisy one.
- Empirically, this reduces oscillation/divergence in bootstrap methods, even if parameter averaging is not theoretically perfect in non-convex neural nets.

### 4.3 Classic DQN loop (organized)

A standard DQN-style training cycle:

1. Act in environment with epsilon-greedy policy from $Q_\phi$.
2. Store transition $(s,a,r,s')$ in replay buffer.
3. Sample random mini-batch from buffer.
4. Build targets with target network:
   $$
   y_j = r_j + \gamma \max_{a'} Q_{\phi'}(s'_j,a')
   $$
5. Update online network by minimizing:
   $$
   \frac{1}{B}\sum_j\left(Q_\phi(s_j,a_j)-y_j\right)^2
   $$
6. Update target network (hard copy every $N$ steps, or soft update every step).

### 4.4 Step-by-step mini example

Suppose a batch has one sample:

- $r=2$, $\gamma=0.9$
- target network gives $\max_{a'}Q_{\phi'}(s',a')=5$
- online prediction is $Q_\phi(s,a)=4$

Target:

$$
y = 2 + 0.9\cdot 5 = 6.5
$$

Error:

$$
Q_\phi(s,a)-y = 4 - 6.5 = -2.5
$$

SGD update pushes $Q_\phi(s,a)$ upward toward 6.5.
Because $\phi'$ is fixed (for now), the target is stable during this inner step.

### 4.5 Why this is one of the key DQN tricks

Replay buffer + target network is the classic stability pair:

- replay buffer addresses sample correlation
- target network addresses moving labels

Together they made deep Q-learning practical at scale (e.g., Atari-era DQN systems).

## 5. A General View: Data Collection, Training, Target Update

General view: Q-learning is not one loop, but three interacting processes:

1. data collection (generate transitions)
2. regression/training (fit Q with SGD)
3. target update (refresh target network)

Different algorithms mostly differ in how fast each process runs.

![alt_text](/assets/images/reinforcement-learning/08/7.png "image_tooltip")

### 5.1 Central object: replay buffer as shared memory

This three-process view is easiest to understand if replay is the shared memory between collection and training.

- Buffer stores tuples $(s,a,r,s')$
- Buffer has finite capacity (often ring/FIFO behavior)
- Old samples are evicted as new samples arrive (capacity control)

Because of this, data collection and learning are partially decoupled: the agent can collect now, train later, and reuse the same transition many times.

### 5.2 The three processes (organized)

Process 1: **Data collection**

- Interact with environment using behavior policy (often epsilon-greedy from online Q).
- Push transitions into replay buffer.

Process 2: **Target-network update**

- Maintain slower parameters $\phi'$.
- Update by hard copy every $N$ steps or by soft/Polyak update.

Process 3: **Regression (learning)**

- Sample mini-batches from replay.
- Build Bellman targets using target network.
- Run SGD on online Q-network.

A common practical schedule for each environment step:

1. collect 1 new transition and append to replay
2. run 4 SGD updates from random replay mini-batches
3. apply soft target update after each SGD step

This is usually more data-efficient than strict 1:1 (one transition, one update).

### 5.3 Algorithms as different rate configurations

From this perspective, online Q-learning, DQN, and FQI mainly differ by process rates:

- **Online Q-learning**:
  - effectively tiny replay / near-immediate sample usage
  - collection, learning, and target changes are tightly coupled
- **DQN**:
  - large replay buffer
  - target updates are intentionally slow
  - collection and learning run continuously
- **Fitted Q-Iteration (FQI)**:
  - collect a dataset chunk
  - run many regression steps (often inner loop)
  - then refresh targets/data at slower outer-loop cadence

### 5.4 Why this view matters (non-stationarity control)

Each process changes what the other processes see:

- collection changes data distribution
- learning changes $Q_\phi$
- target updates change bootstrapped labels

Stability comes from separating timescales:

- fast enough learning to fit current targets
- slower target updates so labels do not move too quickly
- replay to smooth data-distribution shifts

This is the systems-level reason practical deep Q-learning can work.

## 6. Overestimation and Double Q-Learning

In deep Q-learning, the max in the Bellman target can create a systematic upward bias when Q estimates are noisy.

![alt_text](/assets/images/reinforcement-learning/08/8.png "image_tooltip")

### 6.1 Why overestimation happens

Standard bootstrap target:

$$
y = r + \gamma \max_{a'} Q_{\phi_{\text{tgt}}}(s',a')
$$

Even if each action-value estimate has zero-mean noise, the max tends to select actions with positive noise.
So unbiased per-action noise becomes a biased maximum.

Equivalent intuition from random variables:

$$
\mathbb{E}[\max(X_1, X_2, \dots)] \ge \max(\mathbb{E}[X_1], \mathbb{E}[X_2], \dots)
$$

This is the statistical root of Q overestimation.

### 6.2 Standard vs Double Q target

![alt_text](/assets/images/reinforcement-learning/08/9.png "image_tooltip")

Standard DQN-style target (same network family for select and evaluate):

$$
y_{\text{std}} = r + \gamma Q_{\phi_{\text{tgt}}}\left(s', \arg\max_{a'} Q_{\phi_{\text{tgt}}}(s',a')\right)
$$

Double Q target (decouple select and evaluate):

$$
y_{\text{dbl}} = r + \gamma Q_{\phi_{\text{tgt}}}\left(s', \arg\max_{a'} Q_{\phi}(s',a')\right)
$$

- $Q_\phi$ (online): selects the action index
- $Q_{\phi_{\text{tgt}}}$ (target): evaluates that selected action

The key is to avoid using one noisy estimator for both selection and evaluation.

### 6.3 Step-by-step noise example

Assume true next-action values are both 5.
One update step gives:

- online: $Q_\phi(s',a_1)=6.2$, $Q_\phi(s',a_2)=4.8$ so $\arg\max=a_1$
- target: $Q_{\phi_{\text{tgt}}}(s',a_1)=5.1$, $Q_{\phi_{\text{tgt}}}(s',a_2)=4.9$

Let $r=0, \gamma=0.99$:

- optimistic same-estimator bootstrap can be near $0.99\times 6.2 = 6.138$
- Double Q bootstrap is $0.99\times 5.1 = 5.049$

Gap is $1.089$ for this single sample, showing how max noise can inflate targets.

### 6.4 Practical implementation in DQN

For each sampled transition $(s_i,a_i,r_i,s_{\text{next},i},d_i)$:

1. Select next action with online network:
   $$
   a_i^\star = \arg\max_a Q_{\phi}(s_{\text{next},i}, a)
   $$
2. Evaluate that action with target network:
   $$
   v_i = Q_{\phi_{\text{tgt}}}(s_{\text{next},i}, a_i^\star)
   $$
3. Build target:
   $$
   y_i = r_i + \gamma(1-d_i)\,v_i
   $$
4. Regress online prediction $Q_\phi(s_i,a_i)$ toward $y_i$.

So yes, Q is evaluated at both current and next states, but for different roles:

- current pair $(s_i,a_i)$ gives the prediction being trained
- next state $s_{\text{next},i}$ gives the bootstrap part of the target

### 6.5 When Double Q helps most

Double Q is especially helpful when:

- action space is large (max over many noisy values)
- estimates are noisy early in training
- rewards are sparse and targets rely heavily on bootstrap

It does not solve every instability, but it is a strong overestimation fix.

## 7. Multi-Step Q Targets

Multi-step targets use several real rewards before bootstrapping.
This often improves learning signal quality early in training.

![alt_text](/assets/images/reinforcement-learning/08/10.png "image_tooltip")

### 7.1 One-step vs N-step formulas

One-step target:

$$
y_t^{(1)} = r_t + \gamma\max_{a'}Q(s_{t+1},a')
$$

N-step target:

$$
y_t^{(N)} = \sum_{k=0}^{N-1}\gamma^k r_{t+k} + \gamma^N\max_{a'}Q(s_{t+N},a')
$$

If $N=1$, this is exactly standard Q-learning.

### 7.2 Why it can help

- often learns faster early
- reduces reliance on inaccurate bootstrap values
- propagates reward information farther per update

Reason: larger $N$ puts more weight on observed rewards and less on current Q errors.

### 7.3 Bias-variance-off-policy tradeoff

Increasing $N$ changes three things:

- lower bootstrap bias (good)
- higher target variance from sampled trajectories (cost)
- stronger off-policy mismatch sensitivity (cost)

So practical systems usually choose moderate $N$ (for example 3 or 5), not very large values.

### 7.4 Step-by-step 3-step example

Let $\gamma=0.9$, rewards be $(1,2,3)$, and tail bootstrap value be 4:

$$
y^{(3)} = 1 + 0.9\cdot2 + 0.9^2\cdot3 + 0.9^3\cdot4
$$

$$
= 1 + 1.8 + 2.43 + 2.916 = 8.146
$$

Compare with one-step if $\max_{a'}Q(s_{t+1},a')=6$:

$$
y^{(1)} = 1 + 0.9\cdot6 = 6.4
$$

The 3-step target carries richer near-future reward information than 1-step.

### 7.5 Why off-policy gets tricky for N-step

Why $N=1$ is usually safe in off-policy Q-learning:

- target uses a max over next actions, not the sampled next action

Why $N>1$ is harder:

- intermediate rewards come from behavior-policy actions in replay
- current policy may choose different actions
- then the sampled reward sequence is not exactly from the current target policy

This mismatch introduces extra bias.

### 7.6 Practical fixes for off-policy N-step

Common approaches:

- use small $N$ (e.g., 3 or 5)
- accept small mismatch bias in practice when behavior and target are close
- apply importance sampling corrections (more principled, can increase variance)
- truncate traces when policy mismatch is large

In practice, many deep RL systems pick moderate N-step returns because the speedup is often worth the approximation.

## 8. Q-Learning with Continuous Actions

In discrete action spaces, the bootstrap step

$$
\max_{a'} Q(s',a')
$$

is cheap because we can enumerate all actions.
In continuous action spaces, this inner maximization becomes a real optimization problem.

This affects both:

- action selection at interaction time
- target construction during critic updates

So practical methods focus on making this maximization fast and stable.

### 8.1 Option 1: Optimize action numerically inside the loop

Use a black-box optimizer to approximate

$$
a^\star(s') \approx \arg\max_a Q_\phi(s',a)
$$

Common choices: random shooting, CEM, CMA-ES, or gradient-based action optimization.

#### Step-by-step random-shooting example

Assume 1D action range $a\in[-1,1]$ for one next state $s'$.

1. Sample candidate actions: $\{-1.0,-0.5,0.0,0.5,1.0\}$.
2. Evaluate critic:
   $Q(s',-1.0)=1.2$, $Q(s',-0.5)=1.9$, $Q(s',0.0)=2.4$, $Q(s',0.5)=2.1$, $Q(s',1.0)=1.5$.
3. Pick best candidate: $a^\star=0.0$.
4. Use in target: $y=r+\gamma Q(s',a^\star)$.

This is simple and easy to parallelize, but expensive in high-dimensional action spaces.

#### CEM workflow (high level)

1. Initialize Gaussian over actions.
2. Sample candidates from the Gaussian.
3. Keep top-k actions by Q-value.
4. Refit Gaussian to elites.
5. Repeat for a few iterations; output mean or best sample.

Better than pure random shooting, but still adds nontrivial compute inside each target update.

### 8.2 Option 2: Constrain Q to be easy to maximize (NAF)

NAF (Normalized Advantage Function) uses a critic form where argmax is analytic.

Typical decomposition:

$$
Q(s,a) = V(s) + A(s,a)
$$

with quadratic advantage:

$$
A(s,a) = -\frac{1}{2}(a-\mu(s))^\top P(s)(a-\mu(s)), \quad P(s) \succeq 0
$$

Because the quadratic is maximized at its center,

$$
\arg\max_a Q(s,a)=\mu(s)
$$

No inner optimization loop is needed.

#### Step-by-step intuition

For a fixed state $s$:

1. Network outputs $\mu(s)=0.3$ and quadratic curvature matrix $P(s)$.
2. Any action away from $0.3$ gets a negative advantage penalty.
3. Maximum Q occurs exactly at $a=0.3$.
4. Target computation directly plugs in this maximizing action.

Tradeoff: fast maximization, but less expressive critic class due to quadratic action structure.

### 8.3 Option 3: Learn a separate maximizer network (DDPG/TD3 style)

Learn an actor $\mu_\theta(s)$ that approximates the maximizing action, and a critic $Q_\phi(s,a)$ that evaluates actions.

![alt_text](/assets/images/reinforcement-learning/08/11.png "image_tooltip")

Actor idea:

$$
\mu_\theta(s) \approx \arg\max_a Q_\phi(s,a)
$$

Critic target:

$$
y_i = r_i + \gamma(1-d_i)\,Q_{\phi_{\text{tgt}}}\big(s_{\text{next},i},\mu_{\theta_{\text{tgt}}}(s_{\text{next},i})\big)
$$

Actor objective (maximize critic value):

$$
J(\theta)=\mathbb{E}_{s\sim\mathcal{B}}\big[Q_\phi(s,\mu_\theta(s))\big]
$$

#### Step-by-step training workflow

1. Collect transition $(s,a,r,s_{\text{next}},d)$ and store in replay.
2. Sample mini-batch from replay.
3. Build critic targets using target actor + target critic.
4. Update critic by regression to targets.
5. Update actor to increase $Q_\phi(s,\mu_\theta(s))$.
6. Soft-update target networks with Polyak averaging.

This avoids per-sample inner optimization, so it scales better to higher-dimensional action spaces.

### 8.4 Side-by-side summary

**Numerical optimization (Option 1)**

- Pros: conceptually simple, flexible, no extra actor network required.
- Cons: expensive inner-loop optimization; gets harder as action dimension grows.

**Easily maximizable critic (Option 2 / NAF)**

- Pros: analytic argmax, very fast at runtime.
- Cons: restrictive Q parameterization can limit representational power.

**Actor as learned maximizer (Option 3 / DDPG family)**

- Pros: efficient in continuous control and scalable.
- Cons: introduces actor-critic coupling and extra stability concerns.

In modern continuous-control practice, actor-critic variants (DDPG/TD3/SAC families) are usually preferred.

## 9. Practical Tips from the Lecture

- start on simple tasks first (debug correctness)
- use large replay buffers for stability
- use target networks always
- clip gradients or use Huber loss
- prefer double Q-learning by default
- tune exploration schedule (high epsilon to low epsilon)
- run multiple random seeds (results are variable)

Huber loss (common robust choice):

$$
L_{\delta}(x)=\frac{1}{2}x^2 \quad \text{for } \lvert x \rvert \le \delta
$$

$$
L_{\delta}(x)=\delta\left(\lvert x \rvert-\frac{1}{2}\delta\right) \quad \text{for } \lvert x \rvert > \delta
$$

## 10. End-to-End DQN Style Loop (Concrete Checklist)

1. interact with env using epsilon-greedy policy
2. store transition in replay buffer
3. sample random mini-batch
4. compute targets with target network
5. update online Q-network by SGD
6. update target network (hard or soft)
7. repeat

This is the practical core behind most Q-learning systems in this lecture.

## 11. Key Takeaways

1. FQI is the stable conceptual template; online Q-learning is its fast but unstable limit.
2. Replay buffers + target networks are the foundational stabilization tools.
3. Double Q-learning directly addresses overestimation bias.
4. Multi-step returns can speed learning but add off-policy complexity.
5. Continuous-action Q-learning needs either inner optimization, constrained Q-form, or an actor network.
