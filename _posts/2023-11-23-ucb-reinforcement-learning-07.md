---
layout: post
title: Value Function Methods in Reinforcement Learning
subtitle: Reinforcement Learning Lecture 7
categories: Reinforcement-Learning
tags: [UCB-Deep-Reinforcement-Learning-2023]
banner: "/assets/images/banners/yuanpang-wa-valley.jpg"
---

# Value Function Methods in Reinforcement Learning

This lecture is about one core idea:

- Learn a **value function** that scores actions.
- Act by choosing the action with the highest score.

If the lecture feels heavy, use this reading order:

1. Section 2 (tiny tabular example)
2. Sections 6 and 7 (FQI and online Q-learning)
3. Section 8 (why convergence is hard with deep networks)

[Course Link](https://rail.eecs.berkeley.edu/deeprlcourse/)

# 1. Motivation: Why Value-Only Methods Exist

In actor-critic, we have:

- Actor: policy network $\pi_\theta(a \mid s)$
- Critic: value estimate (usually $V$ or $Q$)

Value-based methods ask:

> If we already know how good each action is, do we still need a separate actor network?

The value-based answer is:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

## Step-by-step mini example

At one state $s$:

- $Q(s,\text{left}) = 1.2$
- $Q(s,\text{right}) = 2.1$

Step 1: compare scores.  
Step 2: choose action with larger score.  
Step 3: policy at this state is `right`.

That is the full value-based control rule.

# 2. Tabular Dynamic Programming (Using the Same Example as Lecture 7 Slides)

The lecture's dynamic programming example assumes:

- **16 discrete states** (a 4x4 gridworld)
- **4 actions per state** (`up`, `down`, `left`, `right`)
- Known transition model $p(s' \mid s,a)$

In slide notation, transitions are stored in a tensor:

$$
T[s,s',a] = p(s' \mid s,a), \quad T \in \mathbb{R}^{16 \times 16 \times 4}
$$

This is exactly why DP is possible: we can enumerate all states/actions and compute exact expectations.

## 2.1 Policy Evaluation: "Given a policy, compute its value table"

For a deterministic policy $\pi(s)$:

$$
V^\pi(s) \leftarrow r(s,\pi(s)) + \gamma \sum_{s'} p(s' \mid s,\pi(s))V^\pi(s')
$$

### Step-by-step update on one grid state (slide-style DP)

Assume:

- discount $\gamma=0.9$
- current state is $s=10$
- current policy chooses `right` at state 10
- reward $r(10,\text{right})=-1$
- transition row from tensor:
  - $T[10,11,\text{right}] = 0.8$
  - $T[10,10,\text{right}] = 0.2$
- current value estimates:
  - $V(11)=4.0$
  - $V(10)=2.0$

Policy-evaluation backup:

$$
V_{\text{new}}(10)= -1 + 0.9\left(0.8\cdot V(11)+0.2\cdot V(10)\right)
$$

$$
= -1 + 0.9\left(0.8\cdot 4.0 + 0.2\cdot 2.0\right)
= -1 + 0.9\cdot 3.6
= 2.24
$$

So one DP backup updates state 10 from `2.0` to `2.24`.

Doing this for all 16 states repeatedly gives $V^\pi$.

## 2.2 Policy Improvement: "Greedify using the value table"

Once we have $V^\pi$, improve policy by one-step lookahead:

$$
\pi_{\text{new}}(s)=\arg\max_a \left[r(s,a)+\gamma\sum_{s'}p(s' \mid s,a)V^\pi(s')\right]
$$

### Step-by-step improvement at one state

Suppose at state $s=10$, from current value table:

- `up` one-step return = $1.1$
- `down` one-step return = $1.7$
- `left` one-step return = $0.9$
- `right` one-step return = $2.24$

Then:

$$
\pi_{\text{new}}(10) = \text{right}
$$

Do this at all 16 states and you complete one policy-improvement step.

## 2.3 Policy Iteration Loop on the 16-State Example

Policy iteration alternates:

1. Evaluate current policy to get $V^\pi$
2. Improve policy with greedy argmax

### Concrete loop

1. Start with a simple policy (for example, random or "always right").
2. Run many policy-evaluation backups over all 16 states until values stabilize.
3. Greedify all states using one-step lookahead.
4. Repeat until policy stops changing.

When policy no longer changes, you have reached an optimal policy for this tabular MDP.

## 2.4 Value Iteration on the Same 16-State Example

Value iteration merges evaluation + improvement into one update:

$$
V_{k+1}(s)=\max_a \left[r(s,a)+\gamma\sum_{s'}p(s' \mid s,a)V_k(s')\right]
$$

### One explicit value-iteration update (same state 10)

Suppose from current table:

- `up`: $1.1$
- `down`: $1.7$
- `left`: $0.9$
- `right`: $2.24$

Then:

$$
V_{k+1}(10)=\max(1.1,1.7,0.9,2.24)=2.24
$$

Perform this at all 16 states each iteration.  
After convergence, derive policy with:

$$
\pi^\star(s)=\arg\max_a \left[r(s,a)+\gamma\sum_{s'}p(s' \mid s,a)V^\star(s')\right]
$$

# 3. Bellman Operator and Contraction (Why Tabular Algorithms Converge)

Define Bellman optimality operator:

$$
(\mathcal{B}V)(s)=\max_a \left[r(s,a)+\gamma\sum_{s'}p(s' \mid s,a)V(s')\right]
$$

Contraction statement:

$$
\lVert \mathcal{B}V-\mathcal{B}\bar V \rVert_\infty \le \gamma \lVert V-\bar V \rVert_\infty
$$

## Step-by-step numeric check (toy MDP)

Take two value vectors:

- $V=[0,0]$ for $(s_0,s_1)$
- $\bar V=[10,10]$

Distance before backup:

$$
\lVert V-\bar V \rVert_\infty = 10
$$

Apply Bellman backup:

- $\mathcal{B}V = [1,3]$
- $\mathcal{B}\bar V = [9,3]$

Distance after backup:

$$
\lVert \mathcal{B}V-\mathcal{B}\bar V \rVert_\infty = 8
$$

And $8 \le 0.9 \cdot 10 = 9$, so distance shrank. Repeating this drives values to one fixed point $V^\star$.

# 4. Why Tabular Methods Break in Real Problems

Tabular means "store one value per state or per state-action pair."

That fails when state spaces are huge.

## Concrete scale example

Suppose input is one grayscale image of size $84\times84$.

- Number of pixels: $7056$
- Each pixel has 256 possibilities
- Number of possible observations: $256^{7056}$

That is astronomically large, so you cannot keep a lookup table.

## What function approximation changes

Instead of a table, learn:

$$
V_\phi(s) \quad \text{or} \quad Q_\phi(s,a)
$$

A neural network shares parameters across states, so similar states can share learned structure.

# 5. Fitted Value Iteration (FVI)

FVI keeps Bellman logic but replaces exact tabular updates with supervised regression.

## 5.1 Update Pattern

$$
y_i=\max_a\left[r(s_i,a)+\gamma \mathbb{E}_{s'\sim p(\cdot \mid s_i,a)}[V_\phi(s')]\right]
$$

Then fit:

$$
\min_\phi \frac{1}{2}\sum_i \left(V_\phi(s_i)-y_i\right)^2
$$

## Step-by-step example

Suppose current predictions are:

- $V_\phi(s_0)=1.2$
- $V_\phi(s_1)=2.0$

Targets from toy MDP:

- At $s_1$: $y_1=\max(0,3)=3$
- At $s_0$: $y_0=\max(1,0+0.9\cdot2.0)=1.8$

Regression wants to move:

- $V_\phi(s_1): 2.0 \to 3.0$
- $V_\phi(s_0): 1.2 \to 1.8$

## 5.2 Practical Limitations

FVI still needs the expectation over next states for each action, so you usually need either:

- known transition model, or
- expensive simulation access per action at chosen states

That is why model-free RL often prefers Q-based methods.

# 6. Fitted Q-Iteration (FQI)

FQI avoids the model expectation by learning $Q(s,a)$ from sampled transitions.

## 6.1 Core Target

Given replay data $\mathcal{D}=\{(s_i,a_i,r_i,s'_i)\}$:

$$
y_i=r_i+\gamma\max_{a'}Q_\phi(s'_i,a')
$$

Then fit:

$$
\min_\phi \frac{1}{2}\sum_i\left(Q_\phi(s_i,a_i)-y_i\right)^2
$$

## 6.2 Why this is useful

- model-free
- off-policy (reuse old data)
- no separate actor needed (policy is greedy over $Q$)

## 6.3 Step-by-step batch example

Use this batch:

$$
\mathcal{D}=\{(s_0,\text{right},0,s_1), (s_0,\text{left},1,\text{terminal}), (s_1,\text{right},3,\text{terminal}), (s_1,\text{left},0,\text{terminal})\}
$$

Assume current network at $s_1$ gives:

- $Q_\phi(s_1,\text{right})=2.2$
- $Q_\phi(s_1,\text{left})=0.4$

Compute targets:

1. $(s_0,\text{right},0,s_1)$:
$$
y=0+0.9\cdot\max(2.2,0.4)=1.98
$$
2. $(s_0,\text{left},1,\text{terminal})$:
$$
y=1
$$
3. $(s_1,\text{right},3,\text{terminal})$:
$$
y=3
$$
4. $(s_1,\text{left},0,\text{terminal})$:
$$
y=0
$$

Then one supervised update pushes each $Q_\phi(s,a)$ toward these labels.

# 7. Online Q-Learning and Exploration

Online Q-learning is the per-transition version of FQI.

## 7.1 Single-step update

For transition $(s_t,a_t,r_t,s_{t+1})$:

$$
y_t=r_t+\gamma\max_{a'}Q_\phi(s_{t+1},a')
$$

$$
\delta_t = Q_\phi(s_t,a_t)-y_t
$$

Then do one gradient step to reduce $\delta_t^2$.

## 7.2 Step-by-step propagation example

Initialize all $Q=0$. Let $\alpha=0.5$.

Observed transitions in order:

- $s_1, \text{right}, r=3, \text{terminal}$
- $s_0, \text{right}, r=0, s_1$

Update 1 on $(s_1,\text{right})$:

$$
y=3,
\quad Q_{\text{new}}(s_1,\text{right})=0+0.5(3-0)=1.5
$$

Update 2 on $(s_0,\text{right})$:

$$
y=0+0.9\cdot\max(Q(s_1,\text{left}),Q(s_1,\text{right}))=0.9\cdot1.5=1.35
$$

$$
Q_{\text{new}}(s_0,\text{right})=0+0.5(1.35-0)=0.675
$$

You can see reward information moving backward from $s_1$ to $s_0$.

## 7.3 Why exploration is necessary

If you always pick current argmax early, you may never try better actions.

### Epsilon-greedy rule

- with probability $1-\epsilon$: greedy action
- with probability $\epsilon$: random action

### Boltzmann rule

$$
\pi(a \mid s) \propto \exp\left(\frac{Q_\phi(s,a)}{\tau}\right)
$$

- lower $\tau$: more greedy
- higher $\tau$: more exploratory

# 8. Why Deep Value Methods Can Be Unstable

This is the mathematically hard part of the lecture.

## 8.1 Two operators in deep value learning

Think of each iteration as two steps:

1. Bellman backup $\mathcal{B}$: compute ideal targets
2. Projection $\Pi$: fit those targets with limited model class (network)

Combined update is roughly:

$$
V \leftarrow \Pi\,\mathcal{B}V
$$

## 8.2 Concrete intuition for "mixed norms" problem

Suppose true optimum is $V^\star=[10,0]$.

- Bellman-like step gives candidate $[9,1]$ (very close to optimum)
- Your model class only allows constant vectors, so projection gives $[5,5]$

What happened:

- Projection may reduce Euclidean fit to target class
- But max-error distance to true optimum can get worse

So "each substep looks reasonable" does not imply global convergence.

## 8.3 Why Q-learning is not full gradient descent

Target depends on the same parameters:

$$
y_i=r_i+\gamma\max_{a'}Q_\phi(s'_i,a')
$$

As $\phi$ changes, $y_i$ changes too. Standard Q-learning treats $y_i$ as fixed within each step, so this is called a **semi-gradient** method.

## 8.4 Step-by-step moving target example

Let $r=1$, $\gamma=0.9$.

If max next-Q estimate changes during training:

- A: max next-Q $=4.0$ gives $y=4.6$
- B: max next-Q $=5.0$ gives $y=5.5$
- C: max next-Q $=3.5$ gives $y=4.15$

The target itself moves. That is why target networks and replay buffers are so important in practice.

# 9. End-to-End Mini Trace (From Data to Policy)

This ties all sections together in one short run.

Step 1: collect transitions  
$(s_0,\text{right},0,s_1)$ and $(s_1,\text{right},3,\text{terminal})$.

Step 2: update $Q(s_1,\text{right})$ upward from reward 3.

Step 3: next time $(s_0,\text{right})$ is updated, it uses bootstrapped value from $s_1$.

Step 4: now $Q(s_0,\text{right})$ exceeds $Q(s_0,\text{left})=1$.

Step 5: greedy policy flips to `right` at $s_0$.

This is the core mechanism of value-based RL:

- learn values from data
- propagate future reward backward
- choose actions via argmax


# 10. Practical Reading Guide

If Section 8 feels hard, this sequence usually works better:

1. Understand Section 2 toy MDP completely.
2. Practice Section 6 target calculations by hand.
3. Follow Section 7 online updates step by step.
4. Only then read Section 8 theory.

# 11. Final Takeaways

1. Value-based RL turns control into repeated Bellman target fitting.
2. Tabular methods are clean and convergent because Bellman backup is a contraction.
3. FQI and online Q-learning make the method model-free and data-efficient.
4. Exploration is mandatory for discovering high-value actions.
5. Deep function approximation is powerful but introduces instability and moving-target issues.
