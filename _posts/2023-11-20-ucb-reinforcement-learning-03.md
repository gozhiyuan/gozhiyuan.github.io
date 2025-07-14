---
layout: post
title: Reinforcement Learning Introduction
subtitle: Reinforcement Learning Lecture 3
categories: Reinforcement-Learning
tags: [UCB-Deep-Reinforcement-Learning-2023]
banner: "/assets/images/banners/yuanpang-wa-valley.jpg"
---

# Introduction to Reinforcement Learning

This lecture covers fundamental definitions, the objective of RL, the anatomy of RL algorithms, and a categorization of various algorithm types along with their trade-offs and assumptions.
[Course Link](https://rail.eecs.berkeley.edu/deeprlcourse/)


## 1. Core Terminology and Concepts

The fundamental difference between reinforcement learning (RL) and imitation learning (IL) lies in their reliance on labeled data to directly train models:
- Imitation Learning (IL): In imitation learning, you collect a dataset of expert demonstrations, typically consisting of observation-action tuples (e.g., humans driving a vehicle provides observation-action pairs). This dataset is then used with supervised learning algorithms to train a policy that learns to take actions resembling those of the expert. The expert's actions serve as the "labels" for the training data.
- Reinforcement Learning (RL): In contrast, reinforcement learning allows you to train policies without having access to expert data. Instead of labeled data, the objective in RL is defined by a reward function. The goal is to find policy parameters that maximize the expected value of the sum of rewards over a trajectory. RL learns through a process of "trial and error," where the policy interacts with the environment, generates its own samples (trajectories), and then improves based on the rewards it receives, rather than directly mimicking expert actions.

![alt_text](/assets/images/reinforcement-learning/03/1.png "image_tooltip")

Reinforcement Learning focuses on training policies to make decisions without expert data, driven by a reward function.

- **Policy** ($\pi_{\theta}$): A distribution over actions ($a_t$) conditioned on observations ($o_t$) or states ($s_t$). In deep RL, this is often represented by a deep neural network, with $\theta$ denoting its parameters.
- **State** ($s_t$): The true underlying configuration of the environment. It satisfies the Markov property, meaning $s_{t+1}$ is independent of $s_{t-1}$ (and prior states) when conditioned on $s_t$.
- **Observation** ($o_t$): What the agent perceives. It is a stochastic function of the state and does not necessarily contain all information to infer the full state. It does not need to satisfy the Markov property.
- **Action** ($a_t$): The output of the policy that influences the environment.
- **Reward Function** ($R(s_t, a_t)$): A scalar-valued function that quantifies the desirability of states and actions. The objective in RL is not just immediate high rewards but rather to "take actions that will lead to higher awards later," considering future consequences.
- **Transition Probability / Dynamics** ($P(s_{t+1}|s_t, a_t)$): Specifies the probability of moving to a new state $s_{t+1}$ given the current state $s_t$ and action $a_t$.


## 2. Markov Decision Processes (MDPs) and Related Concepts

- **Markov Chain**: A stochastic process with a set of states ($\mathcal{S}$) and a transition function ($\mathcal{T}$), where $\mathcal{T}$ denotes $P(s_{t+1}|s_t)$.
- **Markov Decision Process (MDP)**: Augments a Markov chain by adding an action space ($\mathcal{A}$) and a reward function ($R$). Transitions now depend on both state and action: $P(s_{t+1}|s_t, a_t)$.
- **Partially Observed Markov Decision Process (POMDP)**: Adds an observation space ($\mathcal{O}$) and emission probability ($P(o_t|s_t)$). Actions are selected based on observations.


## 3. The Reinforcement Learning Objective

The fundamental goal in reinforcement learning is to find the policy parameters (denoted as $\theta$) that maximize the expected value of the sum of rewards over a trajectory.  
A trajectory is a sequence of states and actions over time, like $s_1a_1s_2a_2\ldots s_Ta_T$.  
This expectation accounts for the stochasticity of the policy, the transition probabilities of the environment, and the initial state distribution.

- **Trajectory** ($\tau$): A sequence like $s_1, a_1, s_2, a_2, \dots$
- **Finite Horizon Objective**:
  $$
  \theta^* = \arg \max_{\theta} \mathbb{E}_{\tau \sim p_{\theta}(\tau)} \left[ \sum_{t=1}^T R(s_t, a_t) \right]
  $$
- **Infinite Horizon**:
  - **Average Reward**: Divide sum of expected rewards by $T$
  - **Discounting**: Apply discount factor $\gamma < 1$
- **Stationary Distribution** ($\mu$): Under ergodicity and aperiodicity, the state-action marginal $P_{\theta}(s_t, a_t)$ converges to $\mu$, allowing definition of long-term reward.
- **Gradient Optimization**: RL uses gradient-based optimization because expectations over non-smooth functions under smooth distributions are smooth and differentiable.

![alt_text](/assets/images/reinforcement-learning/03/2.png "image_tooltip")

### Finite Horizon Objective

📏 **Finite Horizon Problem**  
For a control problem that is "finite horizon," the decision-making task has a fixed number of time steps ($T$), after which it concludes.

📊 **Trajectory Distribution**  
The joint probability distribution over trajectories, which depends on the policy ($\pi_\theta$), can be factorized using the chain rule of probability and by exploiting the Markov property.  
This factorization defines the trajectory distribution as:  
$p(s_1) \cdot \prod_{t=1}^T \pi_\theta(a_t|s_t) \cdot p(s_{t+1}|s_t, a_t)$

🧮 **Rewriting the Objective using Linearity of Expectation**  
While the primary objective is:  
$$
\mathbb{E}_{\tau \sim p_\theta(\tau)} \left[ \sum_t R(s_t, a_t) \right]
$$  
It can be rewritten using linearity of expectation:  
$$
\sum_t \mathbb{E}_{\tau \sim p_\theta(\tau)} [R(s_t, a_t)]
$$  
Since $R(s_t, a_t)$ only depends on state and action at time $t$, we marginalize out other variables:  
$$
\sum_t \mathbb{E}_{(s_t, a_t) \sim p_\theta(s_t, a_t)} [R(s_t, a_t)]
$$

💡 **Significance**  
While this may seem like a small math trick, it's incredibly useful in deriving and understanding RL algorithms.  
Especially for infinite horizon problems, this expression generalizes neatly with stationary distributions.  
It shows that at each time step, the RL objective focuses on the expected reward given the likely state-action pairs from the policy and environment.


### Infinite Horizon Objective

🧠 In the context of reinforcement learning, the "infinite horizon case" addresses scenarios where the decision-making task does not have a fixed end time ($T \to \infty$), introducing the concept of a stationary distribution for the objective.

📆 Finite Horizon Context  
Previously, the RL objective for a finite horizon task was to find policy parameters ($\theta$) that maximize the expected sum of rewards over a trajectory.  
This could be rewritten as a sum over time of the expected reward for every state-action marginal $p_\theta(s_t, a_t)$.

🚧 Challenge of Infinite Horizon  
When $T \to \infty$, summing positive rewards indefinitely may yield an undefined or infinite result.

🧮 Making the Objective Finite  
To handle this, two main strategies are used:

- 📊 **Average Reward Formulation**: Divide the total reward by $T$ to compute average reward per time step.
- 🧾 **Discounting**: Apply a discount factor to future rewards (e.g., $\gamma^t$) so they decay and the infinite sum converges.

🔁 Augmented Markov Chain  
To analyze infinite horizon settings, the policy and environment can be modeled together as an augmented Markov chain, where the state-action pair $(s_t, a_t)$ becomes the new "augmented state".  
The transition operator combines the environment dynamics $p(s_{t+1}|s_t, a_t)$ and the policy $\pi_\theta(a_{t+1}|s_{t+1})$.

📉 Stationary Distribution  
We can now ask whether $p_\theta(s_t, a_t)$ converges to a **stationary distribution** $\mu$ as $t \to \infty$.

- 🔄 A stationary distribution $\mu$ satisfies:  
  $$ \mu = T \cdot \mu $$  
  where $T$ is the transition operator of the augmented Markov chain.

✅ Existence Conditions for Stationarity:

- 🔁 **Ergodicity**: Every state can eventually reach every other state with non-zero probability.
- 🔄 **Aperiodicity**: The chain doesn’t get stuck in cyclic patterns.

🏁 Infinite Horizon Objective  
If a stationary distribution $\mu$ exists, then the RL objective becomes:  
$$ \mathbb{E}_{(s, a) \sim \mu} [R(s, a)] $$  
This is the **expected reward under the stationary distribution**, providing a clean and finite formulation for the infinite horizon setting.


### Summary
- __Defining the Objective with a Reward Function__: In RL, since there's no expert data to mimic, the objective of the policy is defined by a reward function. This reward function is a scalar-valued function, typically dependent on the state and action, and it tells the agent which states and actions are "better". For example, if training a car, a state where the car drives quickly on the road might yield a high reward, while a collision would result in a low reward.
- __Maximizing Future Rewards__: A crucial aspect of the RL objective is that it's not just about getting high rewards now, but about taking actions that will lead to higher rewards later. This means the policy must consider future consequences when choosing current actions. This decision-making problem, balancing immediate and future rewards, is at the core of RL.
- __Formal Objective__: Expected Sum of Rewards: The primary goal in reinforcement learning is to find the policy parameters (theta) that maximize the expected value of the sum of rewards over a trajectory. A trajectory is a sequence of states and actions over time. This objective can be formally written as an expectation of the sum of rewards with respect to the trajectory distribution.
  - The "expectation" part is vital because it accounts for the inherent stochasticity of the policy, the environment's transition probabilities, and the initial state distribution.
  - For a finite horizon problem, where the task lasts for a fixed number of time steps (T), the joint distribution of states and actions depends on the policy, and this can be factorized using the Markov property. The objective can also be expressed equivalently as a sum over time of the expected reward for every state-action marginal.
- __Infinite Horizon Considerations__: When the control problem is infinite horizon (T goes to infinity), the sum of rewards might become infinite if rewards are always positive. To make the objective finite and well-defined, approaches like the average reward formulation can be used (dividing the sum of expected rewards by T). Alternatively, a concept called "discounts" can be applied (which will be discussed later in the course). In the infinite horizon case, assuming conditions like ergodicity and aperiodicity, the objective converges to the expected value of the reward under the stationary distribution.
- __Optimizing Expectations for Smoothness__: A fundamental principle in reinforcement learning is that it's about optimizing expectations. Even if the reward function itself is highly discontinuous or non-differentiable (e.g., +1 for winning a game, -1 for losing), the expected value of that reward with respect to a policy's parameters can be smooth and differentiable. This crucial property allows reinforcement learning algorithms to utilize smooth optimization methods, such as gradient descent, to train policies even with seemingly non-smooth or sparse reward functions.


## 4. Value Functions: Q-functions and Value Functions

🎯 In reinforcement learning (RL), the primary goal is to find a policy ($\pi_\theta$) that maximizes the expected total rewards over a trajectory. This inherently involves dealing with expectations due to the stochastic nature of both the policy and the environment.

### **Objective and the Chain Rule**  
The RL objective for a finite horizon (fixed time steps $T$) is:  
$$ \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[ \sum_{t=1}^T R(s_t, a_t) \right] $$  
This expectation can be expanded using the chain rule:
- Expectation over the initial state $p(s_1)$
- Then over action $a_1$ using policy $\pi(a_1|s_1)$
- Include reward $R(s_1, a_1)$
- Nest additional expectations over future states/actions

![alt_text](/assets/images/reinforcement-learning/03/4.png "image_tooltip")

Though it seems more complex, this expansion motivates the introduction of value functions that simplify optimization.

### Q-functions (Action-Value Functions)  

🧠 *If I start in state* $( s_t $), *take action* $( a_t )$, *and then follow policy* $( \pi )$, *what is my total expected return from now on?*
- ✅ The **immediate reward** from taking $( a_t )$ in $( s_t )$
- ➕ The **expected future rewards** from following the policy after that

A Q-function $Q^\pi(s_t, a_t)$ is:  
$$ Q^\pi(s_t, a_t) = \mathbb{E} \left[ \sum_{k=t}^T R(s_k, a_k) \middle| s_t, a_t, \pi \right] $$  

$Q^\pi(s_t, a_t)$ is the general definition: the expected return at any time $t$, given you follow policy $\pi$. $Q(s_1, a_1)$ is just a specific instance of the above at time step $t = 1$. So it's shorthand for:

$$
Q(s_1, a_1) = Q^\pi(s_1, a_1) = \mathbb{E}\left[ \sum_{k=1}^T R(s_k, a_k) \,\middle|\, s_1, a_1, \pi \right]
$$


It estimates the total expected reward when starting at $(s_t, a_t)$ and following policy $\pi$ thereafter. So it includes future rewards, not just the immediate reward (reward function). 

### Value Functions (State-Value Functions)  

🧠 *If I’m in state* $( s_t )$, *and I follow policy* $( \pi )$ *from now on, without deciding on the action myself, what is my expected return?*

- 📊 It **averages** over the actions the policy $( \pi )$ would choose at that state


A value function $V^\pi(s_t)$ is:  
$$ V^\pi(s_t) = \mathbb{E} \left[ \sum_{k=t}^T R(s_k, a_k) \middle| s_t, \pi \right] $$  
Also,  
$$ V^\pi(s_t) = \mathbb{E}_{a_t \sim \pi(a_t|s_t)} \left[ Q^\pi(s_t, a_t) \right] $$  
And the total objective can be viewed as:  
$$ \mathbb{E}_{s_1 \sim p(s_1)} \left[ V^\pi(s_1) \right] $$

### Summary

![alt_text](/assets/images/reinforcement-learning/03/5.png "image_tooltip")

| Concept           | Depends On     | Meaning                                                       |
| ----------------- | -------------- | ------------------------------------------------------------- |
| $Q^\pi(s_t, a_t)$ | State + Action | Return from $s_t$ if you take action $a_t$, then follow $\pi$ |
| $V^\pi(s_t)$      | Just State     | Average return from $s_t$, following $\pi$                    |


$$
\begin{aligned}
\mathbb{E}_{\tau \sim p_\theta(\tau)} \left[ \sum_{t=1}^T R(s_t, a_t) \right]
&= \mathbb{E}_{s_1 \sim p(s_1)} \left[ \mathbb{E}_{a_1 \sim \pi(a_1|s_1)} \left[ R(s_1, a_1) + \mathbb{E}_{s_2, a_2, \dots} \left[ \sum_{t=2}^T R(s_t, a_t) \right] \right] \right] \\
&= \mathbb{E}_{s_1 \sim p(s_1)} \left[ \mathbb{E}_{a_1 \sim \pi(a_1|s_1)} \left[ Q(s_1, a_1) \right] \right] \\
&= \mathbb{E}_{s_1 \sim p(s_1)} \left[ \mathbb{E}_{a_1 \sim \pi(a_1|s_1)} \left[ Q^\pi(s_1, a_1) \right] \right] \quad \text{(Explicitly denoting policy dependence)} \\
&= \mathbb{E}_{s_1 \sim p(s_1)} \left[ V^\pi(s_1) \right]
\end{aligned}
$$

### **Utility of Q and V Functions**

🛠️ Policy Improvement using Q-functions  
If $Q^\pi(s, a)$ is known, you can form a better policy $\pi'$ via:  
$$ \pi'(a|s) = 
\begin{cases}
1 & \text{if } a = \arg\max_{a'} Q^\pi(s, a') \\
0 & \text{otherwise}
\end{cases} $$  
➡️ This idea is the backbone of policy iteration and Q-learning (e.g., DQN for Atari).

### Gradient Estimation (Advantage Functions)  
The advantage function is:  
$$ A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s) $$  
If $A^\pi(s, a) > 0$, the action $a$ is better than average — increase its probability.  
This is used in actor-critic methods like TRPO, where gradients are more accurate and stable than pure REINFORCE.

### Summary  
Q and V functions:
- Simplify computation of expectations
- Enable better policy derivation (argmax for Q, gradient for advantage)
- Play critical roles in nearly every modern RL algorithm


- **Q-function** ($Q^{\pi}(s_t, a_t)$): Expected cumulative reward if starting from $(s_t, a_t)$ and following policy $\pi$.
- **Value Function** ($V^{\pi}(s_t)$): Expected cumulative reward from $s_t$ under $\pi$.  
  $$
  V^{\pi}(s_t) = \mathbb{E}_{a_t \sim \pi(a_t|s_t)} [Q^{\pi}(s_t, a_t)]
  $$
- **Utility**: Q-functions simplify policy improvement by selecting actions maximizing $Q^{\pi}(s_t, a_t)$.


## 5. Anatomy of a Reinforcement Learning Algorithm

The three basic parts are:
1. 🟧 Generate Samples (Orange Box)  
2. 🟩 Fit a Model / Estimate the Return (Green Box)  
3. 🟦 Improve the Policy (Blue Box)  

![alt_text](/assets/images/reinforcement-learning/03/3.png "image_tooltip")

This process is repeated to continually refine the policy.


### 🟧 Generate Samples (Orange Box)

This part involves the "trial" aspect of "trial and error" in RL. It means running your policy in the environment to collect samples—trajectories like:  
$( s_1, a_1, s_2, a_2, \dots, s_T, a_T )$

**Cost:**
- Real-world systems (e.g., robots, cars): Expensive, slow, real-time data collection.
- Simulators (e.g., MuJoCo): Cheap, can run up to 10,000× faster than real-time.


### 🟩 Fit a Model / Estimate the Return (Green Box)

In this phase, the algorithm evaluates or models how well the current policy is performing.

**Options:**
- Learn a dynamics model (model-based RL)
- Learn value functions $( V^\pi(s) )$ or Q-functions $( Q^\pi(s, a) )$

**Q-function:**  
$( Q^\pi(s_t, a_t) )$: Expected total reward from $( s_t )$, taking action $( a_t )$, then following $( \pi )$

**Value function:**  
$( V^\pi(s_t) )$: Expected total reward from $( s_t )$, averaging over $( \pi(a_t|s_t) )$

**Cost:**
- Simple in policy gradients (just sum rewards)
- Expensive in model-based RL (train large neural nets)

### 🟦 Improve the Policy (Blue Box)

Update the policy using insights from the green box.

**Cost:**
- Cheap: A single gradient update
- Expensive: Backprop through learned models

## 6. Types of RL Algorithms

### 6.1. 🎯 Policy Gradients

- **Core Idea:**  
  Directly differentiate the RL objective (expected reward) with respect to the policy parameters $( \theta )$, and perform gradient ascent. Even if the reward is discontinuous, the expectation is often smooth and differentiable.

- **Anatomy:**  
  - 🟩 *Green Box (Estimate Return):* Sum rewards along trajectories.  
  - 🟦 *Blue Box (Improve Policy):* Compute and apply gradient of expected reward w.r.t. policy parameters.

- **Sample Efficiency:**  
  On-policy — new samples needed after each policy change → less efficient.

- **Stability & Convergence:**  
  Performs true gradient ascent, but convergence is not guaranteed in practice.

- **Assumptions:**  
  Often assumes episodic learning.

- **Examples:**  
  REINFORCE, Natural Policy Gradient, TRPO, PPO


### 6.2. 📊 Value-Based Methods

- **Core Idea:**  
  Learn a value function or Q-function for the optimal policy. The policy is often *implicit*, e.g., choosing the action with the highest Q-value.

- **Anatomy:**  
  - 🟩 *Green Box (Estimate Return):* Fit $( V(s) )$ or $( Q(s,a) )$ using neural networks.  
  - 🟦 *Blue Box (Improve Policy):* Use $( \arg\max_a Q(s, a) )$ — no gradient needed.

- **Sample Efficiency:**  
  Off-policy — can reuse past samples → more efficient.

- **Stability & Convergence:**  
  Not guaranteed to converge with non-linear approximators like deep networks.

- **Assumptions:**  
  Assumes full observability and Markov property.

- **Examples:**  
  Q-learning, DQN, TD learning, Fitted Value Iteration

### 6.3. 🧠 Actor-Critic Methods

- **Core Idea:**  
  Combine value-based and policy gradient methods. A critic estimates values, and an actor updates the policy using these estimates.

- **Anatomy:**  
  - 🟩 *Green Box (Estimate Return):* Learn $( V^\pi(s) )$ or $( Q^\pi(s, a) )$.  
  - 🟦 *Blue Box (Improve Policy):* Policy updates use the critic's gradient signal (e.g., advantage function $( A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s) )$).

- **Sample Efficiency:**  
  Can be on-policy or off-policy; typically more efficient than pure policy gradients.

- **Assumptions:**  
  Supports both episodic and full-observability variants.

- **Examples:**  
  A3C, SAC, DDPG, TRPO (with value function)


### 6.4. 🧩 Model-Based RL Algorithms

- **Core Idea:**  
  Learn a model $( p(s_{t+1}|s_t, a_t) )$ of the environment, then use it for planning or policy improvement.

- **Anatomy:**  
  - 🟩 *Green Box (Fit Model):* Learn dynamics model (e.g., next state prediction).  
  - 🟦 *Blue Box (Improve Policy):* Options:
    1. Direct planning (e.g., MCTS, trajectory optimization)  
    2. Backprop through model for gradients  
    3. Use model to learn value function  
    4. Simulate data and train model-free RL

- **Sample Efficiency:**  
  Highly efficient — leverage simulated data.

- **Stability & Convergence:**  
  Model training converges, but better model ≠ better policy.

- **Assumptions:**  
  Often assumes smooth, continuous environments; episodic setups.

- **Examples:**  
  Dyna, Guided Policy Search, MPPO, SVG

### 6.5 📊 Summary Table

| Feature                   | Policy Gradients      | Value-Based Methods      | Actor-Critic Methods     | Model-Based RL            |
|---------------------------|------------------------|---------------------------|---------------------------|----------------------------|
| Core Idea                 | Optimize reward via gradients | Estimate $( Q )$/$( V )$ for optimal policy | Combine critic + actor     | Learn model; plan/improve policy |
| Green Box (Estimation)    | Sum of rewards         | Fit $( Q(s,a) )$/$( V(s) )$ | Fit $( Q^\pi(s,a) )$/$( V^\pi(s) )$ | Fit dynamics model $( p(s'|s,a) )$ |
| Blue Box (Improvement)    | Gradient ascent        | $( \arg\max Q(s,a) )$     | Gradient ascent using critic | Planning / backprop / simulated data |
| Policy Representation     | Explicit (e.g., NN)    | Implicit (via Q-function) | Explicit (Actor NN)       | Both explicit and implicit |
| Sample Efficiency         | Low (on-policy)        | High (off-policy)         | Medium (on/off-policy)    | Very high                  |
| Stability / Convergence   | True gradient ascent   | No guarantees w/ NN       | Same as value-based       | Model may not yield better policy |
| Common Assumptions        | Episodic resets        | Full observability        | Episodic or continuous    | Smoothness, episodic       |
| Examples                  | REINFORCE, TRPO, PPO   | Q-learning, DQN, TD       | A3C, SAC, DDPG            | Dyna, Guided Policy Search |


### 💡 Additional Trade-offs

- **Sample Cost vs. Compute:**  
  If data is expensive (e.g., real robot), prefer sample-efficient methods (model-based or off-policy).  
  If data is cheap (e.g., fast simulator), computation time becomes the bottleneck.

- **Convergence Guarantees:**  
  Unlike supervised learning, many deep RL methods lack strong theoretical guarantees and may diverge.

- **Environment Assumptions:**  
  Method choice depends on state/action space, stochasticity, and whether the task is episodic or infinite horizon.