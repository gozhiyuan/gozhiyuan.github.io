---
layout: post
title: Policy Gradients in Reinforcement Learning
subtitle: Reinforcement Learning Lecture 5
categories: Reinforcement-Learning
tags: [UCB-Deep-Reinforcement-Learning-2023]
banner: "/assets/images/banners/yuanpang-wa-valley.jpg"
---

# Policy Gradients in Reinforcement Learning

This lecture summarizes the core concepts, derivations, practical considerations, and advanced topics related to Policy Gradients, a fundamental algorithm in reinforcement learning (RL).
[Course Link](https://rail.eecs.berkeley.edu/deeprlcourse/)


## 1. Introduction to Policy Gradients

Policy gradients are a foundational RL algorithm that directly optimize the policy parameters through gradient descent to maximize the expected sum of rewards. They are considered "model-free" because they do not require knowledge of the environment's transition probabilities or initial state distributions.

**Key Idea:**

The goal of RL is to find policy parameters $\theta$ that maximize the expected value of the sum of rewards under the trajectory distribution $P_\theta(\tau)$.

$$
J(\theta) = \mathbb{E}_{\tau \sim P_\theta(\tau)} \left[\sum_{t=1}^T R(s_t, a_t)\right]
$$

The policy $\pi$ (with parameters $\theta$) defines a distribution over actions given a state or observation. If represented by a deep neural network, $\theta$ are the network's weights.

## 2. Evaluating the RL Objective and Policy Gradient

Since the transition probabilities and initial state distribution are typically unknown in model-free RL, evaluation relies on sampling.

![alt_text](/assets/images/reinforcement-learning/05/1.png "image_tooltip")

### 2.1. Evaluating the Objective

Estimate $J(\theta)$ by collecting $N$ sampled trajectories $\tau_i$:

$$
J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T R(s_{i,t}, a_{i,t})
$$

More samples lead to a more accurate estimate.

### 2.2. Evaluating the Policy Gradient

We aim to estimate the gradient $\nabla_\theta J(\theta)$ without access to transition probabilities.

**Log-Gradient Trick (Likelihood Ratio Trick):**

$$
\nabla_\theta P_\theta(\tau) = P_\theta(\tau) \nabla_\theta \log P_\theta(\tau)
$$

Applying it:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim P_\theta(\tau)} [\nabla_\theta \log P_\theta(\tau) \cdot R(\tau)]
$$

Since only the policy depends on $\theta$:

$$
\log P_\theta(\tau) = \log P(s_1) + \sum_{t=1}^T \left(\log \pi_\theta(a_t \mid s_t) + \log P(s_{t+1} \mid s_t, a_t)\right)
$$

$$
\nabla_\theta \log P_\theta(\tau) = \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_t \mid s_t)
$$

So:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim P_\theta(\tau)} \left[\left(\sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_t \mid s_t)\right) \cdot R(\tau)\right]
$$

**REINFORCE Algorithm Steps:**

1. Sample trajectories from current policy.
2. Evaluate the policy gradient.
3. Gradient ascent: $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$


## 3. Understanding Policy Gradients Intuitively

### 3.1. Comparison to Maximum Likelihood

- Supervised learning increases log probs of all actions.
- Policy gradient **weights** log probs by total reward.
- High-reward trajectories get reinforced more than low-reward ones.

![alt_text](/assets/images/reinforcement-learning/05/2.png "image_tooltip")

📌 Policy gradients can be intuitively understood as a **weighted version of the maximum likelihood gradient**, where the weights are determined by the rewards of the trajectories.

Here’s a breakdown:

- 🎯 **Reinforcement Learning Objective**:  
  The goal of reinforcement learning (RL) is to find policy parameters (theta, $\theta$) that maximize the expected value of the sum of rewards under the trajectory distribution $P_\theta(\tau)$.  
  A trajectory ($\tau$) is a sequence like $(s_1, a_1, s_2, a_2, \dots)$, and its total reward is denoted as $r(\tau)$.

- 📐 **Policy Gradient Calculation**:  
  The policy gradient algorithm directly differentiates the RL objective and performs gradient ascent to improve the policy.  
  The REINFORCE formula:

  $$
  \nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim P_\theta(\tau)} \left[ \left( \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right) \cdot r(\tau) \right]
  $$

- 🔍 **Comparison to Maximum Likelihood**:
  - In **maximum likelihood** (e.g. supervised/imitation learning), the goal is to maximize $\log \pi(a_t \mid s_t)$ for observed actions.  
  - The gradient simply sums $\nabla \log \pi(a_t \mid s_t)$ over all actions, increasing the probability of everything observed.  
  - In **policy gradients**, actions are sampled by the policy itself, and their gradient is **weighted by the reward** — i.e.,

    $$
    \nabla \log \pi(a_t \mid s_t) \cdot r(\tau)
    $$

    rather than being treated equally.

- ⚖️ **The "Weighted" Aspect**:
  - High-reward trajectories → 📈 increase their log probabilities → make them more likely.  
  - Low-reward trajectories → 📉 decrease their log probabilities → make them less likely.  
  - This is like a **weighted version of the max-likelihood gradient**, where each action’s contribution is scaled by how good the outcome was.  
  - This intuitively captures **trial and error** learning:
    > "Good stuff is made more likely, and bad stuff is made less likely." 💡

### 3.2. Trial and Error Learning

Policy gradients formalize trial-and-error learning: "Make good stuff more likely, bad stuff less likely."

### 3.3. Partial Observability

Policy gradients apply to POMDPs with minimal change (replace states $s$ with observations $o$).


## 4. Challenges: High Variance

Policy gradient estimators have high variance.

### 4.1. The Problem

- Adding a constant to all rewards affects sample estimates.
- Rare high-reward or negative-reward trajectories distort gradients.
- Variance increases with reward scale.


## 5. Reducing Variance: Practical Tricks

### 5.1. Causality (Reward-to-Go)


![alt_text](/assets/images/reinforcement-learning/05/3.png "image_tooltip")

Use only future rewards at each step $t$:

$$
Q_{i,t} = \sum_{t'=t}^T R(s_{i,t'}, a_{i,t'})
$$

Reduces variance while maintaining unbiasedness.

### 5.2. Baselines

Subtract a baseline $b$ from the reward:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim P_\theta(\tau)} \left[\left(\sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_t \mid s_t)\right) \cdot (R(\tau) - b)\right]
$$

- Helps center rewards: above-average trajectories get boosted.
- Average reward is a decent baseline.
- Optimal baseline involves gradient magnitude but is rarely used.

🎯 The concept of an **optimal baseline** in policy gradient algorithms addresses the challenge of **high variance** in the gradient estimator — a major hurdle in making these methods practical.

Here’s a detailed breakdown:

- 🧠 **Purpose of a Baseline**:
  - Policy gradients aim to **increase** the log-probabilities of actions with **high rewards**, and **decrease** those with **low rewards** — mimicking trial-and-error learning.  
  - But if all rewards are positive ➕, even "bad" trajectories might still get their probabilities slightly increased 🤨.  
  - A **baseline** (denoted as `b`) is subtracted from the reward in the policy gradient formula:  
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim P_\theta(\tau)} \left[\sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot (r(\tau) - b)\right]
$$  
  - This **centers** the reward:  
    - Better-than-average rewards ⬆️ get reinforced.  
    - Worse-than-average rewards ⬇️ get discouraged.  
  - Subtracting a constant baseline **does not bias** the estimator — it changes variance but not expectation ✔️.

- 📐 **Deriving the Optimal Baseline**:
  - Goal: **Minimize the variance** of the policy gradient estimator.  
  - Take the derivative of the variance w.r.t. `b`, set it to zero, solve for `b`.  
  - Use the identity:  
$$
P(\tau) \cdot \nabla \log P(\tau) = \nabla P(\tau)
$$  
  - Result:  
$$
b^* = \frac{\mathbb{E}[g^2 \cdot r]}{\mathbb{E}[g^2]}
$$  
    where:  
    - $g = \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)$  
    - $r = r(\tau)$ (total reward)


![alt_text](/assets/images/reinforcement-learning/05/4.png "image_tooltip")

- 🧮 **Interpretation of the Optimal Baseline**:
  - This formula shows that the optimal baseline is a **reweighted average** of rewards.  
  - The weights are based on the **squared magnitude of the gradient** $g^2$.  
  - For policies with many parameters, this suggests:  
    - 🧩 Each parameter could have its own **separate optimal baseline**, depending on the gradient magnitude for that parameter.

- 🛠️ **Practical Considerations**:
  - The **average reward** is a good (but suboptimal) baseline — widely used.  
  - The theoretically **optimal baseline** is rarely used because:  
    - It requires computing the gradient before computing the baseline 😵  
    - Adds complexity with limited practical benefit.  
  - Still, **reducing variance is critical** to make policy gradients work well:  
    - Helps with smaller batch sizes  
    - Stabilizes training  
    - Improves learning rate robustness  
  - ✅ Combine baselines with tricks like **reward-to-go** for better performance.


## 6. Off-Policy Policy Gradients

### 6.1. Why On-Policy

Standard gradients require new samples after every parameter update, which is inefficient.

📌 **Policy gradients** are generally considered **on-policy** algorithms — they require new samples every time the policy changes because the gradient is an expectation under the **current policy**, $P_\theta(\tau)$.  
If $\theta$ changes, the distribution $P_\theta(\tau)$ changes, and old samples are no longer valid ➡️ **extremely inefficient** when sampling is expensive (e.g., real-world robots or slow simulators).

⚡ **Off-policy policy gradients** solve this by allowing reuse of data collected from previous policies or external sources like human demonstrations.

- 🚫 **The Problem with On-Policy Learning**:
  - Neural networks usually make **tiny parameter updates** per step.  
  - On-policy methods require **fresh samples at every update**, making training **slow and expensive** when environment interaction is costly.


- 🔄 **Introducing Importance Sampling for Off-Policy Learning**:
  - Use **importance sampling (IS)** to reuse off-policy data.  
  - IS adjusts expectations under the new policy using samples from a different policy:  
$$
\mathbb{E}_{\tau \sim P_\theta(\tau)}[f(\tau)] = \mathbb{E}_{\tau \sim P_{\bar{\theta}}(\tau)} \left[ \frac{P_\theta(\tau)}{P_{\bar{\theta}}(\tau)} f(\tau) \right]
$$  
  - The ratio $\frac{P_\theta(\tau)}{P_{\bar{\theta}}(\tau)}$ is the **importance weight**, used to correct for sampling bias.

- 🧮 **Derivation of Off-Policy Policy Gradient**:
  - Rewrite the objective using importance weights between new and old policies.  
  - Transition probabilities and initial states cancel out (they’re shared across policies):  
$$
\frac{P_{\theta'}(\tau)}{P_{\bar{\theta}}(\tau)} = \prod_{t=1}^T \frac{\pi_{\theta'}(a_t \mid s_t)}{\pi_{\bar{\theta}}(a_t \mid s_t)}
$$  
  - The **off-policy policy gradient** becomes:  
$$
\mathbb{E}_{\tau \sim P_{\bar{\theta}}} \left[\left(\prod_{t=1}^T \frac{\pi_{\theta'}(a_t \mid s_t)}{\pi_{\bar{\theta}}(a_t \mid s_t)}\right) \cdot \left( \sum_{t=1}^T \nabla_{\theta'} \log \pi_{\theta'}(a_t \mid s_t) \cdot r(\tau) \right) \right]
$$  
  - Each gradient term is now **weighted** by the importance ratio.


![alt_text](/assets/images/reinforcement-learning/05/5.png "image_tooltip")

- ⚠️ **Challenges: Variance Explosion**:
  - Importance weights are **products over many time steps** → risk of exploding or vanishing values.  
  - If $\pi_{\theta'}(a_t \mid s_t)$ is very small, the product becomes near-zero → noisy, high-variance estimates.  
  - This makes off-policy policy gradients **even more unstable** than standard ones.

- 🔧 **Approximations to Reduce Variance**:

  - 🕒 **Reward-to-Go (Causality Trick)**:  
    - Only sum future rewards at time $t$:  
      $$
      \hat{Q}_{i,t} = \sum_{t'=t}^T r(s_{t'}, a_{t'})
      $$  
    - Actions cannot affect past rewards → lowers variance.

  - 📉 **Ignore State Marginal Probabilities**:  
    - Common approximation: **ignore full trajectory weighting**, and only reweight based on action probabilities at each step.  
    - Reduces variance from exponential blowup.  
    - This approximation is valid when $\theta'$ is **not too different** from $\bar{\theta}$.

- 🧪 **Practical Algorithms**:
  - 🔁 **TRPO (Trust Region Policy Optimization)** and **PPO (Proximal Policy Optimization)**:  
    - Use **importance sampling** with smart constraints and approximations.  
    - Add techniques like **KL-divergence penalties**, **clipped ratios**, and **natural gradients** for stability.

  - 🧑‍🏫 **Incorporating Demonstrations**:  
    - **Guided Policy Search** (Levine & Koltun, 2013) uses off-policy gradients + importance sampling to integrate human or demonstration data.  

📚 **Summary**:  
Off-policy policy gradients **extend the flexibility** of policy gradient methods by reusing past data, but introduce **high variance** due to importance weights.  
By applying **reward-to-go**, **ignoring marginal states**, and using algorithms like PPO/TRPO, we can make off-policy policy gradients **practical and scalable** in deep reinforcement learning.


### 6.2. Importance Sampling (IS)

To evaluate expectations with off-policy samples:

$$
J(\theta') = \mathbb{E}_{\tau \sim P_{\bar{\theta}}(\tau)} \left[\frac{P_{\theta'}(\tau)}{P_{\bar{\theta}}(\tau)} \cdot R(\tau)\right]
$$

Importance weight:

$$
\frac{P_{\theta'}(\tau)}{P_{\bar{\theta}}(\tau)} = \prod_{t=1}^T \frac{\pi_{\theta'}(a_t \mid s_t)}{\pi_{\bar{\theta}}(a_t \mid s_t)}
$$

Policy gradient becomes:

$$
\nabla_{\theta'} J(\theta') = \mathbb{E}_{\tau \sim P_{\bar{\theta}}(\tau)} \left[\left(\prod_{t=1}^T \frac{\pi_{\theta'}(a_t \mid s_t)}{\pi_{\bar{\theta}}(a_t \mid s_t)}\right) \cdot \left(\sum_{t=1}^T \nabla_{\theta'} \log \pi_{\theta'}(a_t \mid s_t)\right) \cdot R(\tau)\right]
$$

**Problem:** Importance weights can explode or vanish exponentially → high variance.

### 6.3. Approximations

Ignore marginal state distributions to reduce variance. Works when $\theta' \approx \bar{\theta}$.

![alt_text](/assets/images/reinforcement-learning/05/6.png "image_tooltip")

📌 In **off-policy policy gradients**, a **first-order approximation** is a crucial trick used to **reduce the high variance** caused by importance sampling. It makes off-policy learning **practically feasible**, especially in deep reinforcement learning.

- 🔁 **On-Policy Nature of Policy Gradients**:
  - Policy gradient algorithms are inherently **on-policy** — they require **new samples** from the **current policy** after each parameter update.  
  - Since deep RL uses **neural networks**, which require many small updates, this can make on-policy learning **extremely costly and inefficient** when sample generation is expensive.

- 🔄 **Enabling Off-Policy Learning via Importance Sampling**:
  - To reuse samples from **previous policies** or other sources (e.g., human demos), importance sampling is applied.  
  - This allows computing expectations under the **new policy** $\pi_{\theta'}$ using data collected from an **old policy** $\pi_\theta$.  
  - The key is the **importance weight**:  
$$
\frac{P_{\theta'}(\tau)}{P_\theta(\tau)} = \prod_{t=1}^T \frac{\pi_{\theta'}(a_t \mid s_t)}{\pi_\theta(a_t \mid s_t)}
$$

- ⚠️ **The Variance Problem**:
  - While this ratio makes the estimator **unbiased**, it leads to **exponentially large variance**.  
  - If $\pi_{\theta'}$ assigns low probability to actions taken by $\pi_\theta$, the product becomes **close to zero**, making gradients **very noisy**.  
  - This is **even worse** than the already high variance in on-policy gradients.

- 🧮 **The First-Order Approximation**:
  - To **mitigate** this, a **first-order approximation** is applied.  
  - **Idea**: Drop the full product over all time steps and use **only the per-step ratio** of action probabilities:  
$$
\frac{\pi_{\theta'}(a_t \mid s_t)}{\pi_\theta(a_t \mid s_t)}
$$  
  - This approximation **ignores the trajectory-level probability ratio**, focusing only on the **current time step's action probability**.

- 🤔 **Rationale and Implications**:
  - This avoids the **exponential blowup in variance**, making gradient estimates **much more stable**.  
  - It introduces **bias** (no longer exact in expectation), but:
    - The bias is **bounded** if $\theta'$ is **close to** $\theta$.  
    - The resulting update can still **improve the policy** in practice.  
  - It's a **practical trade-off**: give up some theoretical correctness to **gain massive stability** and usability in deep RL.

- 🧪 **When to Use It**:
  - Especially useful when:
    - Sample collection is **expensive**.
    - Policies change **gradually** (e.g., small learning rates).  
  - Many modern RL algorithms (e.g., PPO) implicitly rely on such **approximations**.


✅ **Summary**:  
The **first-order approximation** in off-policy policy gradients replaces full importance weights (across trajectories) with **step-wise ratios of action probabilities**.  
This clever simplification controls variance, enabling **efficient and stable learning**, even at the cost of some bias — a crucial trick in practical deep reinforcement learning.


## 7. Implementing Policy Gradients with Autodiff

### 7.1. The Challenge

Backpropagation needs a differentiable computation graph with many parameters.

### 7.2. Pseudo-Loss for Autodiff

Define a pseudo-loss:

$$
\tilde{J} = \sum_{i} \sum_{t} \log \pi(a_{i,t} \mid s_{i,t}) \cdot Q_{i,t}
$$

When autodiff differentiates $\tilde{J}$, it recovers the true policy gradient.

![alt_text](/assets/images/reinforcement-learning/05/7.png "image_tooltip")

### 7.3. Practical Tips

- **High Variance**: Use large batches (thousands).
- **Optimizer**: Adam is acceptable; SGD is hard to tune.
- **Hyperparameter Tuning**: Crucial, especially for learning rate.


## 8. Advanced Policy Gradients

### 8.1. Poor Conditioning in Continuous Action Spaces

In continuous action policies (e.g., Gaussians), some parameters affect output more. E.g., small $\sigma$ in $\mathcal{N}(k s, \sigma)$ exaggerates gradient wrt $\sigma$.

### 8.2. Natural Policy Gradient

Rescale gradients so updates are equal in **policy space**, not **parameter space**.

- Constrain divergence (KL) between $\pi_{\theta}$ and $\pi_{\theta'}$:

  $$ 
  D_{KL}(\pi_{\theta'} \| \pi_{\theta}) \approx \frac{1}{2} (\theta' - \theta)^T F (\theta' - \theta)
  $$

- Fisher Information Matrix (FIM):

  $$
  F = \mathbb{E}_{\pi_\theta} \left[\nabla_\theta \log \pi_\theta \cdot \nabla_\theta \log \pi_\theta^T\right]
  $$

- Natural Gradient Update:

  $$
  \theta \leftarrow \theta + \alpha F^{-1} \nabla_\theta J(\theta)
  $$

Algorithms like **TRPO** and **PPO** build on this idea for stable training.


## 9. Conclusion and Future Topics

Policy gradients are a direct but high-variance method for optimizing policies in RL.

Key techniques to improve performance:

- Use **reward-to-go** (causality).
- Use **baselines** to reduce variance.
- Careful **off-policy approximations**.
- Address **conditioning** via natural gradients.

**Next Topics:**

- Actor-Critic methods (combine value functions for variance reduction).
- Advanced natural gradient techniques.
- Automatic step size adaptation.
