---
layout: post
title: Imitation Learning
subtitle: Reinforcement Learning Lecture 2
categories: Reinforcement-Learning
tags: [UCB-Deep-Reinforcement-Learning-2023]
banner: "/assets/images/banners/yuanpang-wa-valley.jpg"
---

# Imitation Learning: Challenges and Solutions

This lecture reviews supervised Learning of Behaviors / Imitation Learning  
[Course Link](https://rail.eecs.berkeley.edu/deeprlcourse/)


## 1. Introduction to Imitation Learning and Behavioral Cloning

Imitation Learning (IL), specifically Behavioral Cloning (BC), involves training a policy (a model that maps observations to actions) using a dataset of expert demonstrations. This is essentially a supervised learning problem where the input is an observation $( O_t )$ and the output is the expert's action $( A_t )$ taken in that observation. The goal is for the learned policy $( \pi_\theta )$ to mimic the expert's behavior.

![alt_text](/assets/images/reinforcement-learning/02/1.png "image_tooltip")

**Key Terminology**:

- **Observation** $( O_t )$: The raw input to the policy (e.g., an array of pixels from a camera).
- **State** $( S_t )$: A concise and complete physical description of the world that produced the observation.
- **Action** $( A_t )$: The output of the policy (e.g., steering angle, movement).
- **Policy** $( \pi_\theta )$: A conditional distribution over actions given an observation, $( P(A_t \mid O_t) )$.
- **Transition Probabilities / Dynamics**: $( P(S_{t+1} \mid S_t, A_t) )$.
- **Markov Property**: $( S_{t+1} \perp S_{t-1} \mid S_t )$.

## Observations vs. States in Supervised Learning of Behaviors

In the context of supervised learning of behaviors and sequential decision-making, it's important to distinguish between **observations** ($( O )$) and **states** ($( S )$). This distinction is especially relevant for reinforcement learning algorithms, though it is less critical in basic imitation learning.


### 🔍 Observation ($( O )$)

- An **observation** is denoted as $( O_t )$, indicating the raw input at time step $( t )$.
- Analogous to the input $( X )$ in standard supervised learning (e.g., image classification).
- Example: In autonomous driving, $( O_t )$ might be the dashboard camera image. For a cheetah chase, it’s the pixels from a video recording.
- Policies are typically learned as **distributions over actions given observations**, written as:

  $$
  \pi(A_t \mid O_t)
  $$

### 🌍 State ($( S )$)

- A **state** is denoted as $( S_t )$, representing the full physical description of the world at time $( t )$.
- It includes everything needed to predict the next state — positions, velocities, even latent variables like intent or "mental state".
- In a simulation, $( S_t )$ could represent the complete memory state of the simulator.
- Satisfies the **Markov property**:

  > If you know the current state $( S_t )$, then the past $( (S_{t-1}, S_{t-2}, \dots) )$ provides no additional information for predicting the future $( S_{t+1} )$.

  Mathematically:

  $$
  P(S_{t+1} \mid S_t) = P(S_{t+1} \mid S_t, S_{t-1}, \dots)
  $$


#### Completeness of Information

- You can always go from **state to observation** because the state encodes all observable information.
- But **you can't always go from observation to state**:
  - An observation might lack certain elements of the true state.
  - Example: A cheetah hidden behind a car may be part of $( S_t )$ but not visible in $( O_t )$.

#### Policy Input Requirements

- Some algorithms require **fully observed policies**: inputs must satisfy the Markov property (i.e., must be states).
- Other algorithms operate on **partially observed data**, i.e., observations that may not include all state variables.


### ⚠️ Practical Confusion

- **Terminology overlap**: In RL literature, "state" and "observation" are often used interchangeably — this can be misleading.
- **Benign confusion**: This mix-up is often harmless in simple tasks or with robust models.
- But **some RL algorithms critically depend** on whether input is a full state or a partial observation.

### 📝 Notational Conventions

- In standard RL:
  - $( S )$: State
  - $( A )$: Action
- In control theory and robotics:
  - $( X )$: State
  - $( U )$: Action

These represent the same concepts, just in different academic traditions.

## 2. The Fundamental Problem with Behavioral Cloning: Distributional Shift

While seemingly straightforward, BC is not guaranteed to work in practice. The primary issue is **distributional shift**.

- **Training Distribution**: $( P_{\text{data}}(O_t) )$
- **Policy Distribution**: $( P_{\pi_\theta}(O_t) )$

Because the learned policy behaves differently from the expert, it encounters states it wasn’t trained on, causing compounding errors.

### Tightrope Walker Analysis

Let the cost of a mistake be 1 whenever the policy deviates from the expert:

- Assume error rate $( \epsilon )$ per step on seen data.
- In worst-case, total expected mistakes grow as:

$$
\mathbb{E}[\text{Total Cost}] = O(\epsilon T^2)
$$

This is because early mistakes compound and push the agent into unfamiliar territory.

However in reality, we can often recover from mistakes! A paradox: imitation learning can work better if the data has more mistakes (and recoveries)!


## 3. Practical Solutions to Improve Behavioral Cloning

### 3.1 Smart Data Collection and Augmentation

- **Add Mistakes + Corrections**: Intentionally record expert recovery behaviors.
- **Data Augmentation**: Generate synthetic data (e.g., side cameras simulating drift).

**Examples**:
- **Drone Navigation**: Use multiple cameras (left, forward, right) to simulate corrections.
- **Low-Cost Robot Arms**: Mistakes from teleoperation provide recovery data.


### 3.2 Using Powerful Models to Reduce Mistakes

#### Incorporate History
- Use sequences of observations $( O_{t-k:t} )$ with:
  - **LSTMs** or **Transformers**
  - **Causal Confusion** caveat: avoid learning from spurious cues (e.g., brake light).

#### Model Multimodal Behavior

**Multimodal behavior** arises in imitation learning when an expert can take **multiple valid actions** for a given state. For example, navigating around a tree might involve going left or right—both are acceptable solutions. While this is **easy to handle in discrete action spaces**, it becomes challenging in **continuous action spaces**.

In continuous settings, a policy that outputs a **single Gaussian** (with mean and variance) can only represent **one mode**. Averaging conflicting expert actions (e.g., "left" and "right") will lead to undesirable actions (e.g., "go straight"). To address this, the following solutions are used:


#### A. 🧮 Mixture of Gaussians

- A simple but limited solution.
- The policy outputs multiple Gaussian components:
  - $( \text{means} = \mu_1, \mu_2, \dots, \mu_n )$
  - $( \text{covariances} = \Sigma_1, \Sigma_2, \dots, \Sigma_n )$
  - $( \text{weights} = w_1, w_2, \dots, w_n )$
- The overall distribution is:

  $$
  p(a \mid o) = \sum_{i=1}^n w_i \cdot \mathcal{N}(a \mid \mu_i, \Sigma_i)
  $$

- Training involves **maximum likelihood**, using the log of this mixture.
- ❗ **Limitation**: The number of modes $n$ is fixed in advance. For complex tasks (e.g., robots with many joints), scaling becomes impractical.




```python
"""
Here’s a simple PyTorch implementation of a Mixture of Gaussians (MoG) policy suitable for imitation learning tasks, where the output is a multimodal distribution over continuous actions.

This implementation assumes:
Input: observation vector (obs)
Output: a mixture of Gaussians: multiple means, log variances (for numerical stability), and mixing weights
The policy samples an action from the mixture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

class MoGPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, num_components=5, hidden_size=256):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_components = num_components

        # Shared MLP for feature extraction
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Outputs: one set per mixture component
        self.mean_layer = nn.Linear(hidden_size, num_components * action_dim)
        self.log_std_layer = nn.Linear(hidden_size, num_components * action_dim)
        self.weight_layer = nn.Linear(hidden_size, num_components)  # logits for mixing weights

    def forward(self, obs):
        """
        obs: Tensor of shape (batch_size, obs_dim)
        Returns: sampled action, component index, and full mixture (for analysis or loss)
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))

        # Get mixture parameters
        means = self.mean_layer(x).view(-1, self.num_components, self.action_dim)
        log_stds = self.log_std_layer(x).view(-1, self.num_components, self.action_dim)
        stds = torch.exp(log_stds)

        weights = F.softmax(self.weight_layer(x), dim=-1)  # shape: (batch_size, num_components)

        # Sample mixture component
        cat = Categorical(weights)
        comp_idx = cat.sample()  # shape: (batch_size,)

        # Select corresponding mean and std for each sample in batch
        batch_size = obs.size(0)
        idx = torch.arange(batch_size)

        selected_means = means[idx, comp_idx]    # (batch_size, action_dim)
        selected_stds = stds[idx, comp_idx]      # (batch_size, action_dim)

        # Sample from the selected Gaussian
        normal = Normal(selected_means, selected_stds)
        action = normal.rsample()  # use rsample() for backprop through reparameterization

        return action, comp_idx, (means, stds, weights)

    def log_prob(self, obs, actions):
        """
        Compute the log probability of the given actions under the full mixture distribution.
        Returns shape: (batch_size,)
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))

        means = self.mean_layer(x).view(-1, self.num_components, self.action_dim)
        log_stds = self.log_std_layer(x).view(-1, self.num_components, self.action_dim)
        stds = torch.exp(log_stds)
        weights = F.softmax(self.weight_layer(x), dim=-1)  # (batch_size, num_components)

        # Expand actions to match mixture shape
        actions = actions.unsqueeze(1)  # (batch_size, 1, action_dim)

        # Compute log-likelihoods for each component
        normal = Normal(means, stds)
        log_probs = normal.log_prob(actions)  # (batch_size, num_components, action_dim)
        log_probs = log_probs.sum(dim=-1)     # sum over action dims → (batch_size, num_components)

        # Compute log-sum-exp across components
        weighted_log_probs = log_probs + torch.log(weights + 1e-8)
        log_prob = torch.logsumexp(weighted_log_probs, dim=-1)  # (batch_size,)

        return log_prob

obs_dim = 32
action_dim = 4
policy = MoGPolicy(obs_dim, action_dim, num_components=5)

obs = torch.randn(16, obs_dim)  # batch of observations
action, comp_idx, mixture = policy(obs)

# For training loss:
logp = policy.log_prob(obs, action)
loss = -logp.mean()  # maximize likelihood
loss.backward()
```


#### B. 🌌 Latent Variable Models (e.g., CVAE)

- Allow flexible and **expressive multimodal outputs**.
- The network still outputs a Gaussian, but it also receives a **latent variable** $z \sim \mathcal{N}(0, I)$.
- The latent variable acts like a "random seed" that controls which mode the network produces:

  $$
  a \sim \pi(a \mid o, z)
  $$

- **Training Challenge**: The network might ignore $z$ if it's uninformative.
- **Solution**: Use **Conditional Variational Autoencoders (CVAE)**:
  - During training, assign specific $z$ vectors to examples (e.g., one for "left", another for "right").
  - Train the encoder to encode the input-action pair into $z$, and decoder to reconstruct the action from $o$ and $z$.
- At test time, sample $z \sim \mathcal{N}(0, I)$ to stochastically choose among modes.


#### C. 🌫 Diffusion Models

- Inspired by recent successes in generative models (e.g., DALL·E, Stable Diffusion).
- Key idea: **Denoising as generation**.
- Steps:
  - Add Gaussian noise to expert action: $a_{t, 0} \rightarrow a_{t, 1} \rightarrow \dots \rightarrow a_{t, T}$
  - Train a neural network to **denoise**, i.e., predict either:
    - The less noisy version $a_{t, i-1}$, or
    - The added noise $\epsilon$ such that $a_{t, i-1} = a_{t, i} - \epsilon$
- At test time:
  - Start from random noise.
  - Apply the trained network repeatedly to **iteratively denoise** back to a valid action.
- ✅ This allows capturing **complex, multimodal** distributions in high-dimensional spaces.


#### D. 🔁 Autoregressive Discretization (Bonus)

- Breaks down a high-dimensional action into **per-dimension discrete outputs**, predicted **sequentially**.
- Inspired by language models (token-by-token generation).
- Each dimension is discretized independently:

  $$
  a = (a_1, a_2, \dots, a_n), \quad \text{where each } a_i \in \{d_1, d_2, \dots, d_k\}
  $$

- At each step, the model outputs a distribution over discretized values **conditioned** on previous dimensions.
- Used in **RT-1 Robotics Transformer** for general-purpose robot control.


#### ✅ Summary

| Method                    | Handles Multimodality | Scales to High-Dimensional Actions | Stochastic Sampling | Notes                                                    |
|---------------------------|------------------------|------------------------------------|----------------------|----------------------------------------------------------|
| Mixture of Gaussians      | ✅                     | ❌ (limited by number of modes)    | ✅                   | Easy to implement                                        |
| Latent Variable Models    | ✅                     | ✅                                 | ✅                   | Needs careful training (e.g., CVAE)                      |
| Diffusion Models          | ✅✅✅                 | ✅✅✅                             | ✅                   | State-of-the-art, expressive                             |
| Autoregressive Discretization | ✅               | ✅                                 | ✅                   | Makes control a sequence modeling problem                |


### 3.3 Multi-Task Learning

- **Goal-Conditioned BC**: Policy conditioned on goal image or state.
- **Relabeling**: Use the achieved end state as the goal.
- **Sub-Optimal Data**: Even failures teach how to reach intermediate states.

**Example**:  
- **Learning Latent Plans from Play**: Play data → relabel → learn general behaviors.

**Scalability**:  
- Combine with history and relabeling → train general-purpose robots.

**HER Connection**:  
- Shares relabeling principle with Hindsight Experience Replay.


### 3.4 Algorithmic Change: DAgger

- Interactive version of BC.
- Aggregate data from states visited by the learned policy.
- Reduce distributional shift by querying the expert during execution.


## 4. Recap and Future Outlook

**Why Behavioral Cloning Fails**:

- Distributional shift → compounding errors.
- Error bound: $( O(\epsilon T^2) )$

**Solutions**:

1. **Data Collection**: Diverse states + recovery examples.
2. **Modeling**: Sequence models, multimodal outputs, latent variables.
3. **Multi-task Learning**: Goal conditioning + relabeling.
4. **New Algorithms**: DAgger

**Limitations**:

- Relies on finite demos.
- Humans can't always demonstrate all recoveries.
- Not inherently self-improving.

**Motivation for RL**:

- RL enables autonomous trial and error.
- Learning through experience, guided by reward.

Next lectures will explore reinforcement learning as the next step beyond imitation.
