---
layout: post
title: Imitation Learning via Privileged Teachers and Generative Models like Diffusion
subtitle: Robot Learning Lecture 6
categories: Robotics
tags: [CMU-Robot-Learning-2024]
banner: "/assets/images/banners/yuanpang-wa-iceburg2.jpg"
---

# 🧠 Imitation Learning via Privileged Teachers and Generative Models like Diffusions

This lecture builds upon previous discussions on Imitation Learning (IL) and delves into advanced techniques and current research areas.

[Course link](https://16-831-s24.github.io/)


## 🔁 1. Recap of Previous Lecture (Imitation Learning Part 1)

Lecture 6 begins with a brief review of key concepts from Part 1:

- **Notations, MDP, POMDP**: Reinforcement learning foundations.
- **What is imitation learning?** Learning from expert demonstrations.
- **Difference from supervised learning**: Highlights IL's sequential nature and distribution shift challenges.
- **Does behavior cloning (BC) work?** BC is the simplest IL method but has major limitations.
- **Fixing BC**:
  - **Better data**: Improve expert demonstrations or augment them.
  - **Better algorithm**: DAgger (Dataset Aggregation) helps fix distribution shift.

---

## ⭐ 2. Key Themes of Imitation Learning in this lecture

### 🔧 2.1 Imitation Learning with Powerful Models

#### 🧠 Using History
- **Problem**: Naive IL models act based only on current observation: $p(a_t \mid o_t)$.
- **Limitation**: Human behavior is history-dependent. Reacting identically to the same frame is unnatural.
- **Solution**: Model $p(a_t \mid o_0, \dots, o_t)$ using time-series techniques from Lecture 4.

#### 🔀 Modeling Multi-modal Behavior
- **Motivation**: Expert behavior is often multi-modal (multiple correct actions).
- **Methods**:
  - **Gaussian Mixture Models**
  - **Latent Variable Models** (e.g., conditional VAE)
  - **Diffusion Models** (details in later lectures)
  - **GANs** and other generative techniques


### 🧑‍🏫 2.2 Imitation Learning with Privileged Teachers

![alt_text](/assets/images/robot-learning/06/1.png "image_tooltip")

#### What is the Privileged Teacher Approach?

The **Privileged Teacher** approach is a strategy in Imitation Learning (IL) where:

- A **teacher agent** is trained in a **simulation environment** using **privileged state information** (`s_t`), which may not be available in the real world.
- A **student agent** is then trained to imitate the teacher using only **realistic observations** (`o_t`) — such as camera images or proprioception — to reflect what it would experience in the real world.

#### 💡 Why Use a Privileged Teacher?

| Component       | Description |
|----------------|-------------|
| **Teacher Policy** | $\pi_{\text{teacher}}(a_t \mid s_t)$, trained using full ground-truth state. |
| **Student Policy** | $\pi_{\text{student}}(a_t \mid o_t)$, trained using partial/realistic observation. |
| **Training Flow** | Teacher runs in sim → collects $(o_t, a_t)$ → student imitates this. |
| **Environment** | Simulation gives access to true state $s_t$, which is not available in real-world deployment. |

#### ✅ Benefits of This Approach

- **White-box Supervision**: Teacher policy is transparent and modifiable.
- **Efficient Optimization**: Easier to learn optimal behavior from ground truth.
- **Invariance**: Enables generalization across varied conditions (e.g. day/night driving).
- **Data Augmentation**: Easier to simulate different conditions and viewpoints.
- **Safer and Cheaper**: No need to collect risky real-world data for student training.

#### 🔍 Applications of Privileged Teacher

#### 🚗 Driving
- **Stage 1**: Train a privileged teacher using full information (e.g., traffic lights, other cars’ positions, velocities).
- **Stage 2**: Student only sees sensor inputs like images or LiDAR and learns to imitate the teacher.
- **Advantage**: Safer than letting students learn directly in the real world with limited perception.

#### 🦿  Locomotion
- **Privileged Information**: Contact forces, terrain maps, external disturbances, etc. (available in simulation).
- **Student Inputs**: Proprioceptive sensors — IMU, joint angles, velocities.
- **Training**: Privileged agent (teacher) is trained using RL (e.g., PPO), then student learns to imitate using only partial observation.
- **Example**: Widely used in simulated quadrupeds and humanoids.

#### ✨ Student Learning Not in Action Space
- **Variants**:
  - **Latent Space Learning**: Student learns to **match latent representations** instead of raw actions.
    - Example: RMA (Recurrent Modulation Agent).
  - **Perception Space Learning**: Student learns to predict **intermediate observations** (like sensor rays).
    - Example: ABS (Adaptive Behavior Shaping).
- **Why?**: Useful when action space is too complex, or when alignment in latent/perception space is more robust.

#### 🚁 Drones
- **Teacher**: MPC controller (Model Predictive Control) trained with access to full state (e.g., exact position, velocity, wind).
- **Student**: Learns from camera or IMU data only.
- **Benefit**: Allows training high-performance policies in simulation and transferring them safely to drones with only onboard sensors.


#### 🧠 Summary

Privileged teacher methods elegantly combine:
- **Powerful policy learning** (via RL + full state) in simulation,
- With **safe and realistic student learning** (via IL) using real-world-like observations.

This method is a key tool in **sim-to-real transfer**, **data-efficient learning**, and **robust control** under partial observability.

### 🎨 2.3 Deep Imitation Learning via Generative Models

![alt_text](/assets/images/robot-learning/06/2.png "image_tooltip")

> “A very active research area!”

**Deep Imitation Learning (IL)** via **Generative Models** is a highly active research area in robot learning. This approach integrates generative models into imitation learning to help robots learn complex behaviors from expert demonstrations.

#### 🎲 Generative Models in Imitation Learning

Generative models are characterized by their ability to:

- Learn a distribution $( p_\theta )$ that approximates a given data distribution $( p_{\text{data}} )$.  
  In imitation learning, $( p_{\text{data}} )$ comes from expert demonstrations. These models are often **conditional**.
- Generate novel samples $( \tilde{x} \sim p_\theta )$ once the distribution is learned.

The learned distribution can represent different levels of expert behavior:
- Action level (conditional): $( p(a_t \mid o_t) )$
- Observation-action pair: $( p(o_t, a_t) )$ or state-action: $( p(s_t, a_t) )$
- A sequence of actions
- A full state-action trajectory

**Key Benefit**: These models can capture **multi-modal behavior**, common in expert demonstrations (e.g., multiple valid actions in a driving scenario).

### ⚔️ Generative Adversarial Imitation Learning (GAIL)

GAIL integrates **Generative Adversarial Networks (GANs)** with imitation learning.

**Core Loop:**
1. **Sample** trajectories from the student policy.
2. **Update** the discriminator to distinguish expert vs. student trajectories.
3. **Train** the student policy to fool the discriminator:
   $[
   \pi_{\text{student}} \approx \pi_{\text{expert}} \Rightarrow D(\tau_{\text{student}}) \approx D(\tau_{\text{expert}})
   ]$

Goal: The student learns to imitate the expert by minimizing discriminator accuracy.

> 💡 Open question: Can GAIL model **multi-modal behavior** effectively?

#### 🎛️ VAE + Imitation Learning (VAE+IL)

**Variational Autoencoders (VAEs)** are also used in IL to model latent expert behavior.

**Example**: Action Chunking with Transformers (ACT)

- Based on **Conditional VAE (CVAE)**.
- **Encoder**:
  $[
  z = \text{Encoder}(a_{1:T}, o)
  ]$
- **Decoder**:
  $[
  \hat{a}_{1:T} = \text{Decoder}(z, o')
  ]$
- Supports **action chunking** and **temporal ensemble**, useful for long-horizon planning.


#### 🌫️ Diffusion Models + Imitation Learning

Diffusion models are powerful generative models recently adopted for IL.

#### Two Main Steps:

1. **Forward Diffusion Process**:
   Gradually corrupt real data $( x_0 )$ with Gaussian noise:
   $[
   q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)
   ]$
   repeated over steps $( t = 1, \dots, T )$, until pure noise $( x_T )$.

2. **Reverse Diffusion Process**:
   Learn reverse process:
   $[
   p_\theta(x_{t-1} \mid x_t)
   ]$
   that reconstructs $( x_0 )$ from noise $( x_T )$ by denoising step-by-step.

#### 🧪 Theoretical Insights:
- Related to **stochastic gradient Langevin dynamics**:
  $[
  x_{t+1} = x_t + \frac{\epsilon}{2} \nabla \log p(x_t) + \sqrt{\epsilon} \cdot \eta, \quad \eta \sim \mathcal{N}(0, I)
  ]$
- In the limit $( t \to \infty, D \to 0 )$, the generated samples match the true data distribution.

#### Applications:
- **Trajectory-level modeling**: Learn entire action trajectories (e.g., diffuser).
- Enables **high-fidelity** and **diverse** imitation, capturing subtle expert nuances.

### 🧠 Summary: Generative Models + IL

Generative models integrate well with IL, offering tools for uncertainty, diversity, and realistic sequence generation. Generative models provide a **flexible, expressive framework** to enhance imitation learning — especially in **robotics**, where behavior is structured, dynamic, and multi-modal.

| Model Type | Learn Distribution Over | Example |
|------------|-------------------------|---------|
| GAN        | $( P(o_t, a_t) )$ or full trajectories | GAIL |
| VAE        | $P(a_t \mid o_t)$, action chunks     | ACT |
| Diffusion  | Trajectories, sequences              | Diffuser |

- **IL + GANs**: Learns to imitate via adversarial feedback; matches expert distribution. 
- **IL + VAEs**: Captures latent structure of expert behavior; supports sequence prediction. 
- **IL + Diffusion**: Enables realistic and diverse trajectory generation; well-suited for robotics. 

---

## 3. Diffusion Policy in Imitation Learning More Details

Diffusion Policy treats **trajectory generation** as a **generative modeling** problem:
- It models the full distribution of action sequences conditioned on observations.
- Uses a **denoising diffusion process** to sample high-quality trajectories that imitate expert behavior.

### 🧩 Model Inputs and Outputs

**Input to the model:**
- $( o_{1:T} )$: Observation history (e.g., images, joint positions, proprioception) — usually fixed throughout the denoising process.
- $( a_{1:T}^{(T)} )$: A fully **noised** version of the action trajectory (sampled from Gaussian noise).
- (Optional) time step $( t )$ — used as a conditioning variable to guide denoising steps.

**Output of the model:**
- $( \hat{a}_{1:T}^{(t-1)} )$: The predicted **denoised** action sequence at step $( t-1 )$, given the noisy sequence at step $( t )$.


### 📈 Training Procedure

1. **Data Collection**:
   - Collect expert trajectories: pairs of $( (o_{1:T}, a_{1:T}) )$

2. **Forward Process (Noise Addition)**:
   - Sample noise level $( t \sim \{1, ..., T\} )$
   - Corrupt expert actions:
     $$
     a_{1:T}^{(t)} = \sqrt{\bar{\alpha}_t} \cdot a_{1:T} + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon,\quad \epsilon \sim \mathcal{N}(0, I)
     $$

3. **Reverse Process (Training Objective)**:
   - Train the model to **predict the added noise** $( \epsilon )$ or the clean action:
     $$
     \mathcal{L} = \mathbb{E}_{a_{1:T}, \epsilon, t} \left[ \left\| \hat{\epsilon}_\theta(o_{1:T}, a_{1:T}^{(t)}, t) - \epsilon \right\|^2 \right]
     $$


### 🏁 Inference (Trajectory Generation)

At test time:
1. Sample pure Gaussian noise: $( a_{1:T}^{(T)} \sim \mathcal{N}(0, I) )$
2. Iteratively denoise using the trained model:
   $[
   a_{1:T}^{(t-1)} = f_\theta(a_{1:T}^{(t)}, o_{1:T}, t)
   ]$
   from $( t = T \rightarrow 0 )$

3. Final output: $( \hat{a}_{1:T}^{(0)} )$, the predicted **action trajectory**.


### 📦 Summary Table

| Component         | Description                                                           |
|------------------|-----------------------------------------------------------------------|
| Input             | Observation $( o_{1:T} )$, noisy trajectory $( a_{1:T}^{(t)} )$       |
| Output            | Denoised trajectory $( \hat{a}_{1:T}^{(t-1)} )$ or noise estimate     |
| Conditioned On    | Observation sequence and timestep $( t )$                             |
| Training Target   | Minimize noise prediction or denoising error                          |
| Inference Output  | High-quality full trajectory $( a_{1:T} )$                            |
| Benefit           | Models multi-modal, high-fidelity expert behavior across time         |


### 🧠 Why Diffusion Policy is Useful in Robotics

- **Multi-modality**: Can represent many valid behaviors (e.g., different grasps).
- **Stability**: Better than autoregressive models which accumulate error.
- **Trajectory-level control**: Avoids myopic step-by-step prediction.