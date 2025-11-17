---
layout: post
title: The Actor–Critic Method
subtitle:
categories: Reinforcement-Learning
tags: [YouTube]
banner: "/assets/images/banners/yuanpang-wa-valley.jpg"
---

# 🎭 The Actor–Critic Method: Bridging Policy-Based and Value-Based Reinforcement Learning

The **Actor–Critic (AC)** method is a foundational algorithm in Reinforcement Learning that elegantly **combines the strengths of Policy-Based and Value-Based methods**.  
It uses two neural networks — the **Policy Network (Actor)** and the **Value Network (Critic)** — trained together in a feedback loop.  
The actor learns *how to act*, and the critic learns *how good the action was*.


## 🧩 1. Dual-Network Structure

The Actor–Critic framework introduces two interconnected approximations:

### 🎬 Policy Network (Actor)
- **Role:** Approximates the optimal policy $( \pi(a|s) )$; determines how the agent behaves.  
- **Notation:** $( \pi(a|s; \theta) )$, where $( \theta )$ are the actor’s trainable parameters.  
- **Input/Output:**  
  - Input: current state $( s )$ (e.g., a screenshot of Super Mario).  
  - Output: probability distribution over the action set $( A )$.  
  - Uses **Softmax** activation to ensure probabilities sum to one:
    $[
    \sum_{a \in A} \pi(a|s; \theta) = 1
    ]$
- **Example Output:**
  - $( \pi(\text{"left"}|s) = 0.2 )$
  - $( \pi(\text{"right"}|s) = 0.1 )$
  - $( \pi(\text{"up"}|s) = 0.7 )$


### 🧠 Value Network (Critic)
- **Role:** Approximates the **Action-Value Function** $( Q_\pi(s,a) )$, which measures how good an action is given the current state.  
- **Notation:** $( q(s,a; w) )$, where $( w )$ are the critic’s trainable parameters.  
- **Input/Output:**  
  - Input: the current state $( s )$.  
  - Output: predicted Q-values for all possible actions.
- **Example Output:**  
  $( q(s, "left"; w) = 2000 )$,  
  $( q(s, "right"; w) = 1000 )$,  
  $( q(s, "up"; w) = 3000 )$

The **State-Value Function** can be approximated by combining both networks:
$[
V(s; \theta, w) \approx \sum_{a} \pi(a|s; \theta) \, q(s,a; w)
]$

## 🔁 2. Training Iteration: One Step of the Actor–Critic Algorithm

Each iteration consists of **two key update phases** — first the **Critic**, then the **Actor** — based on a single environment interaction.

### 🧭 A. Interaction and Data Gathering

1. **Observe State and Sample Action:**  
   - Observe the current state $( s_t )$.  
   - The Actor samples an action according to the current policy:  
     $[
     a_t \sim \pi(\cdot | s_t; \theta_t)
     ]$

2. **Perform Action:**  
   - Execute $( a_t )$ in the environment.  
   - Observe **reward** $( r_t )$ and **next state** $( s_{t+1} )$.

3. **Sample Next Action (for TD target):**  
   - For learning only (not executed), sample a next action:  
     $[
     a'_{t+1} \sim \pi(\cdot | s_{t+1}; \theta_t)
     ]$


### ⚖️ B. Update the Value Network (Critic) via Temporal Difference (TD) Learning

The Critic learns to better estimate future returns.

4. **Evaluate Critic:**  
   $[
   q_t = q(s_t, a_t; w_t)
   ]$
   $[
   q'_{t+1} = q(s_{t+1}, a'_{t+1}; w_t)
   ]$

5. **Compute TD Error (δₜ):**  
   Measures how far the prediction is from the target return:
   $[
   \delta_t = q_t - (r_t + \gamma \cdot q'_{t+1})
   ]$
   where $( \gamma )$ is the discount factor.

6. **Differentiate Critic:**  
   Compute the gradient of the Q-function with respect to $( w )$:  
   $[
   d_{g,t} = \frac{\partial q(s_t, a_t; w)}{\partial w}
   ]$

7. **Update Critic Parameters (Gradient Descent):**
   $[
   w_{t+1} = w_t - \alpha \cdot \delta_t \cdot d_{g,t}
   ]$
   $( \alpha )$ is the Critic’s learning rate.

This update helps the critic produce more accurate estimates of $( Q_\pi(s,a) )$.


### 🎯 C. Update the Policy Network (Actor) via Policy Gradient

The Actor learns to choose actions that **increase the critic’s estimated value**.

8. **Differentiate Actor:**  
   Compute the gradient of the log-policy with respect to $( \theta )$:  
   $[
   d_{\pi,t} = \frac{\partial \log \pi(a_t | s_t; \theta)}{\partial \theta}
   ]$

9. **Update Actor Parameters (Gradient Ascent):**  
   The policy parameters are updated toward actions the critic deems better:
   $[
   \theta_{t+1} = \theta_t + \beta \cdot q_t \cdot d_{\pi,t}
   ]$
   $( \beta )$ is the Actor’s learning rate.  
   The product $( q_t \cdot d_{\pi,t} )$ represents the **stochastic policy gradient estimate** $( g(a_t, \theta_t) )$.

## ⚙️ 3. The Actor–Critic Feedback Loop

The two networks **learn from each other**:

- The **Actor** produces actions and learns from the Critic’s evaluations.  
- The **Critic** evaluates those actions and improves using the rewards from the environment.  
- Over time:
  - The Actor becomes better at selecting rewarding actions.
  - The Critic becomes better at judging those actions.

This tight feedback loop allows **continuous learning during interaction** — faster and smoother than pure Monte Carlo methods like REINFORCE.


## 📈 4. Summary of the Full Algorithm
| Step | Operation | Network | Learning Type |
|------|------------|----------|----------------|
| 1 | Observe state $s_t$ | — | — |
| 2 | Sample action $a_t \sim \pi(\cdot \mid s_t; \theta_t)$ | Actor | — |
| 3 | Execute $a_t$, observe $s_{t+1}$, $r_t$ | Environment | — |
| 4 | Sample $a'_{t+1} \sim \pi(\cdot \mid s_{t+1}; \theta_t)$ | Actor | — |
| 5 | Compute TD error $\delta_t = q_t - (r_t + \gamma q'_{t+1})$ | Critic | Supervised (TD) |
| 6 | Update $w_{t+1} = w_t - \alpha \, \delta_t \, d_{g,t}$ | Critic | Gradient Descent |
| 7 | Compute $d_{\pi,t} = \frac{\partial}{\partial \theta} \log \pi(a_t \mid s_t; \theta_t)$ | Actor | Policy Gradient |
| 8 | Update $\theta_{t+1} = \theta_t + \beta \, q_t \, d_{\pi,t}$ | Actor | Gradient Ascent |


## 🧠 5. Intuitive Understanding

- **Actor (Policy Network):** Learns the *strategy* — which actions to take.
- **Critic (Value Network):** Learns to *evaluate* that strategy by predicting how rewarding each action is.
- **Together:** The Actor adjusts its behavior based on the Critic’s feedback; the Critic adjusts its judgment based on real rewards.

This design makes Actor–Critic methods more **sample-efficient and stable** than pure policy gradient methods, and more **flexible** than value-based methods like DQN.


> **In short:**  
> The Actor–Critic method unifies decision-making and evaluation.  
> The **Actor** improves its policy using gradients guided by the **Critic’s value estimates**,  
> while the **Critic** refines its judgment using real environmental rewards —  
> creating a closed loop of continuous, mutually beneficial learning.

