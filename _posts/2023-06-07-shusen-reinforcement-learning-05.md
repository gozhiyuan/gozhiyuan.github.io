---
layout: post
title: AlphaGo, AlphaGo Zero, and AlphaZero - Deep Reinforcement Learning Meets Search
subtitle:
categories: Reinforcement-Learning
tags: [YouTube]
banner: "/assets/images/banners/yuanpang-wa-valley.jpg"
---

# ♟️ AlphaGo, AlphaGo Zero, and AlphaZero - Deep Reinforcement Learning Meets Search

This blog explores the **structure and training process** of **AlphaGo** and its successors, **AlphaGo Zero** and **AlphaZero**, illustrating how deep reinforcement learning and search are combined to achieve **superhuman performance** in complex board games.


## 🧩 1. Introduction and Go Game Complexity

- **Game Description:**  
  Go is played on a 19×19 grid with 361 points.  
  A game **state** $s$ represents the arrangement of black, white, and empty intersections.

- **State Representation:**  
  A simplified state can be modeled as a tensor $19 \times 19 \times 2$ — one plane for black stones, one for white.  
  The original AlphaGo used $19 \times 19 \times 48$ feature planes to encode richer context (e.g., liberties, ko, move history).

- **Action Space:**  
  Each move corresponds to placing a stone on an empty intersection.  
  $$
  A = \{1, 2, 3, \dots, 361\}
  $$  
  The “pass” move is also a valid action.

- **Complexity:**  
  The number of possible Go game sequences is on the order of  
  $$
  10^{170}
  $$  
  — vastly greater than Chess ($\sim 10^{47}$), making brute-force search infeasible.

  | **Game**        | **State-Space Complexity** | **Game-Tree Complexity** | 
  |------------------|----------------------------|---------------------------
  | **Chess**        | $\sim 10^{43}$             | $\sim 10^{120}$           
  | **Go (19 X 19)** | $\sim 10^{170}$–$10^{180}$ | $\sim 10^{360}$           
  | **Checkers**     | $\sim 10^{20}$             | $\sim 10^{40}$            


## 🧠 2. AlphaGo Training Strategy (High-Level)

![alt_text](/assets/images/shusen-rl/05/1.png "image_tooltip")

AlphaGo’s training involves **three key stages**:

1. **Supervised Learning (Behavior Cloning):**  
   Train an initial **policy network** to imitate human expert moves.

2. **Reinforcement Learning (Self-Play Policy Gradient):**  
   Refine the policy network by playing against itself.

3. **Value Network Training:**  
   Train a **value network** to predict the win probability of a given state.

During play, **Monte Carlo Tree Search (MCTS)** combines the policy and value networks to choose moves.

![alt_text](/assets/images/shusen-rl/05/2.png "image_tooltip")

## 🏗️ 3. Policy and Value Network Architecture

- **Input Representation:**  
  For AlphaGo Zero, input is a $19 \times 19 \times 17$ tensor:
  - 8 planes for black’s last 8 moves  
  - 8 planes for white’s last 8 moves  
  - 1 plane indicating whose turn it is

- **Policy Network Output:**  
  $\pi(a \mid s; \theta)$  
  Outputs a probability distribution over all legal actions via **Softmax**:
  $$
  \sum_{a \in A} \pi(a \mid s; \theta) = 1
  $$

- **Value Network Output:**  
  $v(s; w) \in [-1, 1]$  
  A scalar estimating the expected game outcome (win = +1, loss = −1).  
  Often, the policy and value heads share convolutional layers.


## 🧍‍♂️ 4. Training Step 1 — Behavior Cloning (Imitation Learning)

![alt_text](/assets/images/shusen-rl/05/3.png "image_tooltip")

- **Goal:** Initialize a policy network to imitate human expert moves.  
- **Method:** Treat the problem as **multi-class classification** (361 possible moves).

1. Observe a state $s_t$ from human games.  
2. Predict action distribution $p_t = \pi(\cdot \mid s_t; \theta)$.  
3. True label $y_t$ = one-hot vector of human move $a_t^\ast$.  
4. Minimize **Cross-Entropy Loss**:
   $$
   L = - \sum_a y_t(a) \log p_t(a)
   $$

- **Limitation:**  
  When the model encounters unseen states, it performs poorly — hence the need for **self-play reinforcement learning** to generalize beyond human data.


## ⚙️ 5. Training Step 2 — Policy Gradient (Self-Play RL)

![alt_text](/assets/images/shusen-rl/05/4.png "image_tooltip")

After imitation learning, the agent improves via **self-play** using the policy gradient method.

- **Setup:**  
  The current network (Player) plays against an older snapshot (Opponent).  
  Only the Player’s parameters are updated.

- **Reward:**  
  $$
  r_T =
  \begin{cases}
  +1, & \text{if win} \\
  -1, & \text{if lose}
  \end{cases}
  $$  
  Intermediate rewards are 0.

- **Return:**  
  Since Go’s outcome is binary, every state in a game has the same return:
  $$
  U_t = r_T
  $$

- **Policy Gradient Objective:**  
  $$
  \nabla_\theta J(\theta)
  = \mathbb{E}_t \!\left[
  \nabla_\theta \log \pi(a_t \mid s_t; \theta) \, U_t
  \right]
  $$

- **Parameter Update:**  
  $$
  \theta \leftarrow \theta + \beta \, \nabla_\theta J(\theta)
  $$


## 📈 6. Training Step 3 — Value Network Training

![alt_text](/assets/images/shusen-rl/05/5.png "image_tooltip")

- **Goal:** Learn $v(s; w) \approx V^\pi(s) = \mathbb{E}[U_t \mid S_t = s]$.  
- **Loss Function:**
  $$
  L(w) = \sum_{t=0}^{T} \big(v(s_t; w) - U_t\big)^2
  $$
- **Optimization:**  
  Use gradient descent to minimize mean squared error between predicted and actual returns.  
  This transforms the value network into a learned evaluator of board positions.


## 🌳 7. Execution — Monte Carlo Tree Search (MCTS)

![alt_text](/assets/images/shusen-rl/05/6.png "image_tooltip")

At runtime, AlphaGo uses **MCTS** to plan moves.

Each simulation repeats four stages:

### 1️⃣ Selection
Choose action $a$ with highest **PUCT score**:
$$
\text{score}(a) = Q(a) + \eta \cdot \frac{\pi(a \mid s; \theta)}{1 + N(a)}
$$
- $Q(a)$ — mean value from previous rollouts  
- $N(a)$ — visit count  
- $\pi(a \mid s; \theta)$ — prior probability from policy network

### 2️⃣ Expansion
Expand the tree by sampling the opponent’s response  
$a_t' \sim \pi(\cdot \mid s_t'; \theta)$ to form next state $s_{t+1}$.

### 3️⃣ Evaluation
Estimate the value of $s_{t+1}$ using:
$$
V(s_{t+1}) = \tfrac{1}{2} v(s_{t+1}; w) + \tfrac{1}{2} r_T
$$
— averaging the network’s value and a rollout result.

### 4️⃣ Backup
Propagate $V(s_{t+1})$ back up the tree to update:
$$
Q(a_t) = \frac{1}{N(a_t)} \sum V(s_{t+1})
$$

**Move Selection:**  
After many simulations, choose the move $a_t$ with the **highest visit count** $N(a_t)$.


## 🧬 8. AlphaGo Zero and AlphaZero

### 🌀 AlphaGo Zero
![alt_text](/assets/images/shusen-rl/05/7.png "image_tooltip")

- Removes human data — learns purely from **self-play**.  
- Uses **MCTS visit counts** as the target distribution for training the policy:
  $$
  L = \text{CrossEntropy}(n, p)
  $$  
  where $n$ = normalized visit counts from MCTS.

### 🌍 AlphaZero
![alt_text](/assets/images/shusen-rl/05/8.png "image_tooltip")
A general version capable of mastering **Go, Chess, and Shogi** from scratch.

- **Unified Network:**
  $$
  f_\theta(s) = (p, v)
  $$
  where  
  - $p$: policy probabilities over actions  
  - $v$: expected game outcome

- **Training Loss:**
  $$
  l = (z - v)^2 - \pi^\top \log p + c \, \|\theta\|^2
  $$
  where  
  - $z$ = actual game outcome (+1 / −1)  
  - $\pi$ = search probabilities from MCTS  
  - $c$ = regularization coefficient

- **Search Efficiency:**  
  AlphaZero evaluates ≈ 80 000 positions/sec in Chess (vs. Stockfish’s 70 million), demonstrating the power of **neural-guided selective search**.


| **Feature** | **AlphaGo (vs. Lee Sedol)** | **AlphaGo Zero** |
|--------------|-----------------------------|------------------|
| **Input Data** | Human expert games **+** self-play | Purely self-play from random moves |
| **Neural Networks** | Separate **Policy** and **Value** networks | A single, unified **Policy–Value** network |
| **Training** | Complex multi-stage pipeline | Single reinforcement learning loop |
| **Search Features** | Used fast rollouts for evaluation | Relied solely on the value network output |
| **Performance** | Defeated world champion **Lee Sedol (4–1)** | Defeated **AlphaGo 100–0** |


## 🏁 Summary

| **Stage** | **Objective** | **Learning Type** | **Description** |
|------------|----------------|--------------------|------------------|
| 1 | Behavior Cloning | Supervised | Learn to mimic human moves |
| 2 | Policy Gradient | Reinforcement | Improve policy via self-play |
| 3 | Value Network | Supervised (Regression) | Predict win probability |
| — | MCTS Execution | Search | Combine policy + value for decision-making |

