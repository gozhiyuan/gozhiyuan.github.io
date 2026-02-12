---
layout: post
title: Value-Based Reinforcement Learning Foundations
subtitle:
categories: Reinforcement-Learning
tags: [YouTube]
banner: "/assets/images/banners/yuanpang-wa-valley.jpg"
---

# Value-Based Reinforcement Learning Foundations

Value-based reinforcement learning (RL) focuses on estimating how *valuable* it is to take a particular action in a given state — quantified as the **expected discounted future reward**.  
This approach underpins algorithms like **Q-learning** and **Deep Q-Networks (DQN)**.

## 1️. Value-Based Reinforcement Learning Foundations

### 🧩 The Action-Value Function $( Q_\pi(s, a) )$

The **Action-Value Function** for a policy $( \pi )$, denoted $( Q_\pi(s_t, a_t) )$, is the **expected discounted return** given that the agent starts from state $( s_t )$, takes action $( a_t )$, and then follows policy $( \pi )$ thereafter.

$[
Q_\pi(s_t, a_t) = E[U_t \mid S_t = s_t, A_t = a_t]
]$

The **discounted return** $( U_t )$ is defined as:
$[
U_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \dots
]$
where $( \gamma )$ (gamma) is the *discount factor* that determines how much the agent values future rewards.

The expectation $( E[\cdot] )$ accounts for:
- **Future actions** $( A_{t+1}, A_{t+2}, … )$ drawn from the policy $( \pi )$.
- **Future states** $S_{t+1}, S_{t+2}, \ldots$ drawn from the environment’s transition probability $p(s' \mid s, a)$.


#### 🧠 Example:
Imagine an agent playing *Super Mario*.  
If Mario is at position $( s_t )$ and chooses the action “jump,” then $( Q_\pi(s_t, \text{jump}) )$ is the *expected total score* (coins, enemies defeated, etc.) Mario can still accumulate if he continues playing according to policy $( \pi )$.


### 🌟 The Optimal Action-Value Function $Q^\ast(s, a)$

The **goal of value-based RL** is to find the **optimal** action-value function, $Q^\ast(s, a)$.

$$
Q^\ast(s_t, a_t) = \max_\pi Q_\pi(s_t, a_t)
$$

- **Definition:** $Q^\ast(s, a)$ represents the *maximum expected discounted return* achievable by taking action $a$ in state $s$, and then behaving optimally thereafter.  
- **Significance:** No policy can outperform $Q^\ast(s, a)$; it represents the *best possible outcome*.  
- **Optimal Action Selection:**  
  $$
  a^\ast = \arg\max_a Q^\ast(s, a)
  $$  
  The optimal policy $\pi^\ast$ always chooses the action with the highest $Q^\ast(s, a)$.  
- **Challenge:** $Q^\ast(s, a)$ is **unknown**, and directly estimating it for complex environments is computationally infeasible.


#### 🎮 Example:
If Mario’s $( Q^\ast(s, a) )$ values for the current frame are:

| Action | $Q^\ast(s,a)$ |
|--------|---------------|
| Jump   | 2500          |
| Left   | 1000          |
| Right  | 3000          |

Then the optimal policy will choose **“Right”**, since it yields the maximum expected reward.


## 2️. Deep Q-Network (DQN)

The **Deep Q-Network (DQN)** is a breakthrough solution that uses a **neural network** to *approximate* $( Q^\ast(s, a) )$.  
Introduced by DeepMind (2015), it allowed agents to play Atari games directly from pixels — achieving human-level performance.

### 🧮 Network Structure

- **Function Approximation:**  
  DQN uses a neural network $( Q(s, a; w) )$, parameterized by weights $( w )$, to approximate $( Q^\ast(s, a) )$.
- **Input:** The **state** $( s )$ (e.g., the current game screen).  
- **Output:** A vector of Q-values for all possible actions $( a \in A )$.

#### Example:
If the action space is `{left, right, jump}`, the network might output:

| Action | Q(s, a; w) |
|--------|------------|
| Left   | 2000       |
| Right  | 1000       |
| Jump   | 3000       |


Here, the agent selects:
$[
a_t = \arg\max_a Q(s_t, a; w)
]$
→ **“Jump”**, since it gives the highest predicted value.

### 🧭 Exploration vs. Exploitation
To balance **exploration** (trying new actions) and **exploitation** (choosing the best-known action), DQN often uses an **ε-greedy policy**:
- With probability **ε**, choose a random action (explore).
- With probability **1 - ε**, choose $( \arg\max_a Q(s, a; w) )$ (exploit).


## 3️. Temporal Difference (TD) Learning

**Temporal Difference (TD) Learning** is the key algorithm used to train the DQN.  
It updates the Q-function *incrementally*, based on new experience, without waiting for the episode to finish.

### 🔁 Recursive Identity

The return $( U_t )$ can be defined recursively:
$[
U_t = R_t + \gamma U_{t+1}
]$

Since $( Q(s_t, a_t; w) )$ approximates $( U_t )$, we have:
$[
Q(s_t, a_t; w) \approx R_t + \gamma Q(s_{t+1}, a_{t+1}; w)
]$

This forms the foundation for **bootstrapping** — updating current estimates using future estimates.


### 🧠 The Principle of Temporal Difference (TD) Learning

#### 1️⃣ The Need for Early Updates
In traditional supervised learning, a model $Q(w)$ estimating travel time (e.g., from **NYC → Atlanta**) must complete the trip before learning from the actual total duration.

For example:
- Predicted total time: $q = 1000$ minutes  
- Actual total time: $y = 860$ minutes  

The loss is computed after completion:  
$\mathcal{L} = (q - y)^2$  

However, **Temporal Difference (TD) Learning** asks:  
> Can we update earlier, before finishing the trip?

Yes — TD allows updating $w$ after each **intermediate milestone**, making learning more efficient.

#### 2️⃣ The Recursive Identity (Bellman Equation)
TD learning is built on the recursive property of **expected return**:

$$
U_t = R_t + \gamma \, U_{t+1}
$$

Here:
- $U_t$ = cumulative discounted future reward  
- $R_t$ = immediate reward  
- $\gamma$ = discount factor $(0 < \gamma < 1)$  

This identity says:  
> The total return equals the immediate reward plus the discounted expected return of the next step.

#### 3️⃣ TD Target (Bootstrapping)
TD constructs a **TD Target** $(y)$ — a better estimate of the true return:

$$
y_t = R_t + \gamma \, Q(s_{t+1}, a_{t+1}; w)
$$

This mixes:
- the **actual** reward ($R_t$), and  
- the **predicted** value of the next state ($Q(s_{t+1}, a_{t+1}; w)$).

This method is called **bootstrapping** because it “pulls itself up” using its own future estimate.


### 🚗 TD Example: Driving from NYC to Atlanta

We can model this as:

$$
T_{NYC→Atlanta} \approx T_{NYC→DC} + T_{DC→Atlanta}
$$

| Element | Prediction (Model $Q(w)$) | Actual Experience |
|----------|----------------------------|------------------|
| Initial Prediction ($q$) | $1000$ min (NYC→Atlanta) | N/A |
| Intermediate Step | $600$ min (DC→Atlanta) | $r = 300$ min (NYC→DC) |

Now that we reached **DC**, we can compute a **TD Target**:

$$
y = (\text{Actual NYC→DC}) + (\text{Predicted DC→Atlanta})
$$

$$
y = 300 + 600 = 900 \text{ minutes}
$$

Thus, the **TD target** $y = 900$ is a more reliable estimate than the initial $1000$ minutes.


#### 4️⃣ The TD Error ($\delta$)
The **TD error** measures how wrong the model was:

**Method A: Prediction vs. TD Target**

$$
\delta = q - y = 1000 - 900 = 100
$$

This 100-minute error guides the gradient update:

$$
\nabla_w \mathcal{L} = (q - y) \, \nabla_w Q(w)
$$


**Method B: Implied Segment Comparison**

1. Model implied NYC→DC = $1000 - 600 = 400$ minutes  
2. Actual NYC→DC = $300$ minutes  
3. $\delta = 400 - 300 = 100$

Both perspectives agree — the model **overestimated** by 100 minutes and must adjust downward.


## 4. 🤖 TD Learning in Deep Q-Networks (DQN)

In DQN, the neural network $Q(s, a; w)$ approximates the optimal function $Q^\ast(s, a)$.

| Component | Formula | Description |
|------------|----------|-------------|
| **Prediction** | $q_t = Q(s_t, a_t; w)$ | Network’s output for current state-action |
| **TD Target** | $y_t = r_t + \gamma \max_a Q(s_{t+1}, a; w)$ | From Bellman optimality |
| **TD Error** | $\delta_t = q_t - y_t$ | Difference between prediction and target |
| **Loss** | $L_t = \tfrac{1}{2} (q_t - y_t)^2$ | Used for gradient descent |

Weight update rule:

$$
w_{t+1} = w_t - \alpha \, (q_t - y_t) \, \nabla_w Q(s_t, a_t; w_t)
$$

This iterative process — predicting, bootstrapping, and correcting via TD error — enables DQN to **progressively refine** $Q^\ast(s, a)$ across experience.

### 🔍 Practical Intuition (Common Questions)

#### Does the Q-function output one score per action?
Yes. For a given state $s_t$, DQN outputs a vector of values:
$$
[Q(s_t, a_1; w), Q(s_t, a_2; w), \dots, Q(s_t, a_n; w)]
$$
Each value estimates the **expected discounted return** of taking that action now, then following the learned policy.

#### Does each action produce a reward $r_t$?
Yes. After every action, the environment returns an immediate reward $r_t$ (possibly positive, zero, or negative).

- **Taxi (Gym-style example):**
  - step penalty: $-1$
  - illegal pickup/drop-off: $-10$
  - successful drop-off: $+20$
- **Robotics (typical shaping):**
  - progress term (e.g., closer to goal): positive
  - control effort penalty (large torques): negative
  - collision/safety penalty: negative
  - task completion bonus: positive terminal reward

#### What is the TD target $y_t$?
$$
y_t = r_t + \gamma \max_{a'} Q_{\text{target}}(s_{t+1}, a')
$$

- $r_t$: immediate reward at current step  
- $\gamma \max_{a'} Q_{\text{target}}(s_{t+1}, a')$: discounted estimate of the **best future return** from the next state

So $y_t$ is not just “current return.” It is a **one-step Bellman target**: reward now + estimated future value.

#### Why can reward be added to a Q score?
Because both represent return in the same unit. By Bellman recursion:
$$
G_t = r_t + \gamma G_{t+1}
$$
and Q-values are expectations of this return. So adding immediate reward to discounted future value is exactly the correct decomposition.

#### Is $Q_{\text{target}}$ the same as $Q$?
Same network architecture, different parameters.

- **Online network** $Q(s,a;w)$: updated every gradient step.
- **Target network** $Q_{\text{target}}(s,a;w^-)$: a delayed copy, updated periodically (or softly).

This delay stabilizes training and reduces feedback loops from chasing a rapidly moving target.

#### What does $\max_{a'}$ mean?
It means: evaluate all possible next actions in $s_{t+1}$ and pick the largest predicted Q-value:
$$
\max_{a'} Q_{\text{target}}(s_{t+1}, a')
$$
This is “the value of the best next action,” not necessarily the sampled exploratory action.


Then:
$[
y_t = 10 + 0.9 \times 100 = 100
]$
If the current prediction $( Q(s_t, a_t; w) = 80 )$,  
the **TD error** is $( (80 - 100) = -20 )$,  
meaning the agent underestimated the value of this action.


### ⚙️ Training Algorithm

Each training step updates the neural network parameters $( w )$ using gradient descent on the **TD loss**:

1. **Observe & Predict:**  
   Predict $( q_t = Q(s_t, a_t; w_t) )$
2. **Environment Response:**  
   Observe reward $( r_t )$ and next state $( s_{t+1} )$
3. **Compute TD Target:**  
   $( y_t = r_t + \gamma \max_a Q(s_{t+1}, a; w_t) )$
4. **Compute Loss:**  
   $[
   L_t = \frac{1}{2}(q_t - y_t)^2
   ]$
5. **Update Weights:**  
   $[
   w_{t+1} = w_t - \alpha (q_t - y_t) \nabla_w Q(s_t, a_t; w_t)
   ]$
   where $( \alpha )$ is the learning rate.

This process repeats as the agent interacts with the environment, **gradually improving** its estimation of $( Q^\ast(s, a) )$.


### Why the TD Target Uses the Maximum Operator

| Concept | Formula | Explanation |
|----------|----------|-------------|
| **TD Target** | $y_t = r_t + \gamma \max_a Q(s_{t+1}, a; w)$ | From Bellman optimality |

1. The network predicts $Q(s_{t+1}, a; w)$ for all actions.  
2. The **maximum** represents the **best possible future decision**:
   $$
   \max_a Q(s_{t+1}, a; w)
   $$
3. This expresses the Bellman **Optimality Equation**, meaning the update always assumes the agent acts optimally in the future.

Thus, DQN learns toward $Q^\ast(s,a)$ — the *optimal* value function — instead of just following the current policy $Q_\pi(s,a)$.

### Why DQN Only Looks One Step Ahead but Learns the Whole Path

DQN bootstraps from **one-step lookahead**:
$$
y_t = r_t + \gamma \max_a Q(s_{t+1}, a; w)
$$
Even though it updates using only the next state, this **recursive update propagates future knowledge backward**.

Over many episodes:
- Reward information moves backward through earlier states.
- The Q-values across time become consistent.
- The network converges toward the global optimum $Q^\ast(s,a)$.

This is why **a one-step TD update** can still produce a **long-horizon optimal policy**.

### Training Dynamics and Number of Steps

If you have **100 episodes** with **10 time steps** each:
- Total transitions: $100 \times 10 = 1000$
- Each transition provides one TD update.

During training:
1. Each $(s_t, a_t, r_t, s_{t+1})$ forms a sample.
2. Compute the target $y_t = r_t + \gamma \max_a Q(s_{t+1}, a; w)$.
3. Update $Q(s_t, a_t; w)$ via gradient descent.

Hence, one episode contributes 10 updates, and 100 episodes → 1000 TD updates.
However, these transitions can be **reused multiple times** through a **Replay Buffer**.

### Replay Buffer (Experience Replay)

The **Replay Buffer** stores past experiences to improve stability and efficiency.

Each experience tuple:
$$
(s_t, a_t, r_t, s_{t+1}, \text{done})
$$
is appended to a buffer of fixed capacity (e.g., 100,000 samples).

During training:
- Random mini-batches are drawn from this buffer.
- Each batch is used to compute TD targets and perform gradient descent.

#### Benefits

| Benefit | Explanation |
|----------|-------------|
| **Breaks correlation** | Random sampling removes temporal dependence between transitions. |
| **Improves stability** | Training resembles i.i.d. supervised data. |
| **Boosts sample efficiency** | Experiences can be reused multiple times. |
| **Enables off-policy learning** | Allows training with experiences from older policies. |

### Prioritized Replay

Not all experiences are equally useful.  
**Prioritized Experience Replay** samples transitions based on their **TD error magnitude**.  
Using priority weights $P_i$ for each experience $i$:
$$
P_i = \frac{|\delta_i|^\alpha}{\sum_k |\delta_k|^\alpha}
$$
Larger errors → more surprising transitions → more frequent updates.

This focuses learning on states where the network is most uncertain or wrong.

### 🧭 Summary of DQN + TD Learning

| Component | Concept | Key Equation |
|------------|----------|--------------|
| **Bootstrapping** | One-step update using current network | $y_t = r_t + \gamma \max_a Q(s_{t+1}, a; w)$ |
| **Error Signal** | Measures model inaccuracy | $\delta_t = q_t - y_t$ |
| **Optimization** | Gradient descent on squared error | $L_t = \frac{1}{2}(q_t - y_t)^2$ |
| **Replay Buffer** | Reuses experience, improves stability | Sample random mini-batches |
| **Goal** | Learn $Q^\ast(s,a)$ to act optimally | $a_t^\ast = \arg\max_a Q(s_t, a; w)$ |

Through many one-step TD updates — sampled from diverse replay experiences —  
DQN gradually propagates reward information backward, **learning the optimal long-term policy** even though each update looks ahead just one step.

## 5. 🎯 Why DQN Learns $Q^\ast(s, a)$ (Optimal Action–Value Function) — Not $V(s)$ or $\pi(a \mid s)$

Let’s break down this important distinction between **DQN**, **Value Function**, and **Policy Learning** — since they all aim to “choose good actions,” but in **different ways**.


### The Core Difference: What Is Being Learned

| **Method Type** | **Learns...** | **Output Meaning** | **Goal** |
|------------------|----------------|--------------------|-----------|
| **Value-Based (DQN)** | $Q^\ast(s, a)$ | Expected discounted return for each possible action in state $s$ | Pick $\arg\max_a Q^\ast(s, a)$ |
| **Policy-Based (PG / Actor)** | $\pi(a \mid s; \theta)$ | Probability distribution over actions | Sample $a \sim \pi(a \mid s; \theta)$ |
| **Value Function (V)** | $V(s)$ | Expected return under the current policy | Evaluates how good a state is (but not which action to take) |


So:
- $V(s)$ → evaluates **states**.
- $Q(s,a)$ → evaluates **state–action pairs**.
- $\pi(a \mid s)$ → directly defines the **action probabilities**.

### Why DQN Learns $Q^\ast(s,a)$, Not $V(s)$

The **Bellman optimality equation** defines the optimal action-value function:
$$
Q^\ast(s,a) = \mathbb{E} \big[ r_t + \gamma \max_{a'} Q^\ast(s',a') \mid s,a \big]
$$

- The $\max_{a'}$ term ensures the function encodes the **best possible future actions**.
- $Q^\ast(s,a)$ directly represents _“the total future reward if I take this action now and then act optimally.”_

Hence, if you know $Q^\ast(s,a)$, you can **derive both the value function and the optimal policy**:

$$
V^\ast(s) = \max_{a} Q^\ast(s,a)
$$

$$
\pi^\ast(a \mid s) =
\begin{cases}
1, & \text{if } a = \arg\max_{a'} Q^\ast(s,a') \\
0, & \text{otherwise}
\end{cases}
$$

That’s why DQN doesn’t need a separate policy network —  
the **policy emerges** automatically as a *greedy function* over $Q^\ast$.

### Does DQN Output the “Best Action”?

Yes — the **output layer** of the DQN predicts a *Q-value (score)* for **each possible action** in the current state:
$$
[Q(s, a_1; w), Q(s, a_2; w), \dots, Q(s, a_n; w)]
$$

- The **input** is the state (e.g., an image in Atari or a vector in control tasks).
- The **output** is a vector of Q-scores, one for each action.
- The **chosen action** is:
  $$
  a_t = \arg\max_a Q(s_t, a; w)
  $$

This is how DQN *acts greedily* based on predicted Q-values.

### ⚙️ Is DQN Deterministic? What About State Transitions?

Let’s clarify two separate aspects here — **the agent’s policy (DQN)** and **the environment’s dynamics (state transitions)**.

#### DQN’s Policy Is *Deterministic* (by Design)

In standard **Deep Q-Networks (DQN)**:
- The network outputs **a Q-value for each discrete action**:  
  $$
  Q(s,a;w)
  $$
- The agent chooses the **action with the highest Q-value**:
  $$
  a_t = \arg\max_a Q(s_t,a;w)
  $$

This means the **policy is deterministic** once the network parameters $w$ are fixed — it always picks the same action for the same state.

✅ **However**, during training we add randomness for exploration:
- The most common strategy is **ε-greedy**:
  - With probability ε, pick a **random action** (exploration).
  - With probability 1 − ε, pick the **best action** (exploitation).

So the **behavior policy** during training is *stochastic*, but the **target policy** (the one being learned) is deterministic:
$$
\pi^\ast(s) = \arg\max_a Q^\ast(s,a)
$$


#### DQN Works Best for **Discrete Action Spaces**

- Each output neuron of the Q-network corresponds to one possible action.
- Computing $\max_a Q(s,a)$ is easy when $a$ is discrete.

But when the **action space is continuous** (like steering angles or forces),  
you cannot take a simple max over all $a$ — it’s infinite.

For that reason, **vanilla DQN cannot handle continuous control**.

➡️ Solutions for continuous actions:
- **DDPG (Deep Deterministic Policy Gradient)** — uses an actor–critic structure where the actor outputs continuous actions.
- **TD3** and **SAC** — improved variants for stability and exploration.


#### The Environment (State Transitions) Can Be Stochastic

The *state transition function* is defined as:
$$
p(s_{t+1} \mid s_t, a_t)
$$
This represents the **probability distribution** of the next state given the current state and action.

- In **deterministic environments**, the same action in the same state always leads to the same next state.
- In **stochastic environments**, transitions are *sampled* from this distribution.

Thus:
- DQN’s *policy* is **deterministic** (it picks a fixed action per state).
- The *environment* can still be **stochastic** — DQN learns the **expected return** under that randomness.


#### 🧭 Summary

| Aspect | Deterministic? | Notes |
|---------|----------------|-------|
| **DQN Policy** | ✅ Yes (greedy $\arg\max_a Q$) | Stochastic only during exploration (ε-greedy) |
| **Action Space** | 🚫 Discrete only | Continuous actions require DDPG/TD3/SAC |
| **Environment Transition $p(s' \mid s,a)$** | ❌ Usually stochastic | DQN learns expected Q-values over these outcomes |
| **Q-function target** | Deterministic estimate of expectation | $Q^\ast(s,a) = \mathbb{E}[R_t + \gamma \max_a Q(s',a)]$ |

💡 **Intuition:**  
DQN itself acts deterministically once trained — but it *learns in a probabilistic world*, averaging over random transitions and rewards to approximate the optimal deterministic action for each state.



### So Is It “Like” Policy-Based RL?

Yes — conceptually, both aim to **find the best action**.  
But they differ fundamentally in *what they learn and how they update*:

| **Aspect** | **Value-Based (DQN)** | **Policy-Based (PG, Actor–Critic)** |
|-------------|------------------------|------------------------------------|
| **What is learned** | Deterministic value function $Q(s, a)$ | Stochastic policy $\pi(a \mid s; \theta)$ |
| **Output** | Scores (Q-values) for each action | Probability distribution over actions |
| **How to pick actions** | $\arg\max_a Q(s, a)$ | Sample from $\pi(a \mid s)$ |
| **Gradient signal** | TD error $(q_t - y_t)$ | Policy gradient $Q(s, a) \nabla_\theta \log \pi(a \mid s; \theta)$ |
| **Exploration** | $\epsilon$-greedy (add randomness externally) | Built-in stochasticity in $\pi(a \mid s)$ |
| **Advantage** | Stable, off-policy, efficient | Works for continuous actions |
| **Limitation** | Discrete actions only | Higher variance, harder to optimize |


So although both methods choose actions that *maximize expected return*,  
- **DQN** does so **indirectly** (via $\max_a Q$).  
- **Policy-based RL** does so **directly** (via parameterizing $\pi(a \mid s)$).
