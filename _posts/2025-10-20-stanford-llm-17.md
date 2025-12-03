---
layout: post
title: LLM Alignment - GRPO Implementation
subtitle: Language Modeling from Scratch Lecture 17
categories: Large-Language-Model Reinforcement-Learning
tags: [Stanford-LLM-From-Scratch-2025]
banner: "/assets/images/banners/yuanpang-wa-iceburg2.jpg"
---

# LLM Alignment - GRPO Implementation

The blog transitions to a practical walkthrough illustrating **policy gradient mechanics** through the lens of **GRPO (Group Relative Policy Optimization)**.  
GRPO simplifies PPO by removing the **Value function (critic)** and instead leveraging the **group structure** of LLM rollouts — i.e., multiple responses per prompt — to compute baselines internally.


[Course link](https://stanford-cs336.github.io/spring2025/)

![alt_text](/assets/images/llm-from-scratch/17/1.png "image_tooltip")


## 1. The Simple Task and Reward Design

A toy example is used where a simple, non-autoregressive model learns to **sort $n$ numbers**.

### Task Definition

```python
def sort_distance_reward(prompt: list[int], response: list[int]) -> float:  # @inspect prompt, @inspect response
    """
    Return how close response is to ground_truth = sorted(prompt).
    In particular, compute number of positions where the response matches the ground truth.
    """
    assert len(prompt) == len(response)
    ground_truth = sorted(prompt)
    return sum(1 for x, y in zip(response, ground_truth) if x == y)

def sort_inclusion_ordering_reward(prompt: list[int], response: list[int]) -> float:  # @inspect prompt, @inspect response
    """
    Return how close response is to ground_truth = sorted(prompt).
    """
    assert len(prompt) == len(response)
    # Give one point for each token in the prompt that shows up in the response
    inclusion_reward = sum(1 for x in prompt if x in response)  # @inspect inclusion_reward
    # Give one point for each adjacent pair in response that's sorted
    ordering_reward = sum(1 for x, y in zip(response, response[1:]) if x <= y)  # @inspect ordering_reward
    return inclusion_reward + ordering_reward

def simple_task():
    # Task: sorting n numbers
    # Prompt: n numbers
    prompt = [1, 0, 2]
    # Response: n numbers
    response = [0, 1, 2]
    # Reward should capture how close to sorted the response is.
    # Define a reward that returns the number of positions where the response matches the ground truth.
    reward = sort_distance_reward([3, 1, 0, 2], [0, 1, 2, 3])  # @inspect reward
    reward = sort_distance_reward([3, 1, 0, 2], [7, 2, 2, 5])  # @inspect reward  @stepover
    reward = sort_distance_reward([3, 1, 0, 2], [0, 3, 1, 2])  # @inspect reward  @stepover
    # Define an alternative reward that gives more partial credit.
    reward = sort_inclusion_ordering_reward([3, 1, 0, 2], [0, 1, 2, 3])  # @inspect reward
    reward = sort_inclusion_ordering_reward([3, 1, 0, 2], [7, 2, 2, 5])  # @inspect reward  @stepover
    reward = sort_inclusion_ordering_reward([3, 1, 0, 2], [0, 3, 1, 2])  # @inspect reward  @stepover
    #Note that the second reward function provides more credit to the 3rd response than the first reward function.

```

### Reward Components

- The goal is to define a reward that provides **partial credit** to mitigate the sparse reward problem.
- Two types of reward components are defined:
  - **Inclusion reward:** Gives points for tokens from the prompt that appear in the response.
  - **Ordering reward:** Gives points for adjacent token pairs that are correctly sorted.

This yields a scalar reward $R$ for each sampled completion.


## 2. Computing Deltas ($\delta$)

Different formulations for the *advantage-like* quantity $\delta$ are explored:

### Centered Rewards

   $$
   \delta = R - \mu_{\text{group}}
   $$
   Here $\mu_{\text{group}}$ is the mean reward within the group (responses to the same prompt).  
   - If all rewards are equal → no update occurs.  
   - Centering stabilizes training by pushing the model away from below-average completions.

2. **Normalized Rewards (GRPO style):**
   $$
   \delta = \frac{R - \mu_{\text{group}}}{\sigma_{\text{group}}}
   $$
   Dividing by the group’s standard deviation $\sigma_{\text{group}}$ makes the update **scale-invariant** to reward magnitudes.

### Max Rewards

   $$
   \delta =
   \begin{cases}
   R - \mu_{\text{group}}, & \text{if } R = \max(R_{\text{group}}) \\
   0, & \text{otherwise}
   \end{cases}
   $$
   Only the **best-performing completions** in a group receive gradient updates — helping the model focus on “top answers” instead of partial-credit responses.

```python
def compute_reward(prompts: torch.Tensor, responses: torch.Tensor, reward_fn: Callable[[list[int], list[int]], float]) -> torch.Tensor:
    """
    Args:
        prompts (int[batch pos])
        responses (int[batch trial pos])
    Returns:
        rewards (float[batch trial])
    """
    batch_size, num_responses, _ = responses.shape
    rewards = torch.empty(batch_size, num_responses, dtype=torch.float32)
    for i in range(batch_size):
        for j in range(num_responses):
            rewards[i, j] = reward_fn(prompts[i, :], responses[i, j, :])
    return rewards

def compute_deltas(rewards: torch.Tensor, mode: str) -> torch.Tensor:  # @inspect rewards
    """
    Args:
        rewards (float[batch trial])
    Returns:
        deltas (float[batch trial]) which are advantage-like quantities for updating
    """
    if mode == "rewards":
        return rewards
    if mode == "centered_rewards":
        # Compute mean over all the responses (trial) for each prompt (batch)
        mean_rewards = rewards.mean(dim=-1, keepdim=True)  # @inspect mean_rewards
        centered_rewards = rewards - mean_rewards  # @inspect centered_rewards
        return centered_rewards
    if mode == "normalized_rewards":
        mean_rewards = rewards.mean(dim=-1, keepdim=True)  # @inspect mean_rewards
        std_rewards = rewards.std(dim=-1, keepdim=True)  # @inspect std_rewards
        centered_rewards = rewards - mean_rewards  # @inspect centered_rewards
        normalized_rewards = centered_rewards / (std_rewards + 1e-5)  # @inspect normalized_rewards
        return normalized_rewards
    if mode == "max_rewards":
        # Zero out any reward that isn't the maximum for each batch
        max_rewards = rewards.max(dim=-1, keepdim=True)[0]
        max_rewards = torch.where(rewards == max_rewards, rewards, torch.zeros_like(rewards))
        return max_rewards
    raise ValueError(f"Unknown mode: {mode}")

```


## 3. Computing the Loss

### Model Definition

```python
class Model(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, prompt_length: int, response_length: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # For each position, we have a matrix for encoding and a matrix for decoding
        self.encode_weights = nn.Parameter(torch.randn(prompt_length, embedding_dim, embedding_dim) / math.sqrt(embedding_dim))
        self.decode_weights = nn.Parameter(torch.randn(response_length, embedding_dim, embedding_dim) / math.sqrt(embedding_dim))
    def forward(self, prompts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prompts: int[batch pos]
        Returns:
            logits: float[batch pos vocab]
        """
        # Embed the prompts
        embeddings = self.embedding(prompts)   # [batch pos dim]
        # Transform using per prompt position matrix, collapse into one vector
        encoded = einsum(embeddings, self.encode_weights, "batch pos dim1, pos dim1 dim2 -> batch dim2")
        # Turn into one vector per response position
        decoded = einsum(encoded, self.decode_weights, "batch dim2, pos dim2 dim1 -> batch pos dim1")
        # Convert to logits (input and output share embeddings)
        logits = einsum(decoded, self.embedding.weight, "batch pos dim1, vocab dim1 -> batch pos vocab")
    return logits
```

### Log-Probability Computation

```python
def compute_log_probs(prompts: torch.Tensor, responses: torch.Tensor, model: Model) -> torch.Tensor:
    """
    Args:
        prompts (int[batch pos])
        responses (int[batch trial pos])
    Returns:
        log_probs (float[batch trial pos]) under the model
    """
    # Compute log prob of responses under model
    logits = model(prompts)  # [batch pos vocab]
    log_probs = F.log_softmax(logits, dim=-1)  # [batch pos vocab]
    # Replicate to align with responses
    num_responses = responses.shape[1]
    log_probs = repeat(log_probs, "batch pos vocab -> batch trial pos vocab", trial=num_responses)  # [batch trial pos vocab]
    # Index into log_probs using responses
    log_probs = log_probs.gather(dim=-1, index=responses.unsqueeze(-1)).squeeze(-1)  # [batch trial pos]
    return log_probs
```

### Loss Variants

```python
def compute_loss(log_probs: torch.Tensor, deltas: torch.Tensor, mode: str, old_log_probs: torch.Tensor | None = None) -> torch.Tensor:
    if mode == "naive":
        return -einsum(log_probs, deltas, "batch trial pos, batch trial -> batch trial pos").mean()
    if mode == "unclipped":
        ratios = log_probs / old_log_probs  # [batch trial]
        return -einsum(ratios, deltas, "batch trial pos, batch trial -> batch trial pos").mean()
    if mode == "clipped":
        epsilon = 0.01
        unclipped_ratios = log_probs / old_log_probs  # [batch trial]
        unclipped = einsum(unclipped_ratios, deltas, "batch trial pos, batch trial -> batch trial pos")
        clipped_ratios = torch.clamp(unclipped_ratios, min=1 - epsilon, max=1 + epsilon)
        clipped = einsum(clipped_ratios, deltas, "batch trial pos, batch trial -> batch trial pos")
        return -torch.minimum(unclipped, clipped).mean()
    raise ValueError(f"Unknown mode: {mode}")

```

The basic loss for policy optimization is computed as:

$$
\mathcal{L}_{\text{policy}} = - \sum \log \pi_\theta(a \mid s) \, \delta
$$

This encourages the model to increase log-probabilities of tokens with positive $\delta$ (above-average responses) and decrease them for negative $\delta$.


### 🧱 Clipped Loss (PPO / GRPO Variant)

To keep the new policy close to the old one, the **importance ratio** is introduced:

$$
r_t(\theta) = 
\frac{\pi_\theta(a_t \mid s_t)}
{\pi_{\text{old}}(a_t \mid s_t)}
$$

The final **clipped objective** is:

$$
\mathcal{L}_{\text{clip}}(\theta)
=
- \mathbb{E}_t
\Big[
\min
\big(
r_t(\theta) \, \delta_t,\;
\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \, \delta_t
\big)
\Big]
$$

Typical clipping range: $\epsilon \in [0.1, 0.2]$

**Important implementation detail:**  
When computing $r_t(\theta)$, the old policy $\pi_{\text{old}}$ must be **frozen** — you should not differentiate through it, or the gradient will collapse to zero.


### ⚖️ KL Penalty Regularization

```python
def compute_kl_penalty(log_probs: torch.Tensor, ref_log_probs: torch.Tensor) -> torch.Tensor:
    """
    Compute an estimate of KL(model | ref_model), where the models are given by:
        log_probs [batch trial pos vocab]
        ref_log_probs [batch trial pos vocab]
    Use the estimate:
        KL(p || q) = E_p[q/p - log(q/p) - 1]
    """
    return (torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1).sum(dim=-1).mean()
```

An optional **KL penalty** is often added to prevent excessive deviation from a reference model $\pi_{\text{ref}}$:

$$
\mathcal{L}_{\text{KL}} =
\lambda \cdot
D_{\text{KL}}(\pi_\theta \, \| \, \pi_{\text{ref}})
$$

where:

$$
D_{\text{KL}}(\pi_\theta \, \| \, \pi_{\text{ref}})
=
\sum_t
\pi_\theta(a_t \mid s_t)
\log
\frac{\pi_\theta(a_t \mid s_t)}
{\pi_{\text{ref}}(a_t \mid s_t)}
$$

This regularization term preserves the model’s **existing capabilities** while encouraging new behaviors.


## 4. The Full GRPO Algorithm and Infrastructure
![alt_text](/assets/images/llm-from-scratch/17/3.png "image_tooltip")

The GRPO training setup typically involves **nested optimization loops** and multiple concurrent model versions.

```python
def generate_responses(prompts: torch.Tensor, model: Model, num_responses: int) -> torch.Tensor:
    """
    Args:
        prompts (int[batch pos])
    Returns:
        generated responses: int[batch trial pos]
    Example (batch_size = 3, prompt_length = 3, num_responses = 2, response_length = 4)
    p1 p1 p1 r1 r1 r1 r1
             r2 r2 r2 r2
    p2 p2 p2 r3 r3 r3 r3
             r4 r4 r4 r4
    p3 p3 p3 r5 r5 r5 r5
             r6 r6 r6 r6
    """
    logits = model(prompts)  # [batch pos vocab]
    batch_size = prompts.shape[0]
    # Sample num_responses (independently) for each [batch pos]
    flattened_logits = rearrange(logits, "batch pos vocab -> (batch pos) vocab")
    flattened_responses = torch.multinomial(softmax(flattened_logits, dim=-1), num_samples=num_responses, replacement=True)  # [batch pos trial]
    responses = rearrange(flattened_responses, "(batch pos) trial -> batch trial pos", batch=batch_size)
    return responses

def simple_model():
  """
    Define a simple model that maps prompts to responses
    
    Assume fixed prompt and response length
        
    Captures positional information with separate per-position parameters
        
    Decode each position in the response independently (not autoregressive)
  """
      model = Model(vocab_size=3, embedding_dim=10, prompt_length=3, response_length=3)
      # Start with a prompt s
      prompts = torch.tensor([[1, 0, 2]])  # [batch pos]
      # Generate responses a
      torch.manual_seed(10)
      responses = generate_responses(prompts=prompts, model=model, num_responses=5)  # [batch trial pos]  @inspect responses
      # Compute rewards R of these responses:
      rewards = compute_reward(prompts=prompts, responses=responses, reward_fn=sort_inclusion_ordering_reward)  # [batch trial]  @inspect rewards
      # Compute deltas δ given the rewards R (for performing the updates)
      deltas = compute_deltas(rewards=rewards, mode="rewards")  # [batch trial]  @inspect deltas
      deltas = compute_deltas(rewards=rewards, mode="centered_rewards")  # [batch trial]  @inspect deltas
      deltas = compute_deltas(rewards=rewards, mode="normalized_rewards")  # [batch trial]  @inspect deltas
      deltas = compute_deltas(rewards=rewards, mode="max_rewards")  # [batch trial]  @inspect deltas
      # Compute log probabilities of these responses:
      log_probs = compute_log_probs(prompts=prompts, responses=responses, model=model)  # [batch trial]  @inspect log_probs
      # Compute loss so that we can use to update the model parameters
      loss = compute_loss(log_probs=log_probs, deltas=deltas, mode="naive")  # @inspect loss
      freezing_parameters()
      old_model = Model(vocab_size=3, embedding_dim=10, prompt_length=3, response_length=3)  # Pretend this is an old checkpoint @stepover
      old_log_probs = compute_log_probs(prompts=prompts, responses=responses, model=old_model)  # @stepover
      loss = compute_loss(log_probs=log_probs, deltas=deltas, mode="unclipped", old_log_probs=old_log_probs)  # @inspect loss
      loss = compute_loss(log_probs=log_probs, deltas=deltas, mode="clipped", old_log_probs=old_log_probs)  # @inspect loss
      # Sometimes, we can use an explicit KL penalty to regularize the model.
      # This can be useful if you want RL a new capability into a model, but you don't want it to forget its original capabilities.
      # KL(p || q) = E_{x ~ p}[log(p(x)/q(x))]
      # KL(p || q) = E_{x ~ p}[-log(q(x)/p(x))]
      # KL(p || q) = E_{x ~ p}[q(x)/p(x) - log(q(x)/p(x)) - 1] because E_{x ~ p}[q(x)/p(x)] = 1
      kl_penalty = compute_kl_penalty(log_probs=log_probs, ref_log_probs=old_log_probs)  # @inspect kl_penalty
      # Summary:
      # Generate responses    
      # - Compute rewards R and δ (rewards, centered rewards, normalized rewards, max rewards)   
      # - Compute log probs of responses   
      # - Compute loss from log probs and δ (naive, unclipped, clipped)
```


The **GRPO (Group Relative Policy Optimization)** algorithm organizes training into **two nested loops** to separate:
- **Expensive work** → *Inference / sampling rollouts*
- **Cheap work** → *Gradient updates on already-collected data*

This is the core idea behind PPO-style on-policy optimization.

```python
def run_policy_gradient(num_epochs: int = 100,
                        num_steps_per_epoch: int = 10,
                        compute_ref_model_period: int = 10,
                        num_responses: int = 10,
                        deltas_mode: str = "rewards",
                        loss_mode: str = "naive",
                        kl_penalty: float = 0.0,
                        reward_fn: Callable[[list[int], list[int]], float] = sort_inclusion_ordering_reward,
                        use_cache: bool = False) -> tuple[str, str]:
    """Train a model using policy gradient.
    Return:
    - Path to the image of the learning curve.
    - Path to the log file
    """
    torch.manual_seed(5)
    image_path = f"var/policy_gradient_{deltas_mode}_{loss_mode}.png"
    log_path = f"var/policy_gradient_{deltas_mode}_{loss_mode}.txt"
    # Already ran, just cache it
    if use_cache and os.path.exists(image_path) and os.path.exists(log_path):
        return image_path, log_path
    # Define the data
    prompts = torch.tensor([[1, 0, 2], [3, 2, 4], [1, 2, 3]])
    vocab_size = prompts.max() + 1
    prompt_length = response_length = prompts.shape[1]
    model = Model(vocab_size=vocab_size, embedding_dim=10, prompt_length=prompt_length, response_length=response_length)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    records = []
    ref_log_probs = None
    ref_model = None
    old_log_probs = None
    if use_cache:
        out = open(log_path, "w")
    else:
        out = sys.stdout
    for epoch in tqdm(range(num_epochs), desc="epoch"):
        # If using KL penalty, need to get the reference model (freeze it every few epochs)
        if kl_penalty != 0:
            if epoch % compute_ref_model_period == 0:
                ref_model = model.clone()
        # Sample responses and evaluate their rewards
        responses = generate_responses(prompts=prompts, model=model, num_responses=num_responses)  # [batch trial pos]
        rewards = compute_reward(prompts=prompts, responses=responses, reward_fn=reward_fn)  # [batch trial]
        deltas = compute_deltas(rewards=rewards, mode=deltas_mode)  # [batch trial]
        if kl_penalty != 0:  # Compute under the reference model
            with torch.no_grad():
                ref_log_probs = compute_log_probs(prompts=prompts, responses=responses, model=ref_model)  # [batch trial]
        if loss_mode != "naive":  # Compute under the current model (but freeze while we do the inner steps)
            with torch.no_grad():
                old_log_probs = compute_log_probs(prompts=prompts, responses=responses, model=model)  # [batch trial]
        # Take a number of steps given the responses
        for step in range(num_steps_per_epoch):
            log_probs = compute_log_probs(prompts=prompts, responses=responses, model=model)  # [batch trial]
            loss = compute_loss(log_probs=log_probs, deltas=deltas, mode=loss_mode, old_log_probs=old_log_probs)  # @inspect loss
            if kl_penalty != 0:
                loss += kl_penalty * compute_kl_penalty(log_probs=log_probs, ref_log_probs=ref_log_probs)
            # Print information
            print_information(epoch=epoch, step=step, loss=loss, prompts=prompts, rewards=rewards, responses=responses, log_probs=log_probs, deltas=deltas, out=out)
            global_step = epoch * num_steps_per_epoch + step
            records.append({"epoch": epoch, "step": global_step, "loss": loss.item(), "mean_reward": rewards.mean().item()})
            # Backprop and update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    if use_cache:
        out.close()
    if use_cache:
        # Plot step versus loss and reward in two subplots
        steps = [r["step"] for r in records]
        losses = [r["loss"] for r in records]
        rewards = [r["mean_reward"] for r in records]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        # Loss subplot
        ax1.plot(steps, losses)
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Train Loss")
        ax1.set_title("Train Loss")
        # Reward subplot
        ax2.plot(steps, rewards)
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Mean Reward")
        ax2.set_title("Mean Reward")
        plt.tight_layout()
        plt.savefig(image_path)
        plt.close()
    return image_path, log_path
def print_information(epoch: int, step: int, loss: torch.Tensor, prompts: torch.Tensor, rewards: torch.Tensor, responses: torch.Tensor, log_probs: torch.Tensor, deltas: torch.Tensor, out):
    print(f"epoch = {epoch}, step = {step}, loss = {loss:.3f}, reward = {rewards.mean():.3f}", file=out)
    if epoch % 1 == 0 and step % 5 == 0:
        for batch in range(prompts.shape[0]):
            print(f"  prompt = {prompts[batch, :]}", file=out)
            for trial in range(responses.shape[1]):
                print(f"    response = {responses[batch, trial, :]}, log_probs = {tstr(log_probs[batch, trial])}, reward = {rewards[batch, trial]}, delta = {deltas[batch, trial]:.3f}", file=out)
def tstr(x: torch.Tensor) -> str:
    return "[" + ", ".join(f"{x[i]:.3f}" for i in range(x.shape[0])) + "]"
```


### 4.1 Outer Loop (Epoch-Level Operations)

The **outer loop** defines how often new responses are sampled and when reference models are updated.

#### **A. Rollout / Sampling**
The current policy  $\pi$
is used to **generate responses (rollouts)** for a batch of prompts.

This step:
- Requires **inference** (expensive)
- Produces the raw data used for training

#### **B. Reward & Delta Computation**

For each sampled response, compute:
- Reward $R$
- Delta (advantage-like signal)  
  Common GRPO choice: **group-centered rewards**

$$
\delta_i = R_i - \mu_{\text{group}}
$$

or normalized rewards:

$$
\delta_i = \frac{R_i - \mu_{\text{group}}}{\sigma_{\text{group}} + \epsilon}
$$

These **reduce variance** without needing a critic.


#### **C. Reference Model Updates (Optional)**

If using KL regularization, the **reference model**  $\pi_{\text{ref}}$ may be:
- **Frozen**
- Or periodically updated (e.g., every 10 epochs)

#### **D. Cache Old Log-Probabilities**

Compute: 
$$
\log \pi_{\text{old}}(a \mid s)
$$

using `torch.no_grad()`, because:
- These log-probs must stay **fixed**
- Used for the ratio inside PPO/GRPO clipping

This avoids storing the full old model.


### 4.2 Inner Loop (Gradient Update Steps)

The **inner loop** performs many updates using the *same* sampled data.

#### **A. Compute Loss**

GRPO uses PPO-style clipped objectives:

Let:

$$
r = \exp\!\left( \log \pi(a \mid s) \;-\; \log \pi_{\text{old}}(a \mid s) \right)
$$

Then:

$$
L_{\text{GRPO}}
= -\,\min\!\Big(
    r \cdot \delta,\;
    \text{clip}\!\left(r,\; 1 - \epsilon,\; 1 + \epsilon\right)\cdot \delta
\Big)
$$

Clipping prevents large, unstable steps.

#### **B. KL Regularization (Optional)**

If using a reference model:

$$
\mathrm{KL}\!\left(\pi \;\|\; \pi_{\text{ref}}\right)
=
\sum_a 
\pi(a \mid s)\;
\Big[
    \log \pi(a \mid s)
    \;-\;
    \log \pi_{\text{ref}}(a \mid s)
\Big]
$$


Then:

$$
L = L_{\text{GRPO}} + \beta \cdot \text{KL}
$$

This keeps the model from drifting too far.


#### **C. Optimizer Step**

Standard PyTorch steps:
1. `optimizer.zero_grad()`
2. `loss.backward()`
3. `optimizer.step()`

This inner loop is cheap → so we run it many times.

## 5. Use and Update of π<sub>old</sub> and π<sub>ref</sub> in GRPO

![alt_text](/assets/images/llm-from-scratch/17/2.png "image_tooltip")

### 5.1 Use and Update of the Old Model (π<sub>old</sub>)

The **Old Model** (or old policy)  
$$\pi_{\text{old}}$$  
is essential for computing **importance ratios** in the clipped GRPO (and PPO) loss. It ensures that the updated policy does not shift too far from the policy that generated the sampled data.

#### **A. Usage**

During optimization, GRPO uses the log-prob ratio:

$$
r = \exp\!\left( \log \pi(a \mid s)\;-\;\log \pi_{\text{old}}(a \mid s) \right)
$$


This ratio is multiplied by the delta and fed into the clipped objective:

$$
L = -\min \big( r\delta,\; \text{clip}(r, 1-\epsilon, 1+\epsilon)\delta \big)
$$


#### **B. Updating / Freezing Behavior**

##### **Outer Loop (Data Collection)**

- Compute log-probs of sampled actions under the *current* policy:  
  $$\log\pi_{\text{old}}$$  
- Cache these values using `torch.no_grad()`.
- These cached values serve as **π<sub>old</sub>** for the entire inner loop.

##### **Inner Loop (Optimization)**

- π remains trainable.
- π<sub>old</sub> stays **frozen** as cached log-probs.
- Used for all gradient steps within the epoch.

##### **Conclusion**

You do **not** store a full copy of π<sub>old</sub>.  
Only the cached log-probs:

$$
\log \pi_{\text{old}}
$$

are needed.  
π<sub>old</sub> is “updated” implicitly **once per outer loop**.


### 5.2 Use and Update of the Reference Model (π<sub>ref</sub>)

The **Reference Model**  
$$\pi_{\text{ref}}$$  
is used **only when KL regularization** is applied:

$$
\lambda \cdot \mathrm{KL}(\pi \;\|\; \pi_{\text{ref}})
$$

This helps prevent catastrophic drift or capability loss.

#### **A. Usage**

Compute:

$$
\mathrm{KL}\!\left(\pi \;\|\; \pi_{\text{ref}}\right)
=
\sum_a
\pi(a \mid s)\;
\Big[
    \log \pi(a \mid s)
    \;-\;
    \log \pi_{\text{ref}}(a \mid s)
\Big]
$$

The KL penalty is added to the clipped objective.


#### **B. Updating / Freezing Behavior**
##### **Outer Loop (Reference Update)**

- π<sub>ref</sub> is **cloned** from the current model only when  
  $$
  \text{epoch} \bmod \text{compute\_ref\_model\_period} = 0
  $$
  e.g., every 10 epochs.
- Otherwise, π<sub>ref</sub> remains frozen.

##### **Outer Loop (Data Collection)**

- If π<sub>ref</sub> exists, compute & cache:  
$$
\log \pi_{\text{ref}}
$$
  over sampled actions (`torch.no_grad()`).

##### **Inner Loop (Optimization)**

- The KL penalty uses the **cached** logπ<sub>ref</sub>.
- π<sub>ref</sub> is completely frozen.

### 🔶 Summary: How the Loops Use Each Model

| Loop | Model | Action | Frequency |
|------|--------|---------|-----------|
| **Outer Loop** | π<sub>ref</sub> | Freeze/update by cloning π every *N* epochs | Infrequent (e.g., every 10) |
| **Outer Loop** | π<sub>old</sub> | Cache logπ of current model → becomes π<sub>old</sub> | Every epoch |
| **Inner Loop** | π<sub>old</sub> | Used for clipped ratio; stays constant | Fixed for all inner steps |
| **Inner Loop** | π<sub>ref</sub> | Provides logπ<sub>ref</sub> for KL penalty | Fixed for all inner steps |
| **Inner Loop** | π (current) | Updated by backprop & optimizer | Many times per epoch |


This structure is what allows GRPO to:
- Reuse expensive rollout data efficiently  
- Stabilize training  
- Control policy drift via clipping and KL penalties  


## 6. Required Infrastructure & Model Management

RL training is much more complex than pre-training because you must coordinate **multiple models** and manage **heavy inference workloads**.

### Managing Multiple Model Copies

| Model | Role | Purpose | Memory Cost |
|-------|------|---------|-------------|
| **Current Model** $$\pi$$ | Policy being trained | Generates responses, computes current log-probs | 1× model |
| **Old Log-Prob Cache** $$\pi_{\text{old}}$$ | Reference for clipping | Need only `log π_old`, not full model | Tiny |
| **Reference Model** $$\pi_{\text{ref}}$$ | KL anchor | Prevents capability drift | +1× model |
| **Critic (Not used in GRPO)** | Used in PPO | Estimates value baseline | Extra model |

GRPO removes the critic to simplify infra.

### Infrastructure Challenges

#### **Expensive Inference Workloads**

You must repeatedly run rollouts:

- Very expensive
- Often needs:
  - VLM inference clusters  
  - Parallel model workers  
  - Async rollout pipelines  

This becomes the bottleneck.

#### **Coordination Between Multiple Models**

Need to coordinate:
- Current model updates
- π_old snapshots (log-probs)
- π_ref freezing or updating

All must remain consistent **across machines and GPUs**.

#### **Distributed Complexity**

Everything must be:
- Parallel
- Synchronized
- Fault-tolerant

RL training involves:
- Distributed inference
- Distributed optimization
- Rollout queues
- Logging/monitoring pipelines

This is why RLHF/GRPO infrastructure is **significantly harder** than pre-training.
