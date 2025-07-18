---
layout: post
title: Robot Basic Pick and Place III - Differential kinematics via optimization 
subtitle: Lecture 3 Basic Pick and Place (Pt. 1)
categories: Robotic
tags: [MIT-Robotic-Manipulation-2023]
banner: "/assets/images/banners/yuanpang-wa-valley.jpg"
---

# 🤖 Basic Pick and Place - Differential kinematics via optimization 

[Course Link](https://manipulation.csail.mit.edu/Fall2023/index.html#description)


## Optimization-Based Differential Inverse Kinematics (DIK-QP)

### Problem with Pseudo-Inverse
- **Limitation**: Does not handle joint limits or other real-world constraints.
- **Consequence**: Can lead to *clipped velocities* and *off-course end-effector* trajectories.

### Pseudo-Inverse as Optimization
- Equivalent to solving an unconstrained least-squares problem:
  
  $$
  \min_v \left|J^G(q)v - V^{G_d}\right|^2_2
  $$

### Adding Constraints: Framing as Optimization
- **Velocity Constraints**:
  $$
  \min_v \left|J^G(q)v - V^{G_d}\right|^2_2 \quad \text{subject to } v_{min} \le v \le v_{max}
  $$
  - A *convex Quadratic Programming (QP)* problem.

- **Position and Acceleration Constraints**:
  Using Euler approximation with time step $( h )$:
  $$
  q_{min} \le q + h v_n \le q_{max}
  $$
  $$
  \dot{v}_{min} \le \frac{v_n - v}{h} \le \dot{v}_{max}
  $$

- **Force/Torque Constraints**:
  Incorporated using *manipulator dynamics*, typically *linearized* with known $( q )$, $( v )$.

### Benefits of QP-based DIK
- More robust.
- Avoids extreme commands.
- Explicitly respects robot limits.
- Superior to clipping after solving the unconstrained problem.

### Robust Optimization Formulation

- **QP Formulation**:
  $$
  \min_v \left|J^G(q)v - V^{G_d}\right|^2_2 \quad \text{subject to } v_{min} \le v \le v_{max}
  $$

- **Joint Limits via Euler Approximation**:
  $$
  q_{min} \le q + h v_n \le q_{max}
  $$
  $$
  \dot{v}_{min} \le \frac{v_n - v}{h} \le \dot{v}_{max}
  $$

- **Force/Torque and Collision Constraints**:
  - Can also be added via linearization.


## Role of Eigenvalues and SVD in Optimization

### 1. Jacobian Matrix $( J )$ and Its Role

The **Jacobian matrix** $( J )$ relates **joint velocities** $( \mathbf{v} )$ to the **end-effector velocity** $( \mathbf{V} )$:

$$
\mathbf{V} = J \mathbf{v}
$$

To compute the joint velocities needed to achieve a desired end-effector velocity $( \mathbf{V}_d )$:

$$
\mathbf{v} = J^+ \mathbf{V}_d
$$

where $( J^+ )$ is the **pseudo-inverse** of the Jacobian matrix.


### 2. What Happens Near Singularities?

- A **singularity** occurs when $( J )$ **loses rank**, meaning it compresses some directions in space to zero.
- Mathematically: **one or more singular values of $( J )$ approach zero.**


### 3. What Are Singular Values and Eigenvalues Here?

- The **singular values** $( \sigma_1, \sigma_2, ..., \sigma_r )$ of $( J )$ measure how much $( J )$ stretches/shrinks space along different directions.
- These are computed via **Singular Value Decomposition (SVD)**:

$$
J = U \Sigma V^T, \quad \text{where } \Sigma = \text{diag}(\sigma_1, \sigma_2, \dots)
$$

- The **eigenvalues** of $( J^T J )$ are $( \lambda_i = \sigma_i^2 )$, so:

$$
J^T J \mathbf{v} = \lambda \mathbf{v}
$$

### 4. 🧭 Intuition: Singular Values as "Stretching Factors"

- Think of $( J )$ mapping a **unit sphere** in joint velocity space to an **ellipsoid** in end-effector velocity space.
- Each **singular value** $( \sigma_i )$ is the **length of an ellipsoid axis**.
- If $( \sigma_i )$ is small → the ellipsoid is **flattened** → **compressed direction**.


### 5. Why Do Small Singular Values Cause Large Velocities?

- If desired velocity $( \mathbf{V}_d )$ has a component along a compressed direction:
  - Then $( \mathbf{v} = J^+ \mathbf{V}_d )$ must be **very large**.
- Recall:

$$
\mathbf{v} = V \Sigma^+ U^T \mathbf{V}_d
$$

- Where $( \Sigma^+ )$ contains $( \frac{1}{\sigma_i} )$ terms.
  - As $( \sigma_i \to 0 )$, $( \frac{1}{\sigma_i} \to \infty )$
  - ⇒ **Joint velocity blows up**

### 6. 🧪 Example: Simplified 2D Case

Suppose the Jacobian is:

$$
J = \begin{bmatrix}
1 & 0 \\
0 & \epsilon
\end{bmatrix}, \quad \text{where } \epsilon \ll 1
$$

Desired end-effector velocity:

$$
\mathbf{V}_d = \begin{bmatrix} 0 \\ 1 \end{bmatrix}
$$

Pseudo-inverse:

$$
J^+ = \begin{bmatrix}
1 & 0 \\
0 & \frac{1}{\epsilon}
\end{bmatrix}
$$

Then required joint velocity:

$$
\mathbf{v} = J^+ \mathbf{V}_d = \begin{bmatrix}
0 \\
\frac{1}{\epsilon}
\end{bmatrix}
$$

As $( \epsilon \to 0 )$, $( v_2 \to \infty )$. ⚠️ **Not feasible for real robots!**

### 7. 🔐 How Constraints Help Limit Large Velocities

Real robots have physical limitations:

- Max/min joint velocities
- Torque limits
- Acceleration limits

So we solve an **optimization problem**:

$$
\min_{\mathbf{v}} \| J \mathbf{v} - \mathbf{V}_d \|^2 \quad \text{subject to } \mathbf{v}_{\text{min}} \leq \mathbf{v} \leq \mathbf{v}_{\text{max}}
$$

- This gives a **feasible joint velocity** $( \mathbf{v} )$ that **avoids singularity-induced explosion**.
- May **sacrifice tracking accuracy**, but ensures **safety**.


## Task Prioritization

### 🔍 What is it?
In robotics, multiple tasks may need to be accomplished **at the same time** — for example:
- Moving the end-effector to a target point
- Keeping the robot's elbow away from obstacles
- Maintaining balance or posture

Some tasks are more **important** than others. **Task prioritization** is a strategy where **high-priority tasks are strictly enforced**, and **lower-priority tasks are optimized within the leftover degrees of freedom**.

### ⚙️ How It Works (Null Space Projection)
A **null space** of a matrix $( J )$ is the set of all vectors $( v )$ such that:

$$
Jv = 0
$$

This means: Moving along vectors in the **null space** does **not affect** the result of $( Jv )$. In robotics, $( J )$ is often the **Jacobian matrix**, which maps **joint velocities** to **end-effector velocities**.

So, joint movements in the null space of $( J )$ **don't move the end-effector** at all!


Let:
- $( J_1 )$: Jacobian of the high-priority task
- $( v_1 )$: joint velocity solving the high-priority task
- $( N_1 = I - J_1^+ J_1 )$: null space projector of $( J_1 )$

Then, for a **secondary task** with Jacobian $( J_2 )$ and desired velocity $( V_2 )$, the overall velocity command becomes:

$$
v = v_1 + J_2^+ (V_2 - J_2 v_1)
$$

Or, in **null-space formulation**:

$$
v = J_1^+ V_1 + (I - J_1^+ J_1) J_2^+ V_2
$$

This ensures:
- $( v )$ satisfies the **primary task**
- The secondary task is **projected into the null space** of the primary task — so it **does not interfere** with the first task

### 🧪 Example
- Task 1 (high priority): Move gripper to a target
- Task 2 (low priority): Keep elbow close to a "natural" position

Even if the elbow task can't be fully satisfied, the robot **won’t fail the main job**.


## Joint Centering (Secondary Task)

### 🤔 Why?
Robots often have **joint limits**. If joints get too close to these limits, motion becomes:
- Unstable
- Hard to reverse
- Risky for hardware

So we want joints to **stay near the center** of their range if possible.

### 🎯 Objective
Define a secondary task:

$$
q_{\text{desired}} = \frac{q_{\text{min}} + q_{\text{max}}}{2}
$$

Define an artificial velocity task:

$$
v_2 = -k (q - q_{\text{desired}})
$$

This "pulls" the joint back toward its center.

### 🧠 Implementation (in null space of primary task):
Use the null-space projection:

$$
v = v_1 + N_1 v_2
$$

Where:
- $( v_1 = J_1^+ V_1 )$: primary task velocity
- $( v_2 = -k (q - q_{\text{desired}}) )$: secondary joint-centering velocity
- $( N_1 = I - J_1^+ J_1 )$: null space projector

This way, **joint centering only happens if it doesn't interfere** with the high-priority task.


### ✅ Summary

| Concept               | Purpose                                          | Priority Level | Implementation                    |
|----------------------|--------------------------------------------------|----------------|------------------------------------|
| Task Prioritization  | Respect task importance order                    | Multiple       | Null-space projection              |
| Joint Centering       | Avoid joint limits, improve manipulability      | Usually lower  | Secondary task in null-space      |


### 📌 Real-World Use
- **Humanoids**: Balance is high-priority; posture and gestures are lower.
- **Manipulators**: End-effector motion is primary; joint centering is secondary.
- **Mobile robots**: Obstacle avoidance might override goal reaching when in conflict.



## Alternative Formulation: Drake's Default DIK

### Pose Tracking
- Convert desired pose $( X^{G_d} )$ into spatial velocity:
  $$
  V^{G_d} = \frac{1}{h}(X^G X^{G_d^{-1}})
  $$

### Linear Program (LP) Formulation
- Drake's DIK implementation (e.g., `addDiffIK`):
  $$
  \max_{v_n, \alpha} \alpha
  $$
  $$
  \text{subject to } J^G(q)v_n = \alpha V^{G_d}, \quad 0 \le \alpha \le 1
  $$

- **Properties**:
  - Robot moves in correct direction.
  - Slows down (via $( \alpha < 1 )$) if constraints are hit.
  - If infeasible: robot stops ($( \alpha = 0 )$).

- **Relaxed Formulation** (Operator Intuition):
  - Scale individual Cartesian components (x, y, z, roll, pitch, yaw).


## Drake's Mathematical Program Solver

- **MathematicalProgram class**:
  - Define decision variables.
  - Add constraints: `AddConstraint()`.
  - Add objectives: `AddCost()`.

- **Efficiency**:
  - Solves QP or LP nearly instantly.
  - Suitable for real-time (100–500 Hz control loops).

