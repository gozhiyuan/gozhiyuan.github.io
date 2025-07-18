---
layout: post
title: Robot Basic Pick and Place II - Differential kinematics
subtitle: Lecture 3 Basic Pick and Place (Pt. 1)
categories: Robotic
tags: [MIT-Robotic-Manipulation-2023]
banner: "/assets/images/banners/yuanpang-wa-valley.jpg"
---

# 🤖 Basic Pick and Place - Differential kinematics

[Course Link](https://manipulation.csail.mit.edu/Fall2023/index.html#description)

The fundamental goal is to move a "red brick from one bin to the second bin" using a robot. This requires:

- **Defining target poses**: Ideal gripper positions/orientations for picking and placing
- **Generating trajectories**: Compose keyframes into smooth paths
- **Robot control**: Convert desired gripper motion into joint commands

![alt_text](/assets/images/robotic-manipulation/04/1.png "image_tooltip")


## 1. 🏛️ Recap: Monogram Notation and Spatial Algebra

A robust notation helps prevent errors. "Monogram Notation" is used for clarity.

### • Points and Positions
- $(^A p^C_F)$: Position of point C, measured from A, expressed in frame F
- Shorthand:
  - $( ^F p^A \equiv {}^F p^A_F )$ (if measured & expressed in F)
  - $( p^A \equiv {}^W p^A_W )$ (if measured from world)

### • Frames
- **World frame (W)**: Global reference (x-forward, y-left, z-up)
- **Body frames (B\_i)**: Attached to each physical body

### • Rotations
- $(^B R^A)$: Orientation of frame A measured from B

### • Spatial Pose/Transform
- $(^B X^A)$: Pose of frame A from B
- Drake: `RigidTransform`, denoted by $(X)$
- $(X^O \Rightarrow {}^W X^O)$

### • Spatial Algebra Rules
- **Position Addition**: $(^A p^B_F + {}^B p^C_F = {}^A p^C_F)$
- **Change Expressed Frame**: $(^A p^B_G = {}^G R^F {}^A p^B_F)$
- **Rotation Composition**:
  - $(^A R^B \cdot {}^B R^C = {}^A R^C)$
  - $(({}^A R^B)^{-1} = {}^B R^A = {}^B R^{A^T})$
- **Transform Composition**:
  - $(^A X^B \cdot {}^B X^C = {}^A X^C)$
  - $(({}^A X^B)^{-1} = {}^B X^A)$

## 2. 🔄 Representations for 3D Rotations

### ◻ Rotation Matrices (3x3)
- Pros: Fast, GPU optimized
- Cons: 9 values, must maintain orthonormality

### ◻ Euler Angles (Roll-Pitch-Yaw)
- Pros: Intuitive, only 3 numbers
- Cons: Gimbal lock at $(\frac{\pi}{2})$ pitch

### ◻ Axis-Angle (Exponential Coordinates)
- Pros: 3 numbers, intuitive
- Cons: Undefined axis at 0 rotation

### ◻ Unit Quaternions (wxyz)
- Pros: 4 numbers, singularity-free, good for interpolation
- Cons: Less intuitive; $((w,x,y,z) \equiv -(w,x,y,z))$


## 3. 📉 Forward and Differential Kinematics

### ✨ Forward Kinematics (FK)
Mapping joint angles $((q))$ to gripper pose $((X^G))$:

$$
X^G = f_{\text{kin}}^G(q)
$$

- Drake uses `MultibodyPlant` for efficient FK.

### ✳ Kinematic Tree
- Robot bodies connected by joints removing DoF

### Differential Kinematics in Robotics

Differential kinematics is a crucial concept in robotics that describes the relationship between the velocities of a robot's joints and the resulting spatial velocity of its end-effector (or any other frame of interest). It's a fundamental step in converting a high-level task, like moving a gripper to a specific pose, into the low-level joint commands that a robot can execute.

- Forward kinematics: Given joint positions, where is the end-effector? Think: “If I bend my arm this way, where’s my hand?”
- Differential kinematics: Given joint velocities, how fast is the end-effector moving? Think: “If I move my elbow and shoulder at certain speeds, how is my hand moving in space?”

#### Core Concept and Relationship to Kinematics

1. **From Forward Kinematics**:  
   Differential kinematics is derived directly from forward kinematics. Forward kinematics is the function that maps the robot's joint angles ($q$) to the pose ($X$) of its end-effector in the world frame:
   
   $$
   X^G = f_{\text{kin}}^G(q)
   $$
   
   Differential kinematics then describes how changes in joint positions relate to changes in gripper pose. Mathematically:
   
   $$
   dX^B = \frac{\partial f_{\text{kin}}^B(q)}{\partial q} \, dq = J^B(q) \, dq
   $$

2. **The Jacobian Matrix**:  
   This partial derivative is known as the *kinematic Jacobian*, denoted as $J(q)$. It's a matrix that provides a linear relationship between generalized velocities (joint velocities) and spatial velocities.


#### Representations for Differential Rotations

While 3D rotations themselves can be represented in various ways (rotation matrices, Euler angles, unit quaternions), which often have singularities or over-parameterization, differential rotations have a canonical, singularity-free representation: **spatial velocity**.

**A: Spatial Velocity**:  
This is a six-component vector:

  $$
  {}^A V^B_C = \begin{bmatrix}
    {}^A \omega^B_C \\
    {}^A \text{v}^B_C
  \end{bmatrix}
  $$

comprising:

**B: Angular Velocity**: Angular velocity tells you how fast and around which axis something is rotating.
  Represented by three numbers (e.g., $w_x, w_y, w_z$). Its direction indicates the instantaneous axis of rotation, and its magnitude represents the rate of rotation around that axis.  
- Angular velocities are **unbounded** ($-\infty$ to $\infty$), avoiding "wrap-around" effects or singularities like Euler angles (e.g., gimbal lock).  
- This makes them efficient and sufficient for representing rotational derivatives everywhere.
- It is a **vector**:
- **Direction** = axis of rotation (right-hand rule)
- **Magnitude** = how fast it’s spinning (in radians per second)
- Units: Radians per second (rad/s)
- Angular velocity is like a **spin speed**.
- It doesn’t tell you how far a point moves, only how the **orientation is changing**.


Example:  
Imagine a robot wrist spinning like a screwdriver:

If it rotates once per second around the z-axis, its angular velocity is:

$$
\omega = 
\begin{bmatrix}
0 \\
0 \\
2\pi
\end{bmatrix}
\text{ rad/s}
$$

**Meaning**: rotating at 1 revolution/second about the z-axis.

  
**C: Translational Velocity**:  
Also a three-component vector representing linear velocity. Translational velocity tells you how fast a **point is moving** in space — i.e., **linear motion**.

It is also a **vector**:
- **Direction** = direction of movement
- **Magnitude** = speed
- Meters per second (m/s)

Example:
A robot gripper is moving straight forward (in the x-direction) at 10 cm/s:

$$
v = 
\begin{bmatrix}
0.1 \\
0 \\
0
\end{bmatrix}
\text{ m/s}
$$


**D: Generalized Velocities vs. Time Derivative of Positions**:  
A subtle point arises because **Drake** uses unit quaternions (4 numbers) to represent orientations within the configuration vector $q$ of floating bodies (e.g., a foam brick or floating hand) to avoid singularities. But for velocities, it uses the **3-component angular velocity**.

- For a free body:
  - $q$ (pose) has 7 components: 3 (position) + 4 (quaternion)
  - Generalized velocity $v$ has 6 components: 3 (translation) + 3 (angular velocity)
  - Therefore, in general:
  
    $$
    \dot{q} \neq v
    $$

- Drake provides:
  - `MapQDotToVelocity`
  - `MapVelocityToQDot`  
  to convert between these representations.


#### ⚙️ Algebraic Rules for Spatial Velocities

**A. 🔄 Changing "Expressed-In" Frames**:
Rotations can be used to convert spatial velocities from one "expressed-in" frame to another.

If you have:
- $( {}^A\text{v}^B_F )$: translational velocity expressed in frame **F**
- $( {}^A\omega^B_F )$: angular velocity expressed in frame **F**

Then to express them in frame **G**, use the rotation matrix $( {}^G R^F )$:

$$
{}^A\text{v}^B_G = {}^G R^F \, {}^A\text{v}^B_F
$$

$$
{}^A\omega^B_G = {}^G R^F \, {}^A\omega^B_F
$$


**B: ➕ Addition of Angular Velocities**  

Angular velocities can be **added** when they are expressed in the **same frame**.

Example:
If you have:
- $( {}^A\omega^B_F )$: angular velocity of B relative to A
- $( {}^B\omega^C_F )$: angular velocity of C relative to B  
Both expressed in frame **F**, then:

$$
{}^A\omega^B_F + {}^B\omega^C_F = {}^A\omega^C_F
$$

Angular velocities also have an **additive inverse**:

$$
{}^A\omega^C_F = -{}^C\omega^A_F
$$


**C: 🔗 Composition of Translational Velocities**

Composing translational velocities involves a **cross-product** term:

$$
{}^A\text{v}^C_F = {}^A\text{v}^B_F + {}^B\text{v}^C_F + {}^A\omega^B_F \times {}^B p^C_F
$$

This arises from **differentiating the position transform rule**.


**D: Additive Inverse for Translational Velocities**

Unlike angular velocities, translational velocity inverses are **not symmetric**:

$$
-{}^A\text{v}^B_F \neq {}^B\text{v}^A_F
$$

Instead, the correct relationship is:

$$
-{}^A\text{v}^B_F = {}^B\text{v}^A_F + {}^A\omega^B_F \times {}^B p^A_F
$$
 

#### Geometric vs. Analytic Jacobian

Due to these different representations, there are many possible kinematic Jacobians.

- **Geometric Jacobian** (used in Drake):  
  Outputs **spatial velocities**, e.g., via:
  ```
  CalcJacobianSpatialVelocity
  ```

- **Analytic Jacobian**:  
  A more literal partial derivative of forward kinematics, often used with specific rotation representations (Euler angles, etc.).




## 4. 🚀 Differential Inverse Kinematics (DIK)

### 🔝 Core Idea
From desired gripper velocity $(V^{G_d})$ to joint velocity $(v)$:

$$
v = [J^G(q)]^{-1} V^{G_d}
$$

### ✴ Jacobian Pseudo-Inverse $(J^+)$
The pseudo-inverse, often referred to as the **Moore-Penrose pseudo-inverse**, is a powerful mathematical concept used to "invert" matrices that are **not square** or **not full-rank**. It provides a robust way to solve linear equations when a traditional inverse does not exist. 
- Used when $(J)$ not square (e.g., 6x7)
- Moore-Penrose inverse returns:
  - Least-norm solution (if redundant)
  - Best-fit (if no exact solution)

#### 🤖 Core Idea and Application in Robotics

- In robotics, the pseudo-inverse is primarily used in **Differential Inverse Kinematics (DIK)**.
- DIK relates **desired changes in the end-effector's pose** (spatial velocity) to **required joint velocities**:

$$
V^G = J^G(q) v
$$

Where:
- $( V^G )$: desired spatial velocity of the gripper
- $( J^G(q) )$: Jacobian matrix
- $( v )$: generalized joint velocity

#### Non-square Jacobian Case:
For a robot like the **KUKA iiwa** (7-DOF), the Jacobian is:

$$
J^G(q) \in \mathbb{R}^{6 \times 7}
$$

It’s not square → no traditional inverse exists.

**Solution**: Use the pseudo-inverse:

$$
v = [J^G(q)]^+ V^{G_d}
$$

#### ✅ Square and Full-Rank:
- If $( J )$ is square and full-rank → pseudo-inverse equals the regular inverse.

#### 🔁 Redundant Robots (More DOF than Tasks):
- If $( \text{cols}(J) > \text{rows}(J) )$ (e.g., 7-DOF robot):
  - Infinite solutions for $( v )$
  - Pseudo-inverse returns the **minimum-norm solution**:
  
    > The one with the smallest joint velocity magnitude.

#### 🚫 No Exact Solution (Underactuated or Infeasible):
- If $( V^{G_d} )$ cannot be exactly achieved:
  - Pseudo-inverse gives the **best-effort** solution in a least-squares sense.

### ⚠️ Invertibility & Kinematic Singularities

- Pseudo-inverse works **only if Jacobian has full row rank**.
- Danger zone: **Near-singular configurations**, where the **smallest singular value** of $( J )$ approaches zero.

#### Why it matters:
- $( \| J^+ \| \rightarrow \infty )$
- This causes **huge velocity commands**, risking:
  - Instability
  - Joint saturation
  - Robot faults

#### Note:
- A **kinematic singularity** is a property of the math — not a physical robot failure.
- Robots can **pass through** singularities in **joint space**, but **task-space control** may break down.


#### 🚫 What Basic Pseudo-Inverse Ignores:
- Joint limits
- Velocity/acceleration/torque constraints
- Integrability (closed-loop consistency)

#### ✅ Reframing as Optimization:

Pseudo-inverse solves:

$$
\min_v \left\| J^G(q) v - V^{G_d} \right\|_2^2
$$

This perspective enables generalization:

#### 🔒 Add Constraints:
- Formulate as **Quadratic Program (QP)**:
  - Add joint velocity, position, acceleration, or torque constraints
  - Solve numerically

#### 🎯 Add Secondary Objectives:
- For example: drive joints toward nominal pose (joint-centering)
- Achieve this by **null space projection**:

> Secondary motion only occurs if it doesn’t interfere with the main task.

#### 🔄 Path Consistency:
- Basic pseudo-inverse doesn't guarantee returning to same joint state after a closed path.
- Advanced controllers (e.g., in **Drake**) enforce:
  - **Same directionality** in motion
  - **Constraint-aware damping**
  - **Component-wise Cartesian saturation**

These enhancements improve safety, stability, and operator usability.



### ⚠ Kinematic Singularities
- When $(\text{rank}(J) < 6)$
- Cause large velocity commands
- Not a failure of robot, but of the $(X \rightarrow q)$ mapping

### ✪ Redundancy Helps
- 7-DOF arms (like iiwa) can avoid singularities better than 6-DOF


## 5. FK, DK, IFK, IDK Summary

### ✅ 1) Forward Kinematics (FK)

#### What is it?
Forward Kinematics computes the **position and orientation (pose)** of the robot's end-effector (gripper/tool) **given joint values**.

#### Formula:
$[
\mathbf{x} = f(q)
]$
- $( q )$: joint angles or displacements
- $( \mathbf{x} )$: end-effector pose (position + orientation)

#### How it works:
- Use a chain of **transformation matrices** from the base to the end-effector.
- These matrices are defined using the **robot's geometry**, typically from Denavit-Hartenberg (DH) parameters.
- For most robots, this is a **closed-form** computation (no iteration).

#### Example:
A 2-joint planar robot arm:
- Link 1 length = 1 m
- Link 2 length = 1 m
- Joint angles: $( q_1 = 45^\circ )$, $( q_2 = 45^\circ )$

Then:
$[
x = \cos(q_1) + \cos(q_1 + q_2) \\
y = \sin(q_1) + \sin(q_1 + q_2)
]$

So:
$[
x \approx 1.41,\quad y \approx 1.41
]$


### 🌀 2) Differential Kinematics

#### What is it?
Differential kinematics relates **joint velocities** $( \dot{q} )$ to **end-effector velocities** (linear and angular) $( \dot{x} )$.

#### Formula:
$[
\dot{x} = J(q) \dot{q}
]$

- $( J(q) )$: Jacobian matrix, derived from FK equations
- $( \dot{x} )$: [linear velocity; angular velocity] in Cartesian space

### Why it matters:
It tells you:
- How fast the gripper is moving
- In what direction
- Based on joint motor speeds

### Example:
If your robot gripper is moving forward at 10 cm/s and rotating at 30°/s, then:
$[
\dot{x} =
\begin{bmatrix}
0.1 \\
0 \\
0 \\
0 \\
0 \\
0.52
\end{bmatrix}
\quad \text{(m/s and rad/s)}
]$

### 🔁 3) Inverse Kinematics (IK)

#### What is it?
Given a **desired end-effector pose**, solve for the **joint angles** that achieve it.

#### Formula:
$[
q = f^{-1}(x)
]$

#### Challenges:
- Often **no closed-form** solution (e.g., for complex 6-DOF arms)
- May have **multiple solutions** or **no solution**
- Requires **iterative methods** like:
  - Newton-Raphson
  - Optimization-based solvers (e.g., IKFast, MoveIt!)

#### Example:
You want your robot hand at position $( x = (1.5, 0.5) )$. What should the joint angles be?

Use an iterative solver to find $( q_1, q_2 )$ such that FK matches the desired pose.


### ♻️ 4) Inverse Differential Kinematics

#### What is it?
Given a **desired end-effector velocity** $( \dot{x} )$, compute the **joint velocities** $( \dot{q} )$ needed to achieve it.

#### Formula (Ideal Case):
$[
\dot{q} = J^{-1}(q) \dot{x}
]$

#### Problem:
- Jacobian $( J )$ might not be **square** (redundant or under-actuated robots)
- Jacobian might be **singular** (robot in a bad configuration)

#### Solution: Use **Pseudo-Inverse**:
$[
\dot{q} = J^+ \dot{x}
]$
- $( J^+ )$: Moore-Penrose pseudo-inverse
- Handles singularities and redundancy
- Provides the **least-squares** optimal joint motion

#### Example:
If you want your robot's end-effector to move 5 cm/s forward:
$[
\dot{x} =
\begin{bmatrix}
0.05 \\
0 \\
0 \\
0 \\
0 \\
0
\end{bmatrix}
\Rightarrow \dot{q} = J^+ \dot{x}
]$
This gives joint velocities $( \dot{q}_1, \dot{q}_2, ..., \dot{q}_n )$ that will make the gripper move as desired.


### 💡 Summary Table

| Concept                      | Input            | Output         | Method               | Notes                         |
|-----------------------------|------------------|----------------|----------------------|-------------------------------|
| Forward Kinematics          | $( q )$          | Pose $( x )$   | Matrix multiplication | Closed-form, geometric model |
| Differential Kinematics     | $( \dot{q} )$    | $( \dot{x} )$  | Jacobian $( J )$     | Linear velocity relationship |
| Inverse Kinematics          | Pose $( x )$     | $( q )$        | Iterative solver     | May not be unique/solvable   |
| Inverse Differential Kinematics | $( \dot{x} )$ | $( \dot{q} )$  | Pseudo-inverse $( J^+ )$ | Best effort solution       |


- FK gives you the **pose** of the robot's hand.
- Differential kinematics gives you the **speed** and **direction** of the robot's hand from joint speeds.
- IK finds **joint angles** to reach a target point — hard if no solution or too many!
- Inverse differential kinematics figures out **how to move** joints to follow a motion path.


## 6. 🤺 Grasp and Pre-Grasp Poses

### 👊 Grasp Pose:

$$
{}^O X^{G_{grasp}} = \text{desired gripper pose relative to object}
$$

**Example**:

$$
{}^{G_{grasp}} R^O = \text{MakeXRotation}(\frac{\pi}{2}) \cdot \text{MakeZRotation}(\frac{\pi}{2})
$$

### 🚕 Pre-Grasp Pose:

$$
{}^O X^{G_{pregrasp}} = \text{same orientation, offset along z-axis}
$$


## ⏳ 7. Trajectory Generation

### 🔹 Keyframe Composition:

- initial, pre-pick, pick, post-pick, clearance, pre-place, place, post-place

### ↔️ Interpolation Techniques
- **Position**: Linear via `PiecewisePolynomial`
- **Rotation**: `PiecewiseQuaternionSlerp`
- **Combining**: `PiecewisePose`
- Can differentiate to obtain spatial velocity for DIK


