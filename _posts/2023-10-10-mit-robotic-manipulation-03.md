---
layout: post
title: Robot Basic Pick and Place I
subtitle: Lecture 3 Basic Pick and Place (Pt. 1)
categories: MIT-Robotic-Manipulation-2023
tags: [robot]
banner: "/assets/images/banners/yuanpang-wa-valley.jpg"
---

## 🤖 Basic Pick and Place - Kinematics and Spatial Algebra 

The lecture covers foundational techniques for robot manipulation, especially pick-and-place tasks, with a focus on kinematics and spatial algebra.

[Course Link](https://manipulation.csail.mit.edu/Fall2023/index.html#description)


## I. Introduction to Robotic Manipulation and the Pick and Place Problem

A core challenge in robotic manipulation is commanding a robot to move an object (e.g., a red brick) from an initial to a goal pose. This "pick and place" task requires geometric, kinematic, and control insights.

📌 **Assumptions**: Perception is deferred — the system assumes perfect knowledge of the object's pose.

### The Recipe:
1. **Kinematic Frames and Spatial Algebra**  
2. **Gripper Frame Plan**  
3. **Forward Kinematics**  
4. **Differential Inverse Kinematics**


## II. Monogram Notation and Spatial Algebra: The Language of Robotics

### A. Key Concepts and Notation

- **Point**: $( p^A )$ = position of point A  
- **Relative Position**:  
  $( {}^A p^C )$ = position of point C measured from A  
- **Expressed-in Frame**:  
  $( {}^A p^C_F )$ = position of C from A, expressed in frame F  

  > "Expressed-in frame is an implementation detail" (Ch. 3)

- **World Frame** $( W )$:  
  Uses vehicle coordinates (x-forward, y-left, z-up)

- **Body Frame** $( B_i )$:  
  Attached to each body in simulation

#### Shorthands:
- $( {}^F p^A \equiv {}^F p^A_F )$  
- $( p^A \equiv {}^W p^A_W )$

- **Rotation**:  
  $( {}^B R^A )$ = orientation of frame A measured from B

- **Pose**:  
  $( {}^B X^A )$ = pose of frame A measured from B  
  > "Pose is a noun, transform is a verb" — they are equivalent.

#### Code Notation:
- $( {}^B p^A_C \rightarrow \texttt{p\_BA\_C} )$  
- $( {}^B R^A \rightarrow \texttt{R\_BA} )$  
- $( {}^B X^A \rightarrow \texttt{X\_BA} )$

### B. Rules of Spatial Algebra

- **Position Addition**:

  $$
  {}^A p^B_F + {}^B p^C_F = {}^A p^C_F
  $$

  $$
  {}^A p^B_F = - {}^B p^A_F
  $$

- **Frame Change**:

  $$
  {}^A p^B_G = {}^G R^F \cdot {}^A p^B_F
  $$

- **Rotation Composition**:

  $$
  {}^A R^B \cdot {}^B R^C = {}^A R^C
  $$

  $$
  \left[ {}^A R^B \right]^{-1} = {}^B R^A = \left( {}^A R^B \right)^T
  $$

- **Transform Composition**:

  $$
  {}^A X^B \cdot {}^B X^C = {}^A X^C
  $$

  $$
  \left[ {}^A X^B \right]^{-1} = {}^B X^A
  $$


### C. Representations for 3D Rotation

- **Rotation Matrices**: 3×3, orthonormal
- **Euler Angles (RPY)**: Order matters, susceptible to gimbal lock
- **Axis-Angle**: Rotation vector
- **Unit Quaternions**:  
  Represented by \( (w, x, y, z) \), with:

  $$
  w^2 + x^2 + y^2 + z^2 = 1
  $$

## III. Gripper Frame Plan "Sketch" and Trajectories

### Keyframes:
Sequence of gripper poses:
- Initial
- Pre-pick
- Pick
- Post-pick
- Pre-place
- Place
- Post-place

### Pre-Grasp Pose:
Safely positions gripper above object for simple vertical motion.

### Grasp Specification:
Pose of object relative to gripper during grasp.

Example:

### 🎯 Grasp Pose Definition in Pick and Place Tasks

When determining the pose of an object relative to the gripper during a grasp, the object frame is aligned with the gripper frame through a specific combination of **position** and **orientation** to ensure a successful and stable grasp. This combined position and orientation is referred to as a **spatial pose** or **transform**.

For the *pick and place* task involving a red foam brick and a gripper, the desired pose of the object in the gripper frame — denoted as $( {}^{G_{\text{grasp}}}X^O )$ — is explicitly defined. This definition includes both translational and rotational components:

![alt_text](/assets/images/robotic-manipulation/03/1.png "image_tooltip")


#### 📍 Position

The desired position (in meters) of the object in the gripper frame is:

$$
{}^{G_{\text{grasp}}}p^O = \begin{bmatrix} 0 \\ 0.11 \\ 0 \end{bmatrix}
$$

This value represents the "happy place for the center of mass of the brick" within the gripper's coordinate system.


#### 🔄 Orientation (Rotation)

The desired orientation of the object relative to the gripper, denoted as $( {}^{G_{\text{grasp}}}R^O )$, is achieved by applying two successive elementary rotations:

```python
MakeXRotation(π/2) @ MakeZRotation(π/2)
```

This rotation scheme ensures:

- The **positive z-axis** of the object aligns with the **negative y-axis** of the gripper frame.
- The **positive x-axis** of the object aligns with the **positive z-axis** of the gripper frame.

The rotation uses the **right-hand rule** to determine the direction and result of each axis transformation.


#### ✋ Pre-Grasp Pose

The **pre-grasp pose** — an intermediate step used to approach the object without collision — is defined to have the **same orientation** as the grasp pose. This



```python
f MakeGripperFrames(X_WG, X_WO):
    """
    Takes a partial specification with X_G["initial"] and X_O["initial"] and
    X_0["goal"], and returns a X_G and times with all of the pick and place
    frames populated.
    """
    # Define (again) the gripper pose relative to the object when in grasp.
    p_GgraspO = [0, 0.11, 0]
    R_GgraspO = RotationMatrix.MakeXRotation(
    np.pi / 2.0
    ) @ RotationMatrix.MakeZRotation(np.pi / 2.0)
    X_GgraspO = RigidTransform(R_GgraspO, p_GgraspO)
```

### Trajectory Generation

- **Timing**:  
  Conservative speeds (e.g., 10 cm/s)

- **Position Interpolation**:  
  Linear interpolation using `PiecewisePolynomial`

- **Rotation Interpolation**:  
  Use **SLERP** (spherical linear interpolation), via `PiecewiseQuaternionSlerp`

- **Pose Trajectory**:  
  Combined in `PiecewisePose`

- **Gripper Command Trajectory**:  
  Controls open/close timing


#### 📐 Why Rotations Cannot Be Linearly Interpolated

When dealing with trajectory rotations, it is crucial to understand that you **cannot simply average rotation matrices or Euler angles** to interpolate between orientations. This is because rotations do not behave linearly in the same way as positions.

- **Rotations Live on a Manifold**:  
  Unlike positions, which exist in Euclidean space (ℝ³), 3D rotations reside on a mathematical manifold called the **Special Orthogonal Group**, denoted as $SO(3)$. In 2D, this is analogous to the **unit circle** $S^1$.

  - If you take two rotation matrices and average them component-wise, the result is **not guaranteed to be orthonormal**, which is a requirement for a valid rotation matrix.
  - **Example in 2D**:  
    Consider two 2D rotations:
    - One at 0 degrees,
    - One at 90 degrees ($\pi/2$ radians).  
    A straight average will point **through the circle**, not along it, resulting in an **invalid** or incorrect rotation.

- **Singularities in Representations**:  
  Euler angles (e.g., roll-pitch-yaw) are a common way to represent 3D rotations, but they suffer from **singularities**.
  - At certain angles (like pitch = $\pi/2$), different combinations of roll and yaw can produce the same orientation.
  - This causes **gimbal lock**, a loss of one degree of freedom and potentially **ambiguous interpolations**.


#### ✅ The Solution: Spherical Linear Interpolation (Slerp)

- **Slerp** stands for **Spherical Linear Interpolation**. It operates on **unit quaternions**, which represent 3D rotations in a smooth and non-singular way.

- **Why Quaternions?**
  - Quaternions are 4D vectors $(w, x, y, z)$ satisfying:
    $$
    w^2 + x^2 + y^2 + z^2 = 1
    $$
  - They **do not suffer from gimbal lock**, unlike Euler angles.
  - However, there is a **"double covering"** issue: both $q$ and $-q$ represent the same rotation.

- **How Slerp Works**:
  - Given two unit quaternions $q_0$ and $q_1$, slerp interpolates **along the shortest arc** on the unit 4D sphere (hypersphere).
  - It linearly interpolates **rotation angle**, not quaternion components.
  - This keeps the interpolation **on the manifold of valid rotations**.

- **In Drake**:  
  Drake provides a class called `PiecewiseQuaternionSlerp` to handle this exact case:
  - You define keyframes as quaternions at specific times.
  - The class generates a **smooth interpolation** between them.


#### 🔄 Position + Orientation: Pose Interpolation

For interpolating full poses (position + orientation), Drake offers:

- `PiecewisePose`: Combines both:
  - **Linear interpolation** for position (via `PiecewisePolynomial`)
  - **Slerp** for orientation (via `PiecewiseQuaternionSlerp`)

This approach ensures **smooth**, **valid**, and **physically consistent** motion of the robot end-effector across keyframes in pick-and-place or other manipulation tasks.


## IV. Kinematics: Mapping Joint Angles to End-Effector Pose

### A. Forward Kinematics

Defines mapping:

$$
X^G = f_{\text{kin}}^G(q)
$$

- $( q )$: joint angles (configuration)  
- $( X^G )$: gripper pose

### MultibodyPlant Structure:

- Tree of bodies and joints
- Each joint defines a transform dependent on joint angle

> "Joints are constraints" — they reduce DoF added by bodies

- **Revolute Joint**:  
  Removes 5 DoF, leaves 1 rotational DoF

### B. The Kinematic Tree in Robotics

The **kinematic tree** is a fundamental concept in robotics, especially for multi-body systems. It organizes all the bodies in a simulated world into a tree topology for efficient computations.

#### Purpose and Structure

- The **MultibodyPlant** in Drake organizes all bodies in the world into this tree topology.
- Every body in the system, except for the **world body**, has a **parent body**.
- Bodies connect to their parents via either a **Joint** or a **floating base**.
- The **world frame** ($W$) serves as the canonical root for the tree—a fixed reference point in space.
- Every body in the physics engine has a unique frame attached to it called a **body frame**.


#### Joints and Degrees of Freedom

- **Joints** are transforms that depend on the joint angle $q$ and define how to go from one link’s coordinate frame to the next.  
  For example, a **revolute joint** defines a pure rotation between frames depending on $q$.
- **Important clarification:**  
  Adding a joint **does not** add degrees of freedom (DoF).  
  Instead:
  - Adding a body typically **adds many DoF** (e.g., 6 for a free-floating body).
  - Joints act as **constraints** that remove DoF by specifying fixed or limited motion relationships between bodies.
  - For example, welding a robot’s base to the world frame removes all floating DoF.
- Robot description files (URDF, SDF) specify different joint types (revolute, prismatic, planar) with corresponding mathematical transforms as functions of $q$.


#### Role in Forward Kinematics

- The kinematic tree is essential for **forward kinematics**:  
  Computing the pose (position and orientation) of any frame in the world (e.g., the gripper frame) as a function of joint positions $q$.
- This is done by recursively composing spatial transforms through joint-dependent transforms along the chain.
- Drake’s MultibodyPlant automates this spatial algebra, often with caching, efficiently returning body poses in the world.
- **Floating bodies** (e.g., foam bricks, floating hands) also have DoF contributing to the full configuration vector $q$.  
  - A floating body’s pose is represented by 7 numbers:  
    - 3 for position  
    - 4 for orientation as a **unit quaternion** (to avoid singularities like gimbal lock).

#### Examples and Visualization

- The **KUKA iiwa robot** is a serial manipulator, so its kinematic tree resembles a **vine**.
- Dexterous robot hands have more complex **branching tree** structures.
- Free objects like a foam brick appear as branches connected directly to the world root node.
- Robot visualizations often display coordinate frames with colors:  
  - Red for X-axis  
  - Green for Y-axis  
  - Blue for Z-axis  
  These frames illustrate transforms between bodies.

### C. Degrees of Freedom in the Example System (Allegro Hand + Foam Brick)

![alt_text](/assets/images/robotic-manipulation/03/2.png "image_tooltip")

The total configuration vector, $( q )$, for the system you referenced — which includes the Allegro hand and a foam brick — sums to **30 degrees of freedom**. This total comes from adding the contributions of each component's degrees of freedom:

#### Degrees of Freedom from Floating Bodies (Foam Brick and Hand Base)

- Adding a body to the MultibodyPlant inherently adds many degrees of freedom.
- A **free-floating body** typically has **6 DoF**:  
  - 3 for position  
  - 3 for orientation  
- To avoid singularities (like gimbal lock), the pose of a floating body is represented by **7 numbers**:  
  - 3 for position  
  - 4 for orientation as a **unit quaternion**
- **Foam Brick**:  
  - Considered a free object with its own floating pose in the scene.  
  - Contributes **7 DoF** (3 position + 4 quaternion orientation).
- **Allegro Hand Base**:  
  - Treated as a floating hand, not fixed to the world frame.  
  - Contributes **7 DoF** (3 position + 4 quaternion orientation).


#### Degrees of Freedom from Robot Joints (Allegro Hand)

- **Joints are constraints** that reduce DoF of bodies by restricting motion.
- A **revolute joint** constrains 6 DoF down to 1 DoF (rotation around a single axis).
- The Allegro hand has **16 revolute joints** in its kinematic tree.
- These joints contribute **16 DoF** in total.


#### Total Degrees of Freedom Calculation

$[
7 \quad (\text{foam brick}) + 7 \quad (\text{floating hand base}) + 16 \quad (\text{revolute joints}) = \boxed{30 \text{ DoF}}
]$

Thus, the complete configuration vector $( q )$ for the system has 30 degrees of freedom.


### Computing $( X^G )$:

- Compose transforms from gripper frame to world frame
