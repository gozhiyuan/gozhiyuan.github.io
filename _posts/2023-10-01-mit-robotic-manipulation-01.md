---
layout: post
title: Anatomy of a Manipulation System
subtitle: Lecture 1 Introduction
categories: MIT-Robotic-Manipulation-2023
tags: [robot]
banner: "/assets/images/banners/yuanpang-wa-valley.jpg"
---

## 1. Defining Robotic Manipulation

The lecture delves into a nuanced definition of robotic manipulation, moving beyond simple object handling to encompass complex interactions within dynamic and unstructured environments.

[Course Link](https://manipulation.csail.mit.edu/Fall2023/index.html#description)


### A. Core Definition

- **Matt Mason's Definition**:  
  > "Manipulation refers to the agent's control of the environment through selective contact."  
  This emphasizes applying forces to effect change in the world.

- **Beyond Simple Tasks**:  
  Changing an object's position (e.g., holding a brick) falls under manipulation, but the course aims to explore much richer and more complex scenarios like tying shoelaces, involving **super rich dynamics and control**.

### B. "Open-World" Manipulation

- **Concept**: Extends Mason's definition to include **open-world manipulation with autonomy**, meaning robots make decisions and understand the world without human teleoperation. The term comes from open-world video games.

- **Key Requirements**:
  - Rich perceptual understanding
  - Common-sense understanding of object behavior
  - Long-term, task-level planning (e.g., making cereal)
  - Combining high-level plans with low-level control

- **Claim**: This broad definition may be "the best way to think about intelligence."

### C. Examples from Toyota Research Institute (TRI)

- **Dishwasher Loading**:
  - Complexities include opening doors, picking silverware, handling occlusions
  - Must be robust to disturbances like doors closing mid-task

- **Grocery Store Operations**:
  - Mobile manipulation system handling dynamic, diverse inventory
  - Pushes open-world manipulation due to unseen and changing environments

- **Spot Demo**:
  - Boston Dynamics robot dog with an arm
  - Picks up a plush toy, demonstrating mobile manipulation

---

## 2. Systems Theory Perspective: Dynamics and Control

The course emphasizes control theory and how traditional paradigms are challenged by manipulation's complexity.

### A. Challenges in Manipulation Control

- **Beyond Robot Dynamics**:  
  Unlike locomotion, manipulation requires modeling and interacting with external objects, not just self-motion.

- **Scaling State Representations**:  
  Tracking every object or object part is impractical (e.g., mug types, chopped onion pieces).

- **Human vs. Robot**:  
  Tasks like shoelace tying are harder for robots than doing backflips—requiring different "cognitive systems."

### B. Role of Deep Learning and Implicit State Representations

- **Breaking Old Limitations**:  
  Deep learning enables **implicit state representations**—networks learn features without manual state modeling.

- **Visuomotor Policies**:  
  End-to-end policies from image to control. Capable of handling clutter and diverse objects.

- **Fluid Dynamics Example**:  
  Tasks like dough rolling or sauce spreading can now be learned from human demonstrations—no need for full physics.

- **Physics vs. Data**:
  - **Trade-off**: Injecting physics knowledge can reduce data needs but limit learning capacity.
  - **Emerging Area**: Physics-inspired neural networks aim to combine both.

---

## 3. Anatomy of a Modern Manipulation System

Comparison of modular software architectures: ROS vs. control theory block diagram approach via Drake.

### A. ROS

- **ROS (Robot Operating System)**:
  - Open-source, modular, message-based
  - Great for prototyping and community sharing
  - **Limitation**: Non-deterministic simulations; difficult debugging


The **Robot Operating System (ROS)** is a foundational **software framework** for robotics that facilitates the development of **complex, modular robotic systems**. Despite its name, it is **not** an actual operating system, but a **middleware layer** that runs on top of a real OS (like Linux) and coordinates the communication between different parts of a robotic system.

#### 🔧 Modularity and Compartmentalization

- ROS is designed to **break down a robotic system into smaller, manageable components**, called **ROS nodes**.
- Each **node** is an **independent executable** (e.g., a Python or C++ script) that performs a specific function.
- Nodes communicate using **typed messages** over a **publish-subscribe architecture**.

#### 📦 Example System Breakdown

| Node               | Function                                           |
|--------------------|----------------------------------------------------|
| Camera Driver      | Publishes RGB-D images                             |
| Perception System  | Subscribes to images, publishes object pose        |
| Planning System    | Subscribes to object pose and robot state, outputs planned trajectories |
| Controller Node    | Executes robot trajectories                        |


#### 🌐 Ecosystem and Benefits

- **Open Source**: ROS fosters **collaborative development**. Components (e.g., planners, controllers) can be reused across labs or companies.
- **Interoperability**:
  - Components in different languages (e.g., C++, Python) or even **different OS environments** (via Docker) can work together.
  - Communication is solely based on the **message type**, abstracting away implementation details.

#### ⚠️ Limitations vs. Control-Theoretic View

- **Control/Systems Theory View**:
  - Components modeled as blocks with well-defined dynamics
  - Deterministic, analyzable, and debuggable
  - Explicit state declaration aids comprehension
  - Composable like object-oriented code


| Aspect                  | ROS                                       | Control-Theoretic View (e.g., Drake) |
|-------------------------|-------------------------------------------|---------------------------------------|
| Internal Node Logic     | Arbitrary logic, as long as messages match | Defined by **differential/difference equations** |
| Communication           | Message passing over OS/network threads    | Signal flow between system components |
| Determinism             | **Non-deterministic** due to thread timing | **Deterministic**, supports exact replay |
| Debugging/Certification | Can be hard due to timing issues           | Easier due to **explicit state declarations** |

- In ROS, **timing-dependent behaviors** can result in **simulation inconsistencies** (e.g., running the same scenario twice yields different results).
- In contrast, systems like Simulink or **Drake** follow a **signals-and-systems paradigm**, enforcing **state declarations** and allowing formal analysis.

### B. Model-Based Design in Control/Systems Theory (Differential Equations)

In contrast to ROS's "software engineering view," **model-based design** in control/systems theory offers a **rigorous, mathematical framework** for representing and analyzing robotic systems. This approach is grounded in **differential and difference equations**, which describe how system states evolve over time.

#### 🔁 Block Diagram Modeling

- Tools like **Simulink** and **Modelica** embody this philosophy.
- A system is represented as a **network of interconnected blocks**.
- Each block:
  - Has **well-defined inputs and outputs**.
  - Is described by **difference or differential equations**.
  - Maintains an **explicitly declared internal state**.

This contrasts sharply with ROS nodes, which are often:
- **Arbitrary executables** (e.g., scripts or binaries).
- Lacking a formally defined state.
- Operating based on **OS-level timing and threading**, introducing nondeterminism.

#### 📈 Differential Equations and System State

- Differential equations govern how the system **state** (e.g., joint angles, velocities) **evolves over time**.
- Also define how **sensor outputs** are generated from states and inputs.

#### Example:
Let:
- $( x(t) )$: system state at time $( t )$
- $( u(t) )$: control input
- $( y(t) )$: sensor output

Then:
- **State evolution**:  
$$
\frac{dx(t)}{dt} = f(x(t), u(t))
$$
- **Sensor model**:  
$$
y(t) = h(x(t), u(t))
$$


#### ⏱️ Timing Semantics and Determinism

- Control-theoretic frameworks **enforce timing semantics**:
  - Explicit update rates (e.g., 100 Hz).
  - Known state transitions at each timestep.
- This enables:
  - **Deterministic simulations**.
  - **Repeatable experiments**.
  - **Debugging via rewind and replay**.

> 🔁 In contrast, ROS’s reliance on OS-level threading and asynchronous messaging can lead to **nondeterministic behavior**, where running the same simulation twice might yield different outcomes.


#### 🧠 Complex Blocks Still Fit the Framework

- Even sophisticated components can be integrated into this model:

| Component                   | Representation in Control View                             |
|----------------------------|--------------------------------------------------------------|
| Neural Network (Feedforward) | Modeled as a static nonlinear function $( y = f(x) )$       |
| Recurrent Neural Network     | Requires explicit internal state declaration $( h_t )$       |
| Photorealistic Renderer      | Modeled as a black-box function mapping state to sensor image $( I = g(x) )$ |

- Declaring internal state (e.g., in RNNs or video renderers) allows these blocks to participate in **rigorous simulation pipelines**.


#### ✅ Summary: Advantages of Control View

| Feature                        | ROS                          | Control-Theoretic (e.g., Simulink, Drake) |
|-------------------------------|------------------------------|-------------------------------------------|
| State Declaration             | Implicit or arbitrary        | **Explicit and formal**                   |
| Execution Timing              | OS-level, thread-based       | **Deterministic and time-indexed**        |
| Repeatable Simulation         | No                           | **Yes**                                   |
| Suitable for Analysis         | Difficult                    | **Supports formal analysis**              |
| Supports Complex Components   | Yes, but uncontrolled        | **Yes, within block structure**           |

By using model-based design and defining system components through differential equations, **robotic manipulation systems** can be **better analyzed, debugged, and scaled**, providing a level of **mathematical rigor** often missing from purely message-passing systems like ROS.



### C. Drake: The Course’s Core Software

- **Purpose**: Rigorous framework for modeling, control, and simulation developed by the instructor’s group.

- **Key Capabilities**:
  - **Modeling Dynamical Systems**: Controllers, models, neural nets via block diagrams
  - **Mathematical Programs**: Optimization-based controller design
  - **Multibody Kinematics and Dynamics**: For physical interactions

- **User-Friendly**:
  - Runs in the browser via DeepNote
  - No installation required

- **Sim-to-Hardware Transition**:
  - "Hardware Station" abstraction unifies simulation and real-world control
  - Just switch backends to deploy the same system on physical robots

- **ROS 2 Compatibility**:
  - Drake diagrams can be used within ROS 2 systems


#### 🔁 Integration with Drake

**Drake** is a robotics framework for **modeling and controlling dynamical systems** via **differential/difference equations**.

#### 🧩 Compatibility

- **Drake + ROS 2**: Drake diagrams can **live inside a ROS ecosystem**.
- Drake offers a **"hardware station interface"**:
  - In simulation mode: runs entirely within Drake (deterministic).
  - In hardware mode: uses ROS **senders/receivers** to communicate with real robots.

#### ✅ Benefits of Integration

- Seamless **simulation-to-real transition** by **flipping a switch**.
- Maintain the **same controller logic**, whether in simulation or deployed to real hardware.
- **Use ROS for modular communication**, and **Drake for deterministic control logic**.

#### 🔄 Or Use Drake Standalone

- You can also **avoid ROS entirely**, using Drake in **a single-process pipeline** to maintain **low complexity and full determinism**.


### D. 🧠 Summary

#### ✅ What Model-Based Design (MBD) Is

In model-based design, we:

- **Define the system's states and dynamics** using math (e.g., differential equations like  
  $$\dot{x} = f(x, u)$$)
- **Simulate how those states evolve** over time with given **inputs** (like control actions)
- **Explicitly track and update state variables**: e.g., robot joint angles, velocities, object poses
- Often use tools like **Simulink** or **Drake** to compose and simulate complex systems as interlinked blocks with **deterministic behavior**

This is especially useful for:
- Control design  
- Physics simulation  
- Planning and motion generation  
- Formal analysis and debugging


#### 🧠 ROS and MBD Together

You're right: **ROS** handles **software infrastructure**, not modeling itself.

So in practice:

- ROS handles real-world **input/output**, such as:
  - Reading camera images
  - Getting joint encoder values
  - Sending motor commands
  - Logging, visualization, communication

- Inside ROS nodes, **some components may use model-based design** — for example:
  - A control node running a trajectory planner using MBD principles
  - A simulator node (e.g., Gazebo, Isaac Sim, Drake) using physics engines

ROS doesn't enforce mathematical modeling — you can put any arbitrary Python script in a node — but **model-based modules can live inside ROS nodes**


#### 🔄 In the Real Robot

- Camera input → ROS topic `/camera/image_raw`  
- Some **perception node** → processes it  
- **Control node** → receives object pose, uses model-based dynamics to plan motion  
- Motion command → sent to hardware driver via another ROS topic


#### 🤖 In Simulation (Model-Based)

- Camera → simulated sensor in physics engine  
- Perception, control, physics → all modeled mathematically or as differentiable blocks  
- State transitions → computed step-by-step using known physics or learned approximations  
- Same architecture can simulate **1,000 runs deterministically**


#### 🧩 So Is Model-Based "Part of ROS"?

🔸 Not **built-in**, but:

- You can **embed model-based design components** into a ROS-based system
- You can **use Drake**, which supports both **model-based simulation** and **ROS 2 integration**


| Feature                    | ROS                                  | Model-Based Design (MBD)                      |
|---------------------------|--------------------------------------|-----------------------------------------------|
| **Purpose**               | Middleware for modular robotics      | Math-based simulation and control             |
| **Core abstraction**      | Nodes, topics, messages              | States, dynamics, differential equations      |
| **Real-time I/O**         | Yes                                  | Typically simulation-first, but can control real robots too |
| **Simulation determinism**| No (depends on OS threads)           | Yes (step-by-step integration)                |
| **Works best for**        | Sensor integration, hardware control | Planning, control, physics-based behavior     |
| **Example tools**         | ROS, RViz, rclpy/rclcpp              | Drake, Simulink, Modelica                     |

---

## 4. Goals for the Course

### A. Core Competencies to be Developed

- Perception Systems (geometric + deep learning)
- Kinematics and Dynamics
- Motion Planning
- Contact Mechanics
- Higher-Level Task Planning

### B. Pedagogical Approach

- **Spiral Curriculum**: Repeatedly revisit core ideas in greater depth
- **Progressive Complexity**: Building up toward full-stack manipulation
- **Mobile Manipulation Emphasis**
- **Advanced Topics**: Later "boutique lectures" will cover niche research (e.g., tactile sensing, belief space planning)
