---
layout: post
title: The Three Eras of Robot Learning
subtitle: A decade of progress in models, data, and world models
categories: Robotics Robot-Learning Foundation-Models
tags: [Robotics, VLA, World-Models, Reinforcement-Learning]
banner: "/assets/images/banners/yuanpang-wa-iceburg2.jpg"
---

# The Three Eras of Robot Learning

Robot learning has changed dramatically over the last decade. The field started with fragile but inspiring demos, moved into foundation models and vision-language-action policies, and is now entering a scaling phase where the central questions are no longer only about model architecture. They are also about data engines, deployment loops, and whether robots can learn from simulated futures before acting in the real world.

This post is based mainly on two talks:

- Jim Fan, [Robotics' End Game](https://www.youtube.com/watch?v=3Y8aq_ofEVs)
- Ted Xiao, [Three Eras of Robot Learning](https://www.youtube.com/watch?v=etPqBphTgmE&t=1105s)

![alt_text](/assets/images/robot-learning-evolution/1.png "image_tooltip")

The core thesis is simple:

> Robot learning is following a path similar to large language models: pre-training, supervised fine-tuning, and then reinforcement learning or reasoning-time improvement. But robotics has one extra constraint: intelligence must survive contact with the physical world.

There are two ways to understand this evolution:

1. **Model evolution:** from RL and behavior cloning, to VLA foundation models, to world-action models.
2. **Data evolution:** from arm farms and teleoperation, to cross-embodiment datasets, to wearables, egocentric video, autonomous rollouts, and neural simulators.

![alt_text](/assets/images/robot-learning-evolution/2.png "image_tooltip")

This structure is useful because the model story and the data story are linked but not identical. Better models make more kinds of data usable. Better data makes more general models possible.

## Part I: Model Evolution

The model evolution of robot learning can be divided into three eras.

| Era | Years | Main model paradigm | Core question |
| --- | --- | --- | --- |
| Era 1 | 2015-2021 | RL and behavior cloning | Can end-to-end robot learning work at all? |
| Era 2 | 2022-2023 | Foundation models and VLA | Can semantic knowledge from web-scale models transfer into action? |
| Era 3 | 2024-present | Scaled VLA, RL after imitation, and WAMs | Can robots generalize, improve, and imagine consequences before acting? |

### 1. Era of Existence Proofs: 2015-2021

The first era was about proving that data-driven robot learning could work at all.

![alt_text](/assets/images/robot-learning-evolution/3.png "image_tooltip")

Robotics had long relied on classical planning, control, perception pipelines, and hand-engineered task logic. These systems could be powerful in constrained settings, but they were brittle when objects, scenes, or goals changed. The dream of end-to-end learning was to let a robot learn directly from interaction, similar to how deep learning had changed vision and language.

This was the era of the "arm farm": rooms full of robot arms repeatedly grasping, pushing, failing, recovering, and collecting experience. These setups were not elegant. They were expensive, maintenance-heavy, and often produced scrappy demos. But they established an important point: physical interaction data could train generalizable behavior.

#### RL versus behavior cloning

At the time, reinforcement learning (RL) looked like the more serious path. The argument was intuitive:

- Behavior cloning copies human demonstrations.
- If the policy drifts away from the demonstrated state distribution, errors compound.
- RL can learn from its own trials and optimize directly for task success.

So behavior cloning was often treated as a toy baseline, while RL was seen as the way to reach high performance.

In practice, the story became more complicated. RL could discover strategies, but it was unstable, sample-hungry, and operationally painful on real robots. The infrastructure mattered as much as the algorithm. If cameras drifted, grippers wore out, data pipelines broke, or reset procedures were inconsistent, the learning signal became noisy.

This is why the "slow down to speed up" lesson mattered. Some teams paused flashy publication cycles and focused on paying down research debt: stable hardware, better teleoperation, consistent data schemas, reproducible evaluation, and clean infrastructure. Once those foundations improved, large-scale imitation learning became much stronger than expected. Behavior cloning was not fundamentally weak; weak data and weak infrastructure had made it look weak.

#### Key milestones

**QT-Opt** showed that value-based RL could handle high-dimensional continuous robot control at meaningful scale. It was an important proof that RL could operate beyond small toy environments.

**Large-scale expert teleoperation** showed that high-quality demonstrations could sometimes beat noisy autonomous exploration. If the data distribution was good enough, imitation learning could reach high success rates.

**Infrastructure-first robotics** became a hidden milestone. The field learned that robot learning is not just about neural network loss functions. It is also about resets, calibration, logging, uptime, data quality, and evaluation discipline.

The main lesson of Era 1:

> Robot learning was possible, but not yet scalable. The bottleneck was not only intelligence. It was reliable data production.

### 2. Era of Foundation Models and VLA: 2022-2023

The second era began when robotics started absorbing the foundation model wave.

![alt_text](/assets/images/robot-learning-evolution/4.png "image_tooltip")

Large language models and vision-language models had already learned rich semantic representations from internet-scale data. Robotics researchers realized that a robot policy did not need to learn all concepts from physical experience alone. A model trained on web-scale images and text already knows a lot about cups, drawers, apples, tools, containers, spatial relations, and common human goals.

The question became: can that semantic knowledge be grounded into robot actions?

This gave rise to the Vision-Language-Action (VLA) paradigm. A VLA model takes visual observations and language instructions, then predicts robot actions. Conceptually, actions become another sequence to model, just as words are tokens in an LLM.

#### SayCan: language as high-level planning

SayCan was one of the first major bridges between LLMs and robots. The key idea was to combine:

- a language model that proposes useful high-level steps, and
- an affordance model that estimates what the robot can actually do.

This matters because language models can suggest plausible plans that are physically impossible for a specific robot. A mobile manipulator may know that "put the apple in the drawer" is semantically meaningful, but if the apple is out of reach or the drawer cannot be opened, the plan should be rejected. SayCan grounded language plans in robot affordances.

#### RT-1 and RT-2: from robot transformer to VLA

RT-1 showed that transformers could scale for robot control when trained on large robot datasets. Instead of hand-designing a controller for each task, the model learned a broad policy from many demonstrations.

RT-2 pushed the idea further by connecting robot control to internet-scale vision-language pre-training. This was important because it suggested that semantic reasoning could transfer into physical action. A robot could respond to concepts that were not narrowly encoded as task labels, such as identifying an object that could be used as a tool.

#### ACT and action chunking

A separate but important idea was action chunking. Instead of predicting a single low-level action at every frame, the policy predicts a short sequence of future actions. This helps with compounding error because the model commits to a coherent local behavior rather than constantly re-deciding from noisy observations.

Action chunking also matches how many physical tasks work. Opening a drawer, folding cloth, or placing an object is not a single action. It is a short behavior segment.

#### Open X-Embodiment

Open X-Embodiment marked another shift: robot learning needed cross-robot data. A single lab's robot fleet was too small and too narrow. The collaboration aggregated data across many labs, robot types, tasks, and environments.

This was a robotics version of the foundation model recipe:

> More diverse data creates more general models, but only if the model and data schema can absorb the diversity.

The main lesson of Era 2:

> Robotics started inheriting semantic knowledge from foundation models, and actions began to look like another modality to scale.

### 3. Era of Scaling and the End Game: 2024-Present

The current era is about scaling three things at once:

1. Model capacity and architecture.
2. Data diversity and data volume.
3. Closed-loop improvement after deployment.

![alt_text](/assets/images/robot-learning-evolution/5.png "image_tooltip")

This is where robotics begins to look even more like the LLM pipeline:

| LLM stage | Robotics analogue | What it means |
| --- | --- | --- |
| Pre-training | Learn physical and semantic priors | Use web images, video, robot data, simulation, and human activity data |
| Supervised fine-tuning | Align to useful robot tasks | Train on demonstrations, instructions, and task-specific behaviors |
| Reasoning RL | Improve beyond imitation | Use rollouts, rewards, world models, and self-improvement loops |

The important difference is that robot reasoning is grounded in action. A language model can think longer by generating more tokens. A robot must also decide whether an action is safe, feasible, reversible, and useful in the physical scene.

Era 3 has two model tracks:

1. **Scaled VLA policies:** stronger reflexive policies that map perception and instructions to action.
2. **World Action Models:** predictive models that imagine possible futures and use them for planning, RL, or synthetic data generation.

#### 3.1 Scaled VLA: The Physical Intelligence π Series

The π series from Physical Intelligence is best understood as the scaling story of VLA-style robot policies. π0 through π*0.6 are primarily about making the VLA policy stronger: better action generation, faster training, more heterogeneous data, and reinforcement learning after imitation.

π0.7 is still a VLA policy, but it starts to bridge into the WAM era because it uses richer conditioning, including visual subgoals that can come from a lightweight world model. So I would classify it as **VLA with world-model assistance**, not as a pure World Action Model.

##### π0: generalist policy as the foundation

π0, released in October 2024, combined a vision-language backbone with an action expert for continuous control. The architecture separated broad scene and instruction understanding from precise motor output.

This is a meaningful design pattern. A robot policy needs both:

- high-level semantic understanding: what is the object, what is the goal, what matters in the scene?
- low-level control: how should the gripper, wrist, arm, or base move right now?

π0 used flow matching for action generation. Compared with classical diffusion-style iterative denoising, flow matching can produce continuous actions more efficiently, which matters for real-time control.

##### π0-FAST: action tokenization for faster training

π0-FAST introduced a faster action representation. The core idea was to compress continuous action trajectories into a token-like format that is easier to train with large autoregressive models.

This is part of a broader trend: robotics is trying to reuse the scaling machinery of language models, but robot actions are not naturally words. Good action tokenization is therefore a key interface between continuous control and foundation model training.

##### π0.5: open-world generalization

π0.5 focused on generalization across new homes, rooms, objects, and tasks. The important shift was heterogeneous data mixing: not just one robot, one lab, or one task family, but many data sources that teach broader physical common sense.

This is where non-robot data becomes increasingly important. Human videos, egocentric video, web images, and language annotations can help models understand the world even when they do not provide direct robot motor commands.

##### π*0.6 and RECAP: RL returns

The pendulum then swung back toward reinforcement learning, but in a more mature form. The goal was no longer to replace imitation learning with RL. The goal was to use RL after imitation to break the "imitation ceiling."

The imitation ceiling is the limit imposed by human demonstrations. If a robot only copies human teleoperators, it may inherit their slowness, inconsistency, or suboptimal strategies. RL can let the robot improve from its own experience.

RECAP is important because it adapts RL to VLA-style policies. Standard RL methods often need log probabilities from the policy distribution, but flow-matching action models do not naturally expose them in the same way. RECAP uses advantage conditioning to steer the model toward better outcomes without relying on the exact same mechanics as PPO-style language-model RL.

The conceptual point is bigger than the specific algorithm:

> The next generation of robot policies will not only imitate demonstrations. They will absorb demonstrations, try tasks themselves, learn from failures, and distill the improved behavior back into the generalist model.

##### π0.7: the bridge from VLA to WAM

π0.7, released in April 2026, is best understood as a steerable generalist VLA model rather than a pure world model. It still acts as a policy, but it is conditioned on richer context:

- language instructions,
- step-by-step language coaching,
- metadata such as speed or quality,
- control-mode labels,
- visual subgoals, including subgoals generated by a lightweight world model.

This richer prompting solves a real data problem. Large robot datasets are messy: some episodes are fast, some are slow, some are successful, some are partial failures, some come from humans, some from autonomous rollouts, and some from different robot bodies. If those are merged naively, the model may learn an average behavior that is not good at anything.

π0.7's strategy is to annotate how a behavior was produced, not just what task was attempted. That lets the model use lower-quality or suboptimal data without treating it as the ideal behavior. At inference time, the same conditioning becomes a steering mechanism: ask for faster execution, higher precision, a different strategy, or a visual subgoal.

The most interesting claim is compositional generalization: recombining known skills to handle a task not directly present in the training set. Examples include using kitchen appliances with language coaching and transferring laundry folding to a different bimanual robot setup.

This is the bridge to WAMs: π0.7 can be steered by predicted or desired futures, but the core policy is still a VLA executing action chunks.

![alt_text](/assets/images/robot-learning-evolution/6.png "image_tooltip")

#### 3.2 World Action Models: Imagine-and-Do

World Action Models belong under Era 3 because they are the end-game model paradigm after reflexive VLA. VLA models are mostly "see-and-do." They observe the world, read the instruction, and predict actions. WAMs aim for "imagine-and-do."

One useful analogy is System 1 versus System 2 reasoning. A VLA policy is like a fast motor system: it maps the current observation to an action chunk quickly enough for closed-loop control. A WAM is closer to a planning system: it predicts what the world may look like after different actions, then lets the robot choose behavior with some sense of consequence.

Instead of only asking "what action should I take?", a robot with a world model can ask:

- If I push here, what will move?
- If I grasp this object, will it slip?
- If I open this drawer, what space becomes reachable?
- If I pour too quickly, will the liquid spill?
- If I fold this sleeve first, will the shirt end up aligned?

This is a major conceptual shift. The robot is no longer only reacting to pixels. It is using predicted future pixels, latent states, or value estimates as part of decision-making.

![alt_text](/assets/images/robot-learning-evolution/7.png "image_tooltip")

##### DreamZero: world and action as one sequence

[DreamZero](https://dreamzero0.github.io/) is a direct example of the WAM idea. It is built on a pretrained video diffusion backbone and jointly predicts future world states and robot actions. The important design choice is that video and action are modeled together instead of being treated as separate modules.

Traditional VLA models usually predict actions from the current observation. DreamZero instead uses video as a dense representation of how the world evolves. That means the model can learn not only that a gripper should move left, but also what the scene should look like after that movement. If the predicted future looks wrong, the policy has a signal that the action may be wrong.

The project reports two practical results that are worth noting:

- Better generalization to unseen tasks and environments than strong VLA baselines.
- Real-time closed-loop control at about 7Hz through system optimizations such as fewer diffusion steps, action chunk smoothing, asynchronous inference, and KV caching.

That second point matters because world models are often too slow for robotics. Predicting video is expensive. A WAM must close the "reactivity gap": it needs enough imagination to be useful, but enough speed to control a physical system.

##### UniSim: a learned universal simulator

[UniSim](https://universal-simulator.github.io/unisim/) takes a broader simulator view. Instead of focusing only on robot action prediction, it asks whether a generative model can simulate realistic interactions across many kinds of data.

The key observation is that different datasets contain different pieces of the physical world:

- image datasets contain objects, scenes, and appearance diversity;
- robotics datasets contain actions and contact-rich manipulation;
- navigation datasets contain movement through space;
- video datasets contain temporal evolution.

UniSim combines these axes into an action-conditioned video model. Given a scene and an action or instruction, it predicts the visual outcome. For example, it can simulate what happens after an instruction like "open the drawer" or a low-level control like "move by x, y."

The long-horizon part is the most important. A useful simulator cannot only produce a pretty next frame. It must stay coherent over many steps, because planning and RL need rollouts. If the robot is putting an orange into a drawer, the simulator must remember that the drawer is open, the orange has moved, and the task state has changed.

UniSim points toward a future where some robot policies are trained inside learned simulators and then transferred to the real world. This does not remove the need for real robot data, but it changes its role. Real data becomes the grounding signal for a simulator that can generate many more training situations.

##### DreamerV3: model-based RL at scale

[DreamerV3](https://www.nature.com/articles/s41586-025-08744-2) is not a robotics-only WAM, but it is one of the clearest examples of world-model-based reinforcement learning.

Dreamer learns a compact latent dynamics model of the environment, then trains an actor-critic policy inside imagined rollouts. The agent does not need to execute every candidate behavior in the real environment. It can improve behavior by rolling forward inside its learned model.

This is the same principle robotics wants, but in a more general RL form:

1. observe the world;
2. learn a latent model of how it changes;
3. imagine future trajectories;
4. optimize the policy using those imagined futures;
5. return to the real environment with a better policy.

DreamerV3 is important because it showed that one model-based RL algorithm could work across many different domains with the same basic configuration, including Atari, Minecraft, and continuous control tasks. For robotics, the lesson is not that DreamerV3 itself is the final robot foundation model. The lesson is that learned dynamics plus imagined rollouts can scale beyond small toy tasks.

##### NVIDIA Cosmos, GR00T-Dreams, and GR00T N2

NVIDIA's physical AI stack frames world models as a data engine. [GR00T-Dreams](https://github.com/NVIDIA/GR00T-Dreams) uses Cosmos world foundation models to generate synthetic robot trajectory data. The workflow is roughly:

1. start from a real or generated scene;
2. use a world model to generate videos of a robot performing tasks in that scene;
3. extract action tokens or trajectory signals from those generated videos;
4. fine-tune a robot foundation model on the resulting synthetic data.

This is different from classical simulation. In a classical simulator, the world is built from meshes, rigid-body dynamics, collision models, and manually specified physics. In a neural simulator, the model learns regularities from data and generates plausible futures directly.

NVIDIA has also described GR00T N2 as a next-generation robot foundation model based on WAM-style research, reporting that it succeeds at new tasks in new environments more often than leading VLA baselines. Since these are company-reported claims, I treat them as directional evidence rather than settled academic consensus. Still, the strategy is clear: use world models not just for rendering videos, but for generating training data and improving robot generalization.

##### GigaBrain and RAMP: conditioning the policy on imagined futures

[GigaBrain-0.5M*](https://gigabrain05m.github.io/) is a useful example of a tighter integration between a VLA policy and a world model.

The model introduces RAMP: Reinforcement leArning via world Model-conditioned Policy. The idea is to improve a VLA by conditioning it on predictions from a world model. Instead of giving the policy only the current image, language instruction, and robot state, the system also gives it predicted future states and value estimates.

The training loop looks like this:

1. Pretrain a VLA on multimodal, robot manipulation, and web video data.
2. Train a world model to forecast future states and values from interaction data.
3. Fine-tune the VLA policy while conditioning actions on the world model's predicted futures.
4. Deploy the policy and collect human-in-the-loop rollouts.
5. Use the new rollout data to continue improving both the world model and the policy.

This is an important bridge between RECAP-style advantage conditioning and full WAM planning. RECAP conditions the policy on an advantage signal: was this behavior better or worse? RAMP adds richer future information: what state does the world model expect, and how valuable does that future look?

The project reports about a 30% improvement over a RECAP baseline on long-horizon manipulation tasks such as laundry folding, box packing, and espresso preparation. The broader point is more important than the exact number: future prediction becomes an input to policy learning, not just a separate diagnostic tool.

That said, WAMs are not a replacement for VLA policies. They solve a different part of the stack:

| Component | Role |
| --- | --- |
| VLA policy | Fast perception-to-action control |
| World model | Predict possible futures |
| RL or planner | Search, evaluate, and improve decisions |
| Real robot data | Ground the system and correct simulation errors |

The likely end state is a hybrid system. A VLA policy handles fast, reactive control. A world model supports slower planning, counterfactual reasoning, and synthetic data generation. RL connects the two by optimizing behavior against rewards and outcomes.

## Part II: Data Evolution

The model story is only half the story. The deeper bottleneck is data.

![alt_text](/assets/images/robot-learning-evolution/8.png "image_tooltip")

Text is abundant, cheap, and already digital. Robot data is expensive, slow, embodied, and tied to hardware. Because of that, every model era has been paired with a data era.

| Data phase | Main source | Why it mattered | Scaling limit |
| --- | --- | --- | --- |
| Arm farms | Robots collecting trial-and-error data | Proved physical learning was possible | Expensive hardware and maintenance |
| Teleoperation | Humans controlling robots | Produced high-quality demonstrations | Limited by robot hours and operator hours |
| Cross-embodiment datasets | Many robots, labs, tasks, and embodiments | Enabled generalist policies | Hard to normalize across robots |
| Data wearables | Human manipulation traces without full robots | Scales manipulation data beyond robot fleets | Human collection is still required |
| Egocentric video | First-person human activity video | Provides broad physical common sense | Not directly robot-action labeled |
| Autonomous rollouts | Robots learning from deployment | Creates a real-world data flywheel | Failures must be labeled and controlled |
| Neural simulators | World models generating action-conditioned futures | Compute becomes part of the data engine | Simulator errors can mislead policies |

### Phase 1: arm farms and teleoperation

The first data strategy was brute-force physical collection. Robots interacted with the world, failed repeatedly, and generated experience. This was essential for existence proofs, but it was expensive and operationally fragile.

Teleoperation then became the workhorse of robot learning. A human controls the robot through VR, joysticks, motion capture, data gloves, or other rigs. This produces high-quality demonstrations because the robot records actions in its own embodiment.

The problem is cost and scale. A robot can collect at most 24 hours of data per day, and in practice much less after maintenance, resets, failures, battery charging, and supervision. Teleoperation is valuable, but it cannot be the whole data engine.

### Phase 2: cross-embodiment robot data

The foundation-model era needed broader robot datasets. A policy trained on one robot in one lab usually overfits to that robot's camera views, gripper, workspace, and task distribution.

Open X-Embodiment addressed this by aggregating trajectories across many labs, robot types, tasks, and environments. The key idea was that diversity can compensate for the narrowness of any single robot setup.

This data phase made VLA scaling more plausible. If actions are another modality, then the model needs many examples of how different bodies produce useful actions in different scenes.

### Phase 3: data wearables

Wearables move data collection away from the robot. Instead of putting the full robot in the loop, humans wear devices that capture manipulation behavior directly.

The Universal Manipulation Interface (UMI) is a good example: a simple gripper-like device lets humans collect manipulation trajectories without controlling an expensive robot arm. Other systems use exoskeletons or data gloves to capture hand motions.

This breaks part of the 24-hour-per-robot bottleneck. Many humans can collect manipulation data in parallel without requiring a matching robot for every hour of data.

### Phase 4: egocentric video

The next scaling step is ambient human video. Instead of collecting every demonstration in a robotics lab, models can pre-train on egocentric video: first-person footage of humans cooking, cleaning, opening containers, manipulating tools, and solving physical tasks.

This data does not directly provide robot actions. A human hand is not a robot gripper. But it provides something extremely valuable: visual common sense about physical interaction. The model can learn what contact, progress, failure, and task completion look like.

This is analogous to web-scale text for LLMs. It is not perfectly aligned to the final task, but it teaches broad priors that make later fine-tuning much more efficient.

### Phase 5: autonomous rollouts and data flywheels

Once robots are deployed, they can collect their own experience. This is the beginning of a data flywheel:

1. Train a policy on demonstrations and broad pre-training.
2. Deploy it on real tasks.
3. Record successes, failures, recoveries, and edge cases.
4. Label or score the outcomes.
5. Improve the policy and redeploy.

This is where RL and metadata become crucial. If a robot collects many failed attempts, the model needs to know they are failures. Otherwise, autonomous data can poison the policy. But if failures are tagged and used correctly, they become training signal.

### Phase 6: neural simulators as the next data engine

World models also change the data strategy. The old pipeline was mostly physical. The emerging pipeline uses neural simulators to create action-conditioned futures.

This is what Jim Fan means by a robotics data flywheel where compute, environment, and data start to merge. If a robot can train in a learned simulator, then a lab does not need to collect every failure mode physically. It can use real data to ground the model, then use the model to generate many more situations.

There is a risk here: a learned simulator can hallucinate plausible but physically wrong futures. If the WAM gets contact, friction, deformable objects, or force wrong, the robot may learn behaviors that fail in the real world. So the end game is not "replace reality with dreams." It is:

> Use real robot data to ground the world model, use the world model to expand the training distribution, then return to real deployment to correct the model.

## Final Map

The cleanest way to summarize the field is to separate model evolution from data evolution:

| Era | Model evolution | Data evolution | Main lesson |
| --- | --- | --- | --- |
| 2015-2021 | RL and behavior cloning | Arm farms and expert teleoperation | Robot learning works, but infrastructure and data quality dominate |
| 2022-2023 | Foundation models and VLA | Web-scale pre-training plus cross-embodiment robot datasets | Semantic knowledge can transfer into action |
| 2024-present | Scaled VLA, RL after imitation, and WAMs | Wearables, egocentric video, autonomous rollouts, and neural simulators | Generalization needs broad data plus closed-loop improvement |

## My Takeaway

The robotics field is converging on a layered recipe:

1. Pre-train on broad visual, language, video, and robot data to learn semantic and physical priors.
2. Fine-tune on high-quality robot demonstrations to align the model to useful tasks.
3. Use action chunking, diffusion, or flow matching to produce smooth real-time control.
4. Use RL and autonomous rollouts to improve beyond human demonstrations.
5. Use world models to generate subgoals, evaluate futures, and scale training environments.

The important taxonomy is:

- **VLA models** are the fast policy layer. They are the main story from RT-1/RT-2 through π0, π0-FAST, π0.5, and π*0.6.
- **π0.7** is a bridge model: still VLA, but more steerable and world-model-assisted.
- **WAMs** are the end-game planning and simulation layer. They belong under the 2024-present scaling era because they extend robot learning from action prediction to future prediction.

The most important transition is from robot learning as a collection of bespoke task policies to robot learning as a general data engine. In the old regime, each task needed its own data collection, model, tuning, and deployment. In the new regime, a generalist policy can absorb many kinds of data, accept richer prompts, reuse skills across tasks and bodies, and improve through a loop of real-world feedback and imagined futures.

That is why the "end game" is not just a bigger VLA model. It is a full learning loop: broad pre-training, high-quality demonstrations, autonomous experience, world-model imagination, and real-world feedback all feeding the same physical intelligence layer.
