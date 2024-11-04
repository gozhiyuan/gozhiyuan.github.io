---
layout: post
title: LLM Agents Lecture 2
subtitle: brief history and overview
categories: Course-TLDR LLM
tags: [llm, agents]
banner: "/assets/images/banners/yuanpang-wa-sky.jpg"
---

## LLM agents: brief history and overview

Great talk from Shunyu Yao.  
[Course Link](https://rdi.berkeley.edu/llm-agents/f24)

## What is an Agent? 🤖
In the realm of Artificial Intelligence, an "agent" is defined as an intelligent system capable of interacting with its environment. This interaction involves perceiving the environment through observations and acting upon it through actions. The environment can vary significantly, encompassing physical spaces like those navigated by robots and autonomous vehicles, digital realms such as video games and smartphone applications, or even involve interactions with humans, as seen in chatbots. Defining an agent requires defining "intelligence" and the specific "environment" it operates within. It's crucial to recognize that the definition of "intelligent" is dynamic and evolves over time. 🧠

## What is an LLM Agent? 🌐

![alt_text](/assets/images/llm-agents/02/1.png "image_tooltip")

LLM agents can be categorized into three levels:

1. **Text Agent**: The most fundamental type, using text as both input (observations) and output (actions). Early chatbots like ELIZA exemplify this category. 📜
2. **LLM Agent**: Elevates text agents by incorporating Large Language Models (LLMs) for action selection. Examples include systems like SayCan and Language Planner. 📚
3. **Reasoning Agent**: Representing the current forefront of the field, these agents leverage LLMs for action and also for reasoning and planning before acting. Notable examples include ReAct and AutoGPT. 🌟

## A Historical Perspective 📜
Before LLMs, text agents were mainly built with either rule-based systems or reinforcement learning (RL). Rule-based systems like ELIZA relied on pre-defined rules to process input and generate responses. While simple, they lacked flexibility and struggled with complex or open-ended tasks. RL-based agents, on the other hand, learned to act by optimizing rewards within specific environments, succeeding in domains like video games but requiring careful reward engineering and extensive training for each new task.

LLMs marked a paradigm shift by demonstrating remarkable generalization capabilities. Trained on massive text datasets, LLMs can be prompted to perform various tasks without task-specific training data. This prompted exploration into LLMs for building more capable, general-purpose text agents. 🌍

## The Evolution of Question Answering ❓
The field of Question Answering (QA) offers a lens to examine LLM agent evolution. Early QA systems handled simple factual questions, but as questions grew in complexity—encompassing reasoning, knowledge retrieval, and computation—new techniques emerged:

- **Chain-of-Thought (CoT) Prompting**: Guides LLMs to generate reasoning steps before arriving at an answer, enhancing multi-step problem-solving. 🧩
- **Retrieval-Augmented Generation (RAG)**: Combines retrievers (similar to search engines) to fetch relevant external information, expanding the LLM’s knowledge base. 🔍
- **Tool Use**: Integrates tools like calculators, search engines, and APIs, allowing LLMs to execute code, perform calculations, and access external information. 🔧

These specialized techniques emerged, each focused on specific QA aspects, creating a fragmented landscape with solutions tailored to individual benchmarks. 📊

## ReAct: Reasoning and Acting in Harmony ⚖️
ReAct addressed this fragmentation by combining reasoning and acting, enabling LLM agents to plan, interact with external environments, and adapt based on feedback.

![alt_text](/assets/images/llm-agents/02/2.png "image_tooltip")

### Key Features of ReAct 🌟
- **Interleaving Reasoning and Acting**: ReAct agents alternate between generating thoughts (reasoning) and taking actions in the environment.
- **Intuitive Prompt Structure**: Guides the LLM with "Thought," "Action," and "Observation" steps, enabling in-context learning from examples.
- **Generalizability**: Effective across a wide range of tasks, including question answering, fact-checking, and text-based games. 🎮

![alt_text](/assets/images/llm-agents/02/3.png "image_tooltip")

## Beyond Question Answering: Reasoning as an Internal Action 🧠
The significance of ReAct lies in framing reasoning as an internal action for LLM agents. Unlike traditional agents, whose actions directly affect the external environment, reasoning actions refine the agent's understanding and planning internally before acting. This internal action space is unbounded, allowing agents to reason at arbitrary length and depth. 🌌

![alt_text](/assets/images/llm-agents/02/4.png "image_tooltip")

## Long-Term Memory: Overcoming Short-Term Context Limitations 🧩
LLMs possess impressive capabilities but rely on limited short-term context windows. To address this, researchers are incorporating long-term memory mechanisms:

1. **External Memory**: Uses external storage like databases or files to store and retrieve information beyond the LLM's context window. 📂
2. **Episodic Memory**: Mimics human memory, storing sequences of past experiences, allowing agents to recall specific events. 📜
3. **Semantic Memory**: Stores generalized knowledge and concepts from past experiences, enabling inferences and reasoning about new situations. 🧠

![alt_text](/assets/images/llm-agents/02/5.png "image_tooltip")

## A Broader Perspective: Reasoning Agents in Agent Paradigms 🧬
Looking beyond LLM-specific advancements, reasoning agents are positioned within the broader historical context of agent research. From early symbolic AI systems to the rise of deep RL, different paradigms have shaped intelligent agent development.

### Paradigms of Agent Design 🔍
- **Symbolic AI**: Uses symbolic representations and rules for agent behavior, excelling in well-defined domains but struggling with real-world ambiguity.
- **Deep RL**: Agents optimize rewards through trial and error, handling complex sensory input but often requiring extensive training and struggling with task generalization.
- **Reasoning Agents**: Leverage language for planning and explanation, bridging the gap between symbolic AI’s explainability and deep RL’s performance, though still in early development.

## The Future of LLM Agents 🌌
The field of LLM agents is rapidly evolving, with exciting research directions focusing on:

- **Training**: Developing training methods specifically for agent tasks, beyond next-token prediction.
- **Interface Design**: Creating interfaces optimized for LLM-agent interactions with external systems and tools.
- **Robustness**: Ensuring reliability in real-world scenarios, addressing hallucination, bias, and safety.
- **Human Collaboration**: Integrating LLM agents into human workflows for seamless collaboration.
- **Benchmarking**: Developing comprehensive benchmarks to evaluate agent performance across tasks.

---

## Short-Answer Questions 📝
1. Explain the difference between a text agent and an LLM agent.
2. Describe two challenges in building text agents using reinforcement learning (RL).
3. How does Retrieval-Augmented Generation (RAG) address pure language models' limitations in question answering?
4. What is the core idea behind ReAct, and how does it combine reasoning and acting?
5. Explain reasoning as an "internal action" in the context of ReAct agents.
6. Why is long-term memory important for LLM agents? Name two different approaches for implementing it.
7. Briefly contrast symbolic AI agents, deep RL agents, and reasoning agents in terms of their strengths and weaknesses.
8. Describe two key challenges in developing robust LLM agents for real-world applications.
9. Explain "pass@k" and why it might not suit LLM agents in human-in-the-loop tasks.
10. What are two promising future research directions in the field of LLM agents?

## Essay Questions ✍️
1. Compare and contrast rule-based systems, reinforcement learning, and LLM-based approaches for building text agents. Discuss each approach's advantages and disadvantages, and examples of suitable tasks.
2. Discuss the significance of ReAct in LLM agent evolution. How does ReAct address previous limitations, and what new capabilities does it enable? Analyze reasoning as an internal action and its implications.
3. Evaluate different long-term memory approaches for LLM agents, considering strengths, weaknesses, and suitability for agent applications. Discuss challenges and future research directions.
4. Analyze symbolic AI agents, deep RL agents, and reasoning agents. How do these paradigms conceptualize intelligence, knowledge representation, and decision-making? Discuss reasoning agents' potential to create general-purpose AI systems.
5. Explore robustness, safety, and human-AI collaboration challenges for deploying LLM agents in real-world scenarios. How can we ensure LLM agents are reliable, unbiased, and beneficial?

---

## 💡 **Most Important Ideas and Facts**

- **Defining LLM Agents**: Three levels of LLM agents are defined: **text agents, LLM agents,** and **reasoning agents**, with the latter being the focus of current research.
- **Limitations of Early Approaches**: Rule-based and RL-based text agents were **domain-specific**, requiring extensive manual design or training, and struggled to generalize.
- **Synergy of Reasoning and Acting**: **ReAct** demonstrated the power of combining reasoning and acting, allowing agents to **plan, adapt,** and **solve tasks** more effectively.
- **Reasoning as an Internal Action**: The briefing highlights reasoning as an **internal action** that manipulates the agent's internal state, leading to more informed external actions.
- **Importance of Long-term Memory**: Long-term memory enables agents to retain **knowledge, skills,** and **experiences**, leading to continuous learning and improved performance over time.
- **Agent-Computer Interface (ACI)**: The need for designing interfaces optimized for agents, considering their unique capabilities and limitations compared to humans.
- **Robustness and Human-in-the-loop**: Challenges related to ensuring agent reliability in real-world scenarios and the need for incorporating **human feedback** and oversight.
- **Benchmarking LLM Agents**: The importance of developing **comprehensive benchmarks** that evaluate agent performance on practical tasks, considering factors like **robustness** and **human interaction**.

---

## 🗣️ **Key Quotes**

- "The definition of what's 'intelligent' often changes across time."
- "Reasoning is constantly guiding the acting to plan the situation and replan the situation based on exceptions."
- "Reasoning agent is different. It's different because reasoning is an internal action for agents, and reasoning has a very special property because it's an infinite space of language."
- "Language is very different because first, you don't have to do too much because you already have rich priors from LLMs."
- "LLMs and humans are different, so should their interfaces."
- "It's more about getting simple things done reliably."

---

## 🔮 **Future Directions**

- **Training Models for Agents**: Developing models specifically trained for **agent tasks**, potentially through interaction with **simulated environments**.
- **Agent-Computer Interface Design**: Creating interfaces that **leverage the strengths** and address the weaknesses of agents compared to humans.
- **Robustness and Safety**: Developing methods to ensure agent **reliability**, prevent unintended consequences, and incorporate **human oversight**.
- **Human-Agent Collaboration**: Exploring ways for **agents and humans to collaborate effectively**, leveraging their complementary strengths.
- **Standardized Benchmarks**: Establishing **comprehensive benchmarks** that evaluate agent performance on real-world tasks, considering factors like **robustness** and **human interaction**.

---

## 🌟 **Overall Impression**

The briefing provides a **comprehensive overview** of the evolution and current state of LLM agents. It effectively highlights the key concepts, milestones, and challenges in the field, urging further research and development toward robust and reliable LLM agents capable of tackling real-world tasks. The emphasis on **long-term memory, human-agent interaction**, and the need for **robust benchmarks** points toward crucial directions for future research.


## Glossary of Key Terms 📘
- **Agent**: An intelligent system interacting with its environment through observations and actions.
- **Text Agent**: An agent that uses text as both input (observations) and output (actions).
- **LLM Agent**: An agent that uses Large Language Models (LLMs) for action selection.
- **Reasoning Agent**: An agent employing LLMs for reasoning and planning in addition to action selection.
- **Rule-Based System**: Processes input and generates responses using pre-defined rules.
- **Reinforcement Learning (RL)**: A paradigm where agents learn to act by maximizing rewards from the environment.
- **Large Language Model (LLM)**: A deep learning model trained on massive text data, capable of generating human-quality text and performing language-based tasks.
- **Chain-of-Thought (CoT) Prompting**: Guides LLMs to generate reasoning steps before answering.
- **Retrieval-Augmented Generation (RAG)**: Combines LLMs with retrievers to incorporate external knowledge.
- **Tool Use**: Integrates tools like calculators, search engines, and APIs to enhance LLMs.
- **ReAct**: A paradigm combining reasoning and acting for LLM agents.
- **Internal Action**: Primarily affects the agent's internal state (e.g., reasoning) rather than the external environment.
- **Long-Term Memory**: Mechanisms enabling agents to retain information beyond short-term context.
- **External Memory**: Uses databases or files for memory storage.
- **Episodic Memory**: Stores sequences of past experiences for recalling specific events.
- **Semantic Memory**: Stores generalized knowledge and concepts from past experiences.
- **Symbolic AI**: Represents knowledge and reasoning using symbols and logical rules.
- **Deep RL**: Optimizes rewards for complex environments, handling sensory input but needing extensive training.

