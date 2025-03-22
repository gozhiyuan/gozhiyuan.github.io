---
layout: post
title: Recommender System 3 -- Ranking
subtitle:
categories: YouTube
tags: [recommender-system]
banner: "/assets/images/banners/yuanpang-wa-valley.jpg"
---

This guide dives into the essentials of recommendation systems, with a focus on key metrics, experimentation methods, and the underlying architecture. Whether you're a beginner or looking to deepen your understanding of recommendation strategies, this will give you a structured breakdown of critical components.

Follow this awesome [tutorial](https://www.youtube.com/watch?v=5dTOPen28ts&list=PLvOO0btloRntAi-VnV06M1Bu0X1xljUUP&index=1) by 
Shusen Wang

## 🏆 03_Rank_01: Ranking Models

**Ranking models** predict user engagement metrics, such as CTR and like rates, using features like item context, user behavior, and environmental cues. These models prioritize engagement using cross-entropy loss during training.

## 🎛 03_Rank_02: Multi-gate Mixture-of-Experts (MMoE) for Ranking

The **MMoE architecture** uses specialized expert networks to predict specific user behaviors, while gate networks dynamically adjust weights to balance these predictions.

## 📺 03_Rank_04: Modeling Video Playtime and Completion

This section models **playtime** and **completion rates** to understand user interaction with videos, optimizing engagement beyond binary click predictions.

## 📐 03_Rank_05: Data Serving Architecture

Data serving architecture ensures smooth interactions between recall and ranking servers, feature stores, and model serving systems. This pipeline handles user requests in real-time.

## ⚖️ 03_Rank_06: Coarse Ranking vs. Fine Ranking

Coarse ranking aims for **efficiency**, while fine ranking focuses on **precision**. Three-tower models balance these needs by using static and dynamic features.

## 🧮 04_Cross_01: Factorized Machines (FM) for Feature Interaction

FM models capture **second-order feature interactions** efficiently, reducing computational costs while enhancing recommendation quality.

## 🌐 04_Cross_02: Deep & Cross Networks (DCN) for Feature Interaction

**Cross layers** in DCN enable the modeling of complex feature interactions without dramatically increasing model parameters, providing a rich feature set for recommendation.

## 🎤 04_Cross_03: LHUC for Feature Adaptation in Speech Recognition

LHUC (Linear Hidden Unit Contribution) adapts features in speech recognition, using **Hadamard products** to modulate feature representations based on speaker characteristics.

## 📜 05_LastN_02: Deep Interest Network (DIN) for User History Modeling

DIN uses an **attention mechanism** to weigh items based on their relevance to the candidate item, allowing a more personalized interpretation of user history.

## 📅 05_LastN_03: SIM: Modeling Longer Sequences & Time Information

The SIM model integrates **longer user histories** and **time embeddings** to capture evolving user interests, enhancing long-term preference predictions.

--- 

## 📊 Reranking for Diversity and User Experience

## 🎨 The Importance of Diversity
Diversity isn't just a buzzword; it significantly boosts user engagement and overall platform health. Providing a range of items helps prevent boredom, keeping users engaged and driving retention. The right mix can make all the difference! 

## 🧩 Measuring Item Similarity
Item similarity is a key factor in enhancing diversity. This section dives into different ways to measure similarity between items, which sets the stage for algorithms that promote a broader, more engaging user experience.

## 🔄 Methods for Enhancing Diversity
Post-processing techniques come in handy for diversifying results after initial ranking. This reranking step optimizes user experience by considering the lineup as a whole, not just individual item scores.

## 📐 MMR Algorithm for Diversity
The **Maximal Marginal Relevance (MMR)** algorithm is a popular choice for promoting diversity. MMR balances item relevance with dissimilarity, so that each new item chosen is different enough to add variety, without sacrificing quality.

## ⚖️ Combining MMR with Sliding Windows and Rule Constraints
Implementing MMR gets practical with techniques like sliding windows, which limit similarity checks, and rule constraints to ensure diversity while meeting business needs.

## 📏 Determinantal Point Process (DPP) for Diversity
DPP takes a sophisticated approach, considering overall diversity of items rather than pairwise similarities. Concepts like parallelotopes and volume calculations reveal how determinants can measure diversity within the entire set.

---

## 🎥 Content-based Item Representation and Methods for Diversity

## 🖼️ Content-based Item Representation
Content-based features like images and text generate item vectors that help calculate similarity. This allows diverse recommendations based on actual content characteristics, providing a more nuanced reranking process.

## 🔀 Methods for Enhancing Diversity
Reiterating the importance of post-processing, this section underscores the focus on balancing relevance and diversity at the reranking stage to improve the recommendation experience.

---

## 🔄 Maximal Marginal Relevance (MMR) Algorithm

## 🔍 MMR Algorithm for Diversity
A step-by-step explanation of how MMR works, providing clarity on how it balances relevance with diversity.

## 🪟 Sliding Window Technique
Sliding windows help make MMR more scalable, limiting similarity calculations for large datasets by focusing on smaller chunks at a time.

---

## 📐 Mathematical Foundations of Determinantal Point Process (DPP)

## 🧮 Parallelotopes and Volume Calculation
Parallelotopes, or geometric representations of item sets, quantify diversity by calculating volume—larger volumes mean higher diversity.

## 📏 Connecting Volume to Determinants
This section explains how the volume of parallelotopes links to matrix determinants, offering a mathematical approach to measuring diversity within item sets.

---

## 💻 Applying Determinantal Point Process (DPP) in Recommendation Systems

## 🧑‍💻 DPP Formulation for Recommendation
Learn how DPP can select items that balance relevance with diversity, ensuring users are presented with a rich variety of content.

## 🏃 Greedy Algorithm for Solving DPP
This section introduces a greedy algorithm to solve DPP efficiently, making it suitable even for large datasets.

## 🚀 Computational Efficiency of the Algorithm
A deep dive into the computational complexity of the greedy algorithm, showcasing its practicality for real-time recommendations.

---

## ❄️ Evaluating and Experimenting with Item Cold Start Strategies

## 📈 Evaluating Item Cold Start Strategies
Introducing new items can be tricky! A balanced evaluation considers three areas: author-side metrics, user-side metrics, and content-side metrics.

- **Author-Side Metrics**: Focus on content creators' metrics, emphasizing the importance of early exposure.
- **User-Side Metrics**: Evaluates user experience to understand the impact on engagement.
- **Content-Side Metrics**: Ensures healthy diversity in the platform's content.

## 🧪 Experiment Design for Cold Start
Designing experiments for cold start strategies can be challenging. This section explains different methodologies for user and author-side experiments, setting up for robust testing.

---

## 🧑‍💻 Improving User Behavior Sequence Modeling and Online Learning

## 🔄 Improving User Behavior Sequence Modeling
Explore techniques to enhance user behavior models, such as increasing sequence length, using filtering strategies, and incorporating features beyond item IDs.

## 🌐 Online Learning for Continuous Model Updates
Online learning is crucial to keep up with evolving user behavior. This section compares full updates with incremental updates and discusses their resource implications.

---

## 📊 Boosting Metrics through Diversity and Personalized Strategies

## 💡 Multiple Strategies for Metric Improvement
Improving key metrics can come from enhancing model architectures, focusing on diversity, and creating personalized approaches for specific user groups.

## 🧩 Diversity in Ranking and Recall
Learn strategies for diversifying both the ranking and recall stages, using algorithms like MMR and DPP to optimize for a balanced user experience.

---

## 🔄 Leveraging Special Content Pools for Specific User Groups

## 🎯 Creating Special Content Pools
This section introduces specialized content pools, tailored for unique user groups like new or inactive users, to improve engagement.

## 🎒 Recall from Special Content Pools
Dual-tower models can help retrieve items from these pools, adding curated content to personalized recommendations.

---

## 📲 Leveraging Follow, Share, and Comment Interactions for Engagement

## 👥 Utilizing Follow Relationships
Follower dynamics boost engagement. This section shares strategies for promoting follows through ranking tweaks and specialized content.

## 📢 Understanding the Value of Shares
Shares drive external traffic! Identifying high-referral users and adapting strategies can amplify reach.

## 💬 The Role of Comments in Content Engagement
Comments often reflect content quality. Promoting comments can enhance engagement and support new content. 

---

Each of these insights contributes to creating a richer, more engaging recommendation system. Diversifying recommendations and leveraging interactions can optimize user experience and retention, helping your platform stand out! 🚀
