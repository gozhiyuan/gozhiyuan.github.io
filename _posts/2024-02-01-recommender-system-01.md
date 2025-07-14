---
layout: post
title: Recommender System 1 -- Introduction
subtitle:
categories: Recommender-System
tags: [YouTube]
banner: "/assets/images/banners/yuanpang-wa-valley.jpg"
---

# 📊 Recommendation System Overview

This guide dives into the essentials of recommendation systems, with a focus on key metrics, experimentation methods, and the underlying architecture. Whether you're a beginner or looking to deepen your understanding of recommendation strategies, this will give you a structured breakdown of critical components.

Follow this awesome [tutorial](https://www.youtube.com/watch?v=5dTOPen28ts&list=PLvOO0btloRntAi-VnV06M1Bu0X1xljUUP&index=1) by 
Shusen Wang

## 📈 01_Basics: Key Metrics and Experimentation

### Core Metrics
Core metrics in recommendation systems center around **user engagement**. They are pivotal for deciding if a new recommendation strategy is worth deploying. While increasing engagement is often the primary goal, it's essential to balance it with other indicators to ensure a well-rounded approach.

### Non-Core Metrics
Metrics like **click-through rates (CTR)** and **interaction behaviors** serve as indicators of user interest and platform health. While not the main focus for algorithm engineers, these metrics provide insight into user satisfaction and platform stability.


- **Core Metrics**: Daily Active Users (DAU), retention, and content consumption are standard measures. For User-Generated Content (UGC) platforms, the number of user posts is also vital.
- **Non-Core Metrics**: Metrics like click-through rate (CTR) and interaction rate provide additional insight but are less critical.
- **Cold Start Metrics**: Cold start evaluations focus on specific aspects:
  - **Author-Side**: Measures like posting penetration rate capture how new content gets published.
  - **User-Side**: Engagement measures reflecting user responses to new items.
  - **Content-Side**: Evaluates content health, including ratios of high-quality items.


### A/B Testing
A/B testing is crucial for evaluating new features. By segmenting users uniformly, A/B tests reveal the real impact of changes. Running multiple tests simultaneously presents challenges, but layered experimentation can mitigate these issues.

- **Random Bucketing**: Users are divided into random groups using hash functions.
- **Stratified Experiments**: Different layers (e.g., recall, ranking, reranking) are tested independently for nuanced insights.
- **Holdout Buckets**: A control group is used to benchmark new models against existing ones.
- **Rollout and Reversal**: Gradually increases traffic to a new model if initial results are positive, with the option to reverse if needed.
- **Cold Start Challenges**: A/B testing can be misleading for cold start strategies. For instance, showing more new items may reduce engagement rates short-term, even if it improves overall diversity and content creation.

### Layered Experimentation
Organizing experiments into layers—like **recall**, **ranking**, and **reranking**—enables testing of multiple features in parallel. This structure helps isolate effects, making tests more efficient.

### Lagging Indicators and Reversal Experiments
**Lagging indicators** can delay the assessment of changes. Reversal experiments help track long-term impacts, ensuring new strategies don’t have hidden downsides.

## 🛠 01_Basics_02: The Recommendation System Pipeline

### Overview of the Pipeline
The recommendation pipeline filters a large pool of items down to a tailored set for each user. Key stages include **recall**, **coarse ranking**, **fine ranking**, and **reranking**.

### Recall
At the recall stage, techniques like **collaborative filtering** and **dual-tower models** retrieve relevant items. This stage reduces the initial pool to a manageable set for further processing.

### Ranking (Coarse and Fine)
Coarse and fine ranking stages apply models to score and refine item lists. These stages progressively improve the selection using **neural networks** and **statistical features** to predict user engagement.

### Reranking
The final stage, reranking, optimizes user experience through techniques like **diversity sampling** and **ad insertions**, ensuring the recommendations are both relevant and engaging.



### 🧬 Two-Tower Model Training
A two-tower model, commonly used in the recall stage, matches users to items using embeddings. Training approaches include:

- **Pointwise**: Treats each sample independently, using binary classification.
- **Pairwise**: Compares positive and negative samples to push user-item similarity for positive items higher.
- **Listwise**: Takes one positive and multiple negatives to maximize relevance across all options.

### 🔄 Two-Tower Model Sample Selection
Selecting positive and negative samples for two-tower models is essential for accurate recommendations:

- **Positive Samples**: Items that the user clicked or engaged with.
- **Negative Samples**: Items ignored by the user; includes easy negatives (not retrieved items) and hard negatives (retrieved but filtered out).
- **Common Mistakes**: Avoid using shown-but-unselected items as negatives in recall models, as this can reduce recommendation quality.

### 🚀 Two-Tower Model Serving
Once trained, two-tower models are optimized for online serving:

- **Item Embeddings**: Pre-computed and stored in a vector database for efficient nearest neighbor search.
- **User Embedding Calculation**: Generated in real-time for each user session to retrieve the most relevant items based on cosine similarity.

### ❄️ Tackling the Cold Start Problem
The cold start problem is a challenge in recommending new items with limited interaction data, especially for UGC platforms with high item turnover.

- **Why New Items Need Attention**:
  - **Limited Interactions**: Sparse data makes it hard to gauge preferences through collaborative filtering.
  - **Encouraging Creation**: Providing exposure to new items motivates content creators.
- **Cold Start Strategies**:
  - **Content-Based**: Leverages item features (e.g., images, text) to recommend.
  - **Category/Keyword-Based**: Uses item categories or keywords for relevant item retrieval.
  - **Look-Alike Recommendation**: Finds similar users based on features to recommend similar items.
- **A/B Testing for Cold Start**: Specialized tests to evaluate strategies like content-based and look-alike recommendations.

### 🚀 Strategies to Improve Recommender Systems

- **Boosting Recall**: Add or refine recall models for a broader, more relevant item pool.
- **Enhancing Ranking Models**: Improve rough and fine ranking models to predict user engagement.
- **Increasing Diversity**: Diversify results at every stage, from recall to ranking, to keep user interest high.
- **Catering to Specific User Groups**: Tailor recommendations for new or inactive users to enhance their experience.
- **Utilizing Social Interactions**: Leverage follows, shares, and comments to drive engagement and improve recommendations.

By mastering these strategies, your recommender system can balance relevance, diversity, and engagement, ultimately enhancing user satisfaction! 🌟
