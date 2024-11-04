---
layout: post
title: Recommender System 2 -- Retrieval
subtitle:
categories: Course-TLDR recommender-system
tags: [recommender-system]
banner: "/assets/images/banners/yuanpang-wa-valley.jpg"
---

This guide dives into the essentials of recommendation systems, with a focus on key metrics, experimentation methods, and the underlying architecture. Whether you're a beginner or looking to deepen your understanding of recommendation strategies, this will give you a structured breakdown of critical components.

Follow this awesome [tutorial](https://www.youtube.com/watch?v=5dTOPen28ts&list=PLvOO0btloRntAi-VnV06M1Bu0X1xljUUP&index=1) by 
Shusen Wang

## 🔄 02_Retrieval_01: Item-based Collaborative Filtering (ItemCF)

### Principles of ItemCF
ItemCF bases recommendations on item similarity, determined by user interaction overlaps. This approach aligns well with users' prior interests.

### Implementation of ItemCF Recall
The ItemCF recall uses two indexes: **user-to-item** and **item-to-item** lists. This enables efficient retrieval and scoring of items for a personalized experience.

## 👥 02_Retrieval_03: User-based Collaborative Filtering (UserCF)

### Principles of UserCF
UserCF leverages **user similarities** to predict a user's interests. By comparing users' preferences, it provides recommendations based on shared interests.

### Calculating User Similarity
User similarity is calculated with measures like **Jaccard similarity**, which considers overlapping items between users. This helps personalize recommendations further.

### Mitigating Popularity Bias
Techniques like **inverse document frequency weighting** adjust similarity scores to give more importance to less popular items, reducing bias.

## 🔍 02_Retrieval_05: Matrix Factorization & Dual-Tower Models for Recall

### Dual-Tower Models as an Improvement
Dual-tower models improve upon matrix factorization by incorporating item and user attributes beyond IDs, providing a richer, more nuanced recommendation.

### Online Recall with Approximate Nearest Neighbor Search (ANNS)
To handle vast item pools, dual-tower models use ANNS techniques, which allow efficient item retrieval based on similarity.

## 🎓 02_Retrieval_06: Training Dual-Tower Models for Recall

### Training Methods: Pointwise, Pairwise, and Listwise
- **Pointwise**: Treats recall as a binary classification task.
- **Pairwise**: Ranks positive samples higher than negatives.
- **Listwise**: Considers all candidate items, optimizing entire lists for engagement.

#### **Two-Tower Models**
- **Architecture**: Separate **user tower** and **item tower** generate embeddings for users and items, respectively.
- **Training**:
    - **Pointwise**: Treats interactions independently as binary classification.
    - **Pairwise**: Encourages positive item embeddings to be more similar to the user embedding than negative items.
    - **Listwise**: Optimizes ranking for a list of items.
- **Sample Selection**:
    - **Positive Samples**: Typically interactions like clicks or purchases.
    - **Negative Samples**: Items the user didn’t interact with, categorized as:
        - **Easy negatives**: Items never retrieved.
        - **Hard negatives**: Items retrieved but filtered out by later ranking.
    - **Sampling Techniques**: 
        - **Uniform**: Equal chance for any item to be a negative sample.
        - **Non-uniform**: Reduces bias toward popular items by adjusting sampling based on item popularity.
        - **Batch**: Samples negatives from the batch of positive samples.
- **Serving**: Item embeddings stored in a vector database, allowing fast nearest neighbor search to retrieve the best-matching items.
- **Continuous Learning**: User interests change rapidly, so models should update often.
    - **Full updates**: Retrains the network daily with recent data.
    - **Incremental updates**: Refreshes parameters with streaming data.

#### **Beyond Two-Tower Models**

Several other models and techniques enhance retrieval performance:

- **Item-to-Item (I2I)**: Finds items similar to those the user has interacted with, like:
    - **U2I2I**: Items similar to past items the user engaged with.
    - **U2A2I**: Items by authors the user follows.
    - **U2A2A2I**: Extends U2A2I to authors similar to those the user follows.
- **Deep Retrieval**: Represents items as paths in a hierarchy and uses neural networks to estimate interest.
- **Content-Based Retrieval**: Uses item features (images, text) for similarity-based retrieval.
    - **CLIP Model**: Jointly embeds images and text for nuanced content-based recommendations.
- **Category/Keyword-Based Retrieval**: Quickly retrieves items within categories or based on keywords, personalized by user profile.
- **Location-Based Retrieval**: Ideal for services like food delivery or event recommendations, where geography matters.
- **Author-Based Retrieval**: Recommends items from authors the user follows.
- **Cache-Based Retrieval**: Speeds up response for repeat queries by caching recent results.
- **Exposure Filtering**: Uses Bloom filters or other methods to avoid recommending items users have already seen.


## 📥 02_Retrieval_07: Handling Positive & Negative Samples

### Positive Sample Selection
By oversampling less popular items, we can mitigate the **popularity bias** and enrich recommendations.

### Negative Sample Selection
Negative sampling ranges from random to relevance-weighted, balancing exploration and accuracy in item retrieval.

## 💾 02_Retrieval_08: Online Recall & Model Updating

Offline storage of **item embeddings** in a vector database enables fast, real-time retrieval. User embeddings are generated online to match with these item vectors effectively.

## 🔎 02_Retrieval_09: Enhancing Dual-Tower Models with Self-Supervised Learning

**Self-supervised learning** improves representations for long-tail items, addressing the imbalance where popular items dominate recommendations. Techniques like **feature masking** force the model to learn from limited data.

## 🛤 02_Retrieval_10: Path-based Recall

Path-based recall uses **knowledge graph paths** to link items, enhancing relevance. Users’ interactions with these paths help in creating deeper item associations.

## 📚 02_Retrieval_11: Diverse Recall Strategies

Strategies like **author-based recall** retrieve items based on user-author relationships, while **location-based recall** tailors recommendations to geographical contexts.

## 🛠 02_Retrieval_12: Bloom Filters for Efficient Recall

Bloom filters efficiently store item presence data, using hash functions and binary vectors. They trade minimal storage space for a low false-positive rate, which is acceptable for many recommendation applications.



