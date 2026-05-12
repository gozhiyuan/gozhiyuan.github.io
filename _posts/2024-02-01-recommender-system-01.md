---
layout: post
title: Recommender System 1 -- Introduction
subtitle:
categories: Recommender-System
tags: [YouTube]
banner: "/assets/images/banners/yuanpang-wa-valley.jpg"
---

# Recommendation System Overview

This post summarizes the first lecture in Shusen Wang's recommender system series. It covers the basic user funnel, common evaluation metrics, the industrial recommendation pipeline, and the online experimentation workflow used to decide whether a new model or strategy should be launched.

Reference: [Shusen Wang's recommender system tutorial](https://www.youtube.com/watch?v=5dTOPen28ts&list=PLvOO0btloRntAi-VnV06M1Bu0X1xljUUP&index=1).

## 1. What a Recommender System Optimizes

A recommender system is not only a model that predicts what a user may click. In a production product, it is a complete decision system that chooses which items to expose, in what order, under what constraints, and with what business goals.

In content platforms such as Xiaohongshu, the item being recommended is usually a user-generated post, often called a "note" (笔记). The system takes a user request, collects candidate notes, scores them, applies product rules, and finally displays a ranked feed.

The quality of the recommendation can be understood through a user conversion funnel:

| Stage | Meaning | Example signal |
| --- | --- | --- |
| Impression / Exposure (曝光) | The system shows an item to the user | A note appears in the feed |
| Click (点击) | The user opens the item | The user taps the note |
| Valid click | The click shows real interest | The user stays for several seconds instead of bouncing immediately |
| Completion (滑动到底) | The user consumes most or all of the content | The user scrolls to the bottom |
| Interaction | The user gives stronger feedback | Like, collect, share, comment, follow |

Different stages have different strengths. A click is easy to collect but can be noisy. A collect or share is rarer but usually expresses stronger preference. A good recommender system needs to use all of these signals instead of optimizing only one metric.

## 2. Evaluation Metrics

Recommendation metrics can be divided into short-term consumption metrics and higher-level North Star metrics. The first group helps engineers debug models quickly. The second group tells the company whether the product is becoming healthier.

### 2.1 Consumption Metrics

These metrics describe how users respond to recommendations in the current session.

| Metric | Formula | Interpretation |
| --- | --- | --- |
| CTR | clicks / impressions | Measures whether exposed items are attractive enough to open |
| Valid CTR | valid clicks / impressions | Filters accidental or low-quality clicks |
| Like rate | likes / clicks | Measures positive feedback after consumption |
| Collect rate | collects / clicks | Indicates stronger long-term interest |
| Share rate | shares / clicks | Captures social value and external distribution |
| Comment rate | comments / clicks | Captures discussion and engagement |
| Completion rate | completed reads / clicks | Measures whether users actually consume the content |

CTR is important, but it is dangerous to optimize it alone. If the ranking model only chases CTR, the feed may become clickbait-heavy. Users may click more in the short term but leave the product faster in the long term.

Completion rate also needs careful design. Long notes naturally have lower completion rates than short notes, so platforms often normalize completion by content length. A fair metric should not punish long-form content simply because it requires more time.

### 2.2 North Star Metrics

North Star metrics are the final business-level goals. They are slower to move, but they are harder to fake.

Common North Star metrics include:

- **User scale**: DAU, MAU, new user retention, active user retention.
- **Consumption**: average feed time, average number of consumed notes per user, session depth.
- **Creator ecosystem**: publishing penetration rate, average posts per creator, creator retention.
- **Platform health**: content diversity, long-tail exposure, user satisfaction, complaint rate.

A recommendation change is not necessarily good just because one model metric improves. For example, a model may increase CTR by repeatedly showing similar content, but this can reduce diversity and hurt long-term retention. Industrial recommendation is therefore a multi-objective optimization problem.

## 3. The Industrial Recommendation Pipeline

A real recommendation system must search through millions or billions of items under strict latency constraints. It cannot score every item with a large neural network. Instead, production systems use a multi-stage funnel.

```text
Item Pool -> Retrieval -> Coarse Ranking -> Fine Ranking -> Re-ranking -> Feed
```

### 3.1 Retrieval / Recall (召回)

Retrieval quickly reduces the full item pool to a few thousand candidates. It usually uses many channels at the same time because one method cannot cover every user interest.

Common retrieval channels include:

- **Collaborative filtering**: retrieve items similar to what the user or similar users consumed.
- **Two-tower models**: encode users and items into embeddings, then retrieve nearest items.
- **Follow-based retrieval**: retrieve posts from authors the user follows.
- **Content-based retrieval**: retrieve items with similar text, image, category, or topic.
- **Location-based retrieval**: useful for local services, stores, restaurants, or events.
- **Trending retrieval**: retrieve fresh or popular items to support exploration.

The goal of retrieval is high recall. It should avoid missing potentially good items, even if some candidates are noisy.

### 3.2 Coarse Ranking (粗排)

Coarse ranking scores the retrieved candidates with a relatively lightweight model. It may reduce several thousand items to several hundred. The model needs to be faster than the final ranking model because it runs on a larger candidate set.

Typical coarse ranking features include:

- user profile features;
- item popularity and freshness;
- simple user-item matching features;
- historical engagement statistics;
- retrieval channel information.

### 3.3 Fine Ranking (精排)

Fine ranking uses a more expressive model, often a large deep neural network, to score the top candidates more accurately.

The fine ranking model usually predicts multiple targets:

- probability of click;
- probability of valid read;
- expected reading time;
- probability of like, collect, share, or comment;
- probability of negative feedback.

These predictions are combined into a final ranking score. A simple version may look like:

```text
score = w1 * p_click
      + w2 * p_like
      + w3 * p_collect
      + w4 * expected_read_time
      - w5 * p_negative_feedback
```

The weights are not purely technical choices. They reflect product goals. A platform that wants deeper consumption may increase the weight on reading time and completion. A platform that wants more social distribution may give more weight to shares.

### 3.4 Re-ranking (重排)

Re-ranking considers the final list as a whole. Its job is to improve the user experience after item-level scores have already been computed.

Common re-ranking goals include:

- **Diversity**: avoid showing too many near-duplicate items.
- **Freshness**: insert newer content when appropriate.
- **Creator fairness**: avoid overexposing a small number of authors.
- **Business constraints**: insert ads, promotions, or operational content.
- **Rule-based filtering**: remove already-seen, blocked, unsafe, or low-quality content.

Algorithms such as MMR and DPP are often used to balance relevance and diversity.

## 4. A/B Testing in Recommendation Systems

Offline metrics are useful, but they are not enough. A recommender system changes user behavior, and user behavior changes future training data. Because of this feedback loop, online A/B testing is the standard way to decide whether a new strategy should be launched.

### 4.1 Basic Workflow

A typical experiment follows three steps:

1. **Offline experiment**
   - Train and evaluate the model on historical data.
   - Check metrics such as AUC, log loss, recall, NDCG, and calibration.
   - Make sure latency, memory cost, and serving cost are acceptable.

2. **Small-traffic A/B test**
   - Randomly assign a small percentage of users to the new strategy.
   - Compare the treatment group against a control group using online metrics.
   - Watch both positive metrics and guardrail metrics.

3. **Ramp-up and full launch**
   - If the result is positive and statistically reliable, increase traffic gradually.
   - Common ramps are 1%, 5%, 10%, 25%, 50%, and then 100%.
   - If metrics regress, roll back quickly.

### 4.2 Example: Testing a New Retrieval Model

Suppose the team builds a new GNN-based retrieval channel. Offline evaluation shows that it retrieves more clicked items than the old ItemCF channel.

An online experiment may be designed as:

| Group | Traffic | Strategy |
| --- | ---: | --- |
| Control | 10% | Existing retrieval channels |
| Treatment A | 10% | Add 1-layer GNN retrieval |
| Treatment B | 10% | Add 2-layer GNN retrieval |
| Treatment C | 10% | Add 3-layer GNN retrieval |

The team may track:

- CTR and valid CTR;
- average reading time per user;
- like, collect, and share rates;
- retrieval latency;
- duplicate content rate;
- next-day retention.

Possible result:

| Group | CTR diff | Reading time diff | Latency diff | Decision |
| --- | ---: | ---: | ---: | --- |
| Treatment A | +0.4% | +0.2% | +3 ms | weak improvement |
| Treatment B | +1.2% | +0.8% | +7 ms | best trade-off |
| Treatment C | +1.3% | +0.7% | +18 ms | too expensive |

In this case, Treatment B may be selected because it improves user metrics while keeping serving cost reasonable. Treatment C has similar business gains but much higher latency, so it may not be worth launching.

### 4.3 Example: Testing Ranking Score Weights

Assume the current ranking score is heavily optimized for CTR:

```text
score_old = 1.0 * p_click + 0.2 * p_like + 0.1 * p_collect
```

The team believes this creates too much clickbait. They test a new score:

```text
score_new = 0.7 * p_click
          + 0.4 * p_like
          + 0.3 * p_collect
          + 0.2 * completion_score
```

The experiment may show:

- CTR decreases by 0.3%;
- average reading time increases by 1.5%;
- collect rate increases by 1.0%;
- negative feedback decreases by 0.5%;
- 7-day retention is flat or slightly positive.

This can still be a successful experiment. A lower CTR is acceptable if the product goal is to reduce shallow clicks and improve meaningful consumption.

### 4.4 Example: Testing Re-ranking Diversity

Suppose users often see many similar posts in a row, such as five makeup tutorials with almost the same topic. The team adds a diversity rule:

> In every window of 10 feed items, no more than 3 items can come from the same narrow topic.

Metrics to monitor:

- feed CTR;
- reading time;
- topic diversity;
- long-tail item exposure;
- hide/report rate;
- retention.

A diversity experiment may reduce CTR slightly because the top-scoring items are no longer shown purely by score order. However, it may improve session depth and retention because the feed feels less repetitive.

### 4.5 Random Buckets

To compare strategies fairly, users are randomly assigned to buckets. A common method is:

```text
bucket_id = hash(user_id) % 100
```

Each bucket contains roughly 1% of users. An experiment can then reserve buckets for control and treatment:

| Bucket range | Group |
| --- | --- |
| 0-4 | Control |
| 5-9 | Treatment |
| 10-99 | Not in this experiment |

User-level bucketing is important. If the same user sometimes sees the old strategy and sometimes sees the new one, the measurement becomes noisy and the user experience becomes inconsistent.

### 4.6 Metrics and Guardrails

An A/B test should define its decision metrics before launch.

Primary metrics are the main goals:

- CTR;
- valid CTR;
- reading time per user;
- interaction rate;
- retention;
- revenue, if ads or commerce are involved.

Guardrail metrics prevent harmful launches:

- latency;
- crash rate;
- negative feedback rate;
- report rate;
- content diversity;
- creator-side exposure fairness;
- ad load or revenue loss.

For example, a model that increases reading time by 2% but doubles serving latency may still be rejected. A model that improves clicks but increases reports should also be rejected.

### 4.7 Statistical Significance and Practical Significance

An experiment needs enough traffic and enough time to produce a reliable conclusion. Engineers usually care about two kinds of significance:

- **Statistical significance**: the observed difference is unlikely to be random noise.
- **Practical significance**: the difference is large enough to matter for the business.

A tiny CTR lift may be statistically significant on a huge platform but not worth the engineering complexity. Conversely, a promising retention improvement may need a longer test because retention is a delayed metric.

### 4.8 Multi-layer Orthogonal Experiments

Large platforms run many experiments at the same time. Recall, ranking, re-ranking, ads, search, and UI teams may all need traffic. To support this, experiments are organized into layers.

Within the same layer, experiments are usually mutually exclusive. For example, two ranking experiments should not affect the same user at the same time, because it would be hard to know which change caused the result.

Across different layers, experiments can be orthogonal. A user may be in:

- the treatment group for a recall experiment;
- the control group for a ranking experiment;
- the treatment group for a UI experiment.

This design allows many teams to experiment simultaneously while keeping attribution manageable.

### 4.9 Holdout Buckets

A holdout bucket is a clean group of users that does not receive new experimental changes for a longer period. It is used to measure the total impact of many launches.

Example:

- 10% of users stay in the holdout group.
- 90% of users receive normal launches.
- After two months, compare the 90% launch group against the 10% holdout group.

This helps answer a broader question:

> What was the total contribution of the recommendation team this quarter?

Without a holdout group, it is hard to measure cumulative impact because many small experiments may interact with each other.

### 4.10 Reverse Experiments

Some metrics, such as retention and creator ecosystem health, need long observation windows. But engineers also want to launch successful strategies quickly and release experiment buckets.

A reverse experiment solves this conflict:

- Launch the new strategy to most users.
- Keep a small reverse bucket on the old strategy.
- Continue comparing new versus old over several weeks or months.

This is especially useful when the short-term metrics are positive but the team wants to verify long-term effects.

## 5. Key Takeaways

- A recommender system is a full product decision pipeline, not just a prediction model.
- The user funnel moves from impression to click, valid consumption, and stronger interactions.
- CTR is useful but incomplete; recommendation quality must include depth, retention, diversity, and ecosystem health.
- Industrial systems use retrieval, coarse ranking, fine ranking, and re-ranking to balance scale and accuracy.
- A/B testing is the final judge for production changes because offline metrics cannot fully predict user behavior.
- Good experiments need random buckets, clear metrics, guardrails, statistical confidence, and long-term validation.
