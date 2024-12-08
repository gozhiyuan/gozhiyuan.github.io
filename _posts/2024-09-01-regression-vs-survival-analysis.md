---
layout: post
title: Regression vs. Survival Analysis 🚀
subtitle:
categories: ML
tags: [regression, survival-analysis, ml]
banner: "/assets/images/banners/yuanpang-wa-sky.jpg"
---


## Predicting Customer Churn: Regression vs. Survival Analysis 🚀

When it comes to predicting **customer churn**, the choice between **regression** and **survival analysis** depends on your data and objectives. While regression models may seem simpler, survival analysis is often better suited for time-to-event problems, especially when dealing with censored data. Let’s dive into this comparison! 🔍

---

## 🎯 What’s the Problem?

Imagine you want to predict **how many days a customer has left before churning**, with the following scenarios:

1. **Some customers have churned** (we know their churn time). ✅
2. **Some customers haven’t churned yet** (they’re still active, so their churn time is unknown). ❓

### Can Regression Models Work? 🤔
Yes, but only under specific conditions:
- All customers must have churned (no censored data).  
- You only need a **single point estimate** of churn time.  

If these conditions aren’t met, you might face biased results and limited insights.

---

## 💡 Regression Models: The Basics

### How They Work:
1. Train a regression model with **churned customers only**.
2. Predict a **single point estimate** for how many days are left before a customer churns.

### Pros ✅:
- **Simple and direct**: Predicts a single number (e.g., 120 days left).
- **Easy to implement**: Widely available algorithms (e.g., linear regression, random forest).

### Cons ❌:
- **Ignores censored data**: Can’t learn from customers who haven’t churned yet.
- **Biased predictions**: Focuses on shorter churn times, as longer times (from censored data) are excluded.
- **No uncertainty**: Doesn’t provide confidence intervals or probabilities.

### When to Use Regression:
- All customers have churned (no censored data).  
- You only need a **quick and approximate estimate** of churn time.

---

## 🔬 Survival Analysis: The Powerhouse for Time-to-Event Data

### How It Works:
Survival analysis models the **time until an event occurs** (e.g., churn), accounting for censored data. It predicts:
- **Survival probability**: Likelihood a customer survives (doesn’t churn) beyond time \(t\).
- **Hazard function**: Risk of churn at a specific time.

### Pros ✅:
- **Handles censored data**: Learns from both churned and active customers.
- **Probabilistic insights**: Provides survival probabilities and hazard rates.
- **Dynamic predictions**: Tracks how churn risk evolves over time.
- **Confidence intervals**: Offers uncertainty estimates for predictions.

### Cons ❌:
- **Complexity**: More difficult to implement and interpret than regression.
- **Probabilistic outputs**: Requires extra work to convert probabilities into actionable insights.

---

## ⚖️ Regression vs. Survival Analysis: A Comparison Table 📝

| **Feature**                         | **Regression**                                      | **Survival Analysis**                                |
|--------------------------------------|-----------------------------------------------------|------------------------------------------------------|
| **Prediction Type**                  | Single point estimate (e.g., 120 days left)         | Probability of churn over time                      |
| **Handles Censored Data?**            | ❌ No                                               | ✅ Yes                                               |
| **Uncertainty Quantification**       | ❌ No                                               | ✅ Yes (confidence intervals, risk scores)           |
| **Adaptability**                     | ❌ Static                                           | ✅ Dynamic (updates with time)                      |
| **Accuracy with Complex Data**       | ❌ Lower (biased if censored data exists)            | ✅ Higher (especially with censored data)            |
| **Interpretation**                   | Simple (single number)                              | Probabilistic (requires interpretation)             |
| **Actionable Insights**              | Limited to predicted days                           | Rich insights (risk over time, optimal intervention) |

---

## 🔍 How Survival Analysis Predicts Days Left

### Example: Survival Function \(S(t)\)
The survival function \(S(t)\) estimates the probability that a customer **survives (doesn’t churn)** beyond time \(t\):
- \(S(90) = 0.7\): The customer has a 70% chance of staying active after 90 days.

### Predicting Remaining Time:
1. **Expected Remaining Time**:
   Use the survival curve to compute the expected time left until churn:
   $ [
   E(T) = \int_0^\infty S(t) \, dt
   ] $
2. **Median Churn Time**:
   Find the time point where \(S(t) = 0.5\), i.e., a 50% chance of churn.

---

## 🛠️ Practical Scenarios

### **Daily Predictions**
Suppose you want to predict **how many days a customer has left** each day:

- **Regression Model**:
  - Predicts: "This customer will churn in 90 days."
  - Limitation: Static and ignores censored data.
- **Survival Analysis**:
  - Predicts: 
    - "70% chance the customer survives 90 days."  
    - "Median time to churn: 110 days."  
    - "50% chance the customer churns within 20 days."

---

## 🎯 Key Takeaways

1. **Use Regression** if:
   - All customers have churned (no censored data).
   - You only need a rough, single-point estimate.
2. **Use Survival Analysis** if:
   - You have censored data (active customers).
   - You need accurate, dynamic predictions and actionable insights.
   - Churn risk evolves over time.

---

## Survival Analysis Techniques: Kaplan-Meier & Cox Proportional Hazards 📊

When analyzing **time-to-event data** (e.g., customer churn), two powerful techniques are the **Kaplan-Meier Survival Curve** and the **Cox Proportional Hazards Model**. Let’s explore their key concepts, use cases, and Python implementations. 🚀

## 1️⃣ Kaplan-Meier Survival Curve 📈

### What Is It? 🤔
The **Kaplan-Meier estimator** is a non-parametric method to calculate the **survival probability over time**. It makes **no assumptions** about the underlying survival distribution and is excellent for visualizing and exploring survival patterns.

### Key Concepts:
1. **Survival Function \(S(t)\)**:
   - \(S(t) = P(T > t)\): The probability that a subject survives beyond time \(t\).
   
2. **Event Times and Censoring**:
   - Accounts for both **events** (e.g., churn) and **censored data** (e.g., active customers).

### Python Implementation: Kaplan-Meier 🎯
```python
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

# Sample dataset
data = pd.DataFrame({
    'customer_id': [1, 2, 3, 4, 5, 6],
    'time_to_event': [5, 6, 6, 2, 4, 3],  # Days until churn or censoring
    'event_occurred': [1, 0, 1, 1, 0, 1]  # 1 = churned, 0 = censored
})

# Kaplan-Meier Fitter
kmf = KaplanMeierFitter()

# Fit the model
kmf.fit(durations=data['time_to_event'], event_observed=data['event_occurred'])

# Plot the survival function
plt.figure(figsize=(8, 6))
kmf.plot_survival_function()
plt.title("Kaplan-Meier Survival Curve")
plt.xlabel("Time (days)")
plt.ylabel("Survival Probability")
plt.grid(True)
plt.show()

# Output survival probabilities
print("Survival Probabilities:\n", kmf.survival_function_)
```


### Pros ✅:
- Simple and effective for visualizing survival data.
- Handles censored data seamlessly.
- Easy to explain and interpret.

### Cons ❌:
- Doesn’t account for covariates (e.g., customer age, spending).
- Limited predictive power for complex scenarios.

---

## 2️⃣ Cox Proportional Hazards Model 📊

### What Is It? 🤔
The **Cox Proportional Hazards model** is a **semi-parametric method** that estimates how covariates (features) influence the **hazard rate** (instantaneous risk of an event). It provides interpretable results and allows for feature-based survival predictions.

### Key Concepts:

1. **Hazard Function**:
   $[
   h(t|X) = h_0(t) \exp(\beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p)
   ]$
   - \( h_0(t) \): Baseline hazard (shared across all individuals).
   - \( X_i \): Covariates (e.g., customer behavior).
   - \( \beta_i \): Coefficients reflecting the effect of covariates.

2. **Proportional Hazards Assumption**:
   - The ratio of hazard rates between two individuals is constant over time.

---

### Python Implementation: Cox Proportional Hazards 🎯

```python
from lifelines import CoxPHFitter
import pandas as pd
import matplotlib.pyplot as plt

# Sample dataset
data = pd.DataFrame({
    'customer_id': [1, 2, 3, 4, 5, 6],
    'time_to_event': [5, 6, 6, 2, 4, 3],  # Days until churn or censoring
    'event_occurred': [1, 0, 1, 1, 0, 1],  # 1 = churned, 0 = censored
    'age': [25, 40, 35, 50, 23, 30],       # Example covariate
    'spending_score': [60, 80, 70, 50, 90, 65]  # Example covariate
})

# Cox Proportional Hazards Fitter
cph = CoxPHFitter()

# Fit the model
cph.fit(data, duration_col='time_to_event', event_col='event_occurred')

# Summary of the model
cph.print_summary()

# Plot the coefficients
cph.plot()
plt.title("Cox Model Coefficients")
plt.show()

# Predict survival probabilities for new data
new_data = pd.DataFrame({
    'age': [28, 45],
    'spending_score': [75, 85]
})
survival_probabilities = cph.predict_survival_function(new_data)
print("Predicted Survival Probabilities:\n", survival_probabilities)

# Visualize survival predictions
survival_probabilities.plot()
plt.title("Predicted Survival Curves for New Customers")
plt.xlabel("Time (days)")
plt.ylabel("Survival Probability")
plt.grid(True)
plt.show()
```

### Pros ✅:
- Incorporates covariates to model individualized survival predictions.
- Provides interpretable coefficients for feature effects.
- Handles censored data effectively.

### Cons ❌:
- Assumes proportional hazards (may not hold in some datasets).
- Requires more data and preprocessing than Kaplan-Meier.

---

## 🔍 Kaplan-Meier vs. Cox Proportional Hazards

| **Feature**               | **Kaplan-Meier**                           | **Cox Proportional Hazards**             |
|---------------------------|---------------------------------------------|------------------------------------------|
| **Assumptions**           | Non-parametric, no assumptions about data  | Proportional hazards assumption          |
| **Handles Covariates?**   | ❌ No                                       | ✅ Yes                                   |
| **Output**                | Survival probability curve                 | Hazard rates, survival probabilities     |
| **Use Case**              | Exploratory analysis, visualizing survival | Modeling survival with feature effects   |

---

## 🧠 Key Takeaways

### Use **Kaplan-Meier** for:
- Simple survival analysis and visualization.
- Exploring overall survival patterns.

### Use **Cox Proportional Hazards** for:
- Advanced modeling with covariates (e.g., customer behaviors).
- Predicting individualized survival probabilities.
