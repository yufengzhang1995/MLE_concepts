# 📘 Machine Learning Algorithms – Interview Prep

---

## ✅ What is logistic regression?

Logistic Regression is a classification algorithm that models the probability of a binary class using a sigmoid activation over a linear function.

**Forward Calculation:**

$$
z = \mathbf{w}^T \mathbf{x} + b \\
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

**Loss Function (Binary Cross Entropy):**

$$
\mathcal{L} = -\left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]
$$

---

## ✅ What is K-Nearest Neighbors (KNN)?

KNN is a non-parametric, instance-based learning algorithm. Given a test input, it finds the **K closest points** in the training data using a distance metric (e.g., Euclidean distance), and predicts the label by **majority vote** (classification) or **average** (regression).

- No training phase.
- Sensitive to choice of K and distance metric.

---

## ✅ What is decision tree learning?

A Decision Tree recursively splits the data based on feature values to maximize information gain.

**Splitting criteria examples:**

- Gini impurity:
  $$
  Gini = 1 - \sum_{i=1}^{C} p_i^2
  $$
  
- Entropy:
  $$
  Entropy = -\sum_{i=1}^{C} p_i \log_2(p_i)
  $$

---

## ✅ What is random forest, and how does it differ from gradient boosting?

- **Random Forest**: Ensemble of decision trees trained in parallel using **bootstrap sampling (bagging)**. Final prediction is via **majority vote or average**.

- **Gradient Boosting**: Trees are trained **sequentially**, each correcting the previous tree’s errors. It uses gradient descent on a loss function.

**Difference:**

- Random Forest = Parallel + Averaging  
- Gradient Boosting = Sequential + Weighted sum

---

## ✅ What is the difference between bagging and boosting?

- **Bagging** (Bootstrap Aggregating):
  - Trains models in **parallel** on different data subsets.
  - Reduces **variance**.
  - Example: Random Forest.

- **Boosting**:
  - Trains models **sequentially** to correct errors.
  - Reduces **bias** and **variance**.
  - Example: AdaBoost, XGBoost.

---

## ✅ What is K-Means clustering, and how does it work?

K-Means is an unsupervised learning algorithm that partitions data into K clusters to minimize intra-cluster variance.

**Objective function:**

$$
\underset{S}{\arg\min} \sum_{i=1}^{K} \sum_{x \in S_i} \|x - \mu_i\|^2
$$

**Algorithm:**
1. Randomly initialize K centroids.
2. Assign each point to nearest centroid.
3. Update centroids as mean of assigned points.
4. Repeat until convergence.

---

## ✅ What is Support Vector Machine (SVM)?

SVM finds a hyperplane that maximizes the margin between classes.

**Linear SVM optimization:**

$$
\text{maximize } \frac{2}{\|\mathbf{w}\|}, \quad \text{subject to } y_i(\mathbf{w}^T x_i + b) \geq 1
$$

- For non-linear data, kernel functions (e.g., RBF) can be used.

---

## ✅ What is Bayesian learning?

Bayesian learning treats model parameters as random variables and updates beliefs using Bayes' theorem:

$$
P(\theta | D) = \frac{P(D | \theta) P(\theta)}{P(D)}
$$

- Incorporates prior beliefs.
- Posterior reflects updated belief after observing data.

---

## ✅ What is the difference between MAP and MLE?

- **MLE (Maximum Likelihood Estimation):**

$$
\hat{\theta}_{\text{MLE}} = \arg\max_{\theta} P(D | \theta)
$$

- **MAP (Maximum A Posteriori):**

$$
\hat{\theta}_{\text{MAP}} = \arg\max_{\theta} P(D | \theta) P(\theta)
$$

MAP includes a prior \( P(\theta) \), while MLE does not.

---

## ✅ What is Naive Bayes and when is it effective?

Naive Bayes is a probabilistic classifier that applies Bayes’ Theorem under the **naive** assumption that features are **conditionally independent** given the class.

**Prediction Rule:**

$$
P(y | x_1, ..., x_n) \propto P(y) \prod_{i=1}^{n} P(x_i | y)
$$

**Effective when:**
- Features are independent.
- In high-dimensional tasks like spam detection, text classification.

---

## ✅ What are the assumptions behind linear regression?

1. **Linearity**: Linear relationship between features and target.
2. **Independence**: Observations are independent.
3. **Homoscedasticity**: Constant variance of errors.
4. **Normality**: Errors are normally distributed.
5. **No multicollinearity**: Predictors are not highly correlated.

---

## ✅ How does ridge regression differ from lasso regression?

Both are regularized versions of linear regression:

- **Ridge (L2 Regularization):**

$$
\min_{\beta} \|y - X\beta\|^2 + \lambda \|\beta\|_2^2
$$

- **Lasso (L1 Regularization):**

$$
\min_{\beta} \|y - X\beta\|^2 + \lambda \|\beta\|_1
$$

- Lasso encourages **sparsity** (feature selection).
- Ridge shrinks coefficients but **keeps all**.

---

## ✅ What are the strengths and weaknesses of tree-based methods?

**Strengths:**
- Easy to interpret
- Handle both categorical and numeric data
- No feature scaling needed
- Non-linear modeling

**Weaknesses:**
- Overfitting (especially deep trees)
- Sensitive to data variations
- Less accurate as standalone models

---

## ✅ What are ensemble methods, and why do they work?

**Ensemble methods** combine multiple models to improve generalization and performance.

**Types:**
- Bagging (Random Forest)
- Boosting (AdaBoost, XGBoost)
- Stacking (meta-model over multiple base models)

**Why they work:**
- Reduce variance (bagging)
- Reduce bias (boosting)
- Leverage diversity of models ("wisdom of the crowd")

---
