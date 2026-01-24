## 1. What are precision, recall, and F1 score?

Answer:

Precision measures how many of the predicted positive samples are actually positive, while recall measures how many of the actual positive samples are correctly identified. The F1 score is the harmonic mean of precision and recall and balances the trade-off between them.

Optional one-liner:

Precision focuses on false positives; recall focuses on false negatives.

## 2. What is an ROC curve, and how is it used?

Answer:

An ROC curve plots the true positive rate against the false positive rate at different classification thresholds. It is used to evaluate a model’s ability to distinguish between classes across all possible thresholds.

## 3. What is AUC, and how is it interpreted?

Answer:

AUC, or Area Under the ROC Curve, measures the overall ability of a model to rank positive samples higher than negative ones. An AUC of 0.5 indicates random performance, while an AUC of 1.0 indicates perfect discrimination.

Very interview-safe intuition:

AUC represents the probability that a randomly chosen positive sample is ranked higher than a randomly chosen negative sample.

## 4. What is cross-entropy loss?

Answer:

Log loss, also known as cross-entropy loss, measures the difference between predicted probabilities and true labels. It penalizes confident but incorrect predictions more heavily, encouraging well-calibrated probability estimates.



## 5. What is a confusion matrix, and how do you interpret it?

Answer:

A confusion matrix summarizes prediction results by showing true positives, false positives, true negatives, and false negatives. It provides a detailed breakdown of classification errors and helps derive metrics such as precision, recall, and accuracy.

## 6. What is the difference between micro, macro, and weighted averaging?

Answer:

Macro averaging computes metrics independently for each class and then averages them equally, treating all classes the same.
Micro averaging aggregates all predictions across classes and computes a global metric, favoring frequent classes.
Weighted averaging is similar to macro averaging but weights each class by its number of samples.

One-liner to sound senior:

Macro emphasizes minority classes, while micro reflects overall system performance.

## 7. How do you evaluate a model for multi-class vs. multi-label classification?

Answer:

In multi-class classification, each sample belongs to exactly one class, so metrics like accuracy, macro F1, or confusion matrices are commonly used.
In multi-label classification, each sample can belong to multiple classes, so metrics such as Hamming loss, subset accuracy, and micro-averaged precision and recall are more appropriate.

## 8. What metrics are suitable for ranking models?

Answer:

Ranking models are typically evaluated using metrics that consider the order of predictions, such as NDCG, MAP, MRR, and precision@k or recall@k. These metrics focus on how well relevant items are ranked near the top rather than exact class labels.


## Appendix
---
title: "Machine Learning Metrics Cheat Sheet"
output: html_document
---

## Confusion Matrix (Binary Classification)

|                | Predicted + | Predicted - |
|----------------|-------------|-------------|
| **Actual +**   | TP          | FN          |
| **Actual -**   | FP          | TN          |

## Precision, Recall, and F1 Score

### Precision
How many predicted positives are correct?

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

### Recall (True Positive Rate)

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

### F1 Score

$$
\text{F1} = \frac{2TP}{2TP + FP + FN}
$$

### Accuracy

$$
\text{Accuracy} = \frac{TP + TN}{TP + FP + FN + TN}
$$

## ROC Curve and AUC

### True Positive Rate (TPR)

$$
\text{TPR} = \frac{TP}{TP + FN}
$$

### False Positive Rate (FPR)

$$
\text{FPR} = \frac{FP}{FP + TN}
$$

### Area Under the Curve (AUC)

$$
\text{AUC} = P\left( s(x^+) > s(x^-) \right)
$$

## Log Loss / Cross-Entropy Loss

### Binary Cross-Entropy

$$
\mathcal{L}
= -\frac{1}{N}
\sum_{i=1}^{N}
\left[
y_i \log(p_i) + (1 - y_i)\log(1 - p_i)
\right]
$$

### Multi-Class Cross-Entropy

$$
\mathcal{L}
= -\frac{1}{N}
\sum_{i=1}^{N}
\sum_{c=1}^{C}
y_{i,c} \log(p_{i,c})
$$

## Averaging Methods (Multi-Class)

Let $M_c$ be the metric for class $c$.

### Macro Average

$$
M_{\text{macro}} = \frac{1}{C} \sum_{c=1}^{C} M_c
$$

### Micro Average (Precision Example)

$$
\text{Precision}_{\text{micro}}
= \frac{\sum_c TP_c}{\sum_c (TP_c + FP_c)}
$$

### Weighted Average

$$
M_{\text{weighted}}
= \sum_{c=1}^{C}
\frac{n_c}{\sum_c n_c} M_c
$$

## Multi-Label Metrics

### Hamming Loss

$$
\text{Hamming Loss}
= \frac{1}{N \cdot C}
\sum_{i=1}^{N}
\sum_{c=1}^{C}
\mathbf{1}(y_{i,c} \neq \hat{y}_{i,c})
$$

### Subset Accuracy (Exact Match)

$$
\text{Subset Accuracy}
= \frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}(\mathbf{y}_i = \hat{\mathbf{y}}_i)
$$

## Ranking Metrics

### Precision@k

$$
\text{Precision@k}
= \frac{\text{Number of relevant items in top } k}{k}
$$

### Recall@k

$$
\text{Recall@k}
= \frac{\text{Number of relevant items in top } k}
{\text{Total number of relevant items}}
$$




### Average Precision (AP)

$$
\text{AP}
= \frac{1}{R}
\sum_{k=1}^{N}
\text{Precision@k} \cdot \text{rel}(k)
$$

### Mean Average Precision (MAP)

$$
\text{MAP}
= \frac{1}{Q}
\sum_{q=1}^{Q} \text{AP}_q
$$

## NDCG

### DCG

$$
\text{DCG@k}
= \sum_{i=1}^{k}
\frac{2^{rel_i} - 1}{\log_2(i + 1)}
$$

### NDCG

$$
\text{NDCG@k}
= \frac{\text{DCG@k}}{\text{IDCG@k}}
$$

## Mean Reciprocal Rank (MRR)

$$
\text{MRR}
= \frac{1}{Q}
\sum_{q=1}^{Q}
\frac{1}{\text{rank}_q}
$$

