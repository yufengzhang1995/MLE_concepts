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

## 4. What is log loss / cross-entropy loss?

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
