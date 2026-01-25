# 📊 Data Issues – Interview Prep

---

## ✅ How does imbalanced data affect model performance?

Imbalanced data means that one class significantly outnumbers the other(s), leading the model to become biased toward the **majority class**.

**Effects:**
- High overall accuracy can be misleading (e.g., 95% accuracy by predicting only the majority class).
- Poor **recall**, **precision**, or **F1 score** on minority class.
- Model may fail in real-world scenarios where detecting the minority class is critical (e.g., fraud detection, medical diagnosis).

---

## ✅ What are common strategies to handle imbalanced data?

1. **Resampling techniques:**
   - **Oversampling** minority class (e.g., SMOTE)
   - **Undersampling** majority class

2. **Algorithmic adjustments:**
   - Use **class weights** in loss function (e.g., `class_weight='balanced'` in scikit-learn)

3. **Anomaly detection models:**
   - Especially useful when the minority class is extremely rare.

4. **Ensemble methods:**
   - Use **balanced bagging** or boosting (e.g., BalancedRandomForest, XGBoost with `scale_pos_weight`)

5. **Evaluation metrics:**
   - Use **F1-score**, **AUC-ROC**, **precision-recall curve** instead of accuracy.

---

## ✅ What is data drift and how do you handle it?

**Data drift** refers to a change in the input data distribution over time, causing the model's performance to degrade.

**Types of Drift:**
- **Covariate Drift**: Feature distribution \( P(X) \) changes.
- **Concept Drift**: Relationship between features and labels \( P(Y|X) \) changes.
- **Label Drift**: Distribution of labels \( P(Y) \) changes.

**Handling strategies:**
- Continuously **monitor model performance** and feature distributions.
- Set up **alerting thresholds**.
- **Retrain model** periodically with recent data.
- Use **online learning** or **drift detection algorithms** (e.g., DDM, ADWIN).

---

## ✅ How to handle missing data in ML pipelines?

1. **Imputation:**
   - **Mean/median/mode imputation** (simple but can introduce bias)
   - **KNN imputation**
   - **Model-based imputation** (e.g., regression or iterative methods)

2. **Flagging:**
   - Add a binary feature indicating whether a value was missing.

3. **Deletion:**
   - Remove rows or columns with too many missing values (only if justified)

4. **Domain-driven assumptions:**
   - Sometimes, missingness carries meaning (e.g., test not performed = healthy)

5. **Tools:**
   - `SimpleImputer`, `IterativeImputer` in scikit-learn
   - Pipelines ensure that imputation happens consistently during training and inference

---

## ✅ How do you split your data into train/validation/test?

Standard practice:

- **Training Set (60-70%)**: Used to train the model.
- **Validation Set (15-20%)**: Used for hyperparameter tuning and model selection.
- **Test Set (15-20%)**: Used for final unbiased evaluation.

**Guidelines:**
- Use **stratified split** for classification (to preserve class ratios).
- Avoid **data leakage** (future data must not be in training).
- For time series, use **chronological split** (e.g., train on past, validate/test on future).

---

## ✅ What are the benefits and risks of oversampling and undersampling?

### 🔼 **Oversampling (e.g., SMOTE):**

**Benefits:**
- Balances class distribution without losing data.
- Helps models better learn from minority class.

**Risks:**
- Can lead to **overfitting** (especially if duplicating samples).
- Synthetic data may not represent real patterns.

### 🔽 **Undersampling:**

**Benefits:**
- Faster training (less data).
- Reduces majority class dominance.

**Risks:**
- **Loss of information** from discarded samples.
- May miss important patterns in the majority class.

---

