## 1. Bias-Variance Tradeoff
An ideal model typically exhibits **low bias and low variance**. Let's understand these terms:

1. **Bias**:
  - Bias refers to the **error introduced by approximating a real-world problem** with a simplified model.
  - Error between average model predictions and ground truth
  - A model with high bias tends to **oversimplify the problem** and may fail to capture important patterns in the data.
  - The bias of the estimated functions tells us the capacity of the underlying model to predict the values
  - **High bias often leads to underfitting,** where the **model performs poorly both on the training data and unseen data.**
2. **Variance**:
  - Variance refers to **the model's sensitivity to fluctuations in the training data**.
  - Average variability in the model prediction for the given dataset
  - A model with high variance fits the training data too closely, **capturing noise** in the data rather than the underlying patterns.
  - The variance of the estimated function tells you how much the function can adjust to the change in the dataset
  - **High variance often leads to overfitting**, where the **model performs well on the training data but poorly on unseen data.**
3. **Formula**
  - $$\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

The ideal model strikes a balance between bias and variance, achieving low levels of both. Such a model generalizes well to unseen data, capturing the underlying patterns without being overly sensitive to noise. This balance results in good performance on both the training and test datasets.

Finding this balance is a key goal in machine learning model selection and training. Techniques such as **cross-validation, regularization, and ensemble methods like bagging and boosting** are commonly used to achieve models with low bias and low variance.

## 2. Regularization

Regularization prevents overfitting by adding a penalty term to the loss function that discourages overly complex models. This constrains model parameters, reduces variance, and improves generalization to unseen data.

**L1**

- lambda of the summation of the absolute value of the coefficients.
- reduces the complexity of the model by shrinking the coefficients to zero
- tries to estimate the median of the data
- helps to reduce the overfitting in the model
- sparse, and important feature selection
- **helps in feature selection by eliminating the features that are not important.**
    - penalizes features that have low predictive outcomes by shrinking their coefficients closer to zero.
- can be used for classification or regression

**L2** 

- lambda of the summation of squared coefficients.
- reduces the complexity of the model by shrinking the coefficients.
- estimate the mean of the data
- **helpful when the number of feature points are large in number**
- **helpful when data suffer from multicollinearity**
- **helps in feature selection by eliminating the features that are not important.**
        
        

L1 and L2 regularization are common techniques used in machine learning to prevent overfitting by adding a penalty term to the loss function. They are particularly popular in linear models like linear regression and logistic regression, but they can also be applied to other types of models.

### L1 Regularization (Lasso Regression):

L1 regularization adds a penalty term to the loss function that is proportional to the absolute value of the coefficients:

$\text{L1 penalty} = \lambda \sum_{i=1}^{n} |w_i|$ 

where:


L1 regularization encourages sparsity in the model, meaning it tends to force the weights of less important features to **zero**, effectively performing feature selection.

### L2 Regularization (Ridge Regression):

L2 regularization adds a penalty term to the loss function that is proportional to the square of the magnitude of the coefficients:

$\text{L2 penalty} = \lambda \sum_{i=1}^{n} w_i^2$ 

L2 regularization penalizes large coefficients, encouraging them to be **small**. It doesn't usually force coefficients to exactly zero, but it can significantly reduce their values, effectively reducing the model's complexity and making it more robust to outliers in the data.

### Key Differences:

L1 regularization adds the absolute value of weights as a penalty, which encourages sparsity and can perform feature selection.
L2 regularization adds the squared value of weights as a penalty, which discourages large weights but typically keeps all features.
L1 produces sparse models; L2 produces smoother models.

1. **Effect on Coefficients**:
    - L1 regularization can lead to **sparse** models with many coefficients set to zero.
    - L2 regularization **shrinks the coefficients towards zero** but rarely forces them to zero.
2. **Robustness to Outliers**:
    - L1 regularization is **generally more robust to outliers** because it doesn't penalize large individual weights as much as L2 regularization.
3. **Feature Selection**:
    - L1 regularization performs feature selection by forcing less important features to zero.
    - L2 regularization doesn't perform feature selection as aggressively as L1 regularization.

Choose L1 regularization (Lasso) when:

1. **Feature Selection**: You want to perform feature selection and prefer a sparse model with only a subset of the most important features. L1 regularization tends to set less important features' coefficients to zero, effectively performing **automatic feature selection.**
2. **Interpretability**: You need a model that is easy to interpret and want to identify the most influential predictors in your model.
3. **Outlier Handling**: Your dataset contains outliers, and you want the regularization technique to be robust to outliers. L1 regularization can be more robust to outliers due to its ability to shrink less important features' coefficients to zero.
4. **Reducing Overfitting**: You suspect that your model is overfitting and want to reduce the complexity of the model by encouraging sparsity in the coefficients.

Choose L2 regularization (Ridge) when:

1. **No Feature Selection Needed**: You do not need feature selection or prefer to keep all features in the model. L2 regularization penalizes all coefficients uniformly without setting any of them to zero.
2. **Robustness to Collinearity**: Your dataset contains highly correlated features, and you want the regularization technique to be robust to multicollinearity. L2 regularization can help stabilize the model's coefficients when dealing with multicollinearity.
3. **Handling Large Weights**: You want the regularization technique to handle large individual weights more gently. L2 regularization tends to shrink large weights more gradually compared to L1 regularization.
4. **Reducing Variance**: You suspect that your **model is suffering from high variance (overfitting)** and want to **reduce the model's complexity** without eliminating any features entirely.

In practice, you may experiment with both L1 and L2 regularization and choose the one that yields the best performance on your validation or test dataset. Additionally, techniques such as Elastic Net regularization combine L1 and L2 penalties to leverage the benefits of both regularization techniques.

$\text{Elastic Net Penalty} = \alpha \times \text{L1 Penalty} + (1 - \alpha) \times \text{L2 Penalty}$ 

Elastic Net regularization is commonly used in linear regression, logistic regression, and other machine learning models where regularization is necessary to prevent overfitting and improve model generalization.


## 3. Generative vs Discriminative
A generative model will learn categories of data while a discriminative model will simply learn the distinction between different categories of data. 

Generative models learn the **joint probability distribution p(x, y) and can generate new data point**, while discriminative models learn the conditional probability p(x|y) and are used for **predicting the output given an input**. 

Generative models include algorithms like Naive Bayes and GANs, capturing data distributions

Discriminative models, such as logistic regression and neural networks, focus on separating different classes. Discriminative models will generally outperform generative models on classification tasks.

Discriminative model learns the predictive distribution p(y|x) directly while generative model learns the joint distribution p(x, y) then obtains the predictive distribution based on Bayes' rule.

## 4. Supervised vs unsupervised learning
### 1. Supervised Learning

In supervised learning, the model is trained on a **labeled** dataset, which means that each training example is paired with an output label. The model learns to predict the output from the input data during training, aiming to generalize to unseen data.

- Applications: Classification, regression, speech recognition, image recognition.
- Challenges: Requires a large amount of labeled data, which can be expensive or time-consuming to obtain.

### 2. Unsupervised Learning

Unsupervised learning involves training models on data without labeled responses. The goal is to discover underlying patterns, groupings, or distributions in the data.

- Applications: Clustering, dimensionality reduction, association mining, anomaly detection.
- Challenges: Harder to evaluate performance without labeled data; the meaningfulness of the outcomes can be subjective.

### 3. Semi-Supervised Learning

Semi-supervised learning falls between supervised and unsupervised learning. It uses **a small amount of labeled data** alongside a large amount of unlabeled data. This approach leverages the labeled data to guide the learning process in the unlabeled dataset.

- Applications: Image and speech recognition, where labeling large datasets can be impractical.
- Challenges: Designing algorithms that effectively leverage both labeled and unlabeled data.

### 4. Weakly-Supervised Learning

Weakly-supervised learning is a type of learning where **the training labels are noisy, limited, or imprecise.** The goal is still to train a model that can make accurate predictions, despite the lower quality of the training labels.

- Applications: Situations where obtaining precise labels is difficult but rough indications are available.
- Challenges: Developing models robust to inaccuracies in the training data.

### 5. Self-Supervised Learning

Self-supervised learning is a subset of unsupervised learning techniques where the data provides supervision. It involves **creating auxiliary tasks,** such as predicting part of the data from other parts, to learn representations without explicit external labeling.

- Applications: Natural language processing (e.g., predicting missing words), computer vision (e.g., predicting missing parts of images).
- Challenges: Creating effective auxiliary tasks that lead to useful representations.

### 6. Reinforcement Learning

Reinforcement learning is a type of machine learning where **an agent learns** to make decisions by taking actions in an environment to achieve some goals. The agent learns from trial and error, receiving rewards or penalties for actions.

- Applications: Game playing, robotics, recommendation systems, navigation.
- Challenges: Requires designing a suitable environment, reward structure, and balancing exploration vs. exploitation.

### Summary of Differences

- Supervised Learning: Learns from labeled data to predict outcomes for unseen data.
- Unsupervised Learning: Discovers patterns or structures in unlabeled data.
- Semi-Supervised Learning: Combines a small amount of labeled data with a large amount of unlabeled data.
- Weakly-Supervised Learning: Deals with imperfect or incomplete labeled data.
- Self-Supervised Learning: Generates its own supervision from the input data.
- Reinforcement Learning: Learns to make decisions through trials, errors, and rewards in an interactive environment.

Each of these learning paradigms has its own set of tools, algorithms, and best practices designed to tackle specific kinds of problems in the realm of artificial intelligence and machine learning.

State-of-the-art methods or models vary across different domains of machine learning, with advancements continually occurring.

- **Supervised Learning:** Transformers (like BERT for NLP) and advanced CNNs (like EfficientNet for image processing) are leading.
- **Unsupervised Learning:** Autoencoders and GANs (Generative Adversarial Networks) for generating new data instances or feature learning.
- **Semi-Supervised Learning:** MixMatch and FixMatch algorithms leverage both labeled and unlabeled data for training.
- **Weakly-Supervised Learning:** Snorkel uses data programming to generate training labels from weak sources.
- **Self-Supervised Learning:** Contrastive Learning methods, like SimCLR for images and GPT (Generative Pretrained Transformer) for text, learn by predicting parts of the data.
- **Reinforcement Learning:** Advanced models like AlphaZero for games and MuZero for more general applications learn optimal strategies through simulation.

## 5. What is the curse of dimensionality?

The curse of dimensionality refers to the phenomenon where data becomes increasingly sparse as dimensionality increases, making distance metrics less meaningful and requiring exponentially more data to learn reliable models.
