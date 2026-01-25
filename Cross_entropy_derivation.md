---
title: "从伯努利分布推导交叉熵"
---

# 🔍 引言

交叉熵损失（Cross Entropy Loss）广泛用于分类问题。  
在二分类问题中，我们可以从概率的角度，使用 **伯努利分布（Bernoulli Distribution）** 出发推导出交叉熵的形式。

---

# 1️⃣ 问题设定：二分类场景

我们考虑如下变量定义：

- \( y \in \{0, 1\} \)：真实标签
- \( \hat{y} \in (0, 1) \)：模型预测的正类概率，即 \( P(y=1 \mid x) \)

我们假设标签 \( y \) 遵循 **伯努利分布**，概率质量函数如下：

$$
P(y \mid \hat{y}) = \hat{y}^y (1 - \hat{y})^{1 - y}
$$

---

# 2️⃣ 最大似然估计（MLE）

我们的目标是最大化这个观测值的概率，也就是最大化似然函数：

$$
\mathcal{L}(\hat{y}) = P(y \mid \hat{y}) = \hat{y}^y (1 - \hat{y})^{1 - y}
$$

---

# 3️⃣ 对数似然（Log-Likelihood）

为了便于求导和优化，我们对似然函数取对数：

$$
\log \mathcal{L}(\hat{y}) = y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})
$$

这是我们想要最大化的 log-likelihood。

---

# 4️⃣ 损失函数：最小化负对数似然

在深度学习中，我们通常通过 **最小化损失函数** 训练模型。于是我们取负号，得到 **交叉熵损失**：

$$
\mathcal{L}_{\text{CE}} = - \left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]
$$

---

# 5️⃣ 直观解释

- 当 \( y = 1 \) 时：

  $$
  \mathcal{L}_{\text{CE}} = -\log(\hat{y})
  $$

  → 希望模型预测 \( \hat{y} \) 越接近 1 越好。

- 当 \( y = 0 \) 时：

  $$
  \mathcal{L}_{\text{CE}} = -\log(1 - \hat{y})
  $$

  → 希望模型预测 \( \hat{y} \) 越接近 0 越好。

---

# 6️⃣ 推导流程图总结

$$
\begin{align*}
&\text{真实标签 } y \in \{0, 1\} \\
&\Downarrow \\
&\text{建模为伯努利分布：} \quad P(y \mid \hat{y}) = \hat{y}^y (1 - \hat{y})^{1 - y} \\
&\Downarrow \\
&\text{取对数得到 log-likelihood} \\
&\log P(y \mid \hat{y}) = y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \\
&\Downarrow \\
&\text{取负号 → 损失函数（Cross Entropy Loss）} \\
&\mathcal{L} = - \left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]
\end{align*}
$$

---

# 🔄 拓展：多分类交叉熵来自什么分布？

在多分类问题中，我们假设标签 \( y \) 服从 **多项分布（Multinoulli / Categorical distribution）**，其最大似然估计同样可以推导出 Softmax + Cross Entropy 损失函数。

---

# ✅ 总结

- 二分类任务中，交叉熵损失其实是伯努利分布的负对数似然
- 本质上，它是从概率建模角度的最大似然推导而来
- 这为理解深度学习中的损失函数提供了更扎实的理论基础

---
