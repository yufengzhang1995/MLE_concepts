# 🧠 Transformers – Interview Prep

Notion link: https://www.notion.so/chloezh1995/Transformer-e1ef29dc01d145b689b329a8a0ab9de5?source=copy_link
---

## ✅ Why do we divide the attention score by √dₖ in the Transformer?

In scaled dot-product attention, the attention score is:

$$
\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V
$$

We divide by \( \sqrt{d_k} \) to prevent large dot products when the dimensionality \( d_k \) is high, which can push softmax into very small gradients (vanishing), making learning unstable.

---

## ✅ Why are different weight matrices used to compute Q, K, and V?

Each token embedding is projected into three different spaces:

- **Query (Q):** What I'm looking for  
- **Key (K):** What I have  
- **Value (V):** What I’ll return if matched

Each has a different purpose, so they need **learnable linear projections**:
$$
Q = XW^Q,\quad K = XW^K,\quad V = XW^V
$$

This allows the model to **learn asymmetric relationships** between tokens.

---

## ✅ Why do we use Multi-Head Attention?

Multi-Head Attention (MHA) allows the model to **attend to information from different representation subspaces** at different positions:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$

Where each head is:

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

**Benefits:**
- Captures **diverse relationships** (e.g., syntax vs semantics)
- Improves **model expressiveness**

---

## ✅ What is the time complexity of attention?

For a sequence of length \( n \) and hidden size \( d \):

- **Dot-product attention**:  
  \( O(n^2 d) \) — due to the \( QK^T \) operation  
- Memory complexity: \( O(n^2) \)

This becomes a bottleneck for **long sequences**.

---

## ✅ What is KV cache and why is it used?

**KV Cache** stores the **Key and Value tensors** from previous time steps during inference (especially in auto-regressive decoding).

**Why it's useful:**
- Avoids recomputing attention over the full sequence
- Reduces complexity from \( O(n^2) \) to \( O(n) \) per token during generation

Common in LLM inference (e.g., GPT).

---

## ✅ What is Multi-Query Attention (MQA)?

**MQA** uses a **single shared Key and Value** across all attention heads, while each head has its own Query projection.

$$
Q_i = XW^Q_i,\quad K = XW^K,\quad V = XW^V
$$

**Benefits:**
- Reduces memory and compute from \( O(hnd) \) to \( O(nd) \)
- Helps in **scaling to large models** and longer contexts

---

## ✅ What is Grouped Query Attention (GQA)?

**GQA** is a compromise between MHA and MQA:
- Use **fewer groups of shared Key/Value** tensors (not one per head or fully shared)

If you have \( h \) heads and \( g \) groups, then:
- Each group shares one KV pair
- Each group handles \( h/g \) attention heads

**Trade-off:** balance between **efficiency and diversity**

---

## ✅ What are the shapes of Q, K, V in MHA, MQA, and GQA?

Assume:
- \( b \): batch size
- \( n \): sequence length
- \( h \): number of heads
- \( d \): total model hidden size
- \( d_h = d / h \): head size

### 🔹 Multi-Head Attention (MHA):
```text
Q: (b, h, n, d_h)
K: (b, h, n, d_h)
V: (b, h, n, d_h)
