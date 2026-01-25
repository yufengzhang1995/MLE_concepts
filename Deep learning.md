# 🤖 Deep Learning – Interview Prep

---

## ✅ What is the ReLU activation function?

**ReLU (Rectified Linear Unit)** is one of the most commonly used activation functions in deep neural networks.

**Definition:**
$$\text{ReLU}(x) = \max(0, x)$$

**Advantages:**
- Sparse activation
- Fast computation
- Reduces vanishing gradient problem compared to sigmoid or tanh

---

## ✅ How to deal with gradient vanishing?

**Gradient vanishing** happens when gradients become very small during backpropagation, especially in deep networks or RNNs.

**Solutions:**
1. **Use ReLU-like activations** (ReLU, Leaky ReLU, GELU)
2. **Weight initialization** (e.g., He initialization)
3. **Batch normalization** to stabilize activations
4. **Residual connections** (e.g., ResNets)
5. **Use architectures designed to mitigate it**, like GRU/LSTM for sequential data

---

## ✅ What are common activation functions and when to use them?

| Activation | Formula | Use Case |
|------------|---------|----------|
| **ReLU** | $\( \max(0, x) \)$ | Default for hidden layers |
| **Sigmoid** | $\( \frac{1}{1 + e^{-x}} \)$ | Output layer for binary classification |
| **Tanh** | $\( \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \)$ | Hidden layers for zero-centered outputs |
| **Leaky ReLU** | $\( \max(\alpha x, x) \), \( \alpha \approx 0.01 \)$ | Avoids dead neurons |
| **GELU (Gaussian Error Linear Unit)** | $\( x \cdot \Phi(x) \)$ | Used in Transformers (e.g., BERT) for smoother activation |

**Tip:** Use **ReLU** by default, switch to **GELU** in transformer-style architectures.

---

## ✅ What are fully connected layers?

**Fully connected (FC) layers** (also called **dense layers**) are layers where **each neuron is connected to every neuron in the previous layer**.

Mathematically:
$$\mathbf{z} = \mathbf{W} \cdot \mathbf{x} + \mathbf{b}$$

- Common in MLPs (multi-layer perceptrons)
- Usually appear after convolutional layers in CNNs

---

## ✅ What is the role of initialization in deep learning?

Proper weight **initialization** ensures that:
- Gradients don’t vanish or explode
- The training starts with balanced activation variance

**Common initialization methods:**
- **Xavier/Glorot** (for tanh/sigmoid):
  $$\text{Var}(W) = \frac{1}{n_{in} + n_{out}}$$
- **He Initialization** (for ReLU):
  $$\text{Var}(W) = \frac{2}{n_{in}}$$

Bad initialization leads to poor convergence and unstable training.

---

## ✅ What is dropout and how does it prevent overfitting?

**Dropout** is a regularization technique that randomly sets a fraction of neurons' outputs to zero during training.

At each iteration:
$$\text{Dropout}(x) = x \cdot \mathbf{m}, \quad \mathbf{m} \sim \text{Bernoulli}(p)$$

**Purpose:**
- Prevents co-adaptation of neurons
- Forces the network to be more robust
- Works like ensemble of sub-networks

During inference, dropout is **turned off** and weights are scaled accordingly.

---

## ✅ What is an epoch, batch, and iteration?

- **Epoch**: One full pass over the entire training dataset.
- **Batch**: A subset of training data used in one forward/backward pass.
- **Iteration**: One update step; if batch size is $\( B \)$ and dataset has $\( N \)$ samples:

  $$\text{Iterations per epoch} = \frac{N}{B}$$

---

## ✅ What is early stopping and how does it help generalization?

**Early stopping** monitors validation loss or performance and **stops training when the model begins to overfit** (i.e., validation loss starts increasing).

**Benefits:**
- Improves generalization
- Saves training time
- Prevents overfitting on training data

---

## ✅ What is the difference between feedforward networks and recurrent networks?

| Type | Feedforward Neural Networks (FNN) | Recurrent Neural Networks (RNN) |
|------|-----------------------------------|---------------------------------|
| Structure | Directed acyclic graph | Contains cycles / feedback loops |
| Memory | No memory of past inputs | Maintains hidden state over time |
| Input | Fixed size | Sequences or time series |
| Usage | Image classification, tabular data | NLP, speech, time series |

**RNN formula (simplified):**

$$h_t = \sigma(W x_t + U h_{t-1} + b)$$

---

