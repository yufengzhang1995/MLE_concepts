## 1. How does gradient descent work?

- Gradient Descent is an iterative optimization algorithm used to minimize a loss function by updating model parameters in the direction of the negative gradient. 
- At each step, it computes the gradient of the loss with respect to the parameters and takes a step of size proportional to the learning rate in the opposite direction of the gradient. 
- This gradually brings the parameters closer to a local or global minimum of the loss function.

- $\theta := \theta - \eta \cdot \nabla_\theta J(\theta)$


## 2. What is stochastic gradient descent, and how is it different from full-batch gradient descent?

- Stochastic Gradient Descent (SGD) updates the model parameters using the gradient computed from only a **single randomly selected data point (or a small mini-batch)** instead of the full training dataset.
- $\theta := \theta - \eta \cdot \nabla_\theta J(\theta; x^{(i)}, y^{(i)})$
- Key Differences:

  - Full-Batch Gradient Descent: Uses all training samples to compute the gradient at every step → more stable but computationally expensive.

   - SGD: Uses one or a few samples → faster and cheaper per step but introduces noise, which can help escape local minima but may cause instability.

## 3. What is backpropagation, and how does it relate to neural networks?

- Backpropagation is the algorithm used to compute the gradient of the loss function with respect to all weights in a neural network.
- It applies the chain rule of calculus to propagate the error backward from the output layer to the input layer.
- The computed gradients are then used by optimizers (like SGD or Adam) to update the weights.
- $\frac{\partial J}{\partial W^{(l)}} = \delta^{(l)} \cdot (a^{(l-1)})^T$

## 4.  What are common variants of gradient descent?

 - **SGD** – Updates parameters using a single sample or mini-batch. Simple and fast.

- **Momentum** – Adds a fraction of the previous update to the current update to accelerate convergence and smooth out noise.
    -  $v_t = \gamma v_{t-1} + \eta \nabla_\theta J(\theta)$
    - $\theta := \theta - v_t$



- **RMSProp** – Divides the learning rate by a moving average of squared gradients, helping to deal with non-stationary objectives and exploding gradients.
  - $E[g^2]t = \rho E[g^2]{t-1} + (1 - \rho)g_t^2$ 
  - $\theta := \theta - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \cdot g_t$


- Adam (Adaptive Moment Estimation) – Combines Momentum and RMSProp by maintaining both the first (mean) and second (variance) moments of the gradients. It is one of the most widely used optimizers due to its robustness and fast convergence.
  -  $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$ 
  - $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$ 
  - $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$
  - $\theta := \theta - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$


## 5.  What are vanishing and exploding gradients?

These problems occur during backpropagation in deep neural networks:

Vanishing Gradients: Gradients become very small as they are propagated backward, leading to very slow learning or no learning in earlier layers.

Exploding Gradients: Gradients grow exponentially, leading to unstable updates and divergence during training.

They are especially common in RNNs and deep feedforward networks.

## 6. How does learning rate affect training?

The learning rate controls the size of the steps taken during gradient descent:

Too high: May overshoot minima or diverge.

Too low: Training becomes slow and may get stuck in local minima.
An ideal learning rate balances convergence speed and stability.

Some optimizers (like Adam) adapt learning rates per parameter during training.

## 7. What is gradient clipping and why is it used?

Gradient clipping is a technique used to prevent exploding gradients, particularly in RNNs or deep models. It works by scaling the gradients when their norm exceeds a specified threshold. This ensures that updates remain stable and helps prevent divergence during training.

## 8. How do optimizers differ in convergence speed and stability?

SGD: Simple and efficient but may converge slowly and get stuck in local minima.

Momentum: Improves convergence speed, especially in ravine-like loss surfaces.

RMSProp: Adapts the learning rate, helping in non-stationary objectives and reducing oscillations.

Adam: Combines the advantages of Momentum and RMSProp, leading to fast convergence and robust training, especially in problems with sparse gradients or noisy data.

Summary:

For stability: Adam > RMSProp > Momentum > SGD

For simplicity and interpretability: SGD

For practical training: Adam is often the go-to choice.

| Optimizer | Convergence Speed | Stability | Notes                       |
| --------- | ----------------- | --------- | --------------------------- |
| SGD       | Slow              | Low       | Simple, noisy updates       |
| Momentum  | Faster            | Medium    | Smooths updates             |
| RMSProp   | Fast              | High      | Adapts learning rates       |
| Adam      | Very Fast         | Very High | Combines Momentum + RMSProp |
