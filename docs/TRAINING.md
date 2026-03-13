# Training MNIST with JAX

This document explains the training process in `train.py`.

## 1. The Model: Multi-Layer Perceptron (MLP)

The demo uses a simple but effective neural network architecture called a **Multi-Layer Perceptron (MLP)**.

- **Input**: 784 neurons (28x28 grayscale image flattened).
- **Hidden Layer 1**: 1024 neurons + [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) activation.
- **Hidden Layer 2**: 512 neurons + ReLU activation.
- **Output**: 10 neurons (one for each digit 0-9).

### Architecture Summary
Each layer performs a basic linear transformation followed by a non-linear activation function:
$y = \text{ReLU}(Wx + b)$
Where $W$ is a weight matrix, $b$ is a bias vector, and $x$ is the input from the previous layer.

## 2. Using JAX for Performance

[JAX](https://jax.readthedocs.io/) is used for three primary reasons:
1. **Autograd**: JAX can automatically compute the gradient of any Python function (crucial for backpropagation).
2. **XLA (Accelerated Linear Algebra)**: The `@jit` decorator compiles Python functions into optimized machine code.
3. **Vmap**: We use `vmap` (vectorized map) to instantly turn a function that handles *one* image into a function that handles a *batch* of images efficiently.

## 3. The Training Loop

The training process follows the standard stochastic gradient descent (SGD) cycle:

1. **Forward Pass**: Compute the predicted digit probabilities for a batch of images.
2. **Loss Calculation**: We use **Cross-Entropy Loss**, which penalizes the model based on how far its prediction is from the true label.
3. **Backward Pass**: Compute the gradient of the loss with respect to all parameters ($W$ and $b$).
4. **Update**: The [Optax](https://optax.readthedocs.io/) library's **AdamW** optimizer updates the parameters in the direction that minimizes the loss.

## 4. Generalization & Augmentation

To help the model recognize freehand drawings (which often look messier than the official MNIST set), we use **Data Augmentation**:
- **Gaussian Noise**: Small random fluctuations are added to each pixel.
- **Weight Decay**: AdamW uses L2 regularization to keep weights small, preventing the model from over-fitting to specific training images.

## 5. Serialization & Export

Once trained, the JAX model's parameters (the learned $W$ and $b$ matrices) are converted from JAX arrays to nested Python lists and saved as a JSON file. This allows our TypeScript code to load them without needing any Python runtime.

---

### External Resources
- [JAX Quickstart Guide](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)
- [3Blue1Brown: But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk)
- [Understanding ReLU](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/)
- [Cross-Entropy Loss Explained](https://machinelearningmastery.com/cross-entropy-for-machine-learning/)
