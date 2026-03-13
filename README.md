# MNIST JAX Demo

This repository contains a simple implementation of MNIST digit recognition using JAX for training and TypeScript for inference (both CLI and Web).

## 📚 Documentation
- **[New to AI? Start Here](./docs/BEGINNERS_GUIDE.md)**: A programmer-friendly introduction.
- **[The ML Engineer Mindset](./docs/ML_ENGINEER_MINDSET.md)**: A technical guide for developers.
- **[Architecture Overview](./docs/OVERVIEW.md)**: How the project is organized.
- **[Deep Dive: Training with JAX](./docs/TRAINING.md)**: The MLP, Autograd, and XLA.
- **[Deep Dive: Inference Architecture](./docs/INFERENCE.md)**: Native TS vs WebGPU (WGSL).

## 🚀 Setup

- `packages/cli`: Node.js CLI inference demonstration.
- `packages/web`: Vite + React web application.
- `train.py`: JAX training script. Exports weights to the packages.

## Setup

1.  **Full Setup** (Data, Python, Monorepo):
    ```bash
    make setup
    ```

2.  **Train Model**:
    ```bash
    make train
    ```

## Running

- **Web App**: `pnpm --filter web dev` or `make web`
- **CLI Demo**: `pnpm --filter cli start` or `make cli`

## How it works

1.  **Training**: JAX is used to train a 2-layer MLP. The model is then "frozen" by exporting its weight matrices and bias vectors as nested JSON arrays.
2.  **Inference**: Since the model is a simple sequence of linear transformations and ReLU activations, inference is implemented from scratch in TypeScript using basic matrix-vector multiplication. No heavy runtime is needed for the inference phase.
