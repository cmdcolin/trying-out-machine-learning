# MNIST JAX Monorepo Architecture

This project is a full-stack machine learning demonstration:
1. **Train** in Python (JAX).
2. **Export** learned weights as JSON.
3. **Inference** in TypeScript (Browser/Node).

## Documentation
- **[How Training Works](./docs/TRAINING.md)**: JAX, MLP, and Optimization.
- **[How Inference Works](./docs/INFERENCE.md)**: CPU vs WebGPU implementations.

---

## Project Structure
- `packages/core`: Shared math and type definitions.
- `packages/webgpu`: WebGPU compute shader implementation.
- `packages/web`: Interactive React + Vite dashboard.
- `packages/cli`: Fast Node.js verification tool.
- `train.py`: JAX training script.

## Setup
```bash
make setup  # Install all dependencies (Python & Node)
make train  # Train the model and export weights
make web    # Start the web application
```
