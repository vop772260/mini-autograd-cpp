# mini-autograd-cpp
A C++17 Reverse-Mode Automatic Differentiation Engine and Computational Graph for Neural Networks with Empirical Benchmarking.
 C++ Reverse-Mode Automatic Differentiation Engine

An academic implementation of a Reverse-Mode Automatic Differentiation (Autograd) Engine and Computational Graph construction in **C++17**, designed for neural network optimization research.

Abstract
Deep learning frameworks like PyTorch and TensorFlow rely on Dynamic Computational Graphs to perform backpropagation. This project implements a scalar-level autograd system that builds a Directed Acyclic Graph (DAG) during the forward pass and uses **Topological Sorting** ($\mathcal{O}(V + E)$ complexity) to evaluate gradients via the multivariate Chain Rule during the backward pass.

Mathematical Formulation
Given a scalar computational graph with objective function $L$, the gradient with respect to any intermediate node $x$ is accumulated as:

$$\frac{\partial L}{\partial x} = \sum_{y \in \text{Children}(x)} \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}$$

Build & Run

```bash
mkdir build && cd build
cmake ..
cmake --build .
./autograd_enginecd /workspaces/mini-autograd-cpp
