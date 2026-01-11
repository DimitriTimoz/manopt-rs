# manopt-rs

[![Crates.io](https://img.shields.io/crates/v/manopt-rs.svg)](https://crates.io/crates/manopt-rs)
[![Documentation](https://docs.rs/manopt-rs/badge.svg)](https://docs.rs/manopt-rs)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A high-performance Rust library for **manifold optimization** built on top of the [Burn](https://github.com/tracel-ai/burn) deep learning framework. This library provides Riemannian optimization algorithms and manifold structures for constrained optimization problems.

##  Features

- **Riemannian Optimization Algorithms**: Modern optimizers adapted for manifold constraints
  - Riemannian Adam (RiemannianAdam)
  - Riemannian Gradient Descent (ManifoldRGD)
- **Multiple Manifolds**: Built-in support for common manifold structures
  - Euclidean spaces
  - Sphere (unit sphere S^(n-1))
  - Stiefel manifold (matrices with orthonormal columns)
  - Orthogonal group
- **Backend Flexibility**: Works with any Burn backend (NDArray, Torch, WGPU, etc.)
- **Type Safety**: Leverages Rust's type system for safe tensor operations
- **High Performance**: Built on Burn's efficient tensor operations

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
manopt-rs = "0.1"

# Example with Burn backend
burn = { version = "0.17", features = ["tch", "autodiff", "ndarray"] }
```

## Quick Start

```rust
use manopt_rs::prelude::*;
use burn::optim::SimpleOptimizer;

fn main() {
    // Configure Riemannian Adam optimizer
    let config = RiemannianAdamConfig::<Euclidean, burn::backend::NdArray>::new()
        .with_lr(0.01)
        .with_beta1(0.9)
        .with_beta2(0.999);

    let optimizer = RiemannianAdam::new(config);

    // Create optimization problem: minimize ||x - target||²
    let target = Tensor::<burn::backend::NdArray, 1>::from_floats([2.0, -1.0, 3.0], &Default::default());
    let mut x = Tensor::<burn::backend::NdArray, 1>::zeros([3], &Default::default());
    let mut state = None;

    // Optimization loop
    for _step in 0..100 {
        let grad = (x.clone() - target.clone()) * 2.0;
        let (new_x, new_state) = optimizer.step(1.0, x.clone(), grad, state);
        x = new_x;
        state = new_state;
    }

    println!("Optimized result: {}", x);
}
```

## Examples

### Basic Optimization

Run a simple quadratic optimization example:

```bash
cargo run --example optimization_demo
```

This demonstrates minimizing a quadratic function using Riemannian Adam.

### Riemannian Adam Demo

Test the Riemannian Adam optimizer:

```bash
cargo run --example riemannian_adam_demo
```

### Multi-Constraint Optimization

Example with multiple manifold constraints:

```bash
cargo run --example multi_constraints
```

## Architecture

### Manifolds

The library is built around the `Manifold` trait, which defines the geometric structure:

```rust
pub trait Manifold<B: Backend>: Clone + Send + Sync {
    fn project<const D: usize>(point: Tensor<B, D>, vector: Tensor<B, D>) -> Tensor<B, D>;
    fn retract<const D: usize>(point: Tensor<B, D>, direction: Tensor<B, D>) -> Tensor<B, D>;
    fn inner<const D: usize>(point: Tensor<B, D>, u: Tensor<B, D>, v: Tensor<B, D>) -> Tensor<B, D>;
    // ... more methods
}
```

### Optimizers

Riemannian optimizers that respect manifold constraints:

- **RiemannianAdam**: Adam optimizer adapted for Riemannian manifolds
- **ManifoldRGD**: Riemannian gradient descent

## Supported Manifolds

- ✅ **Euclidean**: Standard unconstrained optimization
- ✅ **Sphere**: Unit sphere constraints
- ✅ **Stiefel**: Matrices with orthonormal columns
- ✅ **Orthogonal Group**: Orthogonal matrices
- 📋 **Planned**: Grassmann, Symmetric Positive Definite

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/DimitriTimoz/manopt-rs.git
   cd manopt-rs
   ```

2. Install dependencies:
   ```bash
   cargo build
   ```

3. Run tests:
   ```bash
   cargo test
   ```

## 🔗 Related Projects

- [Manopt](https://www.manopt.org/): MATLAB toolbox for optimization on manifolds
- [Pymanopt](https://pymanopt.org/): Python toolbox for optimization on manifolds
- [Burn](https://github.com/tracel-ai/burn): Deep learning framework in Rust

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by the [Manopt](https://www.manopt.org/) toolbox
- Built on the excellent [Burn](https://github.com/tracel-ai/burn) framework
- Thanks to the Rust community for their amazing ecosystem
