use burn::optim::SimpleOptimizer;
use rust_manopt::prelude::*;

fn main() {
    println!("Testing Riemannian Adam optimizer...");

    // Create a simple test case with Euclidean manifold
    let config = RiemannianAdamConfig::<Euclidean, burn::backend::NdArray>::new()
        .with_lr(0.1)
        .with_beta1(0.9)
        .with_beta2(0.999);

    let optimizer = RiemannianAdam::new(config);

    // Create test tensors
    let tensor = Tensor::<burn::backend::NdArray, 2>::zeros([2, 2], &Default::default());
    let grad = Tensor::<burn::backend::NdArray, 2>::ones([2, 2], &Default::default());

    // Perform one optimization step
    let (new_tensor, state) = optimizer.step(1.0, tensor.clone(), grad, None);

    println!("Original tensor: {}", tensor);
    println!("New tensor: {}", new_tensor);
    println!("State initialized: {}", state.is_some());

    println!("Riemannian Adam test completed successfully!");
}
