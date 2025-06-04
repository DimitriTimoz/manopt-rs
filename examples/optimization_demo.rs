use burn::optim::SimpleOptimizer;
use rust_manopt::prelude::*;

fn main() {
    // Configure the optimizer
    let config = RiemannianAdamConfig::<Euclidean, burn::backend::NdArray>::new()
        .with_lr(0.01)
        .with_beta1(0.9)
        .with_beta2(0.999)
        .with_eps(1e-8);

    let optimizer = RiemannianAdam::new(config);

    // Create a simple quadratic optimization problem: minimize ||x - target||^2
    let target =
        Tensor::<burn::backend::NdArray, 1>::from_floats([2.0, -1.0, 3.0], &Default::default());
    let mut x =
        Tensor::<burn::backend::NdArray, 1>::from_floats([0.0, 0.0, 0.0], &Default::default());
    let mut state = None;

    println!("Target: {}", target);
    println!("Initial x: {}", x);
    println!("\nOptimization steps:");

    // Perform optimization steps
    for step in 0..100 {
        // Compute gradient of ||x - target||^2 = 2 * (x - target)
        let grad = (x.clone() - target.clone()) * 2.0;

        // Perform optimizer step
        let (new_x, new_state) = optimizer.step(1.0, x.clone(), grad, state);
        x = new_x;
        state = new_state;

        // Print progress every 10 steps
        if step % 10 == 0 {
            let loss = (x.clone() - target.clone()).powf_scalar(2.0).sum();
            println!("Step {}: x = {}, loss = {}", step, x, loss);
        }
    }

    println!("\nFinal result:");
    println!("x = {}", x);
    println!("Target = {}", target);
    let final_loss = (x.clone() - target.clone()).powf_scalar(2.0).sum();
    println!("Final loss = {}", final_loss);
}
