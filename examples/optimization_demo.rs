use std::collections::HashMap;

use burn::optim::SimpleOptimizer;
use manopt_rs::{optimizers::LessSimpleOptimizer, prelude::*};

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

    let mut loss_decay: HashMap<usize, f32> = HashMap::new();

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
            let loss_scalar = loss.into_scalar();
            println!("Step {}: x = {}, loss = {:.5}", step, x, loss_scalar);
            loss_decay.insert(step, loss_scalar);
        }
    }

    println!("\nResult after 100:");
    println!("x = {}", x);
    println!("Target = {}", target);
    let final_loss = (x.clone() - target.clone())
        .powf_scalar(2.0)
        .sum()
        .into_scalar();
    println!("Loss after 100 = {:.5}", final_loss);
    loss_decay.insert(100, final_loss);

    // Perform optimization steps
    (x, state) = optimizer.many_steps(|_| 1.0, 400, |x| (x - target.clone()) * 2.0, x, state);

    println!("\nFinal result:");
    println!("x = {}", x);
    println!("Target = {}", target);
    let final_loss = (x.clone() - target.clone())
        .powf_scalar(2.0)
        .sum()
        .into_scalar();
    println!("Final loss = {:.5}", final_loss);
    println!("State is set {}", state.is_some());
    loss_decay.insert(500, final_loss);
    let mut sorted_losses: Vec<(usize, f32)> = loss_decay.into_iter().collect();
    sorted_losses.sort_by_key(|z| z.0);
    println!("The loss decayed as follows: {:?}", sorted_losses);
}
