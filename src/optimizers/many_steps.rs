//! A optimizer that allows for many steps with a given learning schedule
//! and a way of evaluating the gradient function on arbitrary
//! points. This way we can step using `SimpleOptimizer::step` with that gradient
//! several times.
use crate::prelude::*;
use burn::{optim::SimpleOptimizer, LearningRate};

/// A optimizer that allows for many steps with a given learning schedule
/// and a way of evaluating the gradient function on arbitrary
/// points. This way we can step using `SimpleOptimizer::step` with that gradient
/// several times.
pub trait LessSimpleOptimizer<B: Backend>: SimpleOptimizer<B> {
    fn many_steps<const D: usize>(
        &self,
        lr_function: impl FnMut(usize) -> LearningRate,
        num_steps: usize,
        grad_function: impl FnMut(Tensor<B, D>) -> Tensor<B, D>,
        tensor: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>);
}

/// The implementation of `LessSimpleOptimizer` is completely determined
/// by how `SimpleOptimizer` has been implemented because we are
/// just taking gradients using the input `grad_function` and steping with
/// `SimpleOptimizer::step`
impl<B: Backend, T: SimpleOptimizer<B>> LessSimpleOptimizer<B> for T {
    #[inline]
    fn many_steps<const D: usize>(
        &self,
        mut lr_function: impl FnMut(usize) -> LearningRate,
        num_steps: usize,
        mut grad_function: impl FnMut(Tensor<B, D>) -> Tensor<B, D>,
        mut tensor: Tensor<B, D>,
        mut state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        // Perform optimization steps
        for step in 0..num_steps {
            // Compute gradient at tensor
            let cur_grad = grad_function(tensor.clone());
            // The current learning rate for this step
            let cur_lr = lr_function(step);
            // Perform optimizer step
            let (new_x, new_state) = self.step(cur_lr, tensor.clone(), cur_grad, state);
            tensor = new_x.detach().require_grad();
            state = new_state;
        }
        (tensor, state)
    }
}
