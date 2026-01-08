use burn::{
    optim::SimpleOptimizer, prelude::Backend, record::Record, tensor::Tensor, LearningRate,
};

/// TODO document and construct an implementation
pub trait SimpleHessianOptimizer<B>: Send + Sync + Clone + SimpleOptimizer<B>
where
    B: Backend,
{
    /// The state of the optimizer. It also implements [record](Record), so that it can be saved.
    type StateWithHessian<const D: usize, const H: usize>: Record<B>
        + Clone
        + From<Self::State<D>>
        + Into<Self::State<D>>
        + 'static;

    /// The optimizer step is performed for one tensor at a time with its gradient, hessian and state.
    ///
    /// Note that the state is passed as parameter, so implementations don't have to handle
    /// the saving and loading of recorded states.
    fn step_with_hessian<const D: usize, const H: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<B, D>,
        grad: Tensor<B, D>,
        hessian: Tensor<B, H>,
        state: Option<Self::StateWithHessian<D, H>>,
    ) -> (Tensor<B, D>, Option<Self::StateWithHessian<D, H>>);

    /// The optimizer step is performed for one tensor at a time with its gradient, hessian and state.
    ///
    /// Note that the state is passed as parameter, so implementations don't have to handle
    /// the saving and loading of recorded states.
    fn step<const D: usize, const H: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<B, D>,
        grad: Tensor<B, D>,
        hessian: Option<Tensor<B, H>>,
        state: Option<Self::StateWithHessian<D, H>>,
    ) -> (Tensor<B, D>, Option<Self::StateWithHessian<D, H>>) {
        if let Some(hessian) = hessian {
            self.step_with_hessian(lr, tensor, grad, hessian, state)
        } else {
            let (new_pt, new_state) =
                SimpleOptimizer::step(self, lr, tensor, grad, state.map(Into::into));
            (new_pt, new_state.map(Into::into))
        }
    }

    /// Change the device of the state.
    ///
    /// This function will be called accordindly to have the state on the same device as the
    /// gradient and the tensor when the [step](SimpleOptimizer::step) function is called.
    fn to_device<const D: usize, const H: usize>(
        state: Self::StateWithHessian<D, H>,
        device: &B::Device,
    ) -> Self::StateWithHessian<D, H>;
}

/// A `SimpleOptimizer` also works as a `SimpleHessianOptimizer` by ignoring the Hessian information
impl<B: Backend, T: SimpleOptimizer<B>> SimpleHessianOptimizer<B> for T {
    type StateWithHessian<const D: usize, const H: usize> = <T as SimpleOptimizer<B>>::State<D>;

    fn step_with_hessian<const D: usize, const H: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<B, D>,
        grad: Tensor<B, D>,
        _hessian: Tensor<B, H>,
        state: Option<Self::StateWithHessian<D, H>>,
    ) -> (Tensor<B, D>, Option<Self::StateWithHessian<D, H>>) {
        self.step(lr, tensor, grad, state)
    }

    fn to_device<const D: usize, const H: usize>(
        state: Self::StateWithHessian<D, H>,
        device: &<B as Backend>::Device,
    ) -> Self::StateWithHessian<D, H> {
        <T as SimpleOptimizer<B>>::to_device(state, device)
    }
}

/// A optimizer that allows for many steps with a given learning schedule
/// and a way of evaluating the gradient and hessian functions on arbitrary
/// points. This way we can step using `SimpleHessianOptimizer::step` with
/// that gradient and hessian several times.
pub trait LessSimpleHessianOptimizer<B: Backend>: SimpleHessianOptimizer<B> {
    fn many_steps<const D: usize, const H: usize>(
        &self,
        lr_function: impl FnMut(usize) -> LearningRate,
        num_steps: usize,
        grad_function: impl FnMut(Tensor<B, D>) -> Tensor<B, D>,
        hessian_function: impl FnMut(Tensor<B, D>) -> Option<Tensor<B, H>>,
        tensor: Tensor<B, D>,
        state: Option<Self::StateWithHessian<D, H>>,
    ) -> (Tensor<B, D>, Option<Self::StateWithHessian<D, H>>);
}

/// The implementation of `LessSimpleHessianOptimizer` is completely determined
/// by how `SimpleHessianOptimizer` has been implemented because we are
/// just taking gradients using the input `grad_function` and hessians with `hessian_function`
/// and steping with `SimpleHessianOptimizer::step`
impl<B: Backend, T: SimpleHessianOptimizer<B>> LessSimpleHessianOptimizer<B> for T {
    #[inline]
    fn many_steps<const D: usize, const H: usize>(
        &self,
        mut lr_function: impl FnMut(usize) -> LearningRate,
        num_steps: usize,
        mut grad_function: impl FnMut(Tensor<B, D>) -> Tensor<B, D>,
        mut hessian_function: impl FnMut(Tensor<B, D>) -> Option<Tensor<B, H>>,
        mut tensor: Tensor<B, D>,
        mut state: Option<Self::StateWithHessian<D, H>>,
    ) -> (Tensor<B, D>, Option<Self::StateWithHessian<D, H>>) {
        // Perform optimization steps
        for step in 0..num_steps {
            // Compute gradient at tensor
            let cur_grad = grad_function(tensor.clone());
            // Compute hessian at tensor
            let cur_hessian = hessian_function(tensor.clone());
            // The current learning rate for this step
            let cur_lr = lr_function(step);
            // Perform optimizer step
            let (new_x, new_state) = SimpleHessianOptimizer::step(
                self,
                cur_lr,
                tensor.clone(),
                cur_grad,
                cur_hessian,
                state,
            );
            tensor = new_x.detach().require_grad();
            state = new_state;
        }
        (tensor, state)
    }
}
