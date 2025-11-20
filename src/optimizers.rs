//! Riemannian optimizers for manifold-constrained optimization.
//!
//! This module provides Riemannian optimization algorithms that work on manifolds,
//! extending classical optimization methods to handle geometric constraints.
use crate::prelude::*;
use burn::module::AutodiffModule;
use burn::optim::{adaptor::OptimizerAdaptor, LrDecayState, SimpleOptimizer};
use burn::record::Record;
use burn::tensor::backend::AutodiffBackend;
use burn::LearningRate;
use std::marker::PhantomData;
pub mod multiple;

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
            tensor = new_x;
            state = new_state;
        }
        (tensor, state)
    }
}

#[derive(Debug)]
pub struct ManifoldRGDConfig<M, B> {
    _manifold: PhantomData<M>,
    _backend: PhantomData<B>,
}

impl<M, B> Default for ManifoldRGDConfig<M, B>
where
    M: Manifold<B>,
    B: Backend,
{
    fn default() -> Self {
        Self {
            _manifold: PhantomData,
            _backend: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ManifoldRGD<M: Manifold<B>, B: Backend> {
    _manifold: PhantomData<M>,
    _backend: PhantomData<B>,
}

impl<M, B> Default for ManifoldRGD<M, B>
where
    M: Manifold<B>,
    B: Backend,
{
    fn default() -> Self {
        Self {
            _manifold: PhantomData,
            _backend: PhantomData,
        }
    }
}

#[derive(Record, Clone)]
pub struct ManifoldRGDState<B: Backend, const D: usize> {
    lr_decay: LrDecayState<B, D>,
}

impl<M, B> SimpleOptimizer<B> for ManifoldRGD<M, B>
where
    M: Manifold<B>,
    B: Backend,
{
    type State<const D: usize> = ManifoldRGDState<B, D>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<B, D>,
        grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        let direction = M::project(tensor.clone(), -grad);
        let result = M::retract(tensor, direction * lr);
        (result, state)
    }

    fn to_device<const D: usize>(
        _state: Self::State<D>,
        _device: &<B as Backend>::Device,
    ) -> Self::State<D> {
        #[allow(clippy::used_underscore_binding)]
        _state
    }
}

impl<M, B> ManifoldRGDConfig<M, B>
where
    M: Manifold<B>,
    B: Backend,
{
    #[must_use]
    pub fn init<Back: AutodiffBackend, Mod: AutodiffModule<Back>>(
        &self,
    ) -> OptimizerAdaptor<ManifoldRGD<M, Back::InnerBackend>, Mod, Back>
    where
        M: Manifold<Back::InnerBackend>,
    {
        let optim = ManifoldRGD::<M, Back::InnerBackend>::default();

        OptimizerAdaptor::from(optim)
    }
}

/// Configuration for the Riemannian Adam optimizer.
///
/// This optimizer extends the Adam algorithm to work on Riemannian manifolds,
/// following the approach described in "Riemannian adaptive optimization methods"
/// (Bécigneul & Ganea, 2018).
///
/// # Example
///
/// ```rust
/// use manopt_rs::prelude::*;
///
/// let config = RiemannianAdamConfig::<Euclidean, burn::backend::NdArray>::new()
///     .with_lr(0.001)
///     .with_beta1(0.9)
///     .with_beta2(0.999)
///     .with_eps(1e-8)
///     .with_amsgrad(true);
/// ```
#[derive(Debug, Clone)]
pub struct RiemannianAdamConfig<M, B> {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub amsgrad: bool,
    pub stabilize: Option<usize>,
    _manifold: PhantomData<M>,
    _backend: PhantomData<B>,
}

impl<M, B> Default for RiemannianAdamConfig<M, B>
where
    M: Manifold<B>,
    B: Backend,
{
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
            stabilize: None,
            _manifold: PhantomData,
            _backend: PhantomData,
        }
    }
}

impl<M, B> RiemannianAdamConfig<M, B>
where
    M: Manifold<B>,
    B: Backend,
{
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_lr(mut self, lr: f64) -> Self {
        self.lr = lr;
        self
    }

    #[must_use]
    pub fn with_beta1(mut self, beta1: f64) -> Self {
        self.beta1 = beta1;
        self
    }

    #[must_use]
    pub fn with_beta2(mut self, beta2: f64) -> Self {
        self.beta2 = beta2;
        self
    }

    #[must_use]
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    #[must_use]
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    #[must_use]
    pub fn with_amsgrad(mut self, amsgrad: bool) -> Self {
        self.amsgrad = amsgrad;
        self
    }

    #[must_use]
    pub fn with_stabilize(mut self, stabilize: Option<usize>) -> Self {
        self.stabilize = stabilize;
        self
    }
}

/// Riemannian Adam optimizer
#[derive(Debug, Clone)]
pub struct RiemannianAdam<M: Manifold<B>, B: Backend> {
    config: RiemannianAdamConfig<M, B>,
}

impl<M, B> RiemannianAdam<M, B>
where
    M: Manifold<B>,
    B: Backend,
{
    #[must_use]
    pub fn new(config: RiemannianAdamConfig<M, B>) -> Self {
        Self { config }
    }
}

/// State for Riemannian Adam optimizer
#[derive(Record, Clone)]
pub struct RiemannianAdamState<B: Backend, const D: usize> {
    pub step: usize,
    pub exp_avg: Tensor<B, D>,
    pub exp_avg_sq: Tensor<B, D>,
    pub max_exp_avg_sq: Option<Tensor<B, D>>,
    lr_decay: LrDecayState<B, D>,
}

impl<M, B> SimpleOptimizer<B> for RiemannianAdam<M, B>
where
    M: Manifold<B>,
    B: Backend,
{
    type State<const D: usize> = RiemannianAdamState<B, D>;

    fn step<const D: usize>(
        &self,
        _lr: LearningRate,
        tensor: Tensor<B, D>,
        grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        let learning_rate = self.config.lr;

        // Apply weight decay if specified
        let grad = if self.config.weight_decay > 0.0 {
            grad + tensor.clone() * self.config.weight_decay
        } else {
            grad
        };

        // Convert Euclidean gradient to Riemannian gradient
        let rgrad = M::egrad2rgrad(tensor.clone(), grad);

        let mut state = match state {
            Some(mut state) => {
                state.step += 1;
                state
            }
            None => RiemannianAdamState {
                step: 1,
                exp_avg: Tensor::zeros_like(&tensor),
                exp_avg_sq: Tensor::zeros_like(&tensor),
                max_exp_avg_sq: if self.config.amsgrad {
                    Some(Tensor::zeros_like(&tensor))
                } else {
                    None
                },
                lr_decay: LrDecayState::new(0, tensor.clone()),
            },
        };

        // Update exponential moving averages
        state.exp_avg =
            state.exp_avg.clone() * self.config.beta1 + rgrad.clone() * (1.0 - self.config.beta1);

        let inner_product = M::inner(tensor.clone(), rgrad.clone(), rgrad.clone());
        state.exp_avg_sq = state.exp_avg_sq.clone() * self.config.beta2
            + inner_product * (1.0 - self.config.beta2);

        // Compute denominator
        let denom = if self.config.amsgrad {
            let max_exp_avg_sq = state.max_exp_avg_sq.as_ref().unwrap();
            let new_max = Tensor::max_pair(max_exp_avg_sq.clone(), state.exp_avg_sq.clone());
            state.max_exp_avg_sq = Some(new_max.clone());
            new_max.sqrt() + self.config.eps
        } else {
            state.exp_avg_sq.clone().sqrt() + self.config.eps
        };

        // Bias correction
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let bias_correction1 = 1.0 - self.config.beta1.powi(state.step as i32);
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let bias_correction2 = 1.0 - self.config.beta2.powi(state.step as i32);
        let step_size = learning_rate * bias_correction2.sqrt() / bias_correction1;

        // Compute direction
        let direction = state.exp_avg.clone() / denom;

        // Move on manifold using exponential map
        let new_point = M::expmap(tensor.clone(), direction.clone() * (-step_size));
        let new_point = M::proj(new_point);

        // Parallel transport the exponential average to the new point
        let exp_avg_new = M::parallel_transport(tensor, new_point.clone(), state.exp_avg);
        state.exp_avg = exp_avg_new;

        (new_point, Some(state))
    }

    fn to_device<const D: usize>(
        mut state: Self::State<D>,
        device: &<B as Backend>::Device,
    ) -> Self::State<D> {
        state.exp_avg = state.exp_avg.to_device(device);
        state.exp_avg_sq = state.exp_avg_sq.to_device(device);
        if let Some(ref max_exp_avg_sq) = state.max_exp_avg_sq {
            state.max_exp_avg_sq = Some(max_exp_avg_sq.clone().to_device(device));
        }
        state.lr_decay = LrDecayState::to_device(state.lr_decay, device);
        state
    }
}

impl<M, B> RiemannianAdamConfig<M, B>
where
    M: Manifold<B>,
    B: Backend,
{
    #[must_use]
    pub fn init<Back: AutodiffBackend, Mod: AutodiffModule<Back>>(
        &self,
    ) -> OptimizerAdaptor<RiemannianAdam<M, Back::InnerBackend>, Mod, Back>
    where
        M: Manifold<Back::InnerBackend>,
    {
        let optim = RiemannianAdam::<M, Back::InnerBackend>::new(RiemannianAdamConfig {
            lr: self.lr,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            weight_decay: self.weight_decay,
            amsgrad: self.amsgrad,
            stabilize: self.stabilize,
            _manifold: PhantomData,
            _backend: PhantomData,
        });

        OptimizerAdaptor::from(optim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::optim::SimpleOptimizer;

    type TestBackend = NdArray;

    #[test]
    fn test_riemannian_adam_basic() {
        let config = RiemannianAdamConfig::<Euclidean, TestBackend>::new()
            .with_lr(0.1)
            .with_beta1(0.9)
            .with_beta2(0.999);

        let optimizer = RiemannianAdam::new(config);

        // Create test tensors
        let tensor = Tensor::<TestBackend, 1>::zeros([3], &Default::default());
        let grad = Tensor::<TestBackend, 1>::ones([3], &Default::default());

        // Perform one step
        let (new_tensor, state) = optimizer.step(1.0, tensor.clone(), grad, None);

        // Check that the tensor moved in the negative gradient direction
        let scalar_value = new_tensor.slice([0; 1]).into_scalar();
        assert!(
            scalar_value < 0.0,
            "Should move in negative gradient direction"
        );
        assert!(state.is_some(), "State should be initialized");
    }

    #[test]
    fn test_riemannian_adam_convergence() {
        let config = RiemannianAdamConfig::<Euclidean, TestBackend>::new().with_lr(0.1);

        let optimizer = RiemannianAdam::new(config);

        // Target optimization: minimize ||x - target||^2
        let target = Tensor::<TestBackend, 1>::from_floats([1.0, -0.5, 2.0], &Default::default());
        let mut x = Tensor::<TestBackend, 1>::zeros([3], &Default::default());
        let mut state = None;

        let initial_loss = (x.clone() - target.clone()).powf_scalar(2.0).sum();

        // Run optimization for several steps
        for _ in 0..50 {
            let grad = (x.clone() - target.clone()) * 2.0;
            let (new_x, new_state) = optimizer.step(1.0, x, grad, state);
            x = new_x;
            state = new_state;
        }

        let final_loss = (x.clone() - target.clone()).powf_scalar(2.0).sum();

        // Check convergence
        assert!(
            final_loss.into_scalar() < initial_loss.into_scalar(),
            "Loss should decrease"
        );
    }

    #[test]
    fn test_riemannian_adam_amsgrad() {
        let config = RiemannianAdamConfig::<Euclidean, TestBackend>::new()
            .with_lr(0.1)
            .with_amsgrad(true);

        let optimizer = RiemannianAdam::new(config);

        let tensor = Tensor::<TestBackend, 1>::zeros([2], &Default::default());
        let grad = Tensor::<TestBackend, 1>::ones([2], &Default::default());

        let (_, state) = optimizer.step(1.0, tensor, grad, None);

        // Check that AMSGrad state is initialized
        assert!(state.is_some());
        let state = state.unwrap();
        assert!(
            state.max_exp_avg_sq.is_some(),
            "AMSGrad should initialize max_exp_avg_sq"
        );
    }

    #[test]
    fn test_riemannian_adam_weight_decay() {
        let config = RiemannianAdamConfig::<Euclidean, TestBackend>::new()
            .with_lr(0.1)
            .with_weight_decay(0.1);

        let optimizer = RiemannianAdam::new(config);

        let tensor = Tensor::<TestBackend, 1>::ones([2], &Default::default());
        let grad = Tensor::<TestBackend, 1>::zeros([2], &Default::default());

        let (new_tensor, _) = optimizer.step(1.0, tensor.clone(), grad, None);

        // With weight decay and zero gradient, the tensor should shrink
        let original_norm = tensor.powf_scalar(2.0).sum().sqrt();
        let new_norm = new_tensor.powf_scalar(2.0).sum().sqrt();

        assert!(
            new_norm.into_scalar() < original_norm.into_scalar(),
            "Weight decay should reduce tensor magnitude"
        );
    }

    #[test]
    fn test_riemannian_adam_state_persistence() {
        let config = RiemannianAdamConfig::<Euclidean, TestBackend>::new().with_lr(0.1);

        let optimizer = RiemannianAdam::new(config);

        let tensor = Tensor::<TestBackend, 1>::zeros([2], &Default::default());
        let grad = Tensor::<TestBackend, 1>::ones([2], &Default::default());

        // First step
        let (tensor1, state1) = optimizer.step(1.0, tensor, grad.clone(), None);
        assert!(state1.is_some());
        let state1 = state1.unwrap();
        assert_eq!(state1.step, 1);

        // Second step with state
        let (_, state2) = optimizer.step(1.0, tensor1, grad, Some(state1));
        assert!(state2.is_some());
        let state2 = state2.unwrap();
        assert_eq!(state2.step, 2);
    }
}
