use candle_core::{Result, Tensor, Var};
use candle_nn::optim::Optimizer;

use crate::manifolds::Manifold;

/// Projection of x in the tangent space of the Stiefel manifold at the point a
fn stiedel_projection(a: &Tensor, x: &Tensor) -> Result<Tensor> {
    x-a.matmul(&x.transpose(2, 3)?)?.matmul(a)
}

/// StiefelOptimizer is an optimizer that applies the Stiefel manifold constraint
/// It means that the parameters are constrained to be orthogonal matrices.
pub struct StiefelOptimizer {
    learning_rate: f64,
    vars: Vec<Var>,
}

impl Optimizer for StiefelOptimizer {
    type Config = f64;

    fn new(vars: Vec<Var>, learning_rate: f64) -> Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .collect();
        Ok(Self {
            vars,
            learning_rate,
        })
    }

    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn step(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()> {
        for var in self.vars.iter() {
            if let Some(grad) = grads.get(var) {
                
                let dir = stiedel_projection(var, grad)?;
            }
        }
        Ok(())
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr
    }
}

pub struct ManfioldRGD<M: Manifold> {
    learning_rate: f64,
    vars: Vec<Var>,
    _manifold: M,
}

impl<M: Manifold> Optimizer for ManfioldRGD<M> {
    type Config = f64;

    fn new(vars: Vec<Var>, learning_rate: f64) -> Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .collect();
        Ok(Self {
            learning_rate,
            vars,
            _manifold: M::new(),
        })
    }

    fn step(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()> {
        for var in self.vars.iter() {
            if let Some(grad) = grads.get(var) {
                let point = var;
                let dir = self._manifold.project(point, grad)?;
                let step = self.learning_rate;
                let new_point = self._manifold.retract(point, &dir, step)?;
                var.set(&new_point)?;
            }
        }
        Ok(())
    }

    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }
}
