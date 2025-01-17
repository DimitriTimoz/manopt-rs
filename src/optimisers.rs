use crate::{prelude::*, Manifold};

pub trait Optimiser {
    fn get_name(&self) -> &str;
    fn step(&self) -> Result<()>;
    fn euclidean_gradient(&self, point: Tensor) -> Result<Tensor>;
    fn reinmannian_gradient(&self, point: Tensor) -> Result<Tensor>;
}

pub struct ReinmannianGradientDescent<M: Manifold> {
    pub manifold: M,
    pub learning_rate: f64,
}
