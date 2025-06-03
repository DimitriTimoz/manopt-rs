use candle_core::{Tensor, Result};

pub trait Manifold {
    fn new() -> Self;
    fn name(&self) -> &'static str;
    fn dimension(&self) -> usize;

    fn project(&self, point: &Tensor, vector: &Tensor) -> Result<Tensor>;
    fn retract(&self, point: &Tensor, direction: &Tensor, step: f64) -> Result<Tensor>;
}
