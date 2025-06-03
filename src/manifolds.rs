use crate::prelude::*;

pub mod steifiel;
pub use steifiel::SteifielsManifold;

pub trait Manifold<B: Backend, const D: usize>: Clone + Send + Sync {
    fn new() -> Self;
    fn name(&self) -> &'static str;

    fn project(&self, point: &Tensor<B, D>, vector: &Tensor<B, D>) -> Tensor<B, D>;
    fn retract(&self, point: &Tensor<B, D>, direction: &Tensor<B, D>, step: f64) -> Tensor<B, D>;

    /// Check if a point is in the manifold.
    /// By default, this is not implemented and returns `false`.
    fn is_in_manifold(&self, _point: &Tensor<B, D>) -> bool {
        false
    }
}
