use crate::prelude::*;

pub mod steifiel;
pub use steifiel::SteifielsManifold;

pub trait Manifold<B: Backend, const D: usize>: Clone + Send + Sync {
    fn new() -> Self;
    fn name() -> &'static str;

    fn project(point: &Tensor<B, D>, vector: &Tensor<B, D>) -> Tensor<B, D>;
    fn retract(point: &Tensor<B, D>, direction: &Tensor<B, D>, step: f64) -> Tensor<B, D>;

    /// Check if a point is in the manifold.
    /// By default, this is not implemented and returns `false`.
    fn is_in_manifold(_point: &Tensor<B, D>) -> bool {
        false
    }
}
