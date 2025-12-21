use burn::{prelude::Backend, tensor::Tensor};

use crate::prelude::Manifold;

pub trait MonoidManifold<B: Backend>: Clone + Send + Sync + Manifold<B> {
    fn lie_mul<const D: usize>(points0: Tensor<B, D>, points1: Tensor<B, D>) -> Tensor<B, D>;
}
