use crate::prelude::*;

/// Euclidean manifold - the simplest case where no projection is needed
#[derive(Clone, Debug)]
pub struct Sphere;

impl<B: Backend> Manifold<B> for Sphere {
    fn new() -> Self {
        Self
    }

    fn name() -> &'static str {
        "Sphere"
    }

    fn project<const D: usize>(_point: Tensor<B, D>, vector: Tensor<B, D>) -> Tensor<B, D> 
    {
        // Y/||y|
        vector.clone()/(vector.clone().transpose().matmul(vector)).sqrt()
    }

    fn retract<const D: usize>(
        point: Tensor<B, D>,
        direction: Tensor<B, D>,
    ) -> Tensor<B, D> {
        todo!("Implement retract for Sphere manifold")
        
    }

    fn inner<const D: usize>(
        _point: Tensor<B, D>,
        u: Tensor<B, D>,
        v: Tensor<B, D>,
    ) -> Tensor<B, D> {
        u.transpose().matmul(v)
    }

    fn is_in_manifold<const D: usize>(_point: Tensor<B, D>) -> bool {
        true
    }
}
