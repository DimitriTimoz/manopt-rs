//! Riemannian manifolds for constrained optimization.
//!
//! This module defines manifolds and their operations.
//! Each manifold implements geometric operations like projection, retraction,
//! exponential maps, and parallel transport.

use std::fmt::Debug;

use crate::prelude::*;

pub mod steifiel;
pub use steifiel::SteifielsManifold;

pub mod sphere;
pub use sphere::Sphere;

/// A Riemannian manifold defines the geometric structure for optimization.
///
/// This trait provides all the necessary operations for Riemannian optimization:
/// - Tangent space projections
/// - Retraction operations  
/// - Exponential maps
/// - Parallel transport
/// - Riemannian inner products
///
/// # Example Implementation
///
/// ```r u st
/// use manopt_rs::prelude::*;
///
/// #[derive(Clone)]
/// struct MyManifold;
///
/// impl<B: Backend> Manifold<B> for MyManifold {
/// }
/// ```
pub trait Manifold<B: Backend>: Clone + Send + Sync {
    const RANK_PER_POINT: usize;

    fn new() -> Self;
    fn name() -> &'static str;
    fn specific_name(s: &Shape) -> String {
        let dims = &s.dims;
        let num_dims = dims.len();
        let (channel_dims, manifold_dims) = dims.split_at(num_dims - Self::RANK_PER_POINT);
        format!(
            "{channel_dims:?} Channels worth of points in {} with specific n's {manifold_dims:?}",
            Self::name()
        )
    }

    fn acceptable_shape(s: &Shape) -> bool {
        s.num_dims() >= Self::RANK_PER_POINT
    }

    /// Project `vector` to the tangent space at `point`
    fn project<const D: usize>(point: Tensor<B, D>, vector: Tensor<B, D>) -> Tensor<B, D>;

    /// Convert Euclidean gradient `grad` to Riemannian gradient at `point`
    fn egrad2rgrad<const D: usize>(point: Tensor<B, D>, grad: Tensor<B, D>) -> Tensor<B, D> {
        Self::project(point, grad)
    }

    /// Project `vector` to the tangent space at `point`
    fn project_tangent<const D: usize>(point: Tensor<B, D>, vector: Tensor<B, D>) -> Tensor<B, D> {
        Self::project(point, vector)
    }

    /// Parallel transport of a tangent vector `tangent` from `point1` to `point2`
    /// By default, this is not accurately implemented and ignores the metric/connection
    /// just projecting to the tangent space.
    fn parallel_transport<const D: usize>(
        _point1: Tensor<B, D>,
        point2: Tensor<B, D>,
        tangent: Tensor<B, D>,
    ) -> Tensor<B, D> {
        // Default implementation: project to tangent space at point2
        Self::project_tangent(point2, tangent.into())
    }

    /// Move along the manifold from `point` along the tangent vector `direction` with step size
    fn retract<const D: usize>(point: Tensor<B, D>, direction: Tensor<B, D>) -> Tensor<B, D>;

    /// Exponential map: move from `point` along tangent vector `direction` with step size
    fn expmap<const D: usize>(point: Tensor<B, D>, direction: Tensor<B, D>) -> Tensor<B, D> {
        Self::retract(point, direction)
    }

    /// Riemannian inner product at a given `point`
    /// `u` and `v` are in the tangent space at `point`
    fn inner<const D: usize>(point: Tensor<B, D>, u: Tensor<B, D>, v: Tensor<B, D>)
        -> Tensor<B, D>;

    /// Project `point` onto manifold
    fn proj<const D: usize>(point: Tensor<B, D>) -> Tensor<B, D>;

    /// Check if a `point` is in the manifold.
    fn is_in_manifold<const D: usize>(point: Tensor<B, D>) -> Tensor<B, D, burn::tensor::Bool>;

    /// Check if a `vector` is in the tangent space at `point`
    /// given that `point` is in the manifold.
    /// By default, this is not accurately implemented and returns `false`.
    fn is_tangent_at<const D: usize>(
        point: Tensor<B, D>,
        vector: Tensor<B, D>,
    ) -> Tensor<B, D, burn::tensor::Bool>;
}

/// Euclidean manifold - the simplest case where no projection is needed
#[derive(Clone, Debug)]
pub struct Euclidean;

impl<B: Backend> Manifold<B> for Euclidean {
    const RANK_PER_POINT: usize = 1;

    fn new() -> Self {
        Self
    }

    fn name() -> &'static str {
        "Euclidean"
    }

    fn project<const D: usize>(_point: Tensor<B, D>, vector: Tensor<B, D>) -> Tensor<B, D> {
        vector
    }

    fn retract<const D: usize>(point: Tensor<B, D>, direction: Tensor<B, D>) -> Tensor<B, D> {
        point + direction
    }

    fn inner<const D: usize>(
        _point: Tensor<B, D>,
        u: Tensor<B, D>,
        v: Tensor<B, D>,
    ) -> Tensor<B, D> {
        (u * v).sum_dim(D - 1)
    }

    fn is_in_manifold<const D: usize>(
        point: Tensor<B, D>,
    ) -> burn::tensor::Tensor<B, D, burn::tensor::Bool> {
        point
            .clone()
            .detach()
            .is_nan()
            .any_dim(<Self as Manifold<B>>::RANK_PER_POINT)
            .bool_not()
    }

    fn proj<const D: usize>(point: Tensor<B, D>) -> Tensor<B, D> {
        point
    }

    fn is_tangent_at<const D: usize>(
        point: Tensor<B, D>,
        vector: Tensor<B, D>,
    ) -> Tensor<B, D, burn::tensor::Bool> {
        let vector_exists = vector
            .clone()
            .detach()
            .is_nan()
            .any_dim(<Self as Manifold<B>>::RANK_PER_POINT)
            .bool_not();
        Self::is_in_manifold(point).bool_and(vector_exists)
    }
}
