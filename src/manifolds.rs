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
/// ```rust
/// use manopt_rs::prelude::*;
///
/// #[derive(Clone)]
/// struct MyManifold;
///
/// impl<B: Backend> Manifold<B> for MyManifold {
///     type PointOnManifold<const D: usize> = Tensor<B, 1>;
///    
///     type TangentVectorWithoutPoint<const D: usize> = Tensor<B,1>;
///     fn new() -> Self { MyManifold }
///     fn name() -> &'static str { "MyManifold" }
///     
///     fn project<const D: usize>(point: Tensor<B, D>, vector: Tensor<B, D>) -> Tensor<B, D> {
///         // Project vector to tangent space at point
///         vector
///     }
///     
///     fn retract<const D: usize>(point: Tensor<B, D>, direction: Tensor<B, D>) -> Tensor<B, D> {
///         // Move along manifold from point in direction with step size
///         point + direction
///     }
///     
///     fn inner<const D: usize>(_point: Tensor<B, D>, u: Tensor<B, D>, v: Tensor<B, D>) -> Tensor<B, D> {
///         // Riemannian inner product at point
///         u * v
///     }
/// }
/// ```
pub trait Manifold<B: Backend>: Clone + Send + Sync {
    type PointOnManifold<const D: usize>;
    type TangentVectorWithoutPoint<const D: usize>;

    fn new() -> Self;
    fn name() -> &'static str;

    /// Project `vector` to the tangent space at `point`
    fn project<const D: usize>(point: Tensor<B, D>, vector: Tensor<B, D>) -> Tensor<B, D>;

    // Move along the manifold from `point` along the tangent vector `direction` with step size
    fn retract<const D: usize>(point: Tensor<B, D>, direction: Tensor<B, D>) -> Tensor<B, D>;

    /// Convert Euclidean gradient `grad` to Riemannian gradient at `point`
    fn egrad2rgrad<const D: usize>(point: Tensor<B, D>, grad: Tensor<B, D>) -> Tensor<B, D> {
        Self::project(point, grad)
    }

    /// Riemannian inner product at a given `point`
    /// `u` and `v` are in the tangent space at `point`
    fn inner<const D: usize>(point: Tensor<B, D>, u: Tensor<B, D>, v: Tensor<B, D>)
        -> Tensor<B, D>;

    /// Exponential map: move from `point` along tangent vector `direction` with step size
    fn expmap<const D: usize>(point: Tensor<B, D>, direction: Tensor<B, D>) -> Tensor<B, D> {
        Self::retract(point, direction)
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
        Self::project_tangent(point2, tangent)
    }

    /// Project `vector` to the tangent space at `point`
    fn project_tangent<const D: usize>(point: Tensor<B, D>, vector: Tensor<B, D>) -> Tensor<B, D> {
        Self::project(point, vector)
    }

    /// Project `point` onto manifold
    fn proj<const D: usize>(point: Tensor<B, D>) -> Tensor<B, D> {
        point
    }

    /// Check if a `point` is in the manifold.
    /// By default, this is not accurately implemented and returns `false`.
    fn is_in_manifold<const D: usize>(_point: Tensor<B, D>) -> bool {
        false
    }

    /// Check if a `vector` is in the tangent space at `point`
    /// given that `point` is in the manifold.
    /// By default, this is not accurately implemented and returns `false`.
    fn is_tangent_at<const D: usize>(_point: Tensor<B, D>, _vector: Tensor<B, D>) -> bool {
        false
    }
}

/// Euclidean manifold - the simplest case where no projection is needed
#[derive(Clone, Debug)]
pub struct Euclidean;

impl<B: Backend> Manifold<B> for Euclidean {
    type PointOnManifold<const D: usize> = Tensor<B, 1>;

    type TangentVectorWithoutPoint<const D: usize> = Tensor<B, 1>;
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

    fn is_in_manifold<const D: usize>(_point: Tensor<B, D>) -> bool {
        true
    }
}
