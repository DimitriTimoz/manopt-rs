//! Riemannian manifolds for constrained optimization.
//!
//! This module defines manifolds and their operations for Riemannian optimization.
//! Each manifold implements geometric operations like projection, retraction,
//! exponential maps, and parallel transport.

use crate::prelude::*;

pub mod steifiel;
pub use steifiel::SteifielsManifold;

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
    fn new() -> Self;
    fn name() -> &'static str;

    fn project<const D: usize>(point: Tensor<B, D>, vector: Tensor<B, D>) -> Tensor<B, D>;
    fn retract<const D: usize>(
        point: Tensor<B, D>,
        direction: Tensor<B, D>,
    ) -> Tensor<B, D>;

    /// Convert Euclidean gradient to Riemannian gradient
    fn egrad2rgrad<const D: usize>(point: Tensor<B, D>, grad: Tensor<B, D>) -> Tensor<B, D> {
        Self::project(point, grad)
    }

    /// Riemannian inner product at a given point
    fn inner<const D: usize>(point: Tensor<B, D>, u: Tensor<B, D>, v: Tensor<B, D>)
        -> Tensor<B, D>;

    /// Exponential map: move from point along tangent vector u with step size
    fn expmap<const D: usize>(
        point: Tensor<B, D>,
        direction: Tensor<B, D>,
    ) -> Tensor<B, D> {
        Self::retract(point, direction)
    }

    /// Parallel transport of tangent vector from point1 to point2
    fn parallel_transport<const D: usize>(
        _point1: Tensor<B, D>,
        point2: Tensor<B, D>,
        tangent: Tensor<B, D>,
    ) -> Tensor<B, D> {
        // Default implementation: project to tangent space at point2
        Self::project_tangent(point2, tangent)
    }

    /// Project vector to tangent space at point
    fn project_tangent<const D: usize>(point: Tensor<B, D>, vector: Tensor<B, D>) -> Tensor<B, D> {
        Self::project(point, vector)
    }

    /// Project point onto manifold
    fn proj<const D: usize>(point: Tensor<B, D>) -> Tensor<B, D> {
        point
    }

    /// Check if a point is in the manifold.
    /// By default, this is not implemented and returns `false`.
    fn is_in_manifold<const D: usize>(_point: Tensor<B, D>) -> bool {
        false
    }
}

/// Euclidean manifold - the simplest case where no projection is needed
#[derive(Clone, Debug)]
pub struct Euclidean;

impl<B: Backend> Manifold<B> for Euclidean {
    fn new() -> Self {
        Self
    }

    fn name() -> &'static str {
        "Euclidean"
    }

    fn project<const D: usize>(_point: Tensor<B, D>, vector: Tensor<B, D>) -> Tensor<B, D> {
        vector
    }

    fn retract<const D: usize>(
        point: Tensor<B, D>,
        direction: Tensor<B, D>,
    ) -> Tensor<B, D> {
        point + direction
    }

    fn inner<const D: usize>(
        _point: Tensor<B, D>,
        u: Tensor<B, D>,
        v: Tensor<B, D>,
    ) -> Tensor<B, D> {
        u * v
    }

    fn is_in_manifold<const D: usize>(_point: Tensor<B, D>) -> bool {
        true
    }
}
