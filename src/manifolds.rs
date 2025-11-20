//! Riemannian manifolds for constrained optimization.
//!
//! This module defines manifolds and their operations for Riemannian optimization.
//! Each manifold implements geometric operations like projection, retraction,
//! exponential maps, and parallel transport.

use std::{fmt::Debug, marker::PhantomData};

use crate::prelude::*;

pub mod steifiel;
use burn::{
    module::{AutodiffModule, ModuleDisplay},
    tensor::backend::AutodiffBackend,
};
pub use steifiel::SteifielsManifold;

pub mod sphere;

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
    fn retract<const D: usize>(point: Tensor<B, D>, direction: Tensor<B, D>) -> Tensor<B, D>;

    /// Convert Euclidean gradient to Riemannian gradient
    fn egrad2rgrad<const D: usize>(point: Tensor<B, D>, grad: Tensor<B, D>) -> Tensor<B, D> {
        Self::project(point, grad)
    }

    /// Riemannian inner product at a given point
    fn inner<const D: usize>(point: Tensor<B, D>, u: Tensor<B, D>, v: Tensor<B, D>)
        -> Tensor<B, D>;

    /// Exponential map: move from point along tangent vector u with step size
    fn expmap<const D: usize>(point: Tensor<B, D>, direction: Tensor<B, D>) -> Tensor<B, D> {
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

    fn retract<const D: usize>(point: Tensor<B, D>, direction: Tensor<B, D>) -> Tensor<B, D> {
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

#[derive(Clone, Debug)]
pub struct Constrained<M, Man> {
    module: M,
    _manifold: PhantomData<Man>,
}

impl<B, M, Man> Module<B> for Constrained<M, Man>
where
    M: Module<B>,
    B: Backend,
    Man: Clone + Debug + Send,
{
    type Record = M::Record;

    fn collect_devices(&self, devices: burn::module::Devices<B>) -> burn::module::Devices<B> {
        self.module.collect_devices(devices)
    }

    fn fork(self, device: &B::Device) -> Self {
        let module = self.module.fork(device);
        Self {
            module,
            _manifold: PhantomData,
        }
    }

    fn to_device(self, device: &B::Device) -> Self {
        let module = self.module.to_device(device);
        Self {
            module,
            _manifold: PhantomData,
        }
    }

    fn visit<Visitor: burn::module::ModuleVisitor<B>>(&self, visitor: &mut Visitor) {
        self.module.visit(visitor);
    }

    fn map<Mapper: burn::module::ModuleMapper<B>>(self, mapper: &mut Mapper) -> Self {
        let module = self.module.map(mapper);
        Self {
            module,
            _manifold: PhantomData,
        }
    }

    fn load_record(self, record: Self::Record) -> Self {
        let module = self.module.load_record(record);
        Self {
            module,
            _manifold: PhantomData,
        }
    }

    fn into_record(self) -> Self::Record {
        self.module.into_record()
    }
}

impl<B, M, Man> AutodiffModule<B> for Constrained<M, Man>
where
    M: AutodiffModule<B>,
    B: AutodiffBackend,
    Man: Clone + Debug + Send,
{
    type InnerModule = M::InnerModule;

    fn valid(&self) -> Self::InnerModule {
        self.module.valid()
    }
}

impl<M, Man> burn::module::ModuleDisplayDefault for Constrained<M, Man>
where
    M: burn::module::ModuleDisplayDefault,
    Man: Clone + Debug + Send,
{
    fn content(&self, content: burn::module::Content) -> Option<burn::module::Content> {
        self.module.content(content)
    }
}

impl<M, Man> ModuleDisplay for Constrained<M, Man>
where
    M: ModuleDisplay,
    Man: Clone + Debug + Send,
{
    fn format(&self, passed_settings: burn::module::DisplaySettings) -> String {
        format!("Constrained<{}>", self.module.format(passed_settings))
    }
}

impl<M, Man> Constrained<M, Man> {
    pub fn new(module: M) -> Self {
        Self {
            module,
            _manifold: PhantomData,
        }
    }

    /// Get a reference to the inner module
    pub fn inner(&self) -> &M {
        &self.module
    }

    /// Get a mutable reference to the inner module
    pub fn inner_mut(&mut self) -> &mut M {
        &mut self.module
    }

    /// Apply manifold projection to a tensor - requires explicit Backend type
    pub fn project_tensor<B, const D: usize>(
        &self,
        point: Tensor<B, D>,
        vector: Tensor<B, D>,
    ) -> Tensor<B, D>
    where
        B: Backend,
        M: Module<B>,
        Man: Manifold<B> + Clone + Debug + Send,
    {
        Man::project(point, vector)
    }

    /// Apply manifold retraction to a tensor - requires explicit Backend type
    pub fn retract_tensor<B, const D: usize>(
        &self,
        point: Tensor<B, D>,
        direction: Tensor<B, D>,
    ) -> Tensor<B, D>
    where
        B: Backend,
        M: Module<B>,
        Man: Manifold<B> + Clone + Debug + Send,
    {
        Man::retract(point, direction)
    }

    /// Convert Euclidean gradient to Riemannian gradient - requires explicit Backend type
    pub fn euclidean_to_riemannian<B, const D: usize>(
        &self,
        point: Tensor<B, D>,
        grad: Tensor<B, D>,
    ) -> Tensor<B, D>
    where
        B: Backend,
        M: Module<B>,
        Man: Manifold<B> + Clone + Debug + Send,
    {
        Man::egrad2rgrad(point, grad)
    }

    /// Project point onto manifold - requires explicit Backend type
    pub fn project_to_manifold<B, const D: usize>(&self, point: Tensor<B, D>) -> Tensor<B, D>
    where
        B: Backend,
        M: Module<B>,
        Man: Manifold<B> + Clone + Debug + Send,
    {
        Man::proj(point)
    }

    /// Get the manifold name
    pub fn manifold_name<B>(&self) -> &'static str
    where
        B: Backend,
        Man: Manifold<B>,
    {
        Man::name()
    }
}

/// Trait for modules that have manifold constraints
pub trait ConstrainedModule<B: Backend> {
    /// Apply manifold constraints to all parameters in the module
    #[must_use]
    fn apply_manifold_constraints(self) -> Self;

    /// Get information about the manifold constraints
    fn get_manifold_info(&self) -> std::collections::HashMap<String, String>;

    /// Check if this module has manifold constraints
    fn has_manifold_constraints(&self) -> bool {
        true
    }
}

/// Blanket implementation for Constrained wrapper
impl<B, M, Man> ConstrainedModule<B> for Constrained<M, Man>
where
    M: Module<B>,
    B: Backend,
    Man: Manifold<B> + Clone + Debug + Send,
{
    fn apply_manifold_constraints(self) -> Self {
        self
    }

    fn get_manifold_info(&self) -> std::collections::HashMap<String, String> {
        let mut info = std::collections::HashMap::new();
        info.insert("manifold_type".to_string(), Man::name().to_string());
        info
    }
}
