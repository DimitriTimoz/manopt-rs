pub mod prelude;
use prelude::*;

pub trait Manifold {
    fn get_name(&self) -> &str;

    /// The inner product (Riemannian metric) at a point on the manifold.
    /// Vectors a and b are tangent vectors at the point.
    fn inner_product(&self, point: Tensor, tangent_a: Tensor, tangent_b: Tensor ) -> Result<Tensor>;


    /// The used retraction at a point on the manifold.
    /// The tangent vector is projected to the manifold.
    fn retraction(&self, point: Tensor, tangent_vector: Tensor) -> Result<Tensor>;

    /// The used exponential map at a point on the manifold.
    /// The tangent vector is projected to the manifold.
    /// The exponential map is the retraction at the point of the tangent vector.
    fn exponential_map(&self, point: Tensor, tangent_vector: Tensor) -> Result<Tensor>;
}

pub struct EuclideanManifold;

impl Manifold for EuclideanManifold {
    fn get_name(&self) -> &str {
        "euclidean"
    }

    fn inner_product(&self, _point: Tensor, tangent_a: Tensor, tangent_b: Tensor) -> Result<Tensor> {
        tangent_a.t()?.matmul(&tangent_b)
    }

    fn retraction(&self, point: Tensor, tangent_vector: Tensor) -> Result<Tensor> {
        point + tangent_vector
    }

    fn exponential_map(&self, point: Tensor, tangent_vector: Tensor) -> Result<Tensor> {
        self.retraction(point, tangent_vector)
    }
}
