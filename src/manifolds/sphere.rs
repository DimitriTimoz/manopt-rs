use crate::prelude::*;

#[derive(Clone, Debug)]
pub struct Sphere;

impl<B: Backend> Manifold<B> for Sphere {
    type PointOnManifold<const D: usize> = Tensor<B, 1>;

    type TangentVectorWithoutPoint<const D: usize> = Tensor<B, 1>;

    fn new() -> Self {
        Self
    }

    fn name() -> &'static str {
        "Sphere"
    }

    fn project<const D: usize>(point: Tensor<B, D>, vector: Tensor<B, D>) -> Tensor<B, D> {
        // For sphere: project vector orthogonal to point
        let dot_product = (point.clone() * vector.clone()).sum_dim(D - 1);
        vector - point * dot_product
    }

    fn retract<const D: usize>(point: Tensor<B, D>, direction: Tensor<B, D>) -> Tensor<B, D> {
        // For sphere: normalize the result
        let new_point = point + direction;
        let norm = new_point.clone().powf_scalar(2.0).sum().sqrt().unsqueeze();
        new_point / norm
    }

    fn inner<const D: usize>(
        _point: Tensor<B, D>,
        u: Tensor<B, D>,
        v: Tensor<B, D>,
    ) -> Tensor<B, D> {
        u.transpose().matmul(v)
    }

    fn is_in_manifold<const D: usize>(point: Tensor<B, D>) -> bool {
        let r_squared = point.clone().powf_scalar(2.0).sum_dim(D - 1);
        let one = Tensor::<B, D>::ones(r_squared.shape(), &point.device());
        r_squared.all_close(one, None, None)
    }

    fn is_tangent_at<const D: usize>(point: Tensor<B, D>, vector: Tensor<B, D>) -> bool {
        let dot_product = (point * vector).sum();
        let zero = Tensor::<B, 1>::from_floats([0.0], &dot_product.device());
        dot_product.all_close(zero, None, Some(1e-6))
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::Manifold;

    use super::Sphere;
    use burn::{
        backend::{Autodiff, NdArray},
        tensor::Tensor,
    };

    type TestBackend = Autodiff<NdArray>;
    type TestTensor = Tensor<TestBackend, 1>;

    const TOLERANCE: f32 = 1e-6;

    fn assert_tensor_close(a: &TestTensor, b: &TestTensor, tol: f32) {
        let diff = (a.clone() - b.clone()).abs();
        let max_diff = diff.max().into_scalar();
        assert!(
            max_diff < tol,
            "Tensors differ by {}, tolerance: {}",
            max_diff,
            tol
        );
    }

    fn create_test_matrix(rows: usize, values: Vec<f32>) -> TestTensor {
        let device = Default::default();
        let data = &values[0..rows];
        Tensor::from_floats(data, &device)
    }

    #[test]
    fn test_manifold_creation() {
        let _manifold = <Sphere as Manifold<TestBackend>>::new();
        assert_eq!(<Sphere as Manifold<TestBackend>>::name(), "Sphere");
    }

    #[test]
    fn test_projection_tangent_space() {
        // Create a point on the Sphere manifold
        let point = create_test_matrix(6, vec![3.0 / 5.0, 0.0, 0.0, 4.0 / 5.0, 0.0, 0.0]);

        // Create a direction vector
        let direction = create_test_matrix(6, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);

        let projected =
            <Sphere as Manifold<TestBackend>>::project(point.clone(), direction.clone());

        // The projection should be orthogonal to the point
        // i.e., point^T * projected should be 0
        let product = (point.clone() * projected.clone()).sum();
        let max_entry = product.abs().max().into_scalar();
        assert!(
            max_entry < TOLERANCE,
            "Projected direction not in tangent space: absoulte value of the dot product = {}",
            max_entry
        );
    }

    #[test]
    fn test_projection_preserves_tangent_vectors() {
        // Create a point on the Sphere manifold
        let point = create_test_matrix(6, vec![3.0 / 5.0, 0.0, 0.0, 4.0 / 5.0, 0.0, 0.0]);

        assert!(
            Sphere::is_in_manifold(point.clone()),
            "This is a point on the sphere by construction"
        );

        // Create a direction vector
        let direction = create_test_matrix(6, vec![4.0 / 5.0, 0.2, 0.3, -3.0 / 5.0, 0.5, 0.6]);

        assert!(
            Sphere::is_tangent_at(point.clone(), direction.clone()),
            "This direction is orthogonal to point by construction"
        );

        let projected =
            <Sphere as Manifold<TestBackend>>::project(point.clone(), direction.clone());

        // The projection should be orthogonal to the point
        // i.e., point^T * projected should be 0
        let product = (point.clone() * projected.clone()).sum();
        let max_entry = product.abs().max().into_scalar();
        assert!(
            max_entry < TOLERANCE,
            "Projected direction not in tangent space: absoulte value of the dot product = {}",
            max_entry
        );

        assert!(
            Sphere::is_tangent_at(point.clone(), projected.clone()),
            "Projecting something already in the tangent space stays in the tangent space"
        );
        assert_tensor_close(&projected, &direction, TOLERANCE);
    }

    #[test]
    fn test_retraction_preserves_sphere_property() {
        let point = create_test_matrix(6, vec![3.0 / 5.0, 0.0, 0.0, 4.0 / 5.0, 0.0, 0.0]);

        assert!(
            Sphere::is_in_manifold(point.clone()),
            "This is a point on the sphere by construction"
        );

        let direction = create_test_matrix(6, vec![4.0 / 5.0, 0.2, 0.3, -3.0 / 5.0, 0.5, 0.6]);

        let moved = Sphere::retract(point, direction);

        assert!(Sphere::is_in_manifold(moved));
    }

    #[test]
    fn test_parallel_transport() {
        let point = create_test_matrix(6, vec![3.0 / 5.0, 0.0, 0.0, 4.0 / 5.0, 0.0, 0.0]);

        assert!(
            Sphere::is_in_manifold(point.clone()),
            "This is a point on the sphere by construction"
        );

        let direction = create_test_matrix(6, vec![4.0 / 5.0, 0.2, 0.3, -3.0 / 5.0, 0.5, 0.6]);

        let moved_point = Sphere::retract(point.clone(), direction.clone());
        let moved_vector = Sphere::parallel_transport(point, moved_point.clone(), direction);

        assert!(Sphere::is_in_manifold(moved_point.clone()));
        assert!(Sphere::is_tangent_at(moved_point, moved_vector));
    }
}
