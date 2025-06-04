use crate::prelude::*;

#[derive(Clone)]
pub struct SteifielsManifold<B: Backend> {
    _backend: std::marker::PhantomData<B>,
}

impl<B: Backend> Manifold<B> for SteifielsManifold<B> {
    fn new() -> Self {
        SteifielsManifold {
            _backend: std::marker::PhantomData,
        }
    }

    fn name() -> &'static str {
        "Steifels"
    }

    /// Should be of 
    fn project<const D: usize>(point: Tensor<B, D>, direction: Tensor<B, D>) -> Tensor<B, D> {
        let xtv = point.clone().transpose().matmul(direction.clone());
        let sym = (xtv.clone() + xtv.clone().transpose()) * 0.5;
        direction.clone() - point.clone().matmul(sym)
    }

    fn retract<const D: usize>(point: Tensor<B, D>, direction: Tensor<B, D>, step: f64) -> Tensor<B, D> {
        let s = point + direction * step;
        gram_schmidt(&s)
    }
}

fn gram_schmidt<B: Backend, const D: usize>(v: &Tensor<B, D>) -> Tensor<B, D> {
    let n = v.dims()[0];
    let k = v.dims()[1];

    let mut u = Tensor::zeros_like(v);
    let v1 = v.clone().slice([0..n, 0..1]);
    let norm = v1.clone().transpose().matmul(v1.clone()).sqrt();
    u = u.slice_assign([0..n, 0..1], v1.clone() / norm);
    
    for i in 1..k {
        u = u.slice_assign([0..n, i..i + 1], v.clone().slice([0..n, i..i + 1]));
        for j in 0..i {
            let uj = u.clone().slice([0..n, j..j + 1]);
            let ui = u.clone().slice([0..n, i..i + 1]);
            let ui = ui.clone() - (uj.clone().transpose().matmul(ui.clone()))*uj;
            u = u.slice_assign([0..n, i..i + 1], ui);
        }
        // Normalize the vector
        let ui = u.clone().slice([0..n, i..i + 1]);
        let norm = ui.clone().transpose().matmul(ui.clone()).sqrt();
        u = u.slice_assign([0..n, i..i + 1], ui / norm);
    }
    u
}


#[cfg(test)]
mod test {
    use super::*;
    use burn::backend::NdArray;
    
    type TestBackend = NdArray;
    type TestTensor = Tensor<TestBackend, 2>;
    
    const TOLERANCE: f32 = 1e-6;
    
    fn assert_tensor_close(a: &TestTensor, b: &TestTensor, tol: f32) {
        let diff = (a.clone() - b.clone()).abs();
        let max_diff = diff.max().into_scalar();
        assert!(max_diff < tol, "Tensors differ by {}, tolerance: {}", max_diff, tol);
    }
    
    fn create_test_matrix(rows: usize, cols: usize, values: Vec<f32>) -> TestTensor {
        let device = Default::default();
        // Reshape the flat vector into a 2D array
        let mut data = Vec::with_capacity(rows);
        for chunk in values.chunks(cols) {
            data.push(chunk.to_vec());
        }
        
        // Create tensor from nested arrays
        match (rows, cols) {
            (3, 2) => {
                if data.len() >= 3 && data[0].len() >= 2 && data[1].len() >= 2 && data[2].len() >= 2 {
                    Tensor::from_floats([
                        [data[0][0], data[0][1]],
                        [data[1][0], data[1][1]],
                        [data[2][0], data[2][1]],
                    ], &device)
                } else {
                    panic!("Invalid 3x2 matrix data");
                }
            },
            (3, 1) => {
                if data.len() >= 3 && data[0].len() >= 1 && data[1].len() >= 1 && data[2].len() >= 1 {
                    Tensor::from_floats([
                        [data[0][0]],
                        [data[1][0]],
                        [data[2][0]],
                    ], &device)
                } else {
                    panic!("Invalid 3x1 matrix data");
                }
            },
            (3, 3) => {
                if data.len() >= 3 && data[0].len() >= 3 && data[1].len() >= 3 && data[2].len() >= 3 {
                    Tensor::from_floats([
                        [data[0][0], data[0][1], data[0][2]],
                        [data[1][0], data[1][1], data[1][2]],
                        [data[2][0], data[2][1], data[2][2]],
                    ], &device)
                } else {
                    panic!("Invalid 3x3 matrix data");
                }
            },
            (4, 2) => {
                if data.len() >= 4 && data[0].len() >= 2 && data[1].len() >= 2 && data[2].len() >= 2 && data[3].len() >= 2 {
                    Tensor::from_floats([
                        [data[0][0], data[0][1]],
                        [data[1][0], data[1][1]],
                        [data[2][0], data[2][1]],
                        [data[3][0], data[3][1]],
                    ], &device)
                } else {
                    panic!("Invalid 4x2 matrix data");
                }
            },
            (2, 2) => {
                if data.len() >= 2 && data[0].len() >= 2 && data[1].len() >= 2 {
                    Tensor::from_floats([
                        [data[0][0], data[0][1]],
                        [data[1][0], data[1][1]],
                    ], &device)
                } else {
                    panic!("Invalid 2x2 matrix data");
                }
            },
            _ => panic!("Unsupported matrix dimensions: {}x{}", rows, cols),
        }
    }
    
    #[test]
    fn test_manifold_creation() {
        let _manifold = SteifielsManifold::<TestBackend>::new();
        assert_eq!(SteifielsManifold::<TestBackend>::name(), "Steifels");
    }
    
    #[test]
    fn test_gram_schmidt_orthogonalization() {
        // Test with a simple 3x2 matrix
        let input = create_test_matrix(3, 2, vec![
            1.0, 1.0,
            1.0, 0.0,
            0.0, 1.0,
        ]);
        
        let result = gram_schmidt(&input);
        
        // Check that the result has orthonormal columns
        let q1 = result.clone().slice([0..3, 0..1]);
        let q2 = result.clone().slice([0..3, 1..2]);
        
        // Check orthogonality: q1^T * q2 should be close to 0
        let dot_product = q1.clone().transpose().matmul(q2.clone());
        let orthogonality_error = dot_product.abs().into_scalar();
        assert!(orthogonality_error < TOLERANCE, 
                "Columns are not orthogonal: dot product = {}", orthogonality_error);
        
        // Check normalization: ||q1|| = ||q2|| = 1
        let norm1 = q1.clone().transpose().matmul(q1.clone()).sqrt().into_scalar();
        let norm2 = q2.clone().transpose().matmul(q2.clone()).sqrt().into_scalar();
        
        assert!((norm1 - 1.0).abs() < TOLERANCE, 
                "First column not normalized: norm = {}", norm1);
        assert!((norm2 - 1.0).abs() < TOLERANCE, 
                "Second column not normalized: norm = {}", norm2);
    }
    
    #[test]
    fn test_gram_schmidt_single_column() {
        // Test with a single column vector
        let input = create_test_matrix(3, 1, vec![3.0, 4.0, 0.0]);
        let result = gram_schmidt(&input);
        
        // Should be normalized to unit length
        let norm = result.clone().transpose().matmul(result.clone()).sqrt().into_scalar();
        assert!((norm - 1.0).abs() < TOLERANCE, 
                "Single column not normalized: norm = {}", norm);
        
        // Should be proportional to original vector
        let expected = create_test_matrix(3, 1, vec![0.6, 0.8, 0.0]);
        assert_tensor_close(&result, &expected, TOLERANCE);
    }
    
    #[test]
    fn test_projection_tangent_space() {
        // Create a point on the Steifel manifold (orthonormal matrix)
        let point = create_test_matrix(3, 2, vec![
            1.0, 0.0,
            0.0, 1.0,
            0.0, 0.0,
        ]);
        
        // Create a direction vector
        let direction = create_test_matrix(3, 2, vec![
            0.1, 0.2,
            0.3, 0.4,
            0.5, 0.6,
        ]);
        
        let projected = SteifielsManifold::<TestBackend>::project(point.clone(), direction.clone());
        
        // The projection should be orthogonal to the point
        // i.e., point^T * projected should be skew-symmetric
        let product = point.clone().transpose().matmul(projected.clone());
        let symmetric_part = (product.clone() + product.clone().transpose()) * 0.5;
        
        // The symmetric part should be close to zero
        let max_symmetric = symmetric_part.abs().max().into_scalar();
        assert!(max_symmetric < TOLERANCE, 
                "Projected direction not in tangent space: max symmetric component = {}", 
                max_symmetric);
    }
    
    #[test]
    fn test_projection_preserves_tangent_vectors() {
        // Use a true tangent vector at the identity block
        let point = create_test_matrix(3, 2, vec![
            1.0, 0.0,
            0.0, 1.0,
            0.0, 0.0,
        ]);
        // Tangent vector: only the (3,1) and (3,2) entries are nonzero
        let tangent = create_test_matrix(3, 2, vec![
            0.0, 0.0,
            0.0, 0.0,
            1.0, -1.0,
        ]);
        // Project the tangent vector again
        let projected = SteifielsManifold::<TestBackend>::project(point.clone(), tangent.clone());
        // Should be unchanged (idempotent)
        assert_tensor_close(&projected, &tangent, 1e-6);
        // Check the tangent space property: X^T V + V^T X = 0
        let xtv = point.clone().transpose().matmul(tangent.clone());
        let vtx = tangent.clone().transpose().matmul(point.clone());
        let skew = xtv + vtx.transpose();
        let max_skew = skew.abs().max().into_scalar();
        assert!(max_skew < 1e-6, "Tangent space property violated: max skew = {}", max_skew);
    }
    
    #[test]
    fn test_retraction_preserves_stiefel_property() {
        // Start with a point on the Steifel manifold
        let point = create_test_matrix(3, 2, vec![
            1.0, 0.0,
            0.0, 1.0,
            0.0, 0.0,
        ]);
        
        // Create a tangent direction
        let direction = create_test_matrix(3, 2, vec![
            0.0, 0.1,
            0.0, -0.1,
            0.2, 0.3,
        ]);
        
        let step = 0.1;
        let retracted = SteifielsManifold::<TestBackend>::retract(point.clone(), direction.clone(), step);
        
        // Check that the result has orthonormal columns
        let q1 = retracted.clone().slice([0..3, 0..1]);
        let q2 = retracted.clone().slice([0..3, 1..2]);
        
        // Check orthogonality
        let dot_product = q1.clone().transpose().matmul(q2.clone()).into_scalar();
        assert!(dot_product.abs() < TOLERANCE, 
                "Retracted point columns not orthogonal: dot product = {}", dot_product);
        
        // Check normalization
        let norm1 = q1.clone().transpose().matmul(q1.clone()).sqrt().into_scalar();
        let norm2 = q2.clone().transpose().matmul(q2.clone()).sqrt().into_scalar();
        
        assert!((norm1 - 1.0).abs() < TOLERANCE, 
                "First column not normalized after retraction: norm = {}", norm1);
        assert!((norm2 - 1.0).abs() < TOLERANCE, 
                "Second column not normalized after retraction: norm = {}", norm2);
    }
    
    #[test]
    fn test_retraction_zero_step() {
        // Retraction with step size 0 should return the original point (after orthogonalization)
        let point = create_test_matrix(3, 2, vec![
            1.0, 0.0,
            0.0, 1.0,
            0.0, 0.0,
        ]);
        
        let direction = create_test_matrix(3, 2, vec![
            0.1, 0.2,
            0.3, 0.4,
            0.5, 0.6,
        ]);
        
        let retracted = SteifielsManifold::<TestBackend>::retract(point.clone(), direction.clone(), 0.0);
        
        // Should be close to the original point
        assert_tensor_close(&retracted, &point, TOLERANCE);
    }
    
    #[test]
    fn test_gram_schmidt_identity_matrix() {
        // Identity matrix should remain unchanged
        let identity = create_test_matrix(3, 3, vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]);
        
        let result = gram_schmidt(&identity);
        assert_tensor_close(&result, &identity, TOLERANCE);
    }
    
    #[test]
    fn test_manifold_properties() {
        // Test that the manifold preserves the Stiefel property: X^T * X = I
        let sqrt_half = (0.5_f32).sqrt();
        let point = create_test_matrix(4, 2, vec![
            sqrt_half, sqrt_half,
            sqrt_half, -sqrt_half,
            0.0, 0.0,
            0.0, 0.0,
        ]);
        
        // Verify it's on the manifold
        let gram_matrix = point.clone().transpose().matmul(point.clone());
        let identity = create_test_matrix(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        
        assert_tensor_close(&gram_matrix, &identity, TOLERANCE);
        
        // Test projection and retraction preserve this property
        let direction = create_test_matrix(4, 2, vec![
            0.1, 0.0,
            0.0, 0.1,
            0.2, 0.3,
            -0.1, 0.2,
        ]);
        
        let projected = SteifielsManifold::<TestBackend>::project(point.clone(), direction.clone());
        let retracted = SteifielsManifold::<TestBackend>::retract(point.clone(), projected, 0.1);
        
        let retracted_gram = retracted.clone().transpose().matmul(retracted.clone());
        assert_tensor_close(&retracted_gram, &identity, TOLERANCE);
    }
}
