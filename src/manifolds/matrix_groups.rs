use crate::{manifolds::utils::identity_in_last_two, prelude::*};

#[derive(Debug, Clone, Default)]
pub struct OrthogonalGroup<B: Backend, const IS_SPECIAL: bool> {
    _backend: std::marker::PhantomData<B>,
}

impl<B: Backend, const IS_SPECIAL: bool> Manifold<B> for OrthogonalGroup<B, IS_SPECIAL> {
    const RANK_PER_POINT: usize = 2;

    fn new() -> Self {
        OrthogonalGroup {
            _backend: std::marker::PhantomData,
        }
    }

    fn name() -> &'static str {
        if IS_SPECIAL {
            "Special Orthogonal"
        } else {
            "Orthogonal"
        }
    }

    fn acceptable_dims(a_is: &[usize]) -> bool {
        debug_assert!(a_is.len() >= Self::RANK_PER_POINT);
        let num_dims = a_is.len();
        a_is[num_dims - 1] == a_is[num_dims - 2]
    }

    fn project<const D: usize>(_point: Tensor<B, D>, _vector: Tensor<B, D>) -> Tensor<B, D> {
        todo!()
    }

    fn retract<const D: usize>(_point: Tensor<B, D>, _direction: Tensor<B, D>) -> Tensor<B, D> {
        todo!()
    }

    fn inner<const D: usize>(
        _point: Tensor<B, D>,
        u: Tensor<B, D>,
        v: Tensor<B, D>,
    ) -> Tensor<B, D> {
        // For orthogonal manifolds, we use the standard Euclidean inner product
        (u * v).sum_dim(D - 1).sum_dim(D - 2)
    }

    fn proj<const D: usize>(_point: Tensor<B, D>) -> Tensor<B, D> {
        todo!()
    }

    fn is_in_manifold<const D: usize>(point: Tensor<B, D>) -> Tensor<B, D, burn::tensor::Bool> {
        if Self::acceptable_shape(&point.shape()) {
            return point.zeros_like().any_dim(D - 1).any_dim(D - 2);
        }
        let a_transpose_times_a = point.clone().transpose().matmul(point);
        let all_dims = a_transpose_times_a.shape();
        debug_assert!(all_dims.num_dims() >= 2);
        let other = identity_in_last_two(&a_transpose_times_a);
        let in_orthogonal = a_transpose_times_a
            .is_close(other, None, None)
            .all_dim(D - 1)
            .all_dim(D - 2);
        if IS_SPECIAL {
            in_orthogonal
        } else {
            #[allow(unused_variables)]
            let has_det_one = { todo!() };
            #[allow(unreachable_code)]
            in_orthogonal.bool_and(has_det_one);
        }
    }

    fn is_tangent_at<const D: usize>(
        _point: Tensor<B, D>,
        _vector: Tensor<B, D>,
    ) -> Tensor<B, D, burn::tensor::Bool> {
        todo!()
    }
}
