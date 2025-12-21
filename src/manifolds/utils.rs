use burn::{prelude::Backend, tensor::Tensor};

/// Given a tensor of shape `a_1 ... a_k x N x N`
/// create a tensor of the same shape
/// whose `i_1...i_k,m,n` entry is `1` if `m==n` and `0` otherwise
pub(crate) fn identity_in_last_two<B: Backend, const D: usize>(
    example: &Tensor<B, D>,
) -> Tensor<B, D> {
    let shape: [usize; D] = example.shape().dims();
    debug_assert!(D >= 2);
    debug_assert_eq!(shape[D - 1], shape[D - 2]);
    let n = shape[D - 1];
    let identity = Tensor::eye(n, &example.device());
    // Broadcasting is right aligned
    // so this will act like identity in shape (1,...1,N,N)
    // and then broadcast those 1's to a_1 ... a_k
    // even though that is not stated in the docs of `expand`
    // More honestly phrased identity of rank 2 is a function NxN -> R
    // and the return value of type a1xa2...xNxN -> R is being done as
    // the precomposition with the map of sets a1x...akxNxN -> 1x...1xNxN -> NxN which is the
    // terminal map and identity maps in each factor and then monoidal unit laws.
    // Relying on this implicit extraneous structure is the reason do not have to do this manually like in
    // the below implementation of `diag_i`.
    identity.expand(example.shape())
}

/// Given a tensor of shape `a_1 ... a_k x N x N`
/// create a tensor of the same shape
/// whose `i_1...i_k,m,n` entry is `f(m)` if `m==n` and `0` otherwise
#[allow(dead_code)]
pub(crate) fn diag_i<B: Backend, const D: usize>(
    example: &Tensor<B, D>,
    diag_fun: impl Fn(usize) -> f32,
) -> Tensor<B, D> {
    let shape: [usize; D] = example.shape().dims();
    debug_assert!(D >= 2);
    debug_assert_eq!(shape[D - 1], shape[D - 2]);
    let n = shape[D - 1];
    let mut other = example.zeros_like();
    let mut ones_shape = [1usize; D];
    ones_shape[..(D - 2)].copy_from_slice(&shape[..(D - 2)]);
    let ones_patch = Tensor::<B, D>::ones(ones_shape, &example.device());
    for diag in 0..n {
        let ranges: [_; D] = std::array::from_fn(|dim| {
            if dim < D - 2 {
                0..shape[dim]
            } else {
                diag..diag + 1
            }
        });
        other = other.slice_assign(ranges, ones_patch.clone().mul_scalar(diag_fun(diag)));
    }
    other
}

/// Given a tensor of shape `l_1 .. l_a .. l_b .. l_D`
/// The rank is dropped to `D-2`
/// but because of the lack of const operations
/// we leave those dimensions there but with `1`s
/// in the shape instead.
/// `l_1 .. 1 .. 1 .. l_D`
pub(crate) fn ein_sum<B: Backend, const D: usize>(
    mut t: Tensor<B, D>,
    a: usize,
    b: usize,
) -> Tensor<B, D> {
    debug_assert!(a < D);
    debug_assert!(b < D);
    debug_assert_ne!(a, b);
    let t_shape: [usize; D] = t.shape().dims();
    debug_assert_eq!(t_shape[a], t_shape[b]);
    if a != D - 2 || b != D - 1 {
        t = t.swap_dims(a, D - 2);
        t = t.swap_dims(b, D - 1);
    }
    let identity_last_two = identity_in_last_two(&t);
    t = t.mul(identity_last_two);
    t = t.sum_dim(D - 1).sum_dim(D - 2);
    if a != D - 2 || b != D - 1 {
        t = t.swap_dims(b, D - 1);
        t = t.swap_dims(a, D - 2);
    }
    t
}

/// Given a tensor of shape `a_1 ... a_k x N x N`
/// create a tensor of shape `a_1 ... a_k x 1 x 1`
/// whose `i_1...i_k,0,0` entry is the trace
/// of the `N x N` matrix with those previous `k`
/// fixed but the last two remaining free.
#[allow(dead_code)]
pub(crate) fn trace<B: Backend, const D: usize>(t: Tensor<B, D>) -> Tensor<B, D> {
    // burn::tensor::linalg looks like a module according to the docs, but
    // doing `burn::tensor::linalg::trace` does not work
    // Also it reduces the rank by 1 and do not have the const generics
    // to do {D-1}. The docs there also are contradictory with D0 and D-1
    // because of this inability to do {D-1}. Creates things that compile
    // but really should not, but is not visible about it.
    // Easier just to use the 1's as here.
    ein_sum(t, D - 2, D - 1)
}

#[cfg(test)]
pub(crate) mod test {

    use burn::{backend::NdArray, prelude::Backend, tensor::Tensor};

    use crate::manifolds::utils::{diag_i, ein_sum, identity_in_last_two};

    pub(crate) fn assert_matrix_close<TestBackend>(
        a: &Tensor<TestBackend, 2>,
        b: &Tensor<TestBackend, 2>,
        tol: f32,
    ) where
        TestBackend: Backend,
        <TestBackend as Backend>::FloatElem: PartialOrd<f32>,
    {
        let diff = (a.clone() - b.clone()).abs();
        let max_diff = diff.max().into_scalar();
        assert!(
            max_diff < tol,
            "Tensors differ by {}, tolerance: {}",
            max_diff,
            tol
        );
    }

    pub(crate) fn create_test_matrix<TestBackend: Backend>(
        rows: usize,
        cols: usize,
        values: Vec<f32>,
    ) -> Tensor<TestBackend, 2> {
        debug_assert_ne!(rows, 0);
        debug_assert_ne!(cols, 0);
        if rows < cols {
            return create_test_matrix(cols, rows, values).transpose();
        }
        let device = Default::default();
        // Reshape the flat vector into a 2D array
        let mut data = Vec::with_capacity(rows);
        for chunk in values.chunks(cols) {
            data.push(chunk.to_vec());
        }

        // Create tensor from nested arrays
        match (rows, cols) {
            (3, 2) => {
                if data.len() >= 3 && data[0].len() >= 2 && data[1].len() >= 2 && data[2].len() >= 2
                {
                    Tensor::from_floats(
                        [
                            [data[0][0], data[0][1]],
                            [data[1][0], data[1][1]],
                            [data[2][0], data[2][1]],
                        ],
                        &device,
                    )
                } else {
                    panic!("Invalid 3x2 matrix data");
                }
            }
            (3, 1) => {
                if data.len() >= 3
                    && !data[0].is_empty()
                    && !data[1].is_empty()
                    && !data[2].is_empty()
                {
                    Tensor::from_floats([[data[0][0]], [data[1][0]], [data[2][0]]], &device)
                } else {
                    panic!("Invalid 3x1 matrix data");
                }
            }
            (3, 3) => {
                if data.len() >= 3 && data[0].len() >= 3 && data[1].len() >= 3 && data[2].len() >= 3
                {
                    Tensor::from_floats(
                        [
                            [data[0][0], data[0][1], data[0][2]],
                            [data[1][0], data[1][1], data[1][2]],
                            [data[2][0], data[2][1], data[2][2]],
                        ],
                        &device,
                    )
                } else {
                    panic!("Invalid 3x3 matrix data");
                }
            }
            (4, 2) => {
                if data.len() >= 4
                    && data[0].len() >= 2
                    && data[1].len() >= 2
                    && data[2].len() >= 2
                    && data[3].len() >= 2
                {
                    Tensor::from_floats(
                        [
                            [data[0][0], data[0][1]],
                            [data[1][0], data[1][1]],
                            [data[2][0], data[2][1]],
                            [data[3][0], data[3][1]],
                        ],
                        &device,
                    )
                } else {
                    panic!("Invalid 4x2 matrix data");
                }
            }
            (2, 2) => {
                if data.len() >= 2 && data[0].len() >= 2 && data[1].len() >= 2 {
                    Tensor::from_floats(
                        [[data[0][0], data[0][1]], [data[1][0], data[1][1]]],
                        &device,
                    )
                } else {
                    panic!("Invalid 2x2 matrix data");
                }
            }
            (2, 1) => {
                if data.len() >= 2 && !data[0].is_empty() && !data[1].is_empty() {
                    Tensor::from_floats([[data[0][0]], [data[1][0]]], &device)
                } else {
                    panic!("Invalid 2x1 matrix data");
                }
            }
            (1, 1) => {
                if data.len() >= 1 && !data[0].is_empty() {
                    Tensor::from_floats([[data[0][0]]], &device)
                } else {
                    panic!("Invalid 1x1 matrix data");
                }
            }
            _ => panic!("Unsupported matrix dimensions: {}x{}", rows, cols),
        }
    }

    #[test]
    fn small_einsum() {
        let mat = create_test_matrix::<NdArray>(
            3,
            3,
            vec![3.0, 4.0, 5.0, 6.0, 7.0, 3.0, -10.0, -4.0, -1.0],
        );
        let ein_summed = ein_sum(mat.clone(), 0, 1);
        assert_eq!(ein_summed.shape().dims(), [1, 1]);
        let scalar = ein_summed.into_scalar();
        assert!((scalar - 9.0).abs() <= 1e-6, "{}", scalar);
    }

    #[test]
    fn identity_test() {
        {
            let mat = create_test_matrix::<NdArray>(
                3,
                3,
                vec![3.0, 4.0, 5.0, 6.0, 7.0, 3.0, -10.0, -4.0, -1.0],
            );
            let mat = mat.expand([3, 3]);
            let identity_mat = identity_in_last_two(&mat);
            let expected = Tensor::eye(3, &identity_mat.device());
            assert_matrix_close(&identity_mat, &expected, 1e-6);
        }
        let mat = create_test_matrix::<NdArray>(
            3,
            3,
            vec![3.0, 4.0, 5.0, 6.0, 7.0, 3.0, -10.0, -4.0, -1.0],
        );
        let expanded_shape = [3, 3, 3, 3, 3];
        let mat = mat.expand(expanded_shape);
        let identity_mat = identity_in_last_two(&mat);
        for idx in 0..expanded_shape[0] {
            for jdx in 0..expanded_shape[1] {
                for kdx in 0..expanded_shape[2] {
                    let slice = identity_mat
                        .clone()
                        .slice([idx..idx + 1, jdx..jdx + 1, kdx..kdx + 1, 0..3, 0..3])
                        .reshape([3, 3]);
                    let expected = Tensor::eye(3, &slice.device());
                    assert_matrix_close(&slice, &expected, 1e-6);
                }
            }
        }
        let mat = create_test_matrix::<NdArray>(
            3,
            3,
            vec![3.0, 4.0, 5.0, 6.0, 7.0, 3.0, -10.0, -4.0, -1.0],
        );
        let expanded_shape = [29, 483, 2, 3, 3];
        let mat = mat.expand(expanded_shape);
        let identity_mat = identity_in_last_two(&mat);
        for idx in 0..expanded_shape[0] {
            for jdx in 0..expanded_shape[1] {
                for kdx in 0..expanded_shape[2] {
                    let slice = identity_mat
                        .clone()
                        .slice([idx..idx + 1, jdx..jdx + 1, kdx..kdx + 1, 0..3, 0..3])
                        .reshape([3, 3]);
                    let expected = Tensor::eye(3, &slice.device());
                    assert_matrix_close(&slice, &expected, 1e-6);
                }
            }
        }
    }

    #[test]
    fn diag_test() {
        let diag_entries = [2.0, 7.0, 9.0];
        let expected =
            create_test_matrix::<NdArray>(3, 3, vec![2.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 9.0]);
        let expanded_shape = [3, 3, 3, 3, 3];
        let mat = expected.clone().expand(expanded_shape);
        let identity_mat = diag_i(&mat, |i| diag_entries[i]);
        for idx in 0..expanded_shape[0] {
            for jdx in 0..expanded_shape[1] {
                for kdx in 0..expanded_shape[2] {
                    let slice = identity_mat
                        .clone()
                        .slice([idx..idx + 1, jdx..jdx + 1, kdx..kdx + 1, 0..3, 0..3])
                        .reshape([3, 3]);
                    assert_matrix_close(&slice, &expected, 1e-6);
                }
            }
        }
        let expanded_shape = [10, 9, 20, 3, 3];
        let mat = expected.clone().expand(expanded_shape);
        let identity_mat = diag_i(&mat, |i| diag_entries[i]);
        for idx in 0..expanded_shape[0] {
            for jdx in 0..expanded_shape[1] {
                for kdx in 0..expanded_shape[2] {
                    let slice = identity_mat
                        .clone()
                        .slice([idx..idx + 1, jdx..jdx + 1, kdx..kdx + 1, 0..3, 0..3])
                        .reshape([3, 3]);
                    assert_matrix_close(&slice, &expected, 1e-6);
                }
            }
        }
    }
}
