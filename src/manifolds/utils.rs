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
        other = other.slice_assign(ranges, ones_patch.clone());
    }
    other
}
