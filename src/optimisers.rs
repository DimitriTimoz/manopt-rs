use burn::optim::{LrDecayState, SimpleOptimizer};
use burn::record::Record;
use burn::LearningRate;
use std::marker::PhantomData;

use crate::manifolds::Manifold;
use crate::prelude::*;

#[derive(Clone)]
pub struct ManifoldRGD<M: Manifold<B,D>, B: Backend, const D: usize> {
    manifold: PhantomData<M>,
    _backend: PhantomData<B>,
}

#[derive(Record, Clone)]
pub struct ManifoldRGDState<B: Backend, const D: usize> {
    lr_decay: LrDecayState<B, D>,
}

impl<M, B, const D: usize> SimpleOptimizer<B> for ManifoldRGD<M, B, D>
where
   M: Manifold<B, D>,
   B: Backend,
{
    type State<const D2: usize> = ManifoldRGDState<B, D2>;
    
    fn step<const D2: usize>(
            &self,
            lr: LearningRate,
            tensor: Tensor<B, D2>,
            grad: Tensor<B, D2>,
            state: Option<Self::State<D2>>,
        ) -> (Tensor<B, D2>, Option<Self::State<D2>>) {
        // Ensure dimensions match at runtime
        assert_eq!(D, D2, "Manifold dimension D must equal tensor dimension D2");
        
        // Cast tensors to the manifold dimension
        // This is safe because we've verified D == D2
        let tensor_d: Tensor<B, D> = unsafe { std::mem::transmute_copy(&tensor) };
        let grad_d: Tensor<B, D> = unsafe { std::mem::transmute_copy(&grad) };
        
        // Call manifold methods
        let direction = M::project(&tensor_d, &grad_d);
        let new_tensor = M::retract(&tensor_d, &direction, lr);
        
        // Cast back to D2
        let result: Tensor<B, D2> = unsafe { std::mem::transmute_copy(&new_tensor) };
        
        (result, state)
    }
    
    fn to_device<const D2: usize>(_state: Self::State<D2>, _device: &<B as Backend>::Device) -> Self::State<D2> {
        todo!()
    }
    

}
