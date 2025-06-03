use burn::optim::{LrDecayState, SimpleOptimizer};
use burn::record::Record;
use burn::LearningRate;
use std::marker::PhantomData;

use crate::manifolds::Manifold;
use crate::prelude::*;

#[derive(Clone)]
pub struct ManifoldRGD<M: Manifold<B,D>, B: Backend, const D: usize> {
    learning_rate: f64,
    _manifold: PhantomData<M>,
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
            _lr: LearningRate,
            _tensor: Tensor<B, D2>,
            _grad: Tensor<B, D2>,
            _state: Option<Self::State<D2>>,
        ) -> (Tensor<B, D2>, Option<Self::State<D2>>) {
        todo!()
    }
    fn to_device<const D2: usize>(_state: Self::State<D2>, _device: &<B as Backend>::Device) -> Self::State<D2> {
        todo!()
    }
    

}
