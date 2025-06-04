use burn::module::AutodiffModule;
use burn::optim::{LrDecayState, Optimizer, SimpleOptimizer};
use burn::record::Record;
use burn::tensor::backend::AutodiffBackend;
use burn::LearningRate;
use std::marker::PhantomData;

use crate::manifolds::Manifold;
use crate::prelude::*;

#[derive(Clone)]
pub struct ManifoldRGD<M: Manifold<B>, B: Backend> {
    manifold: PhantomData<M>,
    _backend: PhantomData<B>,
}

#[derive(Record, Clone)]
pub struct ManifoldRGDState<B: Backend, const D: usize> {
    lr_decay: LrDecayState<B, D>,
}

impl<M, B> SimpleOptimizer<B> for ManifoldRGD<M, B>
where
   M: Manifold<B>,
   B: Backend,
{
    type State<const D: usize> = ManifoldRGDState<B, D>;
    
    fn step<const D: usize>(
            &self,
            lr: LearningRate,
            tensor: Tensor<B, D>,
            grad: Tensor<B, D>,
            state: Option<Self::State<D>>,
        ) -> (Tensor<B, D>, Option<Self::State<D>>)
    {
        let direction = M::project(tensor.clone(), grad);
        let result = M::retract(tensor, direction, lr);
        (result, state)
    }

    fn to_device<const D: usize>(_state: Self::State<D>, _device: &<B as Backend>::Device) -> Self::State<D> {
        _state
    }
}


// impl<M, B, const D: usize> Optimizer<M, B> for ManifoldRGD<M, B, D>
// where
//     M: AutodiffModule<B>,
//     B: AutodiffBackend,
//     M: Manifold<B, D>,
//     B: Backend,
// {
//     type Record = ManifoldRGDState<B, D>;

//     fn step(&mut self, lr: LearningRate, module: M, grads: burn::optim::GradientsParams) -> M {
        
//     }

//     fn to_record(&self) -> Self::Record {
//         todo!()
//     }

//     fn load_record(self, record: Self::Record) -> Self {
//         todo!()
//     }
// }