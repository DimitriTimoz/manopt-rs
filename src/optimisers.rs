use burn::module::AutodiffModule;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{LrDecayState, SimpleOptimizer};
use burn::record::Record;
use burn::tensor::backend::AutodiffBackend;
use burn::LearningRate;
use std::marker::PhantomData;

use crate::manifolds::Manifold;
use crate::prelude::*;

#[derive(Debug)]
pub struct ManifoldRGDConfig<M, B> {
    _manifold: PhantomData<M>,
    _backend: PhantomData<B>,
}

impl<M, B> Default for ManifoldRGDConfig<M, B>
where
    M: Manifold<B>,
    B: Backend,
{
    fn default() -> Self {
        Self {
            _manifold: PhantomData,
            _backend: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ManifoldRGD<M: Manifold<B>, B: Backend> {
    _manifold: PhantomData<M>,
    _backend: PhantomData<B>,
}

impl<M,B> Default for ManifoldRGD<M, B>
where
    M: Manifold<B>,
    B: Backend,
{
    fn default() -> Self {
        Self {
            _manifold: PhantomData,
            _backend: PhantomData,
        }
    }
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

impl<M, B> ManifoldRGDConfig<M, B> 
where
    M: Manifold<B>,
    B: Backend,
{
    pub fn init<Back: AutodiffBackend, Mod: AutodiffModule<Back>>(
        &self,
    ) -> OptimizerAdaptor<ManifoldRGD<M, Back::InnerBackend>, Mod, Back> 
    where
        M: Manifold<Back::InnerBackend>,
    {
        let optim = ManifoldRGD::<M, Back::InnerBackend>::default();

        OptimizerAdaptor::from(optim)
    }
}
