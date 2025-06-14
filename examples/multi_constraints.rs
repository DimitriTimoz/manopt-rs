use burn::module::ModuleVisitor;
use burn::nn::LinearConfig;
use burn::nn::Linear;
use burn::module::{Module, ModuleMapper};

use manopt_rs::manifolds::Constrained;
use manopt_rs::prelude::*;

#[derive(Module, Debug)]
pub struct TestModel<B: Backend> {
    linear_inner: Constrained<Linear<B>, Euclidean>
}

struct T;
impl<B:Backend> ModuleMapper<B> for T{
    fn map_float<const D: usize>(&mut self, _id: burn::module::ParamId, tensor: Tensor<B, D>) -> Tensor<B, D> {
        tensor
    }

    fn map_int<const D: usize>(
        &mut self,
        _id: burn::module::ParamId,
        tensor: Tensor<B, D, Int>,
    ) -> Tensor<B, D, Int> {
        tensor
    }

    fn map_bool<const D: usize>(
        &mut self,
        _id: burn::module::ParamId,
        tensor: Tensor<B, D, Bool>,
    ) -> Tensor<B, D, Bool> {
        tensor
    }
}

impl<B: Backend> ModuleVisitor<B> for T {
    fn visit_float<const D: usize>(&mut self, _id: burn::module::ParamId, tensor: &Tensor<B, D>) {
        println!("Visiting float tensor with id: {:?}, value: {:?}", _id, tensor);
    }

    fn visit_int<const D: usize>(&mut self, _id: burn::module::ParamId, tensor: &Tensor<B, D, Int>) {
        println!("Visiting int tensor with id: {:?}, value: {:?}", _id, tensor);
    }

    fn visit_bool<const D: usize>(&mut self, _id: burn::module::ParamId, tensor: &Tensor<B, D, Bool>) {
        println!("Visiting bool tensor with id: {:?}, value: {:?}", _id, tensor);
    }
}


fn main() {
    type MyBackend = burn::backend::NdArray;
    type AutoDiffBackend = burn::backend::Autodiff<MyBackend>;

    let device = Default::default();
    let model = TestModel {
        linear_inner: Constrained::new(LinearConfig::new(1, 1).init::<AutoDiffBackend>(&device))
    };
    let mut visitor = T;
    model.visit(&mut visitor);

}
