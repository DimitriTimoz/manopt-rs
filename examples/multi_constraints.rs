use burn::module::Module;
use burn::module::ModuleVisitor;
use burn::nn::Linear;
use burn::nn::LinearConfig;

use manopt_rs::constrained_module::Constrained;
use manopt_rs::optimizers::multiple::{
    ManifoldOptimizable, MultiManifoldOptimizer, MultiManifoldOptimizerConfig,
};
use manopt_rs::prelude::*;

// Example: User-defined custom manifold
#[derive(Clone, Debug)]
pub struct CustomSphereManifold;

impl<B: Backend> Manifold<B> for CustomSphereManifold {
    const RANK_PER_POINT: usize = 1;

    fn new() -> Self {
        Self
    }

    fn name() -> &'static str {
        "CustomSphere"
    }

    fn project<const D: usize>(point: Tensor<B, D>, vector: Tensor<B, D>) -> Tensor<B, D> {
        // For sphere: project vector orthogonal to point
        debug_assert!(point.shape() == vector.shape());
        let dot_product =
            (point.clone() * vector.clone()).sum_dim(D - <Self as Manifold<B>>::RANK_PER_POINT);
        vector - point * dot_product
    }

    fn retract<const D: usize>(point: Tensor<B, D>, direction: Tensor<B, D>) -> Tensor<B, D> {
        // For sphere: normalize the result
        debug_assert!(point.shape() == direction.shape());
        let new_point = point + direction;
        let norm = new_point
            .clone()
            .powf_scalar(2.0)
            .sum_dim(D - <Self as Manifold<B>>::RANK_PER_POINT)
            .sqrt();
        new_point / norm
    }

    fn inner<const D: usize>(
        _point: Tensor<B, D>,
        u: Tensor<B, D>,
        v: Tensor<B, D>,
    ) -> Tensor<B, D> {
        (u * v).sum_dim(D - <Self as Manifold<B>>::RANK_PER_POINT)
    }

    fn proj<const D: usize>(point: Tensor<B, D>) -> Tensor<B, D> {
        // Project point onto unit sphere
        let norm = point
            .clone()
            .powf_scalar(2.0)
            .sum_dim(D - <Self as Manifold<B>>::RANK_PER_POINT)
            .sqrt();
        point / norm
    }

    fn is_in_manifold<const D: usize>(point: Tensor<B, D>) -> Tensor<B, D, Bool> {
        let r_squared = point
            .powf_scalar(2.0)
            .sum_dim(D - <Self as Manifold<B>>::RANK_PER_POINT);
        let one = r_squared.ones_like();
        r_squared.is_close(one, None, None)
    }

    fn is_tangent_at<const D: usize>(
        point: Tensor<B, D>,
        vector: Tensor<B, D>,
    ) -> Tensor<B, D, Bool> {
        let dot_product = (point * vector).sum_dim(D - <Self as Manifold<B>>::RANK_PER_POINT);
        let zeros = dot_product.zeros_like();
        dot_product.is_close(zeros, None, Some(1e-6))
    }

    fn acceptable_dims(a_is: &[usize]) -> bool {
        let n = *a_is.first().expect("The ambient R^n does exist");
        n > 0
    }
}

#[derive(Debug, Clone)]
pub struct TestModel<B: Backend> {
    // Euclidean constrained linear layer
    linear_euclidean: Constrained<Linear<B>, Euclidean>,
    // Custom sphere constrained linear layer
    linear_sphere: Constrained<Linear<B>, CustomSphereManifold>,
    // Regular unconstrained linear layer
    linear_regular: Linear<B>,
}

impl<B: Backend> Module<B> for TestModel<B> {
    type Record = (
        <Constrained<Linear<B>, Euclidean> as Module<B>>::Record,
        <Constrained<Linear<B>, CustomSphereManifold> as Module<B>>::Record,
        <Linear<B> as Module<B>>::Record,
    );

    fn collect_devices(&self, mut devices: burn::module::Devices<B>) -> burn::module::Devices<B> {
        devices = self.linear_euclidean.collect_devices(devices);
        devices = self.linear_sphere.collect_devices(devices);
        devices = self.linear_regular.collect_devices(devices);
        devices
    }

    fn fork(self, device: &B::Device) -> Self {
        Self {
            linear_euclidean: self.linear_euclidean.fork(device),
            linear_sphere: self.linear_sphere.fork(device),
            linear_regular: self.linear_regular.fork(device),
        }
    }

    fn to_device(self, device: &B::Device) -> Self {
        Self {
            linear_euclidean: self.linear_euclidean.to_device(device),
            linear_sphere: self.linear_sphere.to_device(device),
            linear_regular: self.linear_regular.to_device(device),
        }
    }

    fn visit<Visitor: burn::module::ModuleVisitor<B>>(&self, visitor: &mut Visitor) {
        self.linear_euclidean.visit(visitor);
        self.linear_sphere.visit(visitor);
        self.linear_regular.visit(visitor);
    }

    fn map<Mapper: burn::module::ModuleMapper<B>>(self, mapper: &mut Mapper) -> Self {
        Self {
            linear_euclidean: self.linear_euclidean.map(mapper),
            linear_sphere: self.linear_sphere.map(mapper),
            linear_regular: self.linear_regular.map(mapper),
        }
    }

    fn load_record(self, record: Self::Record) -> Self {
        Self {
            linear_euclidean: self.linear_euclidean.load_record(record.0),
            linear_sphere: self.linear_sphere.load_record(record.1),
            linear_regular: self.linear_regular.load_record(record.2),
        }
    }

    fn into_record(self) -> Self::Record {
        (
            self.linear_euclidean.into_record(),
            self.linear_sphere.into_record(),
            self.linear_regular.into_record(),
        )
    }
}

impl<B: Backend> TestModel<B> {
    pub fn new(device: &B::Device) -> Self {
        let linear_euclidean = LinearConfig::new(10, 5).init(device);
        let linear_sphere = LinearConfig::new(5, 3).init(device);
        let linear_regular = LinearConfig::new(3, 1).init(device);

        Self {
            linear_euclidean: Constrained::new(linear_euclidean),
            linear_sphere: Constrained::new(linear_sphere),
            linear_regular,
        }
    }
}

struct ManifoldAwareVisitor;

impl<B: Backend> ModuleVisitor<B> for ManifoldAwareVisitor {
    fn visit_float<const D: usize>(&mut self, id: burn::module::ParamId, tensor: &Tensor<B, D>) {
        println!(
            "Visiting parameter: {:?} with shape: {:?}",
            id,
            tensor.dims()
        );
    }

    fn visit_int<const D: usize>(
        &mut self,
        _id: burn::module::ParamId,
        _tensor: &Tensor<B, D, Int>,
    ) {
    }

    fn visit_bool<const D: usize>(
        &mut self,
        _id: burn::module::ParamId,
        _tensor: &Tensor<B, D, Bool>,
    ) {
    }
}

fn main() {
    type MyBackend = burn::backend::NdArray;
    type AutoDiffBackend = burn::backend::Autodiff<MyBackend>;

    let device = Default::default();

    // Create a model with mixed manifold constraints
    let model = TestModel::<AutoDiffBackend>::new(&device);

    println!("=== Model Structure ===");
    println!(
        "Euclidean layer manifold: {}",
        model.linear_euclidean.manifold_name::<AutoDiffBackend>()
    );
    println!(
        "Sphere layer manifold: {}",
        model.linear_sphere.manifold_name::<AutoDiffBackend>()
    );

    // Create multi-manifold optimizer
    let config = MultiManifoldOptimizerConfig::default();
    let mut optimizer = MultiManifoldOptimizer::new(config);

    // Collect manifold information from the model
    optimizer.collect_manifolds(&model);

    // Register custom manifold for specific parameters (if needed)
    optimizer.register_manifold::<CustomSphereManifold>("linear_sphere.weight".to_string());

    println!("\n=== Manifold Information ===");
    println!(
        "Euclidean info: {:?}",
        model.linear_euclidean.get_manifold_info()
    );
    println!("Sphere info: {:?}", model.linear_sphere.get_manifold_info());

    // Example of applying constraints
    let constrained_model = optimizer.apply_constraints(model);

    // Visit the model to see parameter structure
    println!("\n=== Parameter Structure ===");
    let mut visitor = ManifoldAwareVisitor;
    constrained_model.visit(&mut visitor);

    println!("\n=== Demonstrating Custom Manifold Operations ===");

    // Show how the custom sphere manifold works
    let point = Tensor::<MyBackend, 1>::from_floats([3.0, 4.0, 0.0], &device);
    let vector = Tensor::<MyBackend, 1>::from_floats([1.0, 1.0, 1.0], &device);

    println!("Original point: {:?}", point.to_data());
    println!("Original vector: {:?}", vector.to_data());

    // Project point to sphere
    let projected_point = CustomSphereManifold::proj(point.clone());
    println!("Point projected to sphere: {:?}", projected_point.to_data());

    // Project vector to tangent space
    let projected_vector = CustomSphereManifold::project(projected_point.clone(), vector);
    println!(
        "Vector projected to tangent space: {:?}",
        projected_vector.to_data()
    );

    // Check if point is on manifold
    println!(
        "Is projected point on sphere? {}",
        CustomSphereManifold::is_in_manifold(projected_point.clone())
    );

    // Check if vector is tangent at point on manifold
    println!(
        "Is projected vector tangent to point on sphere? {}",
        CustomSphereManifold::is_tangent_at(projected_point, projected_vector)
    );
}
