use ndarray::prelude::*;
use rand::Rng;
use rand_distr::{Distribution, Normal, Uniform};

use crate::partition::*;

pub trait IsModule {
    fn apply(&self, p: &mut Partition);
    fn forward(&self, x: &ArrayView2<f32>) -> Array2<f32>;
}

/// A linear layer in a neural network
pub struct Linear {
    pub weight: Array2<f32>,
    pub bias: Array1<f32>,
}

fn randn<Sh: ShapeBuilder, R: Rng + ?Sized>(shape: Sh, mean: f32, std: f32, rng: &mut R) -> Array<f32, Sh::Dim> {
    let distr = Normal::new(mean, std).unwrap();
    Array::from_shape_simple_fn(shape, || distr.sample(rng))
}
fn randu<Sh: ShapeBuilder, R: Rng + ?Sized>(shape: Sh, low: f32, high: f32, rng: &mut R) -> Array<f32, Sh::Dim> {
    let distr = Uniform::new(low, high);
    Array::from_shape_simple_fn(shape, || distr.sample(rng))
}

impl Linear {
    pub fn new<R: Rng + ?Sized>(dim_in: usize, dim_out: usize, rng: &mut R) -> Linear {
        let std = 1.0 / f32::sqrt(dim_in as f32);
        let weight = randn((dim_in, dim_out), 0.0, std, rng);
        let bias = randn(dim_out, 0.0, std, rng);
        Linear { weight, bias }
    }
    pub fn new_uniform<R: Rng + ?Sized>(dim_in: usize, dim_out: usize, rng: &mut R) -> Linear {
        let std = 1.0 / f32::sqrt(dim_in as f32);
        let weight = randu((dim_in, dim_out), -std, std, rng);
        let bias = randu(dim_out, -std, std, rng);
        Linear { weight, bias }
    }
}

impl IsModule for Linear {
    fn apply(&self, p: &mut Partition) {
        p.linear_transform(&self.weight.view(), &self.bias.view());
    }

    fn forward(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        (x * &self.weight + &self.bias).into()
    }
}

impl<M:IsModule> IsModule for Vec<M> {
    fn apply(&self, p: &mut Partition) {
        for module in self {
            module.apply(p);
        }
    }

    fn forward(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        let mut x = x.into_owned(); // Note: slightly inefficient
        for module in self {
            x = module.forward(&x.view());
        }
        x
    }
}

/// A generic neural network
pub enum Module {
    Identity,
    Linear(Linear),
    ReLU,
    Sequential(Vec<Module>),
    Residual(Vec<Module>),
}

impl IsModule for Module {
    fn apply(&self, p: &mut Partition) {
        use Module::*;
        match self {
            Identity => (),
            Linear(module) => module.apply(p),
            ReLU => p.relu(),
            Sequential(module) => module.apply(p),
            Residual(module) => {
                let layer = p.num_layers() - 1;
                module.apply(p);
                p.residual(layer);
            },
        }
    }

    fn forward(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        use Module::*;
        match self {
            Identity => x.into_owned(),
            Linear(module) => module.forward(x),
            ReLU => relu(x),
            Sequential(module) => module.forward(x),
            Residual(module) => x + module.forward(x),
        }
    }
}
