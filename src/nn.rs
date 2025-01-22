use ndarray::prelude::*;
use rand::Rng;
use rand_distr::{Distribution, Normal, Uniform};

use crate::partition::*;
use crate::util::*;

/// A neural network module
pub trait NNModule {
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

impl NNModule for Linear {
    fn apply(&self, p: &mut Partition) {
        p.linear(&self.weight.view(), &self.bias.view());
    }

    fn forward(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        x * &self.weight + &self.bias
    }
}

impl<M:NNModule> NNModule for [M] {
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
    LeakyReLU(f32),
    Sequential(Vec<Module>),
    Residual(Vec<Module>),
}

impl NNModule for Module {
    fn apply(&self, p: &mut Partition) {
        use Module::*;
        match self {
            Identity => (),
            Linear(module) => module.apply(p),
            ReLU => p.relu(),
            LeakyReLU(factor) => p.leaky_relu(*factor),
            Sequential(module) => module.apply(p),
            Residual(module) => {
                let layer = p.last_layer();
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
            LeakyReLU(factor) => leaky_relu(x, *factor),
            Sequential(module) => module.forward(x),
            Residual(module) => x + module.forward(x),
        }
    }
}
