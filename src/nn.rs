use std::sync::Arc;

use ndarray::prelude::*;
use ndarray::Data;
use ndarray::OwnedRepr;
use ndarray::RawData;
use ndarray::RawDataClone;
use rand::Rng;
use rand_distr::{Distribution, Normal, Uniform};

use crate::regioncam::*;
use crate::util::*;

/// A neural network module
pub trait NNModule {
    fn add_to(&self, rc: &mut Regioncam, input_layer: LayerNr) -> LayerNr;
    fn forward(&self, x: &ArrayView2<f32>) -> Array2<f32>;
    // fn repr(&self) -> String;
}

/// A linear layer in a neural network
#[derive(Clone, Debug)]
pub struct LinearBase<S: RawData<Elem=f32> + RawDataClone + Data> {
    pub weight: ArrayBase<S, Ix2>,
    pub bias: ArrayBase<S, Ix1>,
}

pub type Linear = LinearBase<OwnedRepr<f32>>;

impl Linear {
    pub fn new<R: Rng + ?Sized>(dim_in: usize, dim_out: usize, rng: &mut R) -> Linear {
        let std = 1.0 / f32::sqrt(dim_in as f32);
        let distr = Normal::new(0.0, std).unwrap();
        Self::from_distr(dim_in, dim_out, distr, rng)
    }
    pub fn new_uniform<R: Rng + ?Sized>(dim_in: usize, dim_out: usize, rng: &mut R) -> Linear {
        let std = 1.0 / f32::sqrt(dim_in as f32);
        let distr = Uniform::new(-std, std);
        Self::from_distr(dim_in, dim_out, distr, rng)
    }
    pub fn from_distr<R: Rng + ?Sized, D: Distribution<f32>>(dim_in: usize, dim_out: usize, distr: D, rng: &mut R) -> Linear {
        let weight =Array::from_shape_simple_fn((dim_in, dim_out),|| distr.sample(rng));
        let bias = Array::from_shape_simple_fn(dim_out,|| distr.sample(rng));
        Linear { weight, bias }
    }
}

impl<S: RawData<Elem=f32> + RawDataClone + Data> NNModule for LinearBase<S> {
    fn add_to(&self, rc: &mut Regioncam, input_layer: LayerNr) -> LayerNr {
        rc.linear_at(input_layer, &self.weight.view(), &self.bias.view());
        rc.last_layer()
    }

    fn forward(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        x.dot(&self.weight) + &self.bias
    }
}

impl<M: NNModule> NNModule for [M] {
    fn add_to(&self, rc: &mut Regioncam, mut layer: LayerNr) -> LayerNr {
        for module in self {
            layer = module.add_to(rc, layer);
        }
        layer
    }

    fn forward(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        let mut x = x.into_owned(); // Note: slightly inefficient
        for module in self {
            x = module.forward(&x.view());
        }
        x
    }
}

impl<M: NNModule> NNModule for Arc<M> {
    fn add_to(&self, rc: &mut Regioncam, input_layer: LayerNr) -> LayerNr {
        self.as_ref().add_to(rc, input_layer)
    }
    
    fn forward(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        self.as_ref().forward(x)
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
    fn add_to(&self, rc: &mut Regioncam, input_layer: LayerNr) -> LayerNr {
        use Module::*;
        match self {
            Identity => input_layer,
            Linear(module) => module.add_to(rc, input_layer),
            ReLU => rc.relu_at(input_layer),
            LeakyReLU(negative_slope) => rc.leaky_relu_at(input_layer, *negative_slope),
            Sequential(module) => module.add_to(rc, input_layer),
            Residual(module) => {
                let layer = rc.last_layer();
                module.add_to(rc, input_layer);
                rc.residual(layer);
                rc.last_layer()
            },
        }
    }

    fn forward(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        use Module::*;
        match self {
            Identity => x.into_owned(),
            Linear(module) => module.forward(x),
            ReLU => relu(x),
            LeakyReLU(negative_slope) => leaky_relu(x, *negative_slope),
            Sequential(module) => module.forward(x),
            Residual(module) => x + module.forward(x),
        }
    }
}
