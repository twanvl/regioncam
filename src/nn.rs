use std::sync::Arc;

use ndarray::prelude::*;
use ndarray::Data;
use ndarray::OwnedRepr;
use ndarray::RawData;
use ndarray::RawDataClone;
use rand::Rng;
use rand_distr::{Distribution, Normal, Uniform};

use crate::util::*;

/// Trait for objects to which neural network layers can be added
pub trait NNBuilder: Sized {
    type LayerNr: Copy;
    /// Append a layer that computes output using the given function.
    /// The function should be linear, without a bias term.
    /// Then `add_bias` should add the bias term to the output.
    fn generalized_linear_at(&mut self, layer: Self::LayerNr, fun: impl Fn(ArrayView2<f32>) -> Array2<f32>, add_bias: impl Fn(ArrayViewMut2<f32>)) -> Self::LayerNr;
    /// Add a ReLU layer, that takes as input the output of the given layer.
    fn relu_at(&mut self, layer_nr: Self::LayerNr) -> Self::LayerNr;
    fn leaky_relu_at(&mut self, layer_nr: Self::LayerNr, negative_slope: f32) -> Self::LayerNr;
    fn max_pool_at(&mut self, layer_nr: Self::LayerNr) -> Self::LayerNr;
    fn argmax_pool_at(&mut self, layer_nr: Self::LayerNr) -> Self::LayerNr;
    fn sign_at(&mut self, layer_nr: Self::LayerNr) -> Self::LayerNr;
    /// Append a classification layer:
    /// with 1 output a sign based classifier, with >1 outputs an argmax layer
    fn decision_boundary_at(&mut self, layer_nr: Self::LayerNr) -> Self::LayerNr;
    /// Append a layer that adds the output of two existing layers
    fn sum(&mut self, layer1: Self::LayerNr, layer2: Self::LayerNr) -> Self::LayerNr;

    /// Layer number of the last layer.
    /// This is used as the input layer when adding new layers with the not `_at` methods.
    fn last_layer(&self) -> Self::LayerNr;
    
    /// Append a layer that computes output using a linear transformation:
    ///   x_{l} = w*x_l' + b.
    fn linear_at(&mut self, layer: Self::LayerNr, weight: &ArrayView2<f32>, bias: &ArrayView1<f32>) -> Self::LayerNr {
        self.generalized_linear_at(layer, |x| x.dot(weight), |mut x| x += bias)
    }
    /// Append a layer that computes output using a linear transformation:
    ///   x_{l+1} = w*x_l + b.
    fn linear(&mut self, weight: &ArrayView2<f32>, bias: &ArrayView1<f32>) {
        self.linear_at(self.last_layer(), weight, bias);
    }
    fn relu(&mut self) {
        self.relu_at(self.last_layer());
    }
    fn leaky_relu(&mut self, negative_slope: f32) {
        self.leaky_relu_at(self.last_layer(), negative_slope);
    }
    fn max_pool(&mut self) {
        self.max_pool_at(self.last_layer());
    }
    fn argmax_pool(&mut self) {
        self.argmax_pool_at(self.last_layer());
    }
    fn sign(&mut self) {
        self.sign_at(self.last_layer());
    }
    fn decision_boundary(&mut self) {
        self.decision_boundary_at(self.last_layer());
    }
    /// Append a layer that adds an earlier layer to the output
    fn residual(&mut self, layer: Self::LayerNr) {
        self.sum(layer, self.last_layer());
    }

    /// Add a NNmodule to the network
    fn add_at(&mut self, layer_nr: Self::LayerNr, module: &impl NNModule) -> Self::LayerNr {
        module.add_to(self, layer_nr)
    }
    /// Add a NNmodule to the network
    fn add(&mut self, module: &impl NNModule) {
        self.add_at(self.last_layer(), module);
    }
}

/// A neural network module
pub trait NNModule {
    fn add_to<B: NNBuilder>(&self, rc: &mut B, input_layer: B::LayerNr) -> B::LayerNr;
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
    /// Create a linear layer with weights and biases initialized using the given distribution
    pub fn from_distr<R: Rng + ?Sized, D: Distribution<f32>>(dim_in: usize, dim_out: usize, distr: D, rng: &mut R) -> Linear {
        let weight = Array::from_shape_simple_fn((dim_in, dim_out),|| distr.sample(rng));
        let bias = Array::from_shape_simple_fn(dim_out,|| distr.sample(rng));
        Linear { weight, bias }
    }
    pub fn dim_in(&self) -> usize {
        self.weight.nrows()
    }
    pub fn dim_out(&self) -> usize {
        self.weight.ncols()
    }
}

impl<S: RawData<Elem=f32> + RawDataClone + Data> NNModule for LinearBase<S> {
    fn add_to<B: NNBuilder>(&self, rc: &mut B, input_layer: B::LayerNr) -> B::LayerNr {
        rc.linear_at(input_layer, &self.weight.view(), &self.bias.view());
        rc.last_layer()
    }

    fn forward(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        x.dot(&self.weight) + &self.bias
    }
}

impl<M: NNModule> NNModule for [M] {
    fn add_to<B: NNBuilder>(&self, rc: &mut B, mut layer: B::LayerNr) -> B::LayerNr {
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
    fn add_to<B: NNBuilder>(&self, rc: &mut B, input_layer: B::LayerNr) -> B::LayerNr {
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
    fn add_to<B: NNBuilder>(&self, rc: &mut B, input_layer: B::LayerNr) -> B::LayerNr {
    //fn add_to(&self, rc: &mut impl NNBuilder, input_layer: LayerNr) -> LayerNr {
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
