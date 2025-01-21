// Utility functions

use ndarray::{Array, ArrayView, Dimension};

pub fn relu<D: Dimension>(arr: &ArrayView<f32, D>) -> Array<f32, D> {
    arr.mapv(|x| x.max(0.0))
}

pub fn leaky_relu<D: Dimension>(arr: &ArrayView<f32, D>, factor: f32) -> Array<f32, D> {
    arr.mapv(|x| if x < 0.0 { factor * x } else { x } )
}
