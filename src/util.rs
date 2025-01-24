// Utility functions

use std::cmp::Ordering;

use ndarray::{Array, Array1, Array2, Array3, ArrayView, ArrayView1, Axis, Dimension, RemoveAxis};
use num_traits::Zero;

pub fn relu<D: Dimension>(arr: &ArrayView<f32, D>) -> Array<f32, D> {
    arr.mapv(|x| x.max(0.0))
}

pub fn leaky_relu<D: Dimension>(arr: &ArrayView<f32, D>, negative_slope: f32) -> Array<f32, D> {
    arr.mapv(|x| if x < 0.0 { negative_slope * x } else { x } )
}

pub(crate) fn argmax(row: ArrayView1<f32>) -> usize {
    let max = row.iter().enumerate().max_by(|x, y| if x.1 < y.1 { Ordering::Less} else { Ordering::Greater });
    max.map_or(0, |(i, _)| i)
}

pub(crate) fn histogram(values: &[usize], len: usize) -> Vec<usize> {
    let mut counts = vec![0; len];
    for value in values {
        counts[*value] += 1;
    }
    counts
}

/// For each row, select the given element in axis
pub(crate) fn select_in_rows<T, D>(axis: Axis, arr: &ArrayView<T, D>, idxs: Array1<usize>) -> Array<T, D::Smaller>
    where
    T: Clone + Zero + std::fmt::Display,
    D: Dimension + RemoveAxis,
    D::Smaller: RemoveAxis,
{
    let mut out = Array::zeros(arr.raw_dim().remove_axis(axis));
    for ((mut out_row, in_row), idx) in out.axis_iter_mut(Axis(0)).zip(arr.axis_iter(Axis(0))).zip(idxs) {
        out_row.assign(&in_row.index_axis(Axis(axis.0-1), idx));
    }
    out
}