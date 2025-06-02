// Utility functions

use std::cmp::Ordering;
use std::ops::{Add, Mul, Range, Sub};

use ndarray::{s, Array, Array1, ArrayBase, ArrayView, ArrayView1, ArrayView2, Axis, Data, Dimension, Ix0, Ix1, Ix2, Ix3, RemoveAxis, Slice, SliceArg, SliceInfo, SliceInfoElem};
use num_traits::Zero;

/// Rectified linear activation function
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

pub(crate) fn histogram(values: impl IntoIterator<Item=usize>, len: usize) -> Vec<usize> {
    let mut counts = vec![0; len];
    for value in values {
        counts[value] += 1;
    }
    counts
}

pub(crate) const EMPTY_RANGE: Range<f32> = f32::INFINITY..f32::NEG_INFINITY;

/// Extend the range to contain the point x.
#[inline]
pub(crate) fn minmax(mut range: Range<f32>, x: f32) -> Range<f32> {
    range.start = f32::min(range.start, x);
    range.end   = f32::max(range.end, x);
    range
}

/// Compute a bounding box from a set of points (rows of the array)
pub fn bounding_box(arr: &ArrayView2<f32>) -> Array1<Range<f32>> {
    arr.fold_axis(Axis(0), EMPTY_RANGE, |r,x| minmax(r.clone(), *x))
}

/// Norm of a vector
pub fn norm<D: Dimension, S: Data<Elem=f32>>(arr: &ArrayBase<S, D>) -> f32 {
    arr.fold(0., |sum, x| sum + x * x).sqrt()
}

/// Normalize an owned vector
pub fn into_normalized<D: Dimension>(arr: Array<f32, D>) -> Array<f32, D> {
    let norm = norm(&arr);
    arr.mapv_into(|x| x / norm)
}

/// For each row, select the given element in axis
pub(crate) fn select_in_rows<T, D>(axis: Axis, arr: &ArrayView<T, D>, idxs: &Array1<usize>) -> Array<T, D::Smaller>
    where
    T: Clone + Zero + std::fmt::Display,
    D: Dimension + RemoveAxis,
    D::Smaller: RemoveAxis,
{
    let mut out = Array::zeros(arr.raw_dim().remove_axis(axis));
    for ((mut out_row, in_row), idx) in out.axis_iter_mut(Axis(0)).zip(arr.axis_iter(Axis(0))).zip(idxs) {
        out_row.assign(&in_row.index_axis(Axis(axis.0-1), *idx));
    }
    out
}

/// Remove the last row from an array
pub(crate) fn pop_array<T, D: Dimension>(axis: Axis, arr: &mut Array<T, D>) {
    arr.slice_axis_inplace(axis, Slice::new(0, Some(-1), 1));
}

// Workaround to construct s![i, .., .., ..] with the right dimension
pub(crate) trait SliceArg0 where Self: Dimension {
    type Arg: SliceArg<Self>;
    fn slice_arg_axis_0(i: usize) -> Self::Arg;
}
impl SliceArg0 for Ix1 {
    type Arg = SliceInfo<[SliceInfoElem; 1], Ix1, Ix0>;

    fn slice_arg_axis_0(i: usize) -> Self::Arg {
        s![i]
    }
}
impl SliceArg0 for Ix2 {
    type Arg = SliceInfo<[SliceInfoElem; 2], Ix2, Ix1>;

    fn slice_arg_axis_0(i: usize) -> Self::Arg {
        s![i, ..]
    }
}
impl SliceArg0 for Ix3 {
    type Arg = SliceInfo<[SliceInfoElem; 3], Ix3, Ix2>;

    fn slice_arg_axis_0(i: usize) -> Self::Arg {
        s![i, .., ..]
    }
}

/// Copy a row/slice in a multidimensional array. Equivalent to:
/// arr[tgt] = arr[src]
pub(crate) fn assign_row<T: Clone, D: Dimension + SliceArg0>(arr: &mut Array<T, D>, tgt: usize, src: usize) {
    let tgt_slice = <D as SliceArg0>::slice_arg_axis_0(tgt);
    let src_slice = <D as SliceArg0>::slice_arg_axis_0(src);
    let (mut tgt_row, src_row) = arr.multi_slice_mut((tgt_slice, src_slice));
    tgt_row.assign(&src_row);
}

/// Remove row i from the array, by replacing row i with the last row and removing the last row.
pub(crate) fn swap_remove_row<T: Clone, D: Dimension + SliceArg0>(arr: &mut Array<T, D>, i: usize) {
    let last = arr.len_of(Axis(0)) - 1;
    if i != last {
        assign_row(arr, i, last);
    }
    pop_array(Axis(0), arr);
}

/// Linear interpolation
pub fn lerp<A>(a: A, b: A, t: f32) -> <A as Add<<f32 as Mul<<A as Sub>::Output>>::Output>>::Output
    where
        A: Copy + Sub,
        f32: Mul<<A as Sub>::Output>,
        A: Add<<f32 as Mul<<A as Sub>::Output>>::Output>
{
    a + t * (b - a)
}

/// Append the row
///   lerp(arr[i], arr[j], t) = arr[i] + t * (arr[j] - arr[i])
/// to the array
pub(crate) fn append_lerp<D: Dimension + RemoveAxis>(arr: &mut Array<f32, D>, i: usize, j: usize, t: f32) {
    let ai = arr.index_axis(Axis(0), i);
    let aj = arr.index_axis(Axis(0), j);
    let ak = lerp(&ai, &aj, t);
    arr.push(Axis(0), ak.view()).unwrap();
}

/// Append the row arr[i] to the array
pub(crate) fn append_row<T: Clone, D: Dimension + RemoveAxis>(arr: &mut Array<T, D>, i: usize) {
    let ai = arr.index_axis(Axis(0), i).to_owned();
    arr.push(Axis(0), ai.view()).unwrap();
}

/// Find the point t between 0.0 and 1.0 at which lerp(a,b,t) == a+t*(b-a) == 0
pub fn find_zero(a: f32, b: f32) -> Option<f32> {
    if (a < 0.0) == (b < 0.0) {
        None
    } else if a == 0.0 {
        Some(0.0)
    } else if b == 0.0 {
        Some(1.0)
    } else {
        let t = -a / (b - a);
        debug_assert!(t >= 0.0 && t <= 1.0);
        Some(t)
    }
}