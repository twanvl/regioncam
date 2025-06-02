// Utility functions

use std::cmp::Ordering;
use std::ops::{Add, Mul, Range, Sub};

use ndarray::{s, Array, Array1, Array2, ArrayBase, ArrayView, ArrayView1, ArrayView2, Axis, Data, Dimension, Ix0, Ix1, Ix2, Ix3, RemoveAxis, Slice, SliceArg, SliceInfo, SliceInfoElem};
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


/// Is a given list a permutation?
pub fn is_permutation<I>(perm: &[I]) -> bool
    where I: Copy, usize: From<I>
{
    let count = histogram(perm.iter().copied().map(usize::from), perm.len());
    count.iter().copied().all(|c| c == 1)
}

/// Permute the elements of a slice.
/// The permutation can be a slice of any index type
pub fn permute<A, I>(vec: &[A], perm: &[I]) -> Vec<A>
    where A: Copy, I: Copy, usize: From<I>
{
    perm.iter().map(|i| vec[usize::from(*i)]).collect()
}

/// Inversely permute the elements of a slice.
pub fn inverse_permute<A, I>(vec: &[A], perm: &[I]) -> Vec<A>
    where A: Copy + Default, I: Copy, usize: From<I>
{
    let mut out = vec![Default::default(); perm.len()];
    for (i, x) in perm.iter().zip(vec) {
        out[usize::from(*i)] = *x;
    }
    out
}

/// Permute the rows of a 2D array.
pub fn permute_rows<A, I>(array: &Array2<A>, perm: &[I]) -> Array2<A>
    where A: Copy + Zero, I: Copy, usize: From<I>
{
    assert_eq!(array.len_of(Axis(0)), perm.len());
    let mut out = Array2::zeros(array.raw_dim());
    for (mut out_row, i) in out.axis_iter_mut(Axis(0)).zip(perm) {
        out_row.assign(&array.row(usize::from(*i)));
    }
    out
}


/// Insert multiple elements into a vector efficiently.
/// 
/// Equivalent to:
///     for (index, element) in insertions {
///         vec.insert(index, element);
///     }
/// 
/// The indices must be in weakly *decreasing* order.
pub(crate) fn insert_many<T: Default + Copy>(vec: &mut Vec<T>, insertions: impl ExactSizeIterator<Item=(usize, T)>) {
    let mut in_pos  = vec.len();
    let mut out_pos = vec.len() + insertions.len();
    // Note: we don't actually need the vec elements to be initialized.
    // An unsafe version could be slightly faster, and avoid a Default constraint.
    vec.resize(vec.len() + insertions.len(), T::default());
    for (index, element) in insertions {
        debug_assert!(index <= in_pos);
        while index < in_pos {
            out_pos -= 1;
            in_pos -= 1;
            vec[out_pos] = vec[in_pos];
        }
        out_pos -= 1;
        vec[out_pos] = element;
    }
}

#[cfg(test)]
mod test {
    use approx::assert_abs_diff_eq;
    use rand::rngs::SmallRng;
    use rand::prelude::*;
    use super::*;

    #[test]
    fn find_zero_is_zero() {
        let mut rng = SmallRng::seed_from_u64(42);
        for _ in 0..100 {
            let a = rng.gen_range(-5.0..5.0);
            let b = rng.gen_range(-5.0..5.0);
            match find_zero(a, b) {
                Some(t) => {
                    assert!(0.0 <= t);
                    assert!(t <= 1.0);
                    assert_abs_diff_eq!(lerp(a, b, t), 0.0, epsilon=1e-5);
                }
                None => {
                    assert_eq!((a < 0.0), (b < 0.0));
                }
            }
        }
    }

    // Generate a random permutation
    fn random_permutation<R: Rng + ?Sized>(n: usize, rng: &mut R) -> Vec<usize> {
        let mut perm: Vec<usize> = (0..n).collect();
        perm.shuffle(rng);
        perm
    }

    #[test]
    fn inverse_permute_test() {
        let mut rng = SmallRng::seed_from_u64(42);
        for n in 0..20 {
            let vec = (0..n).collect::<Vec<_>>();
            let perm = random_permutation(n, &mut rng);
            assert_eq!(vec, inverse_permute(&permute(&vec, &perm), &perm));
            assert_eq!(vec, permute(&inverse_permute(&vec, &perm), &perm));
        }
    }

    fn insert_many_spec<T>(vec: &mut Vec<T>, insertions: impl ExactSizeIterator<Item=(usize, T)>) {
        for (index, element) in insertions {
            vec.insert(index, element);
        }
    }
    #[test]
    fn insert_many_test() {
        let test_cases = vec![
            (vec![], vec![(0, 0)]),
            (vec![5,6,7], vec![(3,0),(3,1),(1,8),(0,9)])];
        for (vec, insertions) in test_cases {
            let mut from_insert_many = vec.clone();
            let mut from_insert_many_spec = vec.clone();
            insert_many(&mut from_insert_many, insertions.iter().copied());
            insert_many_spec(&mut from_insert_many_spec, insertions.iter().copied());
            assert_eq!(from_insert_many, from_insert_many_spec)
        }
    }

}