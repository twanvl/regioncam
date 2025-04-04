use ndarray::{array, stack, Array2, ArrayView2, Axis};

use crate::nn::{Linear, NNModule};
use crate::util::*;


/// A plane in a higher dimensional space.
///
/// Optionally also keeps the points used to construct the plane
#[derive(Clone, Debug)]
pub struct Plane {
    // Ortogonal 2×D matrix
    pub mapping: Linear,
    // Optional: labeled points on the plane
    pub points: Array2<f32>,
    //point_labels: Vec<String>,
}

impl Plane {
    /// Construct a plane through 3 points
    pub fn through_points(points: &ArrayView2<f32>) -> Self {
    //pub fn through_points<S: Data<Elem=f32>>(points: &ArrayBase<S,Ix2>) -> Self {
        let mut mapping = orthogonal_mapping(points);
        let mut points2d = inverse_mapping(&mapping, points);
        // center bounding box on (0,0)
        let bb = bounding_box(&points2d.view());
        let center = bb.map(|range| (range.start + range.end) * 0.5);
        points2d -= &center.broadcast(points2d.dim()).unwrap();
        mapping.bias += &center.dot(&mapping.weight); // adjust bias to compensate for shifting points
        Plane { mapping, points: points2d }
    }

    /// Project points in the output space onto this plane
    pub fn inverse(&self, points: &ArrayView2<f32>) -> Array2<f32> {
        inverse_mapping(&self.mapping, points)
    }

    /// Transform points on the plane into 
    pub fn forward(&self, points2d: &ArrayView2<f32>) -> Array2<f32> {
        self.mapping.forward(points2d)
    }

    /// Size of square around the origin that contains the points generating the plane.
    pub fn size(&self) -> f32 {
        self.points.fold(0.0, |max, x| f32::max(max, x.abs()))
    }
}

impl From<Linear> for Plane {
    fn from(mapping: Linear) -> Self {
        Plane { mapping, points: array![[]] }
    }
}

fn orthogonal_mapping(points: &ArrayView2<f32>) -> Linear {
    assert_eq!(points.nrows(), 3, "Constructing a plane requires exactly 3 points.");
    // Make orthogonal basis
    let d1 = into_normalized(&points.row(1) - &points.row(0));
    let mut d2 = &points.row(2) - &points.row(0);
    let prod = d2.dot(&d1);
    d2.zip_mut_with(&d1, |x2, x1| *x2 -= prod * *x1 );
    let d2 = into_normalized(d2);
    Linear {
        weight: stack![Axis(0), d1, d2],
        bias: points.row(0).into_owned(),
    }
}

fn inverse_mapping(mapping: &Linear, points: &ArrayView2<f32>) -> Array2<f32> {
    let bias = &mapping.bias.broadcast(points.dim()).unwrap();
    (points - bias).dot(&mapping.weight.t())
}

#[cfg(test)]
mod test {
    use approx::assert_abs_diff_eq;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::*;

    #[test]
    fn plane_through_points() {
        let mut rng = SmallRng::seed_from_u64(42);
        for d in 2..5 {
            let points = Array2::from_shape_fn((3,d), |_| rng.gen_range(-5.0..5.0));
            let plane = Plane::through_points(&points.view());
            // Plane should go through points
            let outputs = plane.forward(&plane.points.view());
            assert_abs_diff_eq!(points, outputs, epsilon=1e-5);
            // inverse . forward
            let inputs = Array2::from_shape_fn((5,2), |_| rng.gen_range(-5.0..5.0));
            let outputs = plane.forward(&inputs.view());
            let inv = plane.inverse(&outputs.view());
            assert_abs_diff_eq!(inputs, inv, epsilon=1e-5);
        }
    }
}