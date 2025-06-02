use ndarray::{array, stack, Array1, Array2, ArrayView2, Axis};

use crate::nn::{Linear, NNModule};
use crate::util::*;


/// A N dimensional hyperplane in a higher dimensional space.
///
/// Optionally also keeps the points used to construct the plane
#[derive(Clone, Debug)]
pub struct Hyperplane<const N: usize> {
    // Ortogonal N×D matrix, mapping the plane into the ambient higher dimensional space
    pub mapping: Linear,
    // Optional: labeled points on the plane
    pub points: Array2<f32>,
}

impl<const N: usize> Hyperplane<N> {
    /// Project points in the output space onto this plane
    pub fn project(&self, points: &ArrayView2<f32>) -> Array2<f32> {
        inverse_mapping(&self.mapping, points)
    }

    /// Transform points on the plane into the output space
    pub fn forward(&self, points2d: &ArrayView2<f32>) -> Array2<f32> {
        self.mapping.forward(points2d)
    }

    /// Size of square around the origin that contains the points generating the plane.
    /// Equivalent to the max norm of the points projected onto the plane
    pub fn size(&self) -> f32 {
        self.points.fold(0.0, |max, x| f32::max(max, x.abs()))
    }

    pub fn from_mapping_and_points(mapping: Linear, mapped_points: &ArrayView2<f32>) -> Self {
        let points = inverse_mapping(&mapping, mapped_points);
        Hyperplane { mapping, points }
    }
    /// Move the plane so that the center of the bounding box of points is on (0,0)
    pub fn centered(self) -> Self {
        let bb = bounding_box(&self.points.view());
        let center = bb.map(|range| (range.start + range.end) * 0.5);
        self.center_at(center)
    }
    /// Move the plane so that 0 maps to what center mapped to
    pub fn center_at(mut self, center: Array1<f32>) -> Self {
        self.points -= &center.broadcast(self.points.dim()).unwrap();
        self.mapping.bias += &center.dot(&self.mapping.weight); // adjust bias to compensate for shifting points
        self
    }
}

impl<const N: usize> From<Linear> for Hyperplane<N> {
    fn from(mapping: Linear) -> Self {
        assert_eq!(mapping.dim_in(), N);
        Hyperplane { mapping, points: array![[]] }
    }
}

/// A 2 dimensional plane in a higher dimensional space.
pub type Plane = Hyperplane<2>;

impl Plane {
    /// Construct a plane through 3 points
    pub fn through_points(points: &ArrayView2<f32>) -> Self {
        let mapping = orthogonal_mapping_2d(points);
        Self::from_mapping_and_points(mapping, points).centered()
    }
}

/// A 1 dimensional hyperplane is a line
pub type Plane1D = Hyperplane<1>;

impl Plane1D {
    /// Construct a line through 2 points
    pub fn through_points(points: &ArrayView2<f32>) -> Self {
        let mapping = orthogonal_mapping_1d(points);
        Self::from_mapping_and_points(mapping, points).centered()
    }
    /// Give a line perpendicular to this plane, such that it intersects this plane at 0.
    /// Is only valid if the ambient space is 2 dimensional.
    pub fn perpendicular(&self) -> Plane1D {
        assert_eq!(self.mapping.dim_out(), 2);
        Linear {
            weight: array![[-self.mapping.weight[(0,1)], self.mapping.weight[(0,0)]]],
            bias: self.mapping.bias.clone(),
        }.into()
    }
}


fn orthogonal_mapping_2d(points: &ArrayView2<f32>) -> Linear {
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

fn orthogonal_mapping_1d(points: &ArrayView2<f32>) -> Linear {
    assert_eq!(points.nrows(), 2, "Constructing a line requires exactly 2 points.");
    let d1 = into_normalized(&points.row(1) - &points.row(0));
    Linear {
        weight: d1.insert_axis(Axis(0)),
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
            // project . forward
            let inputs = Array2::from_shape_fn((10,2), |_| rng.gen_range(-5.0..5.0));
            let outputs = plane.forward(&inputs.view());
            let inv = plane.project(&outputs.view());
            assert_abs_diff_eq!(inputs, inv, epsilon=1e-5);
        }
    }

    #[test]
    fn plane1d_through_points() {
        let mut rng = SmallRng::seed_from_u64(42);
        for d in 1..5 {
            let points = Array2::from_shape_fn((2,d), |_| rng.gen_range(-5.0..5.0));
            let plane = Plane1D::through_points(&points.view());
            // Plane should go through points
            let outputs = plane.forward(&plane.points.view());
            assert_abs_diff_eq!(points, outputs, epsilon=1e-5);
            // project . forward
            let inputs = Array2::from_shape_fn((10,1), |_| rng.gen_range(-5.0..5.0));
            let outputs = plane.forward(&inputs.view());
            let inv = plane.project(&outputs.view());
            assert_abs_diff_eq!(inputs, inv, epsilon=1e-5);
            // perpendicular
            if d == 2 {
                let perp = plane.perpendicular();
                // points on plane are at 0 on perpendicular line
                let inputs  = Array2::from_shape_fn((10,1), |_| rng.gen_range(-5.0..5.0));
                let points = plane.forward(&inputs.view());
                let distance_to_plane = perp.project(&points.view());
                assert_abs_diff_eq!(distance_to_plane, Array2::zeros(distance_to_plane.raw_dim()), epsilon=1e-5);
            }
        }
    }
}