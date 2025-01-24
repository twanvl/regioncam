use std::{io::Write, ops::Range};
use colorgrad::Gradient;
use ndarray::{ArrayView2, Axis};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rand_distr::{Distribution, Uniform};
use svg_fmt::*;
use colorgrad::preset::SinebowGradient;

use crate::partition::*;


pub struct SvgOptions {
    pub face_color_range: Range<u8>,
    pub face_color_by_value: f32,
    pub image_size: (f32,f32),
    pub image_border: f32,
    pub line_width: f32,
    pub line_width_decision_boundary: f32,
    pub line_color: Color,
    pub line_color_decision_boundary: Color,
}

impl Default for SvgOptions {
    fn default() -> Self {
        Self {
            face_color_range: 150..255,
            face_color_by_value: 0.6,
            image_size: (800.0, 800.0),
            image_border: 0.0,
            line_width: 0.5,
            line_width_decision_boundary: 1.5,
            line_color: black(),
            line_color_decision_boundary: red(),
        }
    }
}

impl SvgOptions {
    pub fn new() -> Self {
        Default::default()
    }

    /// Output to svg
    pub fn write_svg(&self, p: &Partition, w: &mut dyn Write) -> std::io::Result<()> {
        let mut rng = SmallRng::seed_from_u64(42);
        writeln!(w, "{}", BeginSvg { w: self.image_size.0, h: self.image_size.1 })?;
        // are we visualizing a classifier?
        let last_layer_is_classification = p.activations_last().ncols() == 1;
        // find bounding box
        let bb = bounding_box(p);
        // transformation for coordinates
        let vertex_coord = |vertex| {
            let inputs = p.vertex_inputs(vertex);
            let x = (inputs[0] - bb.0.start) / (bb.0.end - bb.0.start) * (self.image_size.0 - 2.0*self.image_border) + self.image_border;
            let y = (inputs[1] - bb.1.start) / (bb.1.end - bb.1.start) * (self.image_size.1 - 2.0*self.image_border) + self.image_border;
            (x,y)
        };
        // find value range
        let values = p.activations_last();
        let value_range = value_range(&values);
        // draw faces
        let gradient = colorgrad::preset::sinebow();
        let face_color_by_value = if last_layer_is_classification { self.face_color_by_value } else { 0.0 };
        for face in p.faces() {
            let path = self.face_to_path(p, face, vertex_coord);
            let color = 
                if face_color_by_value > 0.0 {
                    let face_centroid = p.face_centroid(face);
                    let face_value = p.face_activation_for(p.last_layer(), face, face_centroid.view())[0];
                    //let face_value = mean(p.vertices_on_face(face).map(|v| values[(v.into(), 0)]));
                    let t = (face_value - value_range.start) / (value_range.end - value_range.start);
                    let [r,g,b,_] = gradient.at(t).to_array();
                    let add_noise = |x: f32, rng: &mut SmallRng| -> u8 {
                        let x = x * face_color_by_value + rng.gen_range(0.0..1.0-face_color_by_value);
                        self.face_color_range.start + (x * (self.face_color_range.end - self.face_color_range.start) as f32) as u8
                    };
                    Color { r: add_noise(r, &mut rng), g: add_noise(g, &mut rng), b: add_noise(b, &mut rng) }
                } else {
                    self.random_color(&mut rng)
                };
            writeln!(w, "{}", path.fill(color))?;
        }
        // draw edges
        for edge in p.edges() {
            let (a,b) = p.endpoints(edge);
            let a = vertex_coord(a);
            let b = vertex_coord(b);
            let label = p.edge_label(edge);
            let is_last = last_layer_is_classification && label.layer == p.last_layer();
            if is_last {
                writeln!(w, "{}", line_segment(a.0, a.1, b.0, b.1).width(self.line_width_decision_boundary).color(self.line_color_decision_boundary))?;
            } else {
                writeln!(w, "{}", line_segment(a.0, a.1, b.0, b.1).width(self.line_width).color(self.line_color))?;
            }
        }
        writeln!(w, "{}", EndSvg)?;
        Ok(())
    }

    fn random_color<R: Rng + ?Sized>(&self, rng: &mut R) -> Color {
        let dist = Uniform::new(self.face_color_range.start, self.face_color_range.end);
        Color {
            r: dist.sample(rng),
            g: dist.sample(rng),
            b: dist.sample(rng),
        }
    }

    fn face_to_path(&self, p: &Partition, face: Face, vertex_coord: impl Fn(Vertex) -> (f32,f32)) -> Path {
        let mut iter = p.halfedges_on_face(face);
        let start_he = iter.next().unwrap();
        let start = vertex_coord(p.start(start_he));
        let path = path().move_to(start.0, start.1);
        let path = iter.fold(path, |path, he| {
            let point = vertex_coord(p.start(he));
            path.line_to(point.0, point.1)
        });
        path.close()
    }
}

const EMPTY_RANGE: Range<f32> = f32::INFINITY..f32::NEG_INFINITY;

#[inline]
fn minmax(mut range: Range<f32>, x: f32) -> Range<f32> {
    range.start = f32::min(range.start, x);
    range.end   = f32::max(range.end, x);
    range
}

fn bounding_box(p: &Partition) -> (Range<f32>, Range<f32>) {
    let mut bounds = vec![EMPTY_RANGE;2];
    for vertex in p.vertices() {
        let pos = p.vertex_inputs(vertex);
        for i in 0..2 {
            bounds[i] = minmax(bounds[i].clone(), pos[i]);
        }
    }
    (bounds[0].clone(), bounds[1].clone())
}

fn value_range(data: &ArrayView2<f32>) -> Range<f32> {
    data.iter().copied().fold(EMPTY_RANGE, minmax)
}

fn mean(iter: impl Iterator<Item=f32>) -> f32 {
    let (count, sum) = iter.fold((0.0, 0.0), |(count, sum), x| (count + 1.0, sum + x));
    sum / count
}