use std::{io::Write, ops::Range};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rand_distr::{Distribution, Uniform};
use svg_fmt::*;

use crate::partition::*;


pub struct SvgOptions {
    pub face_color_range: Range<u8>,
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
            face_color_range: 150..250,
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
        // find bounding box
        let bb = bounding_box(p);
        // transformation for coordinates
        let vertex_coord = |vertex| {
            let inputs = p.vertex_inputs(vertex);
            let x = (inputs[0] - bb.0.start) / (bb.0.end - bb.0.start) * (self.image_size.0 - 2.0*self.image_border) + self.image_border;
            let y = (inputs[1] - bb.1.start) / (bb.1.end - bb.1.start) * (self.image_size.1 - 2.0*self.image_border) + self.image_border;
            (x,y)
        };
        // draw faces
        for face in p.faces() {
            let path = self.face_to_path(p, face, vertex_coord);
            let color = self.random_color(&mut rng);
            writeln!(w, "{}", path.fill(color))?;
        }
        // draw edges
        let last_layer_is_classification = p.activations_last().ncols() == 1;
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

fn bounding_box(p: &Partition) -> (Range<f32>, Range<f32>) {
    let mut bounds = vec![f32::INFINITY..f32::NEG_INFINITY;2];
    for vertex in p.vertices() {
        let pos = p.vertex_inputs(vertex);
        for i in 0..2 {
            bounds[i].start = f32::min(bounds[i].start, pos[i]);
            bounds[i].end   = f32::max(bounds[i].end, pos[i]);
        }
    }
    (bounds[0].clone(), bounds[1].clone())
}

