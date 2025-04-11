use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::Write;
use std::ops::Range;
use ndarray::{Array1, ArrayView1};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use svg_fmt::*;

use crate::partition::*;
use crate::regioncam::*;
use crate::util::*;


#[derive(Clone)]
pub struct SvgOptions {
    pub image_size: (f32,f32),
    pub draw_boundary: bool,
    pub face_color_range: Range<f32>,
    pub face_color_layer_amount: f32,
    pub face_color_by_layer: bool,
    pub line_width: f32,
    pub line_width_decision_boundary: f32,
    pub line_color: Color,
    pub line_color_decision_boundary: Color,
    pub line_color_by_layer: bool,
    pub line_color_by_neuron: bool,
    pub line_color_amount: f32,
    pub point_size: f32,
    pub font_size: f32,
}

impl Default for SvgOptions {
    fn default() -> Self {
        Self {
            image_size: (800.0, 800.0),
            draw_boundary: false,
            face_color_range: 0.666..1.0,
            face_color_layer_amount: 0.666,
            face_color_by_layer: true,
            line_width: 0.8,
            line_width_decision_boundary: 1.6,
            line_color: black(),
            line_color_decision_boundary: red(),
            line_color_by_layer: true,
            line_color_by_neuron: false,
            line_color_amount: 0.333,
            point_size: 8.0,
            font_size: 16.0,
        }
    }
}

impl SvgOptions {
    pub fn new() -> Self {
        Default::default()
    }
    
    /// Output to svg
    pub fn write_svg(&self, p: &Regioncam, w: &mut dyn Write) -> std::io::Result<()> {
        SvgWriter::new(p, self).write_svg(w)
    }
}

pub struct MarkedPoint {
    pub position: [f32;2],
    pub label: String,
    //pub color: Color,
}

/// Helper struct for writing a Partition to an svg file
pub struct SvgWriter<'a> {
    options: &'a SvgOptions,
    regioncam: &'a Regioncam,
    bounding_box: Array1<Range<f32>>,
    decision_boundary_layer: Option<usize>,
    pub points: &'a [MarkedPoint],
}

impl<'a> SvgWriter<'a> {
    pub fn new(regioncam: &'a Regioncam, options: &'a SvgOptions) -> Self {
        let bounding_box: ndarray::ArrayBase<ndarray::OwnedRepr<Range<f32>>, ndarray::Dim<[usize; 1]>> = bounding_box(&regioncam.inputs().view());
        let values = regioncam.activations_last();
        let last_layer_is_classification = values.ncols() == 1;
        let decision_boundary_layer = last_layer_is_classification.then_some(regioncam.last_layer());

        SvgWriter { options, regioncam, bounding_box, decision_boundary_layer, points: &[] }
    }

    /// Output to svg
    pub fn write_svg(&mut self, w: &mut dyn Write) -> std::io::Result<()> {
        self.write_header(w)?;
        self.write_faces(w)?;
        self.write_edges(w)?;
        self.write_points(w)?;
        self.write_footer(w)?;
        Ok(())
    }
    
    pub fn write_header(&self, w: &mut dyn Write) -> std::io::Result<()> {
        // Note: svg_fmt::BeginSvg doesn't support width and height attributes
        let size = self.options.image_size;
        let padding = if self.options.draw_boundary { self.options.line_width * 0.5 } else { 0.0 };
        writeln!(w, r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="{} {} {} {}" width="{}" height="{}">"#,
             -padding, -padding,
             size.0 + 2.0 * padding, size.1 + 2.0 * padding,
             size.0 + 2.0 * padding, size.1 + 2.0 * padding
        )
    }

    pub fn write_footer(&self, w: &mut dyn Write) -> std::io::Result<()> {
        writeln!(w, "{}", EndSvg)
    }
    
    pub fn write_faces(&mut self, w: &mut dyn Write) -> std::io::Result<()> {
        // draw faces
        for face in self.regioncam.faces() {
            let path = self.face_to_path(face);
            let color = self.face_color(face);
            writeln!(w, "{}", path.fill(color))?;
        }
        Ok(())
    }

    pub fn write_edges(&self, w: &mut dyn Write) -> std::io::Result<()> {
        // draw edges
        for edge in self.regioncam.edges() {
            match self.edge_style(edge) {
                None => (),
                Some((color, width)) => {
                    let path = self.edge_to_path(edge);
                    writeln!(w, "{}", path.width(width).color(color))?;
                }
            }
        }
        Ok(())
    }
    
    pub fn write_points(&self, w: &mut dyn Write) -> std::io::Result<()> {
        for point in self.points {
            let (x, y) = self.point_coord(point.position.as_ref().into());
            let radius = self.options.point_size * 0.5;
            if radius > 0.0 {
                let circle = Circle { x, y, radius, style: Style::default(), comment: None };
                let circle = circle.fill(black());
                writeln!(w, "{}", circle)?;
            }
            if !point.label.is_empty() && self.options.font_size > 0.0 {
                writeln!(w, "{}", text(x + radius, y - radius, &point.label).size(self.options.font_size))?;
            }
        }
        Ok(())
    }

    fn vertex_coord(&self, vertex: Vertex) -> (f32, f32) {
        let inputs = self.regioncam.vertex_inputs(vertex);
        self.point_coord(inputs)
    }
    fn point_coord(&self, inputs: ArrayView1<f32>) -> (f32, f32) {
        (
            where_in_range(inputs[0], &self.bounding_box[0]) * self.options.image_size.0,
            where_in_range(inputs[1], &self.bounding_box[1]) * self.options.image_size.1
        )
    }

    fn face_color(&mut self, face: Face) -> Color {
        if self.options.face_color_by_layer {
            // Color based on face hash per layer
            let mut color = ColorF32::white();
            let mut first = true;
            for layer in 0..self.regioncam.num_layers() {
                if self.regioncam.layer(layer).has_activations() {
                    // Color based on face hash
                    let hash = self.regioncam.face_hash_at(layer, face);
                    let face_layer_color = ColorF32::from_hash(hash);
                    if first {
                        color = face_layer_color;
                        first = false;
                    } else {
                        color = ColorF32::uniform_interpolate(color, face_layer_color, self.options.face_color_layer_amount);
                    }
                }
            }
            color
        } else {
            // Color based on unique face hash
            let hash = self.regioncam.face_hash(face);
            ColorF32::from_hash(hash)
        }
        .rescale(self.options.face_color_range.clone()).into()
    }

    pub fn edge_style(&self, edge: Edge) -> Option<(Color, f32)> {
        let label = self.regioncam.edge_label(edge);
        let visible = self.options.draw_boundary || !(self.regioncam.is_boundary(edge) || label.layer == 0);
        if visible {
            let is_decision_boundary = Some(label.layer) == self.decision_boundary_layer;
            let (color, width) = if is_decision_boundary {
                (self.options.line_color_decision_boundary, self.options.line_width_decision_boundary)
            } else {
                (self.options.line_color, self.options.line_width)
            };
            if self.options.line_color_by_neuron || self.options.line_color_by_layer {
                // hash edge label or layer
                let mut hasher = DefaultHasher::new();
                if self.options.line_color_by_neuron {
                    self.regioncam.edge_label(edge).hash(&mut hasher);
                } else {
                    self.regioncam.edge_label(edge).layer.hash(&mut hasher);
                };
                let hash = hasher.finish();
                // create a color for the edge label
                let mix_color = ColorF32::from_hash(hash);
                let color = ColorF32::lerp(color.into(), mix_color, self.options.line_color_amount).into();
                Some((color, width))
            } else {
                Some((color, width))
            }
        } else {
            None
        }
    }

    fn face_to_path(&self, face: Face) -> Path {
        let mut iter = self.regioncam.vertices_on_face(face);
        let start_vertex = iter.next().unwrap();
        let start = self.vertex_coord(start_vertex);
        let path = path().move_to(start.0, start.1);
        let path = iter.fold(path, |path, vertex| {
            let point = self.vertex_coord(vertex);
            path.line_to(point.0, point.1)
        });
        path.close()
    }

    fn edge_to_path(&self, edge: Edge) -> LineSegment {
        let (a,b) = self.regioncam.endpoints(edge);
        let a = self.vertex_coord(a);
        let b = self.vertex_coord(b);
        line_segment(a.0, a.1, b.0, b.1)
    }
}

#[derive(Clone, Debug)]
struct ColorF32 {
    r: f32,
    g: f32,
    b: f32,
}
impl From<ColorF32> for Color {
    fn from(color: ColorF32) -> Self {
        Color {
            r: (color.r * 255.0) as u8,
            g: (color.g * 255.0) as u8,
            b: (color.b * 255.0) as u8,
        }
    }
}
impl From<Color> for ColorF32 {
    fn from(color: Color) -> Self {
        ColorF32 {
            r: color.r as f32 / 255.0,
            g: color.g as f32 / 255.0,
            b: color.b as f32 / 255.0,
        }
    }
}
impl ColorF32 {
    fn from_fn(mut f: impl FnMut() -> f32) -> Self {
        ColorF32 {
            r: f(),
            g: f(),
            b: f(),
        }
    }
    fn map(self, f: impl Fn(f32) -> f32) -> Self {
        ColorF32 {
            r: f(self.r),
            g: f(self.g),
            b: f(self.b),
        }
    }
    fn zip(self, other: Self, f: impl Fn(f32, f32) -> f32) -> Self {
        ColorF32 {
            r: f(self.r, other.r),
            g: f(self.g, other.g),
            b: f(self.b, other.b),
        }
    }
    fn white() -> Self {
        Self::from_fn(|| 1.0)
    }
    fn from_hash(hash: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(hash);
        Self::random(0.0..1.0, &mut rng)
    }
    fn random<R: Rng + ?Sized>(range: Range<f32>, rng: &mut R) -> Self {
        Self::from_fn(|| rng.gen_range(range.clone()))
    }
    fn rescale(self, range: Range<f32>) -> Self {
        Self::map(self, |x| point_in_range(x, &range))
    }
    fn lerp(self, other: Self, t: f32) -> Self {
        Self::zip(self, other, |x, y| lerp(x, y, t))
    }
    fn uniform_interpolate(self, other: Self, t: f32) -> Self {
        Self::zip(self, other, |x, y| uniform_interpolate(x, y, t))
    }
}

// Interpolate between two values with a uniform distribution, such that the result is also uniformly distributed.
// if a,b ~ U(0,1), then
//  lerp(a,b,t) ~ trapezoid([(0,0), (t,1/(1-t)), (1-t, 1/(1-t)), (1,0)])
// rescale to a uniform distribution
fn uniform_interpolate(a: f32, b: f32, t: f32) -> f32 {
    let x = lerp(a,b,t);
    let t = f32::min(t, 1.0 - t);
    if x < t {
        (x * x) / (2.0 * t * (1.0 - t))
    } else if x <= 1.0 - t {
        (x - t * 0.5) / (1.0 - t)
    } else {
        1.0 - (1.0 - x) * (1.0 - x) / (2.0 * t * (1.0 - t))
    }
}

fn point_in_range(x: f32, range: &Range<f32>) -> f32 {
    range.start + x * (range.end - range.start)
}
fn where_in_range(x: f32, range: &Range<f32>) -> f32 {
    if range.start == range.end {
        0.0
    } else {
        (x - range.start) / (range.end - range.start)
    }
}
