use std::f32::consts::TAU;
use std::io::Write;
use std::ops::Range;
//use colorgrad::Gradient;
use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use svg_fmt::*;

use crate::partition::*;
use crate::regioncam::*;
use crate::util::*;


#[derive(Clone)]
pub struct SvgOptions {
    pub face_color_range: Range<f32>,
    pub face_color_by_value: f32,
    pub image_size: (f32,f32),
    pub draw_boundary: bool,
    pub line_width: f32,
    pub line_width_decision_boundary: f32,
    pub line_color: Color,
    pub line_color_decision_boundary: Color,
    pub point_size: f32,
    pub label_size: f32,
}

impl Default for SvgOptions {
    fn default() -> Self {
        Self {
            face_color_range: 0.6..0.95,
            face_color_by_value: 0.6,
            image_size: (800.0, 800.0),
            draw_boundary: false,
            line_width: 0.8,
            line_width_decision_boundary: 1.6,
            line_color: black(),
            line_color_decision_boundary: red(),
            point_size: 8.0,
            label_size: 16.0,
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
}

/// Helper struct for writing a Partition to an svg file
pub struct SvgWriter<'a> {
    options: &'a SvgOptions,
    regioncam: &'a Regioncam,
    bounding_box: Array1<Range<f32>>,
    value_range: Range<f32>,
    decision_boundary_layer: Option<usize>,
    pub points: &'a [MarkedPoint],
    rng: SmallRng,
}

impl<'a> SvgWriter<'a> {
    pub fn new(regioncam: &'a Regioncam, options: &'a SvgOptions) -> Self {
        let bounding_box: ndarray::ArrayBase<ndarray::OwnedRepr<Range<f32>>, ndarray::Dim<[usize; 1]>> = bounding_box(&regioncam.inputs().view());
        let values = regioncam.activations_last();
        let value_range = value_range(&values);
        let last_layer_is_classification = values.ncols() == 1;
        let decision_boundary_layer = last_layer_is_classification.then_some(regioncam.last_layer());
        let rng = SmallRng::seed_from_u64(42);

        SvgWriter { options, regioncam, bounding_box, value_range, decision_boundary_layer, points: &[], rng }
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
            let label = self.regioncam.edge_label(edge);
            if self.options.draw_boundary || !(self.regioncam.is_boundary(edge) || label.layer == 0) {
                let is_decision_boundary = Some(label.layer) == self.decision_boundary_layer;
                let path = self.edge_to_path(edge)
                    .width(if is_decision_boundary { self.options.line_width_decision_boundary } else { self.options.line_width })
                    .color(if is_decision_boundary { self.options.line_color_decision_boundary } else { self.options.line_color });
                 writeln!(w, "{path}")?;
            }
        }
        Ok(())
    }
    
    pub fn write_points(&self, w: &mut dyn Write) -> std::io::Result<()> {
        for point in self.points {
            let (x, y) = self.point_coord(point.position.as_ref().into());
            let radius = self.options.point_size * 0.5;
            let circle = Circle { x, y, radius, style: Style::default(), comment: None };
            let circle = circle.fill(black());
            writeln!(w, "{}", circle)?;
            if !point.label.is_empty() {
                writeln!(w, "{}", text(x + radius, y - radius, &point.label).size(self.options.label_size))?;
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
        if self.decision_boundary_layer.is_some() && self.options.face_color_by_value > 0.0 {
            // base face color on value of face
            let face_centroid = self.regioncam.face_centroid(face);
            let face_value = self.regioncam.face_activation_for(self.regioncam.last_layer(), face, face_centroid.view())[0];
            // map to value range
            let t = where_in_range(face_value, &self.value_range);
            //
            //let gradient = colorgrad::preset::sinebow();
            //let [r, g, b, _] = gradient.at(t * 0.9).to_array();
            // Alternative color map:
            let periodic = |t| f32::cos(t * TAU) * 0.5 + 1.0;
            let [r,g,b] = [periodic(t), periodic(t * 2.5 + 0.33), periodic(t * 5.2 + 0.66)];
            self.random_color([r, g, b], self.options.face_color_by_value)
        } else {
            self.random_color([0.0, 0.0, 0.0], 0.0)
        }
    }

    fn random_color_component(&mut self, base: f32, base_fraction: f32) -> u8 {
        let x = base * base_fraction + self.rng.gen_range(0.0..1.0-base_fraction);
        let x = point_in_range(x, &self.options.face_color_range) * 255.0;
        x as u8
    }
    fn random_color(&mut self, [r, g, b]: [f32;3], base_fraction: f32) -> Color {
        Color {
            r: self.random_color_component(r, base_fraction),
            g: self.random_color_component(g, base_fraction),
            b: self.random_color_component(b, base_fraction),
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

fn value_range(data: &ArrayView2<f32>) -> Range<f32> {
    data.iter().copied().fold(EMPTY_RANGE, minmax)
}
