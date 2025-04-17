use std::hash::{DefaultHasher, Hash, Hasher};
use std::ops::Range;
use ndarray::{Array1, ArrayView1};
use rand::{rngs::SmallRng, Rng, SeedableRng};

use crate::partition::*;
use crate::regioncam::*;
use crate::util::*;


/// Rendering options/configuration
#[derive(Clone)]
pub struct RenderOptions {
    pub image_size: (f32,f32),
    pub draw_boundary: bool,
    pub draw_faces: bool,
    pub draw_edges: bool,
    pub draw_vertices: bool,
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
    pub layer_line_colors: Vec<Color>, // by layer
    pub point_color: Color,
    pub point_size: f32,
    pub font_size: f32,
    pub vertex_size: f32,
    pub text_only_markers: bool,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            image_size: (800.0, 800.0),
            draw_boundary: false,
            draw_faces: true,
            draw_edges: true,
            draw_vertices: false,
            face_color_range: 0.666..1.0,
            face_color_layer_amount: 0.666,
            face_color_by_layer: true,
            line_width: 0.8,
            line_width_decision_boundary: 1.6,
            line_color: Color::new(0.0, 0.0, 0.0),
            line_color_decision_boundary: Color::new(1.0, 0.0, 0.0),
            line_color_by_layer: true,
            line_color_by_neuron: false,
            line_color_amount: 0.333,
            layer_line_colors: default_layer_line_colors(),
            point_color: Color::new(0.0, 0.0, 0.0),
            point_size: 8.0,
            font_size: 16.0,
            vertex_size: 3.0,
            text_only_markers: false,
        }
    }
}

fn default_layer_line_colors() -> Vec<Color> {
    vec![
        Color::new(0.0, 0.0, 0.0),
        Color::new(0.3, 1.0, 0.0),
        Color::new(0.0, 0.5, 1.0),
        Color::new(1.0, 0.0, 0.8),
        Color::new(1.0, 0.8, 0.0),
        Color::new(0.0, 1.0, 0.5),
        Color::new(1.0, 1.0, 1.0),
    ]
}

impl RenderOptions {
    pub fn new() -> Self {
        Default::default()
    }
    
    /*/// Output to svg using the options from self
    pub fn write_svg(&self, regioncam: &Regioncam, w: &mut dyn Write) -> std::io::Result<()> {
        Renderer::new(regioncam, self).write_svg(w)
    }*/
}


/// A point to mark in the rendered images
pub struct MarkedPoint {
    pub position: [f32;2],
    pub label: String,
    pub color: Option<Color>,
}

impl MarkedPoint {
}


/// A color, with 32 bit float components in the range 0.0..1.0
#[derive(Clone, Copy, Debug)]
pub struct Color {
    r: f32,
    g: f32,
    b: f32,
}

impl From<(f32,f32,f32)> for Color {
    fn from(value: (f32, f32, f32)) -> Self {
        let (r, g, b) = value;
        Color::new(r, g, b)
    }
}

impl Color {
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Color { r, g, b }
    }
    pub fn white() -> Self {
        Self::new(1.0, 1.0, 1.0)
    }
    pub fn from_fn(mut f: impl FnMut() -> f32) -> Self {
        Color {
            r: f(),
            g: f(),
            b: f(),
        }
    }
    pub fn map(self, f: impl Fn(f32) -> f32) -> Self {
        Color {
            r: f(self.r),
            g: f(self.g),
            b: f(self.b),
        }
    }
    pub fn zip(self, other: Self, f: impl Fn(f32, f32) -> f32) -> Self {
        Color {
            r: f(self.r, other.r),
            g: f(self.g, other.g),
            b: f(self.b, other.b),
        }
    }
    pub fn from_hash(hash: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(hash);
        Self::random(0.0..1.0, &mut rng)
    }
    pub fn random<R: Rng + ?Sized>(range: Range<f32>, rng: &mut R) -> Self {
        Self::from_fn(|| rng.gen_range(range.clone()))
    }
    pub fn rescale(self, range: Range<f32>) -> Self {
        Self::map(self, |x| point_in_range(x, &range))
    }
    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self::zip(self, other, |x, y| lerp(x, y, t))
    }
    pub fn uniform_interpolate(self, other: Self, t: f32) -> Self {
        Self::zip(self, other, |x, y| uniform_interpolate(x, y, t))
    }
}


/// Helper struct for rendering a Partition
pub struct Renderer<'a> {
    options: &'a RenderOptions,
    regioncam: &'a Regioncam,
    bounding_box: Array1<Range<f32>>,
    decision_boundary_layer: Option<usize>,
    pub points: &'a [MarkedPoint],
}

impl<'a> Renderer<'a> {
    pub fn new(regioncam: &'a Regioncam, options: &'a RenderOptions) -> Self {
        let mut bounding_box = bounding_box(&regioncam.inputs().view());
        let values = regioncam.activations_last();
        let last_layer_is_classification = values.ncols() == 1;
        let decision_boundary_layer = last_layer_is_classification.then_some(regioncam.last_layer());

        // Adjust bounding box to account for drawing the boundary edge
        // An edge drawn at x=0 will cover (-line_width/2 .. line_width/2)
        if options.draw_edges && options.draw_boundary || options.draw_vertices {
            let mut padding = options.line_width * 0.5;
            if options.draw_vertices {
                padding = padding.max(options.vertex_size * 0.5);
            }
            // the image would have extend (-padding .. size + padding) instead of (0 .. size)
            // mapping these points back means (start - padding * width / size .. end + padding * size / width)
            for (range, size) in bounding_box.iter_mut().zip([options.image_size.0, options.image_size.1]) {
                let width = range.end - range.start;
                range.start -= padding * width / size;
                range.end += padding * width / size;
            }
        }
        
        Renderer { options, regioncam, bounding_box, decision_boundary_layer, points: &[] }
    }
    pub fn with_points(mut self, points: &'a impl AsRef<[MarkedPoint]>) -> Self {
        self.points = points.as_ref();
        self
    }
    
    // Generic rendering functions

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

    fn face_color(&self, face: Face) -> Color {
        if self.options.face_color_by_layer {
            // Color based on face hash per layer
            let mut color = Color::new(0.5, 0.8, 1.0);
            let mut first = true;
            for layer in 0..self.regioncam.num_layers() {
                if self.regioncam.layer(layer).has_activations() {
                    // Color based on face hash
                    let hash = self.regioncam.face_hash_at(layer, face);
                    let face_layer_color = Color::from_hash(hash);
                    if first {
                        color = face_layer_color;
                        first = false;
                    } else {
                        color = Color::uniform_interpolate(color, face_layer_color, self.options.face_color_layer_amount);
                    }
                }
            }
            color
        } else {
            // Color based on unique face hash
            let hash = self.regioncam.face_hash(face);
            Color::from_hash(hash)
        }
        .rescale(self.options.face_color_range.clone())
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
            if self.options.line_color_by_layer && !self.options.line_color_by_neuron && label.layer < self.options.layer_line_colors.len() {
                let mix_color = self.options.layer_line_colors[label.layer];
                let color = Color::lerp(color.into(), mix_color, self.options.line_color_amount);
                Some((color, width))
            } else if self.options.line_color_by_neuron || self.options.line_color_by_layer {
                // hash edge label or layer
                let mut hasher = DefaultHasher::new();
                if self.options.line_color_by_neuron {
                    label.hash(&mut hasher);
                } else {
                    label.layer.hash(&mut hasher);
                };
                let hash = hasher.finish();
                // create a color for the edge label
                let mix_color = Color::from_hash(hash);
                let color = Color::lerp(color.into(), mix_color, self.options.line_color_amount);
                Some((color, width))
            } else {
                Some((color, width))
            }
        } else {
            None
        }
    }
}



#[cfg(feature = "svg")]
mod svg {
    use std::io::Write;
    use svg_fmt::*;
    use crate::*;

    impl<'a> Renderer<'a> {
        // SVG output
        
        /// Output to svg
        pub fn write_svg(&self, w: &mut dyn Write) -> std::io::Result<()> {
            self.write_header(w)?;
            if self.options.draw_faces {
                self.write_faces(w)?;
            }
            if self.options.draw_edges {
                self.write_edges(w)?;
            }
            if self.options.draw_vertices {
                self.write_vertices(w)?;
            }
            self.write_points(w)?;
            self.write_footer(w)?;
            Ok(())
        }
        
        fn write_header(&self, w: &mut dyn Write) -> std::io::Result<()> {
            // Note: svg_fmt::BeginSvg doesn't support width and height attributes
            let size = self.options.image_size;
            writeln!(w, r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="{} {} {} {}" width="{}" height="{}">"#,
                0.0, 0.0, size.0, size.1, size.0, size.1
            )
        }

        fn write_footer(&self, w: &mut dyn Write) -> std::io::Result<()> {
            writeln!(w, "{}", EndSvg)
        }

        fn write_faces(&self, w: &mut dyn Write) -> std::io::Result<()> {
            // draw faces
            for face in self.regioncam.faces() {
                let path = self.face_to_svg_path(face);
                let color = self.face_color(face);
                writeln!(w, "{}", path.fill(color))?;
            }
            Ok(())
        }

        fn write_edges(&self, w: &mut dyn Write) -> std::io::Result<()> {
            // draw edges
            for edge in self.regioncam.edges() {
                match self.edge_style(edge) {
                    None => (),
                    Some((color, width)) => {
                        let path = self.edge_to_svg_path(edge);
                        writeln!(w, "{}", path.width(width).color(color.into()))?;
                    }
                }
            }
            Ok(())
        }

        fn write_vertices(&self, w: &mut dyn Write) -> std::io::Result<()> {
            let radius = self.options.vertex_size * 0.5;
            let color = svg_fmt::black();
            for vertex in self.regioncam.vertices() {
                let (x, y) = self.vertex_coord(vertex);
                let circle = Circle { x, y, radius, style: Style::default(), comment: None };
                let circle = circle.fill(color);
                writeln!(w, "{}", circle)?;
            }
            Ok(())
        }

        fn write_points(&self, w: &mut dyn Write) -> std::io::Result<()> {
            for point in self.points {
                let (x, y) = self.point_coord(point.position.as_ref().into());
                let radius = self.options.point_size * 0.5;
                let color = point.color.unwrap_or(self.options.point_color).into();
                if radius > 0.0 {
                    let circle = Circle { x, y, radius, style: Style::default(), comment: None };
                    let circle = circle.fill(color);
                    writeln!(w, "{}", circle)?;
                }
                if !point.label.is_empty() && self.options.font_size > 0.0 {
                    let label_x = x + radius;
                    let mut label_y = y - radius;
                    let mut align  = Align::Left;
                    if self.options.text_only_markers {
                        align = Align::Center;
                        label_y -= self.options.font_size * 0.5;
                    }
                    let text = text(label_x, label_y, &point.label).size(self.options.font_size).color(color).align(align);
                    writeln!(w, "{}", text)?;
                }
            }
            Ok(())
        }

        fn face_to_svg_path(&self, face: Face) -> Path {
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

        fn edge_to_svg_path(&self, edge: Edge) -> LineSegment {
            let (a,b) = self.regioncam.endpoints(edge);
            let a = self.vertex_coord(a);
            let b = self.vertex_coord(b);
            line_segment(a.0, a.1, b.0, b.1)
        }
    }

    impl From<crate::Color> for svg_fmt::Color {
        fn from(color: crate::Color) -> Self {
            svg_fmt::Color {
                r: (color.r * 255.0) as u8,
                g: (color.g * 255.0) as u8,
                b: (color.b * 255.0) as u8,
            }
        }
    }
    impl From<crate::Color> for svg_fmt::Fill {
        fn from(value: crate::Color) -> Self {
            svg_fmt::Color::from(value).into()
        }
    }
    impl From<svg_fmt::Color> for crate::Color {
        fn from(color: svg_fmt::Color) -> Self {
            crate::Color {
                r: color.r as f32 / 255.0,
                g: color.g as f32 / 255.0,
                b: color.b as f32 / 255.0,
            }
        }
    }
    
    impl Regioncam {
        /// Output to svg
        pub fn write_svg(&self, options: &RenderOptions, w: &mut dyn Write) -> std::io::Result<()> {
            Renderer::new(self, options).write_svg(w)
        }
    }
}



#[cfg(feature = "piet")]
mod piet {
    use piet::*;
    use piet::kurbo::*;
    use crate::*;

    impl<'a> Renderer<'a> {
        fn render(&self, ctx: &mut impl RenderContext) {
            if self.options.draw_faces {
                self.render_faces(ctx);
            }
            if self.options.draw_edges {
                self.render_edges(ctx);
            }
            if self.options.draw_vertices {
                self.render_vertices(ctx);
            }
            self.render_points(ctx);
        }

        fn render_faces(&self, ctx: &mut impl RenderContext) {
            for face in self.regioncam.faces() {
                let path = self.face_to_piet_path(face);
                let color = self.face_color(face);
                ctx.fill(path.as_slice(), &piet::Color::from(color));
            }
        }

        fn render_edges(&self, ctx: &mut impl RenderContext) {
            for edge in self.regioncam.edges() {
                match self.edge_style(edge) {
                    None => (),
                    Some((color, width)) => {
                        let path = self.edge_to_piet_path(edge);
                        ctx.stroke(path, &piet::Color::from(color), width.into());
                    }
                }
            }
        }

        fn render_vertices(&self, ctx: &mut impl RenderContext) {
            let radius = f64::from(self.options.vertex_size) * 0.5;
            let color = piet::Color::BLACK;
            for vertex in self.regioncam.vertices() {
                let pos = to_point(self.vertex_coord(vertex));
                ctx.fill(Circle::new(pos, radius), &color);
            }
        }

        fn render_points(&self, ctx: &mut impl RenderContext) {
            for point in self.points {
                let pos = to_point(self.point_coord(point.position.as_ref().into()));
                let radius = f64::from(self.options.point_size) * 0.5;
                let color = piet::Color::from(point.color.unwrap_or(self.options.point_color));
                if radius > 0.0 {
                    ctx.fill(Circle::new(pos, radius), &color)
                }
                if !point.label.is_empty() && self.options.font_size > 0.0 {
                    let font_size_pt = self.options.font_size as f64 * 0.75;
                    let layout = ctx.text().new_text_layout(point.label.clone())
                      .font(FontFamily::SANS_SERIF, font_size_pt)
                      .text_color(color)
                      .build().unwrap();
                    let text_pos =
                        if self.options.text_only_markers {
                            Point::new(pos.x - layout.size().width * 0.5, pos.y - layout.size().height * 0.5)
                        } else {
                            pos + (radius, -radius - layout.line_metric(0).map_or(0.0, |m|m.baseline))
                        };
                    ctx.draw_text(&layout, text_pos);
                }
            }
        }

        fn face_to_piet_path(&self, face: Face) -> Vec<PathEl> {
            self.regioncam.vertices_on_face(face)
                .enumerate()
                .map(|(i, vertex)| {
                    let point = to_point(self.vertex_coord(vertex));
                    if i == 0 {
                        PathEl::MoveTo(point)
                    } else {
                        PathEl::LineTo(point)
                    }
                })
                .chain(std::iter::once(PathEl::ClosePath))
                .collect()
        }

        fn edge_to_piet_path(&self, edge: Edge) -> Line {
            let (a,b) = self.regioncam.endpoints(edge);
            let a = self.vertex_coord(a);
            let b = self.vertex_coord(b);
            Line::new(to_point(a), to_point(b))
        }
    }
    
    #[cfg(feature = "tiny-skia")]
    mod png {
        use super::*;
        use ::png::EncodingError;
        use piet_tiny_skia::Cache;
        use tiny_skia::Pixmap;

        impl<'a> Renderer<'a> {
            fn render_to_pixmap(&self) -> Pixmap {
                let image_size = self.options.image_size;
                let mut pixmap = Pixmap::new(image_size.0 as u32, image_size.1 as u32).unwrap();
                let mut cache = Cache::new();
                let mut ctx = cache.render_context(&mut pixmap);
                self.render(&mut ctx);
                pixmap
            }
            pub fn write_png(&self, path: impl AsRef<std::path::Path>) -> Result<(), EncodingError> {
                self.render_to_pixmap().save_png(path)
            }
            pub fn encode_png(&self) -> Result<Vec<u8>, EncodingError> {
                self.render_to_pixmap().encode_png()
            }
        }
    }

    // Note: piet 0.7.0 implements From<(f32,f32)> for Point, but 0.6.2 does not
    fn to_point(p: (f32,f32)) -> Point {
        Point::new(p.0.into(), p.1.into())
    }
    
    impl From<crate::Color> for piet::Color {
        fn from(color: crate::Color) -> Self {
            piet::Color::rgb8(
                (color.r * 255.0) as u8,
                (color.g * 255.0) as u8,
                (color.b * 255.0) as u8,
            )
        }
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
