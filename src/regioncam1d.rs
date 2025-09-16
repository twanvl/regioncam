use std::ops::Range;
use std::fmt::Debug;

use ndarray::{array, Array, Array1, Array2, ArrayView2, ArrayViewMut2, Axis};
use crate::partition::{declare_index_type, Index, Vertex};
use crate::util::*;
use crate::{EdgeLabel, LayerNr, NNBuilder, Regioncam, Plane1D};

// Edges in a 1D partition are just (vertices[i], vertices[i+1]).
// We identify edges by the position of the first vertex.
declare_index_type!(pub Edge1D);
impl Debug for Edge1D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "e{}", self.0)
    }
}


/// All data for a layer
#[derive(Debug, Clone, PartialEq)]
struct Layer1D {
    vertex_data: Array2<f32>, // vertex -> f32[D_l]
    continuous: bool,         // is the function continuous at this layer?
}

impl Layer1D {
    fn vertices(&self) -> &Array2<f32> {
        &self.vertex_data
    }
}

impl Layer1D {
    /// Remove data for a vertex
    fn swap_remove_vertex(&mut self, vertex: Vertex) {
        swap_remove_row(&mut self.vertex_data, vertex.index());
    }
}

pub type VertexLabel = EdgeLabel;

/// A partition of 1d space, annalogous to the 2d Regioncam
#[derive(Debug, Clone, PartialEq)]
pub struct Regioncam1D {
    // Order of vertices, when sorted by input coordinate.
    // We don't keep the vertex data in this order, because that would involve a lot of moving
    vertices: Vec<Vertex>,
    // Are the vertices sorted?
    sorted: bool,
    /// Data for vertice & edges at each layer.
    layers: Vec<Layer1D>,
    //edge_label: Vec<EdgeLabel>, // vertex -> label
    vertex_label: Vec<VertexLabel>, // vertex -> label
}

impl Regioncam1D {
    /// Construct a 1d regioncam that covers the given range of coordinates.
    pub fn new(range: Range<f32>) -> Self {
        let layer = Layer1D {
            vertex_data: array![[range.start], [range.end]],
            continuous: true,
        };
        Regioncam1D {
            vertices: vec![Vertex::from(0), Vertex::from(1)],
            //vertex_positions: vec![Edge1D::from(0), Edge1D::from(1)],
            sorted: true,
            layers: vec![layer],
            vertex_label: vec![VertexLabel::default(); 2],
        }
    }
    pub fn from_size(size: f32) -> Self {
        Self::new(-size*0.5 .. size*0.5)
    }
    pub fn from_plane(plane: &Plane1D, size: f32) -> Self {
        let mut out = Self::from_size(plane.size() * size);
        out.add(&plane.mapping);
        out
    }

    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }
    pub fn num_edges(&self) -> usize {
        self.num_vertices() - 1
    }
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
    
    pub fn edges(&self) -> impl Iterator<Item=Edge1D> {
        (0..self.num_edges()).map(Edge1D::from)
    }
    pub fn start(&self, edge: Edge1D) -> Vertex {
        self.vertices[edge.index()]
    }
    pub fn end(&self, edge: Edge1D) -> Vertex {
        self.vertices[edge.index() + 1]
    }
    pub fn endpoints(&self, edge: Edge1D) -> (Vertex, Vertex) {
        (self.start(edge), self.end(edge))
    }
    pub fn vertex_label(&self, vertex: Vertex) -> &VertexLabel {
        &self.vertex_label[vertex.index()]
    }
    pub fn activations(&self, layer: usize) -> &Array2<f32> {
        &self.layers[layer].vertex_data
    }
    
    /// Sort all vertex data
    /// Postcondition: self.vertices == (0..self.num_vertices()).collect()
    pub fn sort(&mut self) {
        if !self.sorted {
            self.vertex_label = permute(&self.vertex_label, &self.vertices);
            for layer in self.layers.iter_mut() {
                // reorder data based on vertex order
                layer.vertex_data = permute_rows(&layer.vertex_data, &self.vertices);
            }
            let n = self.num_vertices();
            self.vertices.clear();
            self.vertices.extend((0..n).map(Vertex::from));
            self.sorted = true;
        }
    }

    pub fn check_invariants(&self) {
        if self.sorted {
            assert_eq!(self.vertices, (0..self.num_vertices()).map(Vertex::from).collect::<Vec<_>>());
        }
        assert_eq!(self.vertices.len(), self.num_vertices());
        assert_eq!(self.vertex_label.len(), self.num_vertices());
        for layer in &self.layers {
            assert_eq!(layer.vertex_data.len_of(Axis(0)), self.num_vertices());
        }
        // each vertex id is used exactly once
        assert!(is_permutation(&self.vertices));
    }

    fn split_all<E: EdgeSplitter1D>(&mut self, splitter: &mut E, vertex_label: VertexLabel) {
        let old_num_vertices = self.num_vertices();
        let mut new_num_vertices = old_num_vertices;
        let mut split_edges = vec![];
        for edge in self.edges() {
            let (a,b) = self.endpoints(edge);
            if let Some(t) = splitter.split_point(self, edge, a, b) {
                if t > 0.0 && t < 1.0 {
                    // compute new vertex data at lerp(a,b,t)
                    for layer in self.layers.iter_mut() {
                        append_lerp(&mut layer.vertex_data, a.index(), b.index(), t);
                    }
                    let new_vertex = Vertex::from(new_num_vertices);
                    new_num_vertices += 1;
                    splitter.apply_split(self, new_vertex, t);
                    split_edges.push(edge);
                }
            }
        }
        // update vertex labels
        self.vertex_label.resize(new_num_vertices, vertex_label);
        // update vertex order
        // Note: when splitting an edge(edge.index(),edge.index()+1), the new vertex should be inserted after edge.index(), so before edge.index()+1
        // Note: we have to reverse the iterator to go in decreasing order (needed for `insert_many`)
        insert_many(&mut self.vertices, split_edges.iter().enumerate().map(
            |(i, edge)| (edge.index() + 1, Vertex::from(old_num_vertices + i))
        ).rev());
        // inserting vertices makes the 
        self.sorted = false;
    }

    /// Split all faces in the partition by treating zeros of the given layer as edges.
    fn split_by_zero_at(&mut self, layer: usize) {
        // Split faces on activation = 0
        let acts = self.activations(layer);
        for dim in 0..acts.len_of(Axis(1)) {
            let vertex_label = VertexLabel { layer: self.num_layers(), dim };
            struct ZeroSplitter {
                layer: usize,
                dim: usize,
            }
            impl EdgeSplitter1D for ZeroSplitter {
                fn split_point(&self, rc: &Regioncam1D, _edge: Edge1D, a: Vertex, b: Vertex) -> Option<f32> {
                    let acts = &rc.layers[self.layer].vertex_data;
                    find_zero(acts[(a.index(), self.dim)], acts[(b.index(), self.dim)])
                }
                fn apply_split(&mut self, rc: &mut Regioncam1D, new_vertex: Vertex, _point: f32) {
                    // make sure that new vertices are actually 0 in dimension dim
                    // this may not be true exactly because of rounding errors.
                    let acts = &mut rc.layers[self.layer].vertex_data;
                    acts[(new_vertex.index(), self.dim)] = 0.0;
                }
            }
            let mut splitter = ZeroSplitter { layer, dim };
            self.split_all(&mut splitter, vertex_label);
        }
    }

    /// Split all faces in the partition at points where argmax of dims changes
    /// Returns: (vertex_max)
    fn split_by_argmax_at(&mut self, layer: usize) -> Array1<f32> {
        // It is hard to do this fully correctly
        // approximate algorithm:
        //  * sort dims by most likely to contain maximum (to avoid unnecesary splits)
        //  * for each dim: split_by difference of that dim and current max
        // this might introduce some unnecessary splits
        
        let acts = self.activations(layer);
        let dims = {
            // for each vertex: find argmax
            let vertex_argmax = acts.rows().into_iter().map(argmax);
            // sort dims by frequency
            let histogram_argmax = histogram(vertex_argmax, acts.ncols());
            let mut dims = Vec::from_iter(0..histogram_argmax.len());
            dims.sort_by_key(|i| std::cmp::Reverse(histogram_argmax[*i])); // most frequent first
            dims
        };

        // split dims
        let label_layer = self.num_layers();
        let mut vertex_max = {
            struct MaxSplitter {
                //argmax_so_far: usize,
                layer: usize,
                dim: usize,
                max_so_far: Vec<f32>,
            }
            impl EdgeSplitter1D for MaxSplitter {
                fn split_point(&self, rc: &Regioncam1D, _edge: Edge1D, a: Vertex, b: Vertex) -> Option<f32> {
                    // split by zeros of (acts[:,dim] - max_so_far)
                    let acts = &rc.layers[self.layer].vertex_data;
                    find_zero(acts[(a.index(), self.dim)] - self.max_so_far[a.index()], acts[(b.index(), self.dim)] - self.max_so_far[b.index()])
                }
                fn apply_split(&mut self, rc: &mut Regioncam1D, new_vertex: Vertex, _t: f32) {
                    let acts = &mut rc.layers[self.layer].vertex_data;
                    self.max_so_far.push(acts[(new_vertex.index(), self.dim)]);
                }
            }
            let mut splitter = MaxSplitter {
                layer,
                dim: dims[0],
                max_so_far: acts.column(dims[0]).to_vec(),
                //argmax_so_far: vec![dims[0]; acts.nrows()];
            };
            for &dim in &dims[1..] {
                let vertex_label = VertexLabel { layer: label_layer, dim };
                splitter.dim = dim;
                self.split_all(&mut splitter, vertex_label);
                // update max_so_far
                let acts = self.activations(layer);
                for (m, x) in splitter.max_so_far.iter_mut().zip(acts.index_axis(Axis(1), dim).iter()) {
                    if *m < *x {
                        *m = *x;
                    }
                }
            }
            splitter.max_so_far
        };

        // We might have made unnecessary splits.
        // We can undo these by removing added vertices with the same argmax on both sides
        {
            let edge_argmax = self.argmax_edge_index(layer, &vertex_max);
            filter_vertices(
                &mut self.vertices,
                &mut (VertexData(&mut vertex_max), self.layers.as_mut()),
                |v, e1, e2| {
                    if self.vertex_label[v.index()].layer == label_layer {
                        if let (Some(e1), Some(e2)) = (e1, e2) {
                            // keep new vertices only if argmax of incident edges is different
                            return edge_argmax[e1.index()] != edge_argmax[e2.index()];
                        }
                    }
                    return true; // keep all older vertices
                }
            );
        };

        // sanity check: activations should be maxima
        if cfg!(debug_assertions) {
            let acts = self.activations(layer);
            for (row, max) in acts.axis_iter(Axis(0)).zip(vertex_max.iter()) {
                if row.iter().all(|x| x < max) {
                    println!("Bad maximum: {row} < {max}");
                }
            }
        }

        vertex_max.into()
    }

    fn argmax_edge_index(&self, layer: usize, max: &[f32]) -> Vec<usize> {
        let acts = self.activations(layer);
        let mut delta_from_max = Array1::zeros(acts.ncols());
        self.edges().map(|edge| {
            // for every dim: how far is any vertex in this face below the maximum?
            delta_from_max.fill(0.0);
            for v in <[Vertex;2]>::from(self.endpoints(edge)) {
                let max = max[v.index()];
                delta_from_max.zip_mut_with(&acts.row(v.index()),
                    |delta, x| *delta = f32::min(*delta, *x - max)
                );
            }
            // The argmax dimension is the one with the smallest deviation from the maximum at any vertex
            // Ideally this would be 0, but numberical errors can change that.
            argmax(delta_from_max.view())
        }).collect()
    }
}

impl NNBuilder for Regioncam1D {
    type LayerNr = LayerNr;

    fn last_layer(&self) -> LayerNr {
        self.num_layers() - 1
    }

    fn generalized_linear_at(&mut self, layer: LayerNr, fun: impl Fn(ArrayView2<f32>) -> Array2<f32>, add_bias: impl Fn(ArrayViewMut2<f32>)) -> LayerNr {
        // Compute vertex data (x' = x*w + b)
        let input_layer = &self.layers[layer];
        let mut vertex_data = fun(input_layer.vertex_data.view());
        add_bias(vertex_data.view_mut());
        // Add layer
        let continuous = input_layer.continuous;
        self.layers.push(Layer1D{ vertex_data, continuous });
        self.last_layer()
    }

    fn relu_at(&mut self, layer_nr: LayerNr) -> LayerNr {
        self.split_by_zero_at(layer_nr);
        // Compute relu output for vertices
        let input_layer = &self.layers[layer_nr];
        let vertex_data = relu(&input_layer.vertex_data.view());
        // Add layer
        let continuous = input_layer.continuous;
        self.layers.push(Layer1D{ vertex_data, continuous });
        self.last_layer()
    }

    fn leaky_relu_at(&mut self, layer_nr: LayerNr, negative_slope: f32) -> LayerNr {
        self.split_by_zero_at(layer_nr);
        // Compute relu output for vertices
        let input_layer = &self.layers[layer_nr];
        let vertex_data = leaky_relu(&input_layer.vertex_data.view(), negative_slope);
        // Add layer
        let continuous = input_layer.continuous;
        self.layers.push(Layer1D{ vertex_data, continuous });
        self.last_layer()
    }
    
    fn max_pool_at(&mut self, layer_nr: LayerNr) -> LayerNr {
        let max = self.split_by_argmax_at(layer_nr);
        // New vertex data is just the max value
        let input_layer = &self.layers[layer_nr];
        let vertex_data = max.insert_axis(Axis(1));
        // Add layer
        let continuous = input_layer.continuous;
        self.layers.push(Layer1D{ vertex_data, continuous });
        self.last_layer()
    }
    
    fn argmax_pool_at(&mut self, layer_nr: LayerNr) -> LayerNr {
        let _max = self.split_by_argmax_at(layer_nr);
        // New vertex data is the argmax for each row
        let acts = self.activations(layer_nr);
        let vertex_argmax = Array::from_iter(acts.rows().into_iter().map(|row| argmax(row) as f32));
        let vertex_data = vertex_argmax.insert_axis(Axis(1));
        // Add layer
        let continuous = false;
        self.layers.push(Layer1D{ vertex_data, continuous });
        self.last_layer()
    }
    
    fn sign_at(&mut self, layer_nr: LayerNr) -> LayerNr {
        // Split faces
        self.split_by_zero_at(layer_nr);
        // Compute sign for vertices
        let input_layer = &self.layers[layer_nr];
        let vertex_data = input_layer.vertex_data.mapv(|x| if x >= 0.0 { 1.0 } else { 0.0 });
        // Add layer
        let continuous = false;
        self.layers.push(Layer1D{ vertex_data, continuous });
        self.last_layer()
    }
    
    fn decision_boundary_at(&mut self, layer_nr: LayerNr) -> LayerNr {
        if self.layers[layer_nr].vertex_data.ncols() == 1 {
            self.sign_at(layer_nr)
        } else {
            self.argmax_pool_at(layer_nr)
        }
    }
    
    fn sum(&mut self, layer1: LayerNr, layer2: LayerNr) -> LayerNr {
        let layer1 = &self.layers[layer1];
        let layer2 = &self.layers[layer2];
        self.layers.push(Layer1D {
            vertex_data: &layer1.vertex_data + &layer2.vertex_data,
            continuous: layer1.continuous && layer2.continuous,
        });
        self.last_layer()
    }
}


impl Regioncam {
    /// Return a 1d slice of a 2d regioncam.
    /// The first layer will be a linear mapping from 1d to 2d.
    /// `plane` is a 1d hyperplane in the 2d space of the 2d regioncam (a line).
    pub fn slice(&self, plane: &Plane1D) -> Regioncam1D {
        // signed distance to plane at each vertex
        let distance_to_plane = plane.perpendicular().project(&self.inputs().view());
        assert_eq!(distance_to_plane.shape(), [self.num_vertices(), 1]);
        let distance_to_plane = distance_to_plane.as_slice().unwrap();
        // if intersection is at vertex/vertices, then add the vertex itself
        let mut out_vertices = self.vertices().zip(distance_to_plane)
            .filter(|(_,x)| **x == 0.0)
            .map(|(v,_)| IndexOrLerp::Index(v))
            .collect::<Vec<_>>();
        // for every edge in the 2d regioncam, if it intersects the plane, then create a vertex
        for edge in self.edges() {
            let (a,b) = self.endpoints(edge);
            match find_zero(distance_to_plane[a.index()], distance_to_plane[b.index()]) {
                Some(t) if t > 0.0 && t < 1.0 => {
                    out_vertices.push(IndexOrLerp::Lerp(edge, a, b, t));
                }
                _ => (),
            }
        }
        // vertex positions projected to line
        let position = plane.project(&self.inputs().view());
        assert_eq!(position.shape(), [self.num_vertices(), 1]);
        let position = position.as_slice().unwrap();
        let position_of = |v| match v {
            IndexOrLerp::Index(v) => position[v.index()],
            IndexOrLerp::Lerp(_, a, b, t) => lerp(position[a.index()], position[b.index()], t),
        };
        // sort vertices by position
        out_vertices.sort_by(|&a, &b| position_of(a).partial_cmp(&position_of(b)).unwrap());
        let out_vertices = out_vertices;
        // collect layer data
        let layers = 
            std::iter::once({
                // first layer: input positions
                let vertex_data = out_vertices.iter().map(|v| position_of(*v)).collect::<Vec<_>>();
                let vertex_data = Array2::from_shape_vec((vertex_data.len(),1), vertex_data).unwrap();
                Layer1D{ vertex_data, continuous: true }
            }).chain(self.layers.iter().map(|layer| {
                // other layers: values at vertex positions
                let mut vertex_data = Array2::zeros((out_vertices.len(), layer.dim()));
                for (mut row, v) in vertex_data.rows_mut().into_iter().zip(out_vertices.iter()) {
                    match v {
                        IndexOrLerp::Index(v) => {
                            row.assign(&layer.vertices().row(v.index()));
                        }
                        IndexOrLerp::Lerp(_, a, b, t) => {
                            if layer.is_continuous() {
                                row.assign(&lerp(&layer.vertices().row(a.index()),  &layer.vertices().row(b.index()), *t));
                            } else {
                                // for discrete layers: don't interpolate
                                // note that sometimes vertex values are not well-defined in this case
                                row.assign(&layer.vertices().row(a.index()));
                            }
                        }
                    };
                }
                Layer1D{ vertex_data, continuous: layer.is_continuous() }
            }))
            .collect();
        // collect vertex labels
        let vertex_label = out_vertices.iter().copied()
            .map(|idx| match idx {
                IndexOrLerp::Index(v) => self.vertex_label(v),
                IndexOrLerp::Lerp(edge, _, _, _) => *self.edge_label(edge),
            })
            .collect();
        // build 1D regioncam
        let vertices = (0..out_vertices.len()).map(Vertex::from).collect::<Vec<_>>();
        Regioncam1D { vertices, sorted: true, layers, vertex_label }
    }
}

// An index into an array, or a linear interpolation between two indices
#[derive(Clone, Copy, Debug)]
enum IndexOrLerp {
    Index(Vertex),
    Lerp(crate::Edge, Vertex, Vertex, f32),
}


/// Collections of vertices that support a swap_remove operation
trait SwapRemoveVertex {
    fn swap_remove_vertex(&mut self, vertex: Vertex);
}
impl SwapRemoveVertex for () {
    fn swap_remove_vertex(&mut self, _vertex: Vertex) {}
}
impl<A: SwapRemoveVertex, B: SwapRemoveVertex> SwapRemoveVertex for (A,B) {
    fn swap_remove_vertex(&mut self, vertex: Vertex) {
        self.0.swap_remove_vertex(vertex);
        self.1.swap_remove_vertex(vertex);
    }
}
struct VertexData<'a, T>(&'a mut Vec<T>);
impl<'a, T> SwapRemoveVertex for VertexData<'a, T> {
    fn swap_remove_vertex(&mut self, vertex: Vertex) {
        self.0.swap_remove(vertex.index());
    }
}
impl SwapRemoveVertex for &mut [Layer1D] {
    fn swap_remove_vertex(&mut self, vertex: Vertex) {
        for layer in self.iter_mut() {
            layer.swap_remove_vertex(vertex);
        }
    }
}

/// Given a list of vertices, keep only those for which the predicate returns true
/// Removed vertices are also removed from vertex_data with swap_remove_vertex
fn filter_vertices(vertices: &mut Vec<Vertex>, vertex_data: &mut impl SwapRemoveVertex, keep: impl Fn(Vertex, Option<Edge1D>, Option<Edge1D>) -> bool) {
    let old_num_vertices = vertices.len();
    // the vector of removed vertices, where
    //   removed_vertices[i] = v
    //   means that vertex v was removed with swap_remove.
    //   this means that Vertex(old_num_vertices-1-i) is now called v
    //   note that the vertex that is then named v could be removed as well.
    let mut removed_vertices = vec![];
    // position of current vertex
    let mut i = 0;
    vertices.retain(|v| {
        let edge_before= if i > 0 { Some(Edge1D(i - 1)) } else { None };
        let edge_after = if i + 1 < old_num_vertices as Index { Some(Edge1D(i)) } else { None };
        let keep_this = keep(*v, edge_before, edge_after);
        if !keep_this {
            vertex_data.swap_remove_vertex(*v);
            removed_vertices.push(*v);
        }
        i += 1;
        keep_this
    });
    // relabel vertices that were swapped
    let new_num_vertices = vertices.len();
    for v in vertices.iter_mut() {
        while v.index() >= new_num_vertices {
            // this vertex was removed, and renamed
            *v = removed_vertices[old_num_vertices - 1 - v.index()];
        }
    }
}


/// Trait for deciding to split an edge
trait EdgeSplitter1D {
    fn split_point(&self, rc: &Regioncam1D, edge: Edge1D, a: Vertex, b: Vertex) -> Option<f32>;
    fn apply_split(&mut self, rc: &mut Regioncam1D, new_vertex: Vertex, t: f32);
}


#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_abs_diff_eq;
    use rand::prelude::*;

    #[test]
    fn check_invariants() {
        let rc = Regioncam1D::new(-1.0 .. 1.0);
        rc.check_invariants();
    }

    #[test]
    fn random_nn() {
        let mut rng = SmallRng::seed_from_u64(42);
        let dim_in = 1;
        let dim_out = 100;
        let layer = crate::nn::Linear::new_uniform(dim_in, dim_out, &mut rng);
        // partition
        let mut rc = Regioncam1D::new(-1.0..1.0);
        rc.add(&layer);
        rc.check_invariants();
        rc.relu();
        rc.check_invariants();
        // sort
        rc.sort();
        rc.check_invariants();
    }

    #[test]
    fn max_pool() {
        let mut rng = SmallRng::seed_from_u64(42);
        let dim_in = 1;
        let dim_hidden = 10;
        let dim_out = 10;
        // two layer network
        let mut rc = Regioncam1D::new(-1.0..1.0);
        rc.add(&crate::nn::Linear::new_uniform(dim_in, dim_hidden, &mut rng));
        rc.relu();
        rc.check_invariants();
        rc.add(&crate::nn::Linear::new_uniform(dim_hidden, dim_out, &mut rng));
        rc.check_invariants();
        rc.max_pool();
        rc.check_invariants();
    }

    impl approx::AbsDiffEq for Regioncam1D {
        type Epsilon = f32;
    
        fn default_epsilon() -> Self::Epsilon {
            1e-5
        }
    
        fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
            self.vertices == other.vertices &&
            self.num_layers() == other.num_layers() &&
            self.layers.iter().zip(other.layers.iter()).all(
                |(l1, l2) | l1.vertex_data.abs_diff_eq(&l2.vertex_data, epsilon)
            )
        }
    }
    
    #[test]
    fn from_2d() {
        let mut rng = SmallRng::seed_from_u64(42);
        let points = array![[-1.0,0.4], [1.0,-0.4]];
        let mut rc2d = Regioncam::square(1.0);
        let l = norm(&points.row(0));
        let mut rc1d = Regioncam1D::new(-l..l);
        // mapping 1d->2d
        let line = Plane1D::through_points(&points.view());
        rc1d.add(&line.mapping);
        assert_abs_diff_eq!(rc2d.slice(&line), rc1d);
        // create a random network
        let dim_hidden = 10;
        let num_layers = 2;
        for i in 0..num_layers {
            let dim_in = rc2d.layer(rc2d.last_layer()).dim();
            let layer = crate::nn::Linear::new_uniform(dim_in, dim_hidden, &mut rng);
            rc2d.add(&layer);
            rc1d.add(&layer);
            let mut rc1d_sorted = rc1d.clone();
            rc1d_sorted.sort();
            assert_abs_diff_eq!(rc2d.slice(&line), rc1d_sorted);
            if i == num_layers - 1 {
                // Note: decision boundary layers can have different vertex values, because they are not continuous.
                // that means that for some vertices there can be 2 valid values, and the two regioncam algorithms could pick different ones.
                // max_pool is the continuous alternative to argmax
                rc2d.max_pool();
                rc1d.max_pool();
            } else {
                rc2d.relu();
                rc1d.relu();
            }
            let mut rc1d_sorted = rc1d.clone();
            rc1d_sorted.sort();
            assert_abs_diff_eq!(rc2d.slice(&line), rc1d_sorted);
        }
    }
}