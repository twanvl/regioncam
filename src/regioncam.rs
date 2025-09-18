
use std::fmt::Debug;
use std::hash::{DefaultHasher, Hash, Hasher};
use ndarray::{array, concatenate, s, Array, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayViewMut2, Axis, NewAxis};
use approx::assert_abs_diff_eq;

use crate::util::*;
use crate::partition::*;
use crate::plane::*;
pub use crate::nn::NNBuilder;


#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct EdgeLabel {
    /// Layer on which this edge was introduced
    pub layer: usize,
    /// Dimension of that layer which caused this edge to be created
    pub dim: usize,
}

pub type LayerNr = usize;

#[derive(Debug, Clone, PartialEq)]
enum LayerType {
    Input,
    Linear,
    /// ReLU like layers
    ReLU { face_mask: Array2<bool> },
    /// Max pooling like layers
    MaxPool { face_argmax: Array1<usize> },
}

/// All data for a layer
#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    vertex_data: Array2<f32>, // vertex -> f32[D_l]
    face_data: Array3<f32>,   // face -> f32[D_in, D_F]
    continuous: bool,         // is the function continuous at this layer?
    layer_type: LayerType,    // how this layer was constructed from previous ones
    //shape: Shape, // actual shape of the data
}

impl Layer {
    pub fn is_continuous(&self) -> bool {
        self.continuous
    }
    pub fn vertices(&self) -> &Array2<f32> {
        &self.vertex_data
    }
    pub fn faces(&self) -> &Array3<f32> {
        &self.face_data
    }
    pub fn is_nonlinear(&self) -> bool {
        match self.layer_type {
            LayerType::Input => false,
            LayerType::Linear => false,
            LayerType::ReLU {..} => true,
            LayerType::MaxPool {..} => true,
        }
    }
    pub fn dim(&self) -> usize {
        self.vertex_data.ncols()
    }
    fn hash_face(&self, face: Face, hasher: &mut impl Hasher) {
        match &self.layer_type {
            LayerType::Input => (),
            LayerType::Linear => (),
            LayerType::ReLU { face_mask } => {
                face_mask.index_axis(Axis(0), face.index()).hash(hasher);
            },
            LayerType::MaxPool { face_argmax } => {
                face_argmax.index_axis(Axis(0), face.index()).hash(hasher);
            },
        }
    }
    /// Append a copy of the given face
    fn append_face(&mut self, face: Face) {
        append_row(&mut self.face_data, face.index());
        match &mut self.layer_type {
            LayerType::Input => (),
            LayerType::Linear => (),
            LayerType::ReLU { face_mask } => {
                append_row(face_mask, face.index());
            },
            LayerType::MaxPool { face_argmax } => {
                append_row(face_argmax, face.index());
            },
        }
    }
    /// Append a vertex that is at lerp(a,b,t)
    fn append_vertex_lerp(&mut self, a: Vertex, b: Vertex, t: f32) {
        append_lerp(&mut self.vertex_data, a.index(), b.index(), t);
    }
    /// Remove data for a face
    fn swap_remove_face(&mut self, face: Face) {
        swap_remove_row(&mut self.face_data, face.index());
        match &mut self.layer_type {
            LayerType::Input => (),
            LayerType::Linear => (),
            LayerType::ReLU { face_mask } => {
                swap_remove_row(face_mask, face.index());
            },
            LayerType::MaxPool { face_argmax } => {
                swap_remove_row(face_argmax, face.index());
            },
        }
    }
}

/// A partition of 2d space, with associated vertex and face data per layer.
#[derive(Debug, Clone, PartialEq)]
pub struct Regioncam {
    partition: Partition,
    /// Data for vertice & faces at each layer
    pub(crate) layers: Vec<Layer>,
    edge_label: Vec<EdgeLabel>,    // edge -> label
}


impl Regioncam {

    // Constructors

    /// Construct a Partition with a single region, containing the given convex polygon
    pub fn from_polygon(vertex_data: Array2<f32>) -> Self {
        let n = vertex_data.nrows();
        assert_eq!(vertex_data.ncols(), 2, "Expected two dimensional points");
        let layer = Layer {
            vertex_data,
            face_data: array![[
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]],
            continuous: true,
            layer_type: LayerType::Input,
        };
        Self {
            partition: Partition::polygon(n),
            edge_label: vec![EdgeLabel::default(); n],
            layers: vec![layer],
        }
    }
    /// Construct a partition with a single rectangle
    pub fn rectangle(u0: f32, u1: f32, v0: f32, v1: f32) -> Self {
        Self::from_polygon(array![
            [u0, v0],
            [u0, v1],
            [u1, v1],
            [u1, v0],
        ])
    }
    /// Construct a partition with a single square centered on the origin
    pub fn square(size: f32) -> Self {
        Self::rectangle(-size, size, -size, size)
    }
    /// Construct a partition with a regular polygon centered on the origin, approximating a circle
    pub fn circle(radius: f32) -> Self {
        use std::f32::consts::TAU;
        let n = 90;
        let mut points = Array2::zeros((n, 2));
        for i in 0..n {
            let t = i as f32 * (TAU / n as f32);
            points[(i,0)] = radius * f32::cos(t);
            points[(i,1)] = radius * f32::sin(t);
        }
        Self::from_polygon(points)
    }
    pub fn from_plane(plane: &Plane, size: f32) -> Self {
        let mut out = Self::square(plane.size() * size);
        out.add(&plane.mapping);
        out
    }

    // Accessors

    pub fn num_vertices(&self) -> usize {
        self.partition.num_vertices()
    }
    pub fn num_faces(&self) -> usize {
        self.partition.num_faces()
    }
    pub fn num_halfedges(&self) -> usize {
        self.partition.num_halfedges()
    }
    pub fn num_edges(&self) -> usize {
        self.partition.num_edges()
    }
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
    pub fn activations(&self, layer: usize) -> &Array2<f32> {
        &self.layers[layer].vertex_data
    }
    pub fn activations_last(&self) -> &Array2<f32> {
        &self.layers.last().unwrap().vertex_data
    }
    pub fn inputs(&self) -> &Array2<f32> {
        &self.layers[0].vertex_data
    }
    pub fn vertex_inputs(&self, vertex: Vertex) -> ArrayView1<'_, f32> {
        self.layers[0].vertex_data.row(vertex.index())
    }
    pub fn vertex_activations_at(&self, layer: LayerNr, vertex: Vertex) -> ArrayView1<'_, f32> {
        self.layers[layer].vertex_data.row(vertex.index())
    }
    pub fn face_activations_at(&self, layer: LayerNr) -> &Array3<f32> {
        &self.layers[layer].face_data
    }
    pub fn face_activations(&self) -> &Array3<f32> {
        self.face_activations_at(self.last_layer())
    }

    pub fn opposite(&self, he: Halfedge) -> Halfedge {
        he.opposite()
    }
    pub fn next(&self, he: Halfedge) -> Halfedge {
        self.partition.next(he)
    }
    pub fn prev(&self, he: Halfedge) -> Halfedge {
        self.partition.prev(he)
    }
    pub fn face(&self, he: Halfedge) -> Option<Face> {
        self.partition.face(he)
    }
    pub fn start(&self, he: Halfedge) -> Vertex {
        self.partition.start(he)
    }
    pub fn end(&self, he: Halfedge) -> Vertex {
        self.partition.end(he)
    }
    pub fn endpoints(&self, edge: Edge) -> (Vertex, Vertex) {
        self.partition.endpoints(edge)
    }
    pub fn edge_label(&self, edge: Edge) -> &EdgeLabel {
        &self.edge_label[edge.index()]
    }
    pub fn edge_faces(&self, edge: Edge) -> (Option<Face>, Option<Face>) {
        self.partition.edge_faces(edge)
    }
    pub fn is_boundary(&self, edge: Edge) -> bool {
        self.partition.is_boundary(edge)
    }
    pub fn halfedge_on_face(&self, face: Face) -> Halfedge {
        self.partition.halfedge_on_face(face)
    }
    pub fn face_centroid_at(&self, layer: LayerNr, face: Face) -> Array1<f32> {
        self.vertex_data_for_face(face, layer).mean_axis(Axis(0)).unwrap()
    }
    pub fn face_centroid(&self, face: Face) -> Array1<f32> {
        self.face_centroid_at(0, face)
    }
    pub fn face_activation_for(&self, layer: LayerNr, face: Face, inputs: ArrayView1<f32>) -> Array1<f32> {
        let face_data = self.layers[layer].face_data.index_axis(Axis(0), face.index());
        let inputs = concatenate![Axis(0), inputs, Array1::ones(1)];
        inputs.dot(&face_data)
    }
    /// Compute a hash code for a given face
    /// The hash code is based only on activations, so it is stable to recreating the partition
    pub fn face_hash(&self, face: Face) -> u64 {
        let mut hasher = DefaultHasher::new();
        for layer in &self.layers {
            layer.hash_face(face, &mut hasher);
        }
        hasher.finish()
    }
    pub fn face_hash_at(&self, layer: usize, face: Face) -> u64 {
        let mut hasher = DefaultHasher::new();
        layer.hash(&mut hasher);
        self.layers[layer].hash_face(face, &mut hasher);
        hasher.finish()
    }
    pub fn layer(&self, layer: LayerNr) -> &Layer {
        &self.layers[layer]
    }
    /// Compute the length of an edge in the input space
    pub fn edge_length_at(&self, layer: LayerNr, edge: Edge) -> f32 {
        let (a,b) = self.endpoints(edge);
        let va = self.vertex_activations_at(layer, a);
        let vb = self.vertex_activations_at(layer, b);
        norm(&(&va - &vb))
    }
    pub fn edge_length(&self, edge: Edge) -> f32 {
        self.edge_length_at(0, edge)
    }
    pub fn face_area(&self, face: Face) -> f32 {
        fn cross(v0: ArrayView1<f32>, v1: ArrayView1<f32>, v2: ArrayView1<f32>) -> f32 {
            debug_assert_eq!(v0.len(), 2);
            debug_assert_eq!(v1.len(), 2);
            debug_assert_eq!(v2.len(), 2);
            let a0 = v1[0] - v0[0];
            let a1 = v1[1] - v0[1];
            let b0 = v2[0] - v0[0];
            let b1 = v2[1] - v0[1];
            return a0 * b1 - a1 * b0;
        }
        let vertex_data = &self.layers[0].vertex_data;
        let mut vertices = self.vertices_on_face(face);
        let v0 = vertices.next().unwrap();
        let mut v1 = vertices.next().unwrap();
        let mut sum = 0.0;
        for v2 in vertices {
            sum += cross(vertex_data.row(v0.index()), vertex_data.row(v1.index()), vertex_data.row(v2.index()));
            v1 = v2;
        }
        sum.abs() * 0.5
    }

    // Iterators

    pub fn vertices(&self) -> impl ExactSizeIterator<Item=Vertex> {
        self.partition.vertices()
    }
    pub fn faces(&self) -> impl ExactSizeIterator<Item=Face> {
        self.partition.faces()
    }
    pub fn halfedges(&self) -> impl ExactSizeIterator<Item=Halfedge> {
        self.partition.halfedges()
    }
    pub fn edges(&self) -> impl ExactSizeIterator<Item=Edge> {
        self.partition.edges()
    }
    pub fn halfedges_on_face(&self, face: Face) -> HalfedgesOnFace<'_> {
        self.partition.halfedges_on_face(face)
    }
    pub fn vertices_on_face(&self, face: Face) -> impl '_ + Iterator<Item=Vertex> {
        self.partition.vertices_on_face(face)
    }
    pub fn vertex_data_for_face(&self, face: Face, layer: usize) -> Array2<f32> {
        let indices = Vec::from_iter(self.vertices_on_face(face).map(usize::from));
        self.layers[layer].vertex_data.select(Axis(0), &indices)
    }
    pub fn halfedges_leaving_vertex(&self, vertex: Vertex) -> HalfedgesLeavingVertex<'_> {
        self.partition.halfedges_leaving_vertex(vertex)
    }
    // The label of a vertex is the smallest edge_label of an incident edge
    pub fn vertex_label(&self, vertex: Vertex) -> EdgeLabel {
        self.halfedges_leaving_vertex(vertex)
            .map(|he| self.edge_label(he.edge()))
            .min()
            .copied()
            .unwrap_or(EdgeLabel::default())
    }

    // Invariants

    /// Check the class invariants
    fn check_invariants(&self) {
        // vectors have the right size
        self.partition.check_invariants();
        assert!(self.layers.len() > 0);
        assert_eq!(self.layers.len(), self.num_layers());
        for layer in &self.layers {
            assert_eq!(layer.vertex_data.len_of(Axis(0)), self.num_vertices());
            assert_eq!(layer.face_data.len_of(Axis(0)), self.num_faces());
            assert_eq!(layer.face_data.len_of(Axis(1)), 3);
            assert_eq!(layer.face_data.len_of(Axis(2)), layer.vertex_data.len_of(Axis(1)));
            match &layer.layer_type {
                LayerType::Input => (),
                LayerType::Linear => (),
                LayerType::ReLU { face_mask } => {
                    assert_eq!(face_mask.len_of(Axis(0)), self.num_faces());
                    assert_eq!(face_mask.len_of(Axis(1)), layer.vertex_data.len_of(Axis(1)));
                },
                LayerType::MaxPool { face_argmax } => {
                    assert_eq!(face_argmax.len_of(Axis(0)), self.num_faces());
                },
            }
        }
        assert_eq!(self.edge_label.len(), self.num_edges());
        self.check_face_outputs();
    }

    fn check_face_outputs(&self) {
        // check that the face outputs are correct
        // this is only true for continuous layers, otherwise there are dicontinuities at edges and vertices
        for face in self.faces() {
            let indices = Vec::from_iter(self.vertices_on_face(face).map(usize::from));
            let vertex_coords_in = self.layers[0].vertex_data.select(Axis(0), &indices);
            let vertex_coords_in = concatenate![Axis(1), vertex_coords_in, Array2::ones((indices.len(), 1))];
            for layer in &self.layers {
                if layer.continuous {
                    let vertex_coords_out = layer.vertex_data.select(Axis(0), &indices);
                    let face_data = layer.face_data.index_axis(Axis(0), face.index());
                    let computed_coords_out = vertex_coords_in.dot(&face_data);
                    assert_abs_diff_eq!(&computed_coords_out, &vertex_coords_out, epsilon=1e-4);
                }
            }
        }
    }

    // Mutations

    /// Split an edge at position t (between 0 and 1)
    /// Return the new vertex
    /// If t == 0 || t == 1, then return on of the existing edge endpoints.
    /// The boolean indicates if a new vertex was inserted.
    pub fn split_edge(&mut self, edge: Edge, t: f32) -> (bool, Vertex) {
        if t == 0.0 {
            (false, self.start(edge.halfedge(0)))
        } else if t == 1.0 {
            (false, self.start(edge.halfedge(1)))
        } else {
            let (a, b) = self.endpoints(edge);
            let (new_v, _new_edge) = self.partition.split_edge(edge);
            // copy edge data
            self.edge_label.push(self.edge_label[edge.index()]);
            // compute new vertex data
            for layer in self.layers.iter_mut() {
                layer.append_vertex_lerp(a, b, t);
            }
            (true, new_v)
        }
    }

    /// Split a face by adding an edge between the start vertices of a and b.
    /// A new face is inserted for the loop start(a)...start(b).
    pub fn split_face(&mut self, a: Halfedge, b: Halfedge, edge_label: EdgeLabel) -> (Edge, Face) {
        let face = self.partition.face(a).unwrap();
        let (new_edge, new_face) = self.partition.split_face(a, b);
        // add edge data
        self.edge_label.push(edge_label);
        // add face data
        for layer in self.layers.iter_mut() {
            layer.append_face(face);
        }
        (new_edge, new_face)
    }

    /// Merge two faces by removing the edge between them.
    /// This function only marks the edge as invalid, without actually removing it, because that would mess up the order of edges.
    ///
    /// Returns: removed face. After this call the removed face is replaced by the last face
    pub fn merge_faces(&mut self, edge: Edge) -> Face {
        let removed_face = self.partition.merge_faces(edge);
        self.swap_remove_face_data(removed_face);
        removed_face
    }

    /// Remove all invalid edges. This renumbers existing edges.
    fn remove_invalid_edges(&mut self) {
        self.partition.remove_invalid_edges(|edge| {
            self.edge_label.swap_remove(edge.index());
        });
    }

    // Remove data for an edge that is not being used
    fn swap_remove_edge_data(&mut self, edge: Edge) {
        self.edge_label.swap_remove(edge.index());
    }

    // Remove data for a face that is not being used
    fn swap_remove_face_data(&mut self, face: Face) {
        for layer in self.layers.iter_mut() {
            layer.swap_remove_face(face);
        }
    }

    /// Split all faces in the partition by treating zeros of a function as edges.
    /// Requires that there is at most 1 zero crossing per edge.
    /// The function split(a,b) should return Some(t) if f(lerp(a,b,t)) == 0
    fn split_all<E: EdgeSplitter>(&mut self, splitter: &mut E, edge_label: EdgeLabel) {
        // Split edges such that all zeros of the function go through vertices.
        let mut zero_vertices = vec![false; self.num_vertices()];
        // Note: splitting can add more edges, this only iterates up to number of edges we had before we started
        for edge in self.edges() {
            let (a,b) = self.endpoints(edge);
            if zero_vertices[a.index()] || zero_vertices[b.index()] {
                // already have a zero crossing on this edge
            } else if let Some(t) = splitter.split_point(self, a, b) {
                // split at t
                let (was_split, new_vertex) = self.split_edge(edge, t);
                if was_split {
                    zero_vertices.push(true);
                    splitter.apply_split(self, new_vertex, t);
                } else {
                    zero_vertices[new_vertex.index()] = true;
                }
                debug_assert_eq!(zero_vertices.len(), self.num_vertices());
            }
        }
        // Split faces that have non-consecutive zero vertices
        for face in self.faces() {
            fn find_non_consecutive<T>(mut iter: impl Iterator<Item=T>, prop: impl Copy + Fn(&T)-> bool) -> Option<(T, T)> {
                // find two elements where property is true, separated by elements where it is false, when the iterator is considered as circular
                //   0011100111   or   1110001100
                //     ^    ^          ^     ^
                let x0 = iter.next()?;
                let (x1,first) = if prop(&x0) { (x0, true) } else { (iter.find(prop)?, false) };
                let _  = iter.find(|x| !prop(x))?;
                let x2 = iter.find(prop)?;
                if first {
                    let _  = iter.find(|x| !prop(x))?;
                }
                Some((x1, x2))
            }
            if let Some((he1,he2)) = find_non_consecutive(self.halfedges_on_face(face), |he| zero_vertices[self.start(*he).index()]) {
                self.split_face(he1, he2, edge_label);
            }
        }
    }

    /// Split all faces in the partition by treating zeros of the given layer as edges.
    fn split_by_zero_at(&mut self, layer: usize) {
        // Split faces on activation = 0
        let acts: &ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>> = self.activations(layer);
        for dim in 0..acts.len_of(Axis(1)) {
            let edge_label = EdgeLabel { layer: self.num_layers(), dim };
            struct ZeroSplitter {
                layer: usize,
                dim: usize,
                // Note: could track which vertices are non-negative to make relu mask
            }
            impl EdgeSplitter for ZeroSplitter {
                fn split_point(&self, rc: &Regioncam, a: Vertex, b: Vertex) -> Option<f32> {
                    let acts = &rc.layers[self.layer].vertex_data;
                    find_zero(acts[(a.index(), self.dim)], acts[(b.index(), self.dim)])
                }
                fn apply_split(&mut self, rc: &mut Regioncam, new_vertex: Vertex, _point: f32) {
                    // make sure that new vertices are actually 0 in dimension dim
                    // this may not be true exactly because of rounding errors.
                    let acts = &mut rc.layers[self.layer].vertex_data;
                    acts[(new_vertex.index(), self.dim)] = 0.0;
                }
            }
            let mut splitter = ZeroSplitter { layer, dim };
            self.split_all(&mut splitter, edge_label);
        }
    }

    /// Compute mask of non-negative dimensions for each face,
    /// The mask is true iff all vertices of the face are non-negative (above -eps)
    fn non_negative_face_mask(&self, layer: usize) -> Array2<bool> {
        let acts = self.activations(layer);
        let mut mask = Array2::from_elem((self.num_faces(), acts.ncols()), true);
        let eps = 1e-6;
        for he in self.halfedges() {
            if let Some(face) = self.partition.face(he) {
                mask.row_mut(face.index()).zip_mut_with(&acts.row(self.start(he).index()),
                    |m, x| *m &= *x >= -eps
                );
            }
        }
        /*for face in self.faces() {
            // for every dim: how far is any vertex in this face below the maximum?
            delta_from_max.fill(0.0);
            for v in self.vertices_on_face(face) {
                let max = max[v.index()];
                delta_from_max.zip_mut_with(&acts.row(v.index()),
                    |delta, x| *delta = f32::min(*delta, *x - max)
                );
            }
            // The argmax dimension is the one with the smallest deviation from the maximum at any vertex
            // Ideally this would be 0, but numberical errors can change that.
            argmax(delta_from_max.view())
        }*/
        mask
    }

    /// Split all faces in the partition at points where argmax of dims changes
    /// Returns: (vertex_max, face_argmax)
    fn split_by_argmax_at(&mut self, layer: usize) -> (Array1<f32>, Array1<usize>) {
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
        let initial_num_edges = self.num_edges();
        let label_layer = self.num_layers();
        let vertex_max = {
            struct MaxSplitter {
                //argmax_so_far: usize,
                layer: usize,
                dim: usize,
                max_so_far: Vec<f32>,
            }
            impl EdgeSplitter for MaxSplitter {
                fn split_point(&self, rc: &Regioncam, a: Vertex, b: Vertex) -> Option<f32> {
                    // split by zeros of (acts[:,dim] - max_so_far)
                    let acts = &rc.layers[self.layer].vertex_data;
                    find_zero(acts[(a.index(), self.dim)] - self.max_so_far[a.index()], acts[(b.index(), self.dim)] - self.max_so_far[b.index()])
                }
                fn apply_split(&mut self, rc: &mut Regioncam, new_vertex: Vertex, _t: f32) {
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
                let edge_label = EdgeLabel { layer: label_layer, dim };
                splitter.dim = dim;
                self.split_all(&mut splitter, edge_label);
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

        // We might have made unnecessary splits. We can undo these by merging faces with the same argmax
        let face_argmax = {
            let mut face_argmax = self.argmax_face_index(layer, &vertex_max);
            // Note: we only need to check new edges
            for edge in self.edges().skip(initial_num_edges) {
                if self.edge_label(edge).layer == label_layer {
                    // this is a new edge, consider it for merger
                    // Note: invalid edges will have edge_faces() = (None,None)
                    if let (Some(face1), Some(face2)) = self.partition.edge_faces(edge) {
                        if face1 != face2 && face_argmax[face1.index()] == face_argmax[face2.index()] {
                            let removed_face = self.merge_faces(edge);
                            face_argmax.swap_remove(removed_face.index());
                        }
                    }
                }
            }
            // TODO: we could also unsplit edges.
            self.remove_invalid_edges();
            // TODO: we could also remove isolated vertices
            face_argmax
        };

        // sanity check
        if cfg!(debug_assertions) {
            let acts = self.activations(layer);
            for (row, max) in acts.axis_iter(Axis(0)).zip(vertex_max.iter()) {
                if row.iter().all(|x| x < max) {
                    println!("Bad maximum: {row} < {max}");
                }
            }
        }
        // return max
        (Array1::from_vec(vertex_max), Array1::from_vec(face_argmax))

        // alternative algorithm:
        //  * for every vertex+dim: track if it is equal to max
        //  * for every edge (a,b): if is_max mask(a) ∩ is_max_mask(b) = ∅: find all changes
        //    *
        //  * split faces
        //    * maybe also split the newly introduced edge needed, and repeat
    }

    fn argmax_face_index(&self, layer: usize, max: &[f32]) -> Vec<usize> {
        let acts = self.activations(layer);
        let mut delta_from_max = Array1::zeros(acts.ncols());
        self.faces().map(|face| {
            // for every dim: how far is any vertex in this face below the maximum?
            delta_from_max.fill(0.0);
            for v in self.vertices_on_face(face) {
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

// Neural network operations

impl NNBuilder for Regioncam {
    type LayerNr = LayerNr;

    fn last_layer(&self) -> LayerNr {
        self.num_layers() - 1
    }

    /// Update partition by adding a ReLU layer, that takes as input the output of the given layer.
    /// This means:
    ///  * split all faces at activation[l,_,d]=0 for any dim d
    ///  * set activation[l+1] = max(0,activation[l])
    fn relu_at(&mut self, layer: LayerNr) -> LayerNr {
        // Split faces
        self.split_by_zero_at(layer);
        // Compute relu output for vertices
        let input_layer = &self.layers[layer];
        let vertex_data = relu(&input_layer.vertex_data.view());
        // Apply mask to compute per-face activation
        let face_mask = self.non_negative_face_mask(layer);
        let mut face_data = input_layer.face_data.clone();
        face_data.zip_mut_with(&face_mask.slice(s![.., NewAxis, ..]), |x, m| if !m { *x = 0.0; });
        // Add layer
        let continuous = input_layer.continuous;
        let layer_type = LayerType::ReLU{ face_mask };
        self.layers.push(Layer{ vertex_data, face_data, continuous, layer_type });
        self.last_layer()
    }

    fn leaky_relu_at(&mut self, layer: LayerNr, negative_slope: f32) -> LayerNr {
        self.split_by_zero_at(layer);
        // Compute leaky relu output for vertices
        let input_layer = &self.layers[layer];
        let vertex_data = leaky_relu(&input_layer.vertex_data.view(), negative_slope);
        // Apply mask to compute per-face activation
        let face_mask = self.non_negative_face_mask(layer);
        let mut face_data = input_layer.face_data.clone();
        face_data.zip_mut_with(&face_mask.slice(s![.., NewAxis, ..]), |x, m| if !m { *x *= negative_slope; });
        // Add layer
        let continuous = input_layer.continuous;
        let layer_type = LayerType::ReLU{ face_mask };
        self.layers.push(Layer{ vertex_data, face_data, continuous, layer_type });
        self.last_layer()
    }

    fn max_pool_at(&mut self, layer: LayerNr) -> LayerNr {
        let (max, face_argmax) = self.split_by_argmax_at(layer);
        // New face data takes the given dim
        let input_layer = &self.layers[layer];
        let face_data = select_in_rows(Axis(2), &input_layer.face_data.view(), &face_argmax);
        let face_data = face_data.insert_axis(Axis(2));
        // New vertex data is just the max value
        let vertex_data = max.insert_axis(Axis(1));
        // Add layer
        let continuous = input_layer.continuous;
        let layer_type = LayerType::MaxPool{ face_argmax };
        self.layers.push(Layer{ vertex_data, face_data, continuous, layer_type });
        self.last_layer()
    }

    fn argmax_pool_at(&mut self, layer: LayerNr) -> LayerNr {
        let (_max, face_argmax) = self.split_by_argmax_at(layer);
        // New face data takes the given dim
        let face_data = concatenate![Axis(1),
            Array::zeros((self.num_faces(), 2)),
            face_argmax.view().insert_axis(Axis(1)).mapv(|x| x as f32)
        ];
        let face_data = face_data.insert_axis(Axis(2));
        // New vertex data is argmax of inputs
        let acts = self.activations(layer);
        let vertex_argmax = Array::from_iter(acts.rows().into_iter().map(|row| argmax(row) as f32));
        let vertex_data = vertex_argmax.insert_axis(Axis(1));
        // Add layer
        let continuous = false;
        let layer_type = LayerType::MaxPool{ face_argmax };
        self.layers.push(Layer{ vertex_data, face_data, continuous, layer_type });
        self.last_layer()
    }

    fn sign_at(&mut self, layer: LayerNr) -> LayerNr {
        // Split faces
        self.split_by_zero_at(layer);
        // Compute sign for vertices
        let input_layer = &self.layers[layer];
        let vertex_data = input_layer.vertex_data.mapv(|x| if x >= 0.0 { 1.0 } else { 0.0 });
        // Apply mask to compute per-face activation
        let face_mask = self.non_negative_face_mask(layer);
        let face_data = concatenate![Axis(1),
            Array::zeros((face_mask.len_of(Axis(0)), 2, face_mask.len_of(Axis(1)))),
            face_mask.view().insert_axis(Axis(1)).mapv(|x| if x { 1.0 } else {0.0})
        ];
        // Add layer
        let continuous = false;
        let layer_type = LayerType::ReLU{ face_mask };
        self.layers.push(Layer{ vertex_data, face_data, continuous, layer_type });
        self.last_layer()
    }
    
    /// Append a classification layer: with 1 output a sign based classifier, with >1 outputs an argmax
    fn decision_boundary_at(&mut self, layer: LayerNr) -> LayerNr {
        if self.layers[layer].vertex_data.ncols() == 1 {
            self.sign_at(layer)
        } else {
            self.argmax_pool_at(layer)
        }
    }

    /// Append a layer that computes output using the given function.
    /// The function should be linear, without a bias term.
    /// Then `add_bias` should add the bias term
    fn generalized_linear_at(&mut self, layer: LayerNr, fun: impl Fn(ArrayView2<f32>) -> Array2<f32>, add_bias: impl Fn(ArrayViewMut2<f32>)) -> LayerNr {
        // Compute vertex data (x' = x*w + b)
        let input_layer = &self.layers[layer];
        let mut vertex_data = fun(input_layer.vertex_data.view());
        add_bias(vertex_data.view_mut());
        // Compute new face_data:
        // If old was x = p*u + v,
        //  new is x' = x*w + b = p*(u*w) + (v*w + b)
        // Where v is face_data[..,2,..]
        let input_face_data = &input_layer.face_data;
        let (a,b,c) = input_face_data.dim();
        let input_face_data = input_face_data.to_shape((a*b, c)).unwrap();
        let face_data = fun(input_face_data.view());
        let (_,d) = face_data.dim();
        let mut face_data = face_data.into_shape_with_order((a,b,d)).unwrap();
        add_bias(face_data.index_axis_mut(Axis(1), 2));
        // Add layer
        let continuous = input_layer.continuous;
        let layer_type = LayerType::Linear;
        self.layers.push(Layer{ vertex_data, face_data, continuous, layer_type });
        self.last_layer()
    }

    fn sum(&mut self, layer1: LayerNr, layer2: LayerNr) -> LayerNr {
        let layer1 = &self.layers[layer1];
        let layer2 = &self.layers[layer2];
        self.layers.push(Layer {
            vertex_data: &layer1.vertex_data + &layer2.vertex_data,
            face_data: &layer1.face_data + &layer2.face_data,
            continuous: layer1.continuous && layer2.continuous,
            layer_type: LayerType::Linear,
        });
        self.last_layer()
    }
}

/// Trait for deciding to split an edge
trait EdgeSplitter {
    fn split_point(&self, rc: &Regioncam, a: Vertex, b: Vertex) -> Option<f32>;
    fn apply_split(&mut self, rc: &mut Regioncam, new_vertex: Vertex, t: f32);
}


#[cfg(test)]
mod test {
    use std::f32::consts::TAU;
    use std::fs::File;

    use ndarray::{stack, Array1};
    use rand::Rng;
    use rand::{rngs::SmallRng, SeedableRng};
    use ndarray_npz::NpzReader;

    use super::*;

    #[test]
    fn constructor_invariants() {
        Regioncam::square(2.0).check_invariants();
    }

    #[test]
    fn iterators() {
        let p = Regioncam::square(2.0);
        assert_eq!(p.vertices().len(), p.num_vertices());
        assert_eq!(p.halfedges().len(), p.num_halfedges());
        assert_eq!(p.edges().len(), p.num_edges());
        assert_eq!(p.num_halfedges(), p.num_edges() * 2);
    }

    #[test]
    fn split_edge() {
        fn check_split_edge(mut rc: Regioncam, edge: Edge, t: f32) {
            let faces = rc.partition.edge_faces(edge);
            let (was_split, v) = rc.split_edge(edge, t);
            //println!("split_edge {edge:?} at {t}\n{p:?}");
            rc.check_invariants();
            assert_eq!(was_split, t != 0.0 && t != 1.0);
            assert_eq!(rc.halfedges_leaving_vertex(v).count(), 2);
            assert_eq!(rc.halfedges_leaving_vertex(v).map(|he| rc.face(he)).collect::<Vec<_>>(), [faces.0, faces.1]);
        }
        for rc in [Regioncam::square(2.0)] {
            for edge in rc.edges() {
                for t in [0.0, 0.5, 1.0] {
                    check_split_edge(rc.clone(), edge, t);
                }
            }
        }
    }

    #[test]
    fn split_face() {
        for mut rc in [Regioncam::square(2.0)] {
            let a = rc.partition.halfedge_on_face(Face::from(0));
            let b = rc.partition.next(rc.partition.next(a));
            println!("Before split {a:?}-{b:?}\n{rc:?}");
            rc.split_face(a, b, EdgeLabel::default());
            println!("split_face {a:?}-{b:?}\n{rc:?}");
            rc.check_invariants();
        }
    }

    #[test]
    #[should_panic]
    fn degenerate_split_face() {
        for mut rc in [Regioncam::square(2.0)] {
            let a = rc.partition.halfedge_on_face(Face::from(0));
            let b = rc.next(a);
            rc.split_face(a, b, EdgeLabel::default());
        }
    }

    #[test]
    fn merge_face() {
        let mut rng = SmallRng::seed_from_u64(42);
        for mut rc in test_cases(true, true) {
            let mut failures = 0;
            while failures < 10 {
                // merge random edges until there are only boundary edges
                let edge = Edge::from(rng.gen_range(0..rc.num_edges()));
                if rc.partition.is_boundary(edge) || rc.partition.is_invalid(edge) || rc.edge_faces(edge).0 == rc.edge_faces(edge).1 {
                    failures += 1;
                } else {
                    rc.merge_faces(edge);
                    println!("{:?}", rc.partition);
                    rc.check_invariants();
                    rc.remove_invalid_edges();
                    println!("{:?}", rc.partition);
                    rc.check_invariants();
                }
            }
            println!("");
        }
    }

    // test a merge_faces that creates a degenerate edge
    #[test]
    fn merge_face_simple() {
        let mut rc = Regioncam::square(1.0);
        let a = rc.halfedge_on_face(Face::from(0));
        let b = rc.next(rc.next(a));
        let (new_edge, _) = rc.split_face(a, b, EdgeLabel::default());
        rc.split_edge(new_edge, 0.5);
        println!("Before: {rc:?}");
        rc.merge_faces(new_edge);
        println!("After: {rc:?}");
        rc.check_invariants();
    }

    fn test_cases(edge_splits: bool, face_splits: bool) -> Vec<Regioncam> {
        let mut out = vec![];
        out.push(Regioncam::square(2.0));
        if face_splits {
            let mut rc = Regioncam::square(2.0);
            let a = rc.halfedge_on_face(Face::from(0));
            let b = rc.next(rc.next(a));
            rc.split_face(a, b, EdgeLabel::default());
            out.push(rc);
        }
        let mut rng = SmallRng::seed_from_u64(42);
        for steps in [0,2,4,8,16,32,64,128] {
            out.push(random_partition(steps, edge_splits, face_splits, &mut rng));
        }
        if edge_splits && face_splits {
            out.push(test_case_triangles(2));
            out.push(test_case_triangles(5));
        }
        out
    }

    // Random repeated edge and face splits
    fn random_partition<R: Rng + ?Sized>(steps: usize, edge_splits: bool, face_splits: bool, rng: &mut R) -> Regioncam {
        // start with a random polygon
        let mut rc =
            if rng.gen_bool(0.5) {
                Regioncam::square(rng.gen_range(0.5..5.0))
            } else {
                let n = rng.gen_range(3..7);
                let mut points = Array2::zeros((n,2));
                for i in 0..n {
                    let r = rng.gen_range(0.8..1.2);
                    let a = rng.gen_range(-0.1..0.1) + (i as f32 / n as f32) * TAU;
                    points[(i,0)] = f32::cos(a) * r;
                    points[(i,1)] = f32::sin(a) * r;
                }
                Regioncam::from_polygon(points)
            };
        // repeatedly split edges and faces
        for _ in 0..steps {
            // pick an edge at random to split
            if edge_splits && rng.gen_bool(0.6) {
                let edge = Edge::from(rng.gen_range(0..rc.num_edges()));
                rc.split_edge(edge, rng.gen_range(0.1..0.9));
            } else if face_splits {
                let face = Face::from(rng.gen_range(0..rc.num_faces()));
                // find two halfedges on this face
                let halfedges = Vec::from_iter(rc.halfedges_on_face(face));
                if halfedges.len() >= 4 {
                    let a = rng.gen_range(0..halfedges.len());
                    let b = rng.gen_range(2..halfedges.len() - 1);
                    rc.split_face(halfedges[a], halfedges[(a+b) % halfedges.len()], EdgeLabel::default());
                }
            }
        }
        rc.check_invariants();
        rc
    }

    #[test]
    fn random_partition_test() {
        let mut rng = SmallRng::seed_from_u64(44);
        let p = random_partition(200, true, true, &mut rng);
        println!("{} vertices, {} edges, {} faces", p.num_vertices(), p.num_edges(), p.num_faces());
    }

    #[test]
    fn simple_relu() {
        let mut rc = Regioncam::square(2.0);
        rc.relu();
        println!("Relu\n{rc:?}");
        rc.check_invariants();
        // applying relu again should keep everything the same (except adding a layer)
        let mut p2 = rc.clone();
        p2.relu();
        p2.check_invariants();
        assert_eq!(rc.num_vertices(), p2.num_vertices());
        assert_eq!(rc.num_faces(), p2.num_faces());
        assert_eq!(rc.num_halfedges(), p2.num_halfedges());
    }

    fn test_case_triangles(n: usize) -> Regioncam {
        let dirs = array![[1.0,0.0], [0.0,1.0], [1.0,1.0], [0.5,-0.5]];
        let m = dirs.nrows();
        let pos = Array1::linspace(-1.8, 1.8, n);
        let weight =
            stack(Axis(2), &vec![dirs.t(); n]).unwrap();
        let weight = weight
            .as_standard_layout()
            .into_shape_with_order((2, m * n)).unwrap();
        let bias =
            stack(Axis(0), &vec![pos.view(); m]).unwrap()
            .into_shape_with_order(m * n).unwrap();
        let mut p = Regioncam::square(2.0);
        p.linear(&weight.view(), &bias.view());
        p
    }

    #[test]
    fn triangles() {
        // partition
        let mut p = test_case_triangles(5);
        p.check_invariants();
        p.relu();
        p.check_invariants();
        println!("{} faces", p.num_faces());
    }

    #[test]
    fn random_nn() {
        let mut rng = SmallRng::seed_from_u64(42);
        let dim_in = 2;
        //let dim_out = 1000;
        let dim_out = 100;
        let layer = crate::nn::Linear::new_uniform(dim_in, dim_out, &mut rng);
        // partition
        let mut rc = Regioncam::square(1.0);
        rc.add(&layer);
        rc.check_invariants();
        rc.relu();
        rc.check_invariants();
        println!("{} faces", rc.num_faces());
    }

    #[test]
    fn max_pool() {
        let mut rng = SmallRng::seed_from_u64(42);
        let dim_in = 2;
        let dim_hidden = 10;
        let dim_out = 10;
        // two layer network
        let mut rc = Regioncam::square(1.0);
        rc.add(&crate::nn::Linear::new_uniform(dim_in, dim_hidden, &mut rng));
        rc.relu();
        rc.check_invariants();
        println!("{} faces after relu", rc.num_faces());
        rc.add(&crate::nn::Linear::new_uniform(dim_hidden, dim_out, &mut rng));
        rc.check_invariants();
        rc.max_pool();
        rc.check_invariants();
        println!("{} faces", rc.num_faces());
    }

    #[test]
    fn argmax_pool() {
        let mut rng = SmallRng::seed_from_u64(42);
        let dim_in = 2;
        let dim_hidden = 25;
        let dim_out = 10;
        // two layer network
        let mut rc = Regioncam::square(1.0);
        rc.add(&crate::nn::Linear::new_uniform(dim_in, dim_hidden, &mut rng));
        rc.relu();
        rc.check_invariants();
        println!("{} faces after relu", rc.num_faces());
        for _ in 0..0 {
            rc.add(&crate::nn::Linear::new_uniform(dim_hidden, dim_hidden, &mut rng));
            rc.relu();
            rc.check_invariants();
            println!("{} faces after relu", rc.num_faces());
        }
        rc.add(&crate::nn::Linear::new_uniform(dim_hidden, dim_out, &mut rng));
        rc.check_invariants();
        rc.argmax_pool_at(rc.last_layer());
        rc.check_invariants();
        // Note: face values do not match vertex value after argmax pool
        println!("{} faces", rc.num_faces());
    }

    #[test]
    fn max_pool_triangles() {
        let mut rc = test_case_triangles(5);
        rc.check_invariants();
        rc.max_pool();
        rc.check_invariants();
        println!("{} faces", rc.num_faces());
    }

    // Test case that gave "Face has inconsistent argmax" error
    #[test]
    fn error_case_1() {
        let mut rc = Regioncam::square(2.0);
        rc.linear(&array![
                [-0.43146494, -1.0630932 , -0.75895   , -1.212125  ,  0.52904165],
                [-1.1837014 ,  0.8460558 , -2.002247  ,  0.7702433 ,  1.5294473 ]].view(),
            &array![ 1.4554669 , -0.1446282 ,  0.63949215,  0.01911678, -0.13796857].view()
        );
        rc.relu();
        rc.linear(&array![
                [-0.31510666,  1.6690438 , -1.9769672 ],
                [ 1.4305679 , -1.4525309 , -1.6842408 ],
                [-1.501365  ,  1.7357037 , -1.4884815 ],
                [ 1.9054497 , -1.1933256 , -1.7565324 ],
                [ 0.37124428, -1.8019046 ,  1.9643012 ]].view(),
                &array![ 0.6519267, -0.5254144, -1.1940845].view()
        );
        println!("{} faces", rc.num_faces());
        //println!("{p:?}");
        rc.decision_boundary();
        rc.check_invariants();
    }

    // Test case that gave index out of bounds error
    #[test]
    #[ignore]
    fn error_case_2() {
        let mut data = NpzReader::new(File::open("test/error_case_2.npz").unwrap()).unwrap();
        let points: Array2<f32> = data.by_name("points.npy").unwrap();
        let linear_0 = crate::nn::Linear::new(
            data.by_name("weight_0.npy").unwrap(),
            data.by_name("bias_0.npy").unwrap()
        );
        let linear_2 = crate::nn::Linear::new(
            data.by_name("weight_2.npy").unwrap(),
            data.by_name("bias_2.npy").unwrap()
        );
        let mut rc = Regioncam::from_plane(&Plane::through_points(&points.view()), 1.5);
        rc.add(&linear_0);
        rc.add(&crate::nn::Module::ReLU);
        rc.add(&linear_2);
        rc.check_invariants();
        rc.decision_boundary();
        rc.check_invariants();
    }
}

