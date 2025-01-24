// This module defines a halfedge data structure for partitioning 2d space.

use std::cmp::Ordering;
use std::fmt::Debug;
use ndarray::{array, concatenate, s, stack, Array, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, ArrayViewMut2, Axis, NewAxis};
use approx::assert_abs_diff_eq;

use crate::util::*;

// Index types

type Index = usize;

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Vertex(Index);

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Halfedge(Index);

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Edge(Index);

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Face(Index);

#[derive(Copy, Clone, PartialEq, Eq)]
struct OptFace(Index);

impl From<Vertex> for usize {
    fn from(vertex: Vertex) -> Self {
        vertex.0
    }
}
impl Debug for Vertex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

impl Halfedge {
    pub fn edge(self) -> Edge {
        Edge(self.0 / 2)
    }
    pub fn side(self) -> usize {
        self.0 & 1
    }
    pub fn new(edge: Edge, side: usize) -> Halfedge {
        Halfedge(edge.0 * 2 + (side & 1))
    }
    pub fn opposite(self) -> Self {
        Halfedge(self.0 ^ 1)
    }
}
impl From<Halfedge> for usize {
    fn from(he: Halfedge) -> Self {
        he.0
    }
}
impl Debug for Halfedge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "h{}", self.0)
    }
}

impl Edge {
    pub fn halfedge(self, side: usize) -> Halfedge {
        Halfedge::new(self, side)
    }
    pub fn halfedges(self) -> (Halfedge, Halfedge) {
        (self.halfedge(0), self.halfedge(1))
    }
}
impl From<Halfedge> for Edge {
    fn from(he: Halfedge) -> Edge {
        he.edge()
    }
}
impl From<Edge> for usize {
    fn from(edge: Edge) -> Self {
        edge.0
    }
}
impl Debug for Edge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "e{}", self.0)
    }
}

impl From<Face> for usize {
    fn from(face: Face) -> Self {
        face.0
    }
}
impl Debug for Face {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "f{}", self.0)
    }
}

impl OptFace {
    const NONE: OptFace = OptFace(usize::MAX);
    pub fn some(face: Face) -> Self {
        OptFace(face.0)
    }
    pub fn view(self) -> Option<Face> {
        if self == OptFace::NONE {
            None
        } else {
            Some(Face(self.0))
        }
    }
}
impl From<Face> for OptFace {
    fn from(face: Face) -> Self {
        OptFace::some(face)
    }
}
impl From<OptFace> for Option<Face> {
    fn from(optface: OptFace) -> Self {
        optface.view()
    }
}
impl Debug for OptFace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.view() {
            None => write!(f, "None"),
            Some(face) => write!(f, "{:?}", face),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum LayerFaceMask {
    NoMask,
    ReLU(Array2<bool>),
    LeakyReLU(Array2<bool>, f32),
    //Max(Array2<usize>),
}
enum LayerType {
    Input,
    ReLU,
    LeakyReLU(f32),
    Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EdgeLabel {
    /// Layer on which this edge was introduced
    pub layer: usize,
    /// Dimension of that layer which caused this edge to be created
    pub dim: usize,
}

/// A partition of 2d space.
/// Stored as a half-edge data structure
#[derive(Debug, Clone, PartialEq)]
pub struct Partition {
    /// Data associated with vertices.
    /// This is the output of layer l in the neural network.
    vertex_data: Vec<Array2<f32>>, // layer -> vertex -> f32[D_l]

    face_data:  Vec<Array3<f32>>,  // layer -> face -> f32[D_in, D_F]
    face_he:    Vec<Halfedge>,     // face -> halfedge
    // /// For each (ReLU) layer, for each face: a mask indicating which dimensions are non-negative
    //face_mask:  Vec<LayerMask>,    // layer -> face -> mask of non-negative dimensions

    he_vertex:  Vec<Vertex>,       // halfedge -> vertex
    he_face:    Vec<OptFace>,      // halfedge -> face or NONE
    he_next:    Vec<Halfedge>,     // halfedge -> halfedge
    he_prev:    Vec<Halfedge>,     // halfedge -> halfedge
    
    edge_label: Vec<EdgeLabel>,    // edge -> label
}

// Append to the array the element
//   lerp(arr[i], arr[j], t) = arr[i] + t * (arr[j] - arr[i])
// Return the index of the new element
fn append_lerp(arr: &mut Array2<f32>, i: usize, j: usize, t: f32) -> usize {
    let k = arr.len_of(Axis(0));
    let ai = arr.row(i);
    let aj = arr.row(j);
    let ak = &ai + t * (&aj - &ai);
    arr.push_row(ak.view()).unwrap();
    k
}

/// Find the point t between 0.0 and 1.0 at which a+t*(b-a) == 0
fn find_zero(a: f32, b: f32) -> Option<f32> {
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
impl Partition {

    // Constructors

    /// Construct a Partition with a single region, containing the given convex polygon
    pub fn from_polygon(vertex_data: Array2<f32>) -> Self {
        let n = vertex_data.nrows();
        assert_eq!(vertex_data.ncols(), 2, "Expected two dimensional points");
        Self {
            vertex_data: vec![vertex_data],
            face_data: vec![array![[
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]]],
            //face_mask: vec![LayerMask::NoMask],
            face_he:   vec![Halfedge(0)],
            he_face:   Vec::from_iter((0..n).flat_map(|_| [OptFace(0), OptFace::NONE])),
            he_vertex: Vec::from_iter((0..n).flat_map(|i| [Vertex(i), Vertex((i+1) % n)])),
            he_next:   Vec::from_iter((0..n).flat_map(|i| [Halfedge(2*((i+1)%n)), Halfedge(2*((i+n-1)%n)+1)])),
            he_prev:   Vec::from_iter((0..n).flat_map(|i| [Halfedge(2*((i+n-1)%n)), Halfedge(2*((i+1)%n)+1)])),
            edge_label: vec![EdgeLabel::default(); n],
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

    // Accessors

    pub fn num_vertices(&self) -> usize {
        self.vertex_data[0].len_of(Axis(0))
    }
    pub fn num_faces(&self) -> usize {
        self.face_he.len()
    }
    pub fn num_halfedges(&self) -> usize {
        self.he_vertex.len()
    }
    pub fn num_edges(&self) -> usize {
        self.num_halfedges() / 2
    }
    pub fn num_layers(&self) -> usize {
        self.vertex_data.len()
    }
    pub fn last_layer(&self) -> usize {
        self.num_layers() - 1
    }
    pub fn activations(&self, layer: usize) -> ArrayView2<f32> {
        self.vertex_data[layer].view()
    }
    pub fn activations_last(&self) -> ArrayView2<f32> {
        self.vertex_data.last().unwrap().view()
    }
    pub fn inputs(&self) -> &Array2<f32> {
        &self.vertex_data[0]
    }
    pub fn vertex_inputs(&self, vertex: Vertex) -> ArrayView1<f32> {
        self.vertex_data[0].row(vertex.0)
    }
    pub fn face_activations_at(&self, layer: usize) -> ArrayView3<f32> {
        self.face_data[layer].view()
    }
    pub fn face_activations(&self) -> ArrayView3<f32> {
        self.face_activations_at(self.last_layer())
    }

    pub fn opposite(&self, he: Halfedge) -> Halfedge {
        he.opposite()
    }
    pub fn next(&self, he: Halfedge) -> Halfedge {
        self.he_next[he.0]
    }
    pub fn prev(&self, he: Halfedge) -> Halfedge {
        //self.next(he.opposite()).opposite()
        self.he_prev[he.0]
    }
    fn optface(&self, he: Halfedge) -> OptFace {
        self.he_face[he.0]
    }
    pub fn face(&self, he: Halfedge) -> Option<Face> {
        self.optface(he).into()
    }
    pub fn start(&self, he: Halfedge) -> Vertex {
        self.he_vertex[he.0]
    }
    pub fn end(&self, he: Halfedge) -> Vertex {
        self.start(he.opposite())
    }
    pub fn endpoints(&self, edge: Edge) -> (Vertex, Vertex) {
        (self.start(edge.halfedge(0)), self.start(edge.halfedge(1)))
    }
    fn edge_optfaces(&self, edge: Edge) -> (OptFace, OptFace) {
        (self.optface(edge.halfedge(0)), self.optface(edge.halfedge(1)))
    }
    pub fn edge_label(&self, edge: Edge) -> &EdgeLabel {
        &self.edge_label[edge.0]
    }
    pub fn halfedge_on_face(&self, face: Face) -> Halfedge {
        self.face_he[face.0]
    }

    //fn halfedges(&self
    
    // Iterators

    pub fn vertices(&self) -> impl ExactSizeIterator<Item=Vertex> {
        (0..self.num_vertices()).map(Vertex)
    }
    pub fn faces(&self) -> impl ExactSizeIterator<Item=Face> {
        (0..self.num_faces()).map(Face)
    }
    pub fn halfedges(&self) -> impl ExactSizeIterator<Item=Halfedge> {
        (0..self.num_halfedges()).map(Halfedge)
    }
    pub fn edges(&self) -> impl ExactSizeIterator<Item=Edge> {
        (0..self.num_edges()).map(Edge)
    }
    pub fn halfedges_on_face(&self, face: Face) -> HalfedgesOnFace<'_> {
        HalfedgesOnFace::new(self.halfedge_on_face(face), &self.he_next)
    }
    pub fn vertices_on_face(&self, face: Face) -> impl '_ + Iterator<Item=Vertex> {
        self.halfedges_on_face(face).map(|he| self.start(he))
    }
    pub fn vertex_data_for_face(&self, face: Face, layer: usize) -> Array2<f32> {
        let indices = Vec::from_iter(self.vertices_on_face(face).map(usize::from));
        self.vertex_data[layer].select(Axis(0), &indices)
    }

    // Invariants

    /// check the class invariants
    #[cfg(test)]
    fn check_invariants(&self) {
        // vectors have the right size
        assert!(self.vertex_data.len() > 0);
        assert_eq!(self.vertex_data.len(), self.num_layers());
        assert_eq!(self.face_data.len(), self.num_layers());
        for (vertex_data, face_data) in self.vertex_data.iter().zip(&self.face_data) {
            assert_eq!(vertex_data.len_of(Axis(0)), self.num_vertices());
            assert_eq!(face_data.len_of(Axis(0)), self.num_faces());
            assert_eq!(face_data.len_of(Axis(1)), 3);
            assert_eq!(face_data.len_of(Axis(2)), vertex_data.len_of(Axis(1)));
        }
        assert_eq!(self.face_he.len(), self.num_faces());
        assert_eq!(self.he_vertex.len(), self.num_halfedges());
        assert_eq!(self.he_face.len(), self.num_halfedges());
        assert_eq!(self.he_next.len(), self.num_halfedges());
        assert_eq!(self.he_prev.len(), self.num_halfedges());
        assert_eq!(self.edge_label.len(), self.num_edges());
        // all indices in the vectors are valid
        for he in &self.face_he {
            assert!(he.0 < self.num_halfedges());
        }
        for face in &self.he_face {
            if let Some(face) = face.view() {
                assert!(face.0 < self.num_faces());
            }
        }
        for vertex in &self.he_vertex {
            assert!(vertex.0 < self.num_vertices());
        }
        for he in &self.he_next {
            assert!(he.0 < self.num_halfedges(), "Invalid halfedge: {} out of bounds {}", he.0, self.num_halfedges());
        }
        for he in &self.he_prev {
            assert!(he.0 < self.num_halfedges(), "Invalid halfedge: {} out of bounds {}", he.0, self.num_halfedges());
        }
        // previous/next of halfedges line up
        assert!(self.halfedges().all(|he| self.next(self.prev(he)) == he));
        assert!(self.halfedges().all(|he| self.prev(self.next(he)) == he));
        // next halfedge has correct start
        assert!(self.halfedges().all(|he| self.start(self.next(he)) == self.end(he)));
        // next halfedge has same face
        assert!(self.halfedges().all(|he| self.face(self.next(he)) == self.face(he)));
        // he_face and face_he agree
        assert!(self.faces().all(|f| self.face(self.halfedge_on_face(f)) == Some(f)));
        self.check_face_outputs();
    }

    #[cfg(test)]
    fn check_face_outputs(&self) {
        // check that the face outputs are correct
        for face in self.faces() {
            let indices = Vec::from_iter(self.vertices_on_face(face).map(usize::from));
            let vertex_coords_in = self.vertex_data[0].select(Axis(0), &indices);
            let vertex_coords_in = concatenate![Axis(1), vertex_coords_in, Array2::ones((indices.len(), 1))];
            for layer in 0..self.num_layers() {
                let vertex_coords_out = self.vertex_data[layer].select(Axis(0), &indices);
                let face_data = self.face_data[layer].index_axis(Axis(0), face.0);
                let computed_coords_out = vertex_coords_in.dot(&face_data);
                //println!("diff {face:?} at {layer}: {}", &computed_coords_out - &vertex_coords_out);
                for i in 0..indices.len() {
                    for j in 0..vertex_coords_out.ncols() {
                        if !approx::abs_diff_eq!(computed_coords_out[(i,j)], vertex_coords_out[(i,j)], epsilon=1e-4) {
                            println!("Not equal: {face:?} at {layer}: vertex {}, dim {j}", indices[i]);
                            println!("{} * {} != {}", vertex_coords_in.row(i), face_data.column(j), vertex_coords_out[(i,j)]);
                            println!("was {}", self.vertex_data[layer-1].select(Axis(0), &indices).column(j));
                        }
                    }
                }
                assert_abs_diff_eq!(&computed_coords_out, &vertex_coords_out, epsilon=1e-4);
            }
        }
    }

    // Mutations

    /// Split an edge at position t (between 0 and 1)
    /// Return the new vertex, and the two halfedges that start at the new vertex.
    /// If t == 0 || t == 1, then return on of the existing edge endpoints.
    /// The two returned halfedges are on face(edge.halfedge(0)) and ..(1) respectively
    /// The boolean indicates if a new vertex was inserted.
    pub fn split_edge(&mut self, edge: Edge, t: f32) -> (bool, Vertex, Halfedge, Halfedge) {
        struct SplitEdge {
            /// Was the edge actually split? If t==0 || t==1 then the vertex already existed
            was_split: bool,
            /// Vertex inserted at position t on the split edge
            vertex: Vertex,
            /// The two halfedges that start at `vertex`
            halfedges: (Halfedge, Halfedge),
        }
        if t == 0.0 {
            (false, self.start(edge.halfedge(0)), edge.halfedge(0), self.next(edge.halfedge(1)))
        } else if t == 1.0 {
            (false, self.start(edge.halfedge(1)), self.next(edge.halfedge(0)), edge.halfedge(1))
        } else {
            let (a, b) = self.endpoints(edge);
            // compute new vertex data
            let new_v = Vertex(self.num_vertices());
            for vertex_data in self.vertex_data.iter_mut() {
                append_lerp(vertex_data, a.0, b.0, t);
            }
            // add new edge
            // reuse existing edge for the first part of the split edge:
            //   Old situation:   a --edge--> b --next-->
            //   New situation:   a --edge--> new_v --new_edge--> b --next-->
            //   Old situation:
            //      --prev1--> a --he1--> b --next1-->
            //      --prev2--> b --he2--> a --next2-->
            //   New situation:
            //      --prev1--> a --he1--> new_v --new_edge[0]--> b --next1-->
            //      --prev2--> b --new_edge[1]--> new_v --he2--> a --next2-->
            let (he1, he2) = edge.halfedges();
            let (face1, face2) = (self.he_face[he1.0], self.he_face[he2.0]);
            let next1 = self.next(he1);
            let prev2 = self.prev(he2);
            let new_edge = Edge(self.num_edges());
            self.he_vertex.extend_from_slice(&[new_v, b]);
            self.he_vertex[he2.0] = new_v;
            self.he_face.extend_from_slice(&[face1, face2]);
            self.he_next.extend_from_slice(&[next1, he2]);
            self.he_next[he1.0] = new_edge.halfedge(0);
            self.he_next[prev2.0] = new_edge.halfedge(1);
            self.he_prev.extend_from_slice(&[he1, prev2]);
            self.he_prev[he2.0] = new_edge.halfedge(1);
            self.he_prev[next1.0] = new_edge.halfedge(0);
            self.edge_label.push(self.edge_label[edge.0]);
            (true, new_v, new_edge.halfedge(0), he2)
        }
    }

    /// Split a face by adding an edge between the start vertices of a and b.
    /// A new face is inserted for the loop start(a)...start(b).
    pub fn split_face(&mut self, a: Halfedge, b: Halfedge, edge_label: EdgeLabel) -> Face {
        let face = self.optface(a);
        assert!(self.face(a).is_some(), "Cannot split the outside face");
        assert_eq!(self.face(a), self.face(b));
        // add edge
        let (as_, _ae) = (self.start(a), self.end(a));
        let (bs,  _be) = (self.start(b), self.end(b));
        let (ap, bp) = (self.prev(a), self.prev(b));
        // old situation:
        //    aps --ap--> as --a--> ae  --...-->  bps --bp--> bs --b--> be
        // new situation:
        //    aps --ap--> as --new_edge[0]--> bs --b--> be
        //    bps --bp--> bs --new_edge[1]--> as --a--> ae
        let new_edge = Edge(self.num_edges());
        /*
        println!("Split face: --{ap:?}--> {as_:?} --{a:?}--> {ae:?} --> .. --{bp:?}--> {bs:?} --{b:?}--> {be:?}");
        println!("Into:");
        println!("  --{ap:?}--> {as_:?} --{:?}--> {bs:?} --{b:?}--> {be:?}", new_edge.halfedge(0));
        println!("  --{bp:?}--> {bs:?} --{:?}--> {as_:?} --{a:?}--> {ae:?}", new_edge.halfedge(1));
        */
        self.he_vertex.extend_from_slice(&[as_, bs]);
        self.he_face.extend_from_slice(&[face, face]);
        self.he_next.extend_from_slice(&[b, a]);
        self.he_next[ap.0] = new_edge.halfedge(0);
        self.he_next[bp.0] = new_edge.halfedge(1);
        self.he_prev.extend_from_slice(&[ap, bp]);
        self.he_prev[a.0] = new_edge.halfedge(1);
        self.he_prev[b.0] = new_edge.halfedge(0);
        self.edge_label.push(edge_label);
        // add face
        let new_face = Face(self.num_faces());
        for face_data in self.face_data.iter_mut() {
            let data = face_data.index_axis(Axis(0), face.0).to_owned();
            face_data.push(Axis(0), data.view()).unwrap();
        }
        self.face_he.push(new_edge.halfedge(1));
        self.face_he[face.0] = new_edge.halfedge(0);
        // assign halfedges to new face
        let mut he = new_edge.halfedge(1);
        while self.he_face[he.0] != new_face.into() {
            self.he_face[he.0] = new_face.into();
            he = self.next(he);
        }
        new_face
    }

    /// Split all faces in the partition by treating zeros of a function as edges.
    /// Requires that there is at most 1 zero crossing per edge.
    /// The function split(a,b) should return Some(t) if f(lerp(a,b,t)) == 0
    fn split_all<E: EdgeSplitter>(&mut self, splitter: &mut E, edge_label: EdgeLabel) {
        // Split edges such that all zeros of the function go through vertices.
        let mut zero_vertices = vec![false; self.num_vertices()];
        // Note: splitting can add more edges, only iterate up to number of edges we had before we started
        let num_edges = self.num_edges();
        for edge in (0..num_edges).map(Edge) {
            let (a,b) = self.endpoints(edge);
            if zero_vertices[a.0] || zero_vertices[b.0] {
                // already have a zero crossing on this edge
            } else if let Some(t) = splitter.split_point(self, a, b) {
                // split at t
                let (was_split, new_vertex, _, _) = self.split_edge(edge, t);
                if was_split {
                    zero_vertices.push(true);
                    splitter.apply_split(self, new_vertex, t);
                } else {
                    zero_vertices[new_vertex.0] = true;
                }
                debug_assert_eq!(zero_vertices.len(), self.num_vertices());
            }
        }
        // Split faces that have non-consecutive zero vertices
        let num_faces = self.num_faces();
        for face in (0..num_faces).map(Face) {
            let mut iter = self.halfedges_on_face(face);
            if let Some(he1) = iter.find(|he| zero_vertices[self.start(*he).0]) {
                if let Some(_) = iter.find(|he| !zero_vertices[self.start(*he).0]) {
                    if let Some(he2) = iter.find(|he| zero_vertices[self.start(*he).0]) {
                        // TODO: handle case where he2 is consecutive to more zeros and to he1.
                        // This can only happen when there are multiple vertices in a line.
                        self.split_face(he1, he2, edge_label);
                    }
                }
            }
        }
    }

    /// Split faces by adding an edge between any two non-adjacent vertices a,b that have:
    ///  f(a) = f(b) > 0
    ///  a vertex c on a..b and a vertex d on b..a have f(c) != f(a) and f(c) != f(a)
    fn split_faces() {

    }

    /// Split all faces in the partition by treating zeros of the given layer as edges.
    fn split_by_zero_at(&mut self, layer: usize) {
        // Split faces on activation = 0
        let acts = self.activations(layer);
        for dim in 0..acts.len_of(Axis(1)) {
            let edge_label = EdgeLabel { layer: self.num_layers(), dim };
            struct ZeroSplitter {
                layer: usize,
                dim: usize,
                // Note: could track which vertices are non-negative to make relu mask
            }
            impl EdgeSplitter for ZeroSplitter {
                fn split_point(&self, partition: &Partition, a: Vertex, b: Vertex) -> Option<f32> {
                    let acts = &partition.vertex_data[self.layer];
                    find_zero(acts[(a.0, self.dim)], acts[(b.0, self.dim)])
                }
                fn apply_split(&mut self, partition: &mut Partition, new_vertex: Vertex, _point: f32) {
                    // make sure that new vertices are actually 0 in dimension dim
                    // this may not be true exactly because of rounding errors.
                    let acts = &mut partition.vertex_data[self.layer];
                    acts[(new_vertex.0, self.dim)] = 0.0;
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
            if let Some(face) = self.face(he) {
                mask.row_mut(face.0).zip_mut_with(&acts.row(self.start(he).0),
                    |m, x| *m &= *x >= -eps
                );
            }
        }
        mask
    }

    /// Aplit all faces in the partition at points where argmax of dims changes
    fn split_by_argmax_at(&mut self, layer: usize) -> Array1<f32> {
        // It is hard to do this fully correctly
        // approximate algorithm:
        //  * sort dims by most likely to contain maximum (to avoid unnecesary splits)
        //  * for each dim: split_by difference of that dim and current max
        // this might introduce some unnecessary splits

        // for each vertex: find argmax
        let acts = self.activations(layer);
        let vertex_argmax = acts.rows().into_iter().map(argmax).collect::<Vec<usize>>();
        // sort dims by frequency
        let histogram_argmax = histogram(&vertex_argmax, acts.ncols());
        let mut dims = Vec::from_iter(0..histogram_argmax.len());
        dims.sort_by(|i, j| histogram_argmax[*j].cmp(&histogram_argmax[*i])); // Note: reverse order

        // split dims
        struct MaxSplitter {
            //argmax_so_far: usize,
            layer: usize,
            dim: usize,
            max_so_far: Vec<f32>,
        }
        impl EdgeSplitter for MaxSplitter {
            fn split_point(&self, partition: &Partition, a: Vertex, b: Vertex) -> Option<f32> {
                // split by zeros of (acts[:,dim] - max_so_far)
                let acts = &partition.vertex_data[self.layer];
                find_zero(acts[(a.0, self.dim)] - self.max_so_far[a.0], acts[(b.0, self.dim)] - self.max_so_far[b.0])
            }
            fn apply_split(&mut self, partition: &mut Partition, new_vertex: Vertex, _t: f32) {
                let acts = &mut partition.vertex_data[self.layer];
                self.max_so_far.push(acts[(new_vertex.0, self.dim)]);
            }
        }
        let mut splitter = MaxSplitter {
            layer,
            dim: dims[0],
            max_so_far: acts.column(dims[0]).to_vec(),
            //argmax_so_far: vec![dims[0]; acts.nrows()];
        };
        for &dim in &dims[1..] {
            let edge_label = EdgeLabel { layer: self.num_layers(), dim };
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
        // sanity check
        if cfg!(debug_assertions) {
            let acts = self.activations(layer);
            for (row, max) in acts.axis_iter(Axis(0)).zip(splitter.max_so_far.iter()) {
                if row.iter().all(|x| x < max) {
                    println!("Bad maximum: {row} < {max}");
                }
            }
        }
        // return max
        Array1::from_vec(splitter.max_so_far)
        
        // alternative algorithm:
        //  * for every vertex+dim: track if it is equal to max
        //  * for every edge (a,b): if is_max mask(a) ∩ is_max_mask(b) = ∅: find all changes
        //    * 
        //  * split faces 
        //    * maybe also split the newly introduced edge needed, and repeat
    }

    fn argmax_face_mask(&self, layer: usize, max: &[f32]) -> Array2<bool> {
        let acts = self.activations(layer);
        let mut mask = Array2::from_elem((self.num_faces(), acts.ncols()), true);
        let eps = 1e-6;
        for he in self.halfedges() {
            if let Some(face) = self.face(he) {
                let max = max[self.start(he).0];
                mask.row_mut(face.0).zip_mut_with(&acts.row(self.start(he).0),
                    |m, x| *m &= *x >= max - eps
                );
            }
        }
        mask
    }
    fn argmax_face_index(&self, layer: usize, max: &[f32]) -> Array1<usize> {
        let mask = self.argmax_face_mask(layer, max);
        Array1::from_iter(mask.rows().into_iter().map(
            |row| row.iter().position(|x| *x).expect("Face has inconsistent argmax")
        ))
    }

    // Neural network operations
    
    /// Update partition by adding a ReLU layer, that takes as input the output of the given layer.
    pub fn relu_at(&mut self, layer: usize) {
        // Split faces
        self.split_by_zero_at(layer);
        // Compute relu output for vertices
        let acts = self.vertex_data[layer].view();
        let new_vertex_data = relu(&acts);
        self.vertex_data.push(new_vertex_data);
        // Apply mask to compute per-face activation
        let face_mask = self.non_negative_face_mask(layer);
        let mut face_data = self.face_data[layer].clone();
        face_data.zip_mut_with(&face_mask.slice(s![.., NewAxis, ..]), |x, m| if !m { *x = 0.0; });
        self.face_data.push(face_data);
    }
    /// Update partition by adding a ReLU layer
    /// This means:
    ///  * split all faces at activation[l,_,d]=0 for any dim d
    ///  * set activation[l+1] = max(0,activation[l])
    pub fn relu(&mut self) {
        self.relu_at(self.last_layer());
    }

    pub fn leaky_relu_at(&mut self, layer: usize, negative_slope: f32) {
        self.split_by_zero_at(layer);
        let face_mask = self.non_negative_face_mask(layer);
        // Compute leaky relu output for vertices
        self.vertex_data.push(leaky_relu(&self.vertex_data[layer].view(), negative_slope));
        // Apply mask to compute per-face activation
        let mut face_data = self.face_data[layer].clone();
        face_data.zip_mut_with(&face_mask.slice(s![.., NewAxis, ..]), |x, m| if !m { *x *= negative_slope; });
        self.face_data.push(face_data);
    }
    pub fn leaky_relu(&mut self, negative_slope: f32) {
        self.leaky_relu_at(self.last_layer(), negative_slope);
    }

    pub fn max_pool_at(&mut self, layer: usize) {
        let max = self.split_by_argmax_at(layer);
        // New face data takes the given dim
        let face_argmax = self.argmax_face_index(layer, max.as_slice().unwrap());
        let face_data = &self.face_data[layer];
        let face_data = select_in_rows(Axis(2), &face_data.view(), face_argmax);
        self.face_data.push(face_data.insert_axis(Axis(2)));
        // New vertex data is just the max value
        self.vertex_data.push(max.insert_axis(Axis(1)));
    }
    pub fn max_pool(&mut self) {
        self.max_pool_at(self.last_layer());
    }

    pub fn arg_max_pool_at(&mut self, layer: usize) {
        let max = self.split_by_argmax_at(layer);
        // New face data takes the given dim
        let face_argmax = self.argmax_face_index(layer, max.as_slice().unwrap());
        let face_data = concatenate![Axis(1),
            Array::zeros((self.num_faces(), 2)),
            face_argmax.insert_axis(Axis(1)).mapv(|x| x as f32)
        ];
        self.face_data.push(face_data.insert_axis(Axis(2)));
        // New vertex data is just the max value
        let acts = self.activations(layer);
        let vertex_argmax = Array::from_iter(acts.rows().into_iter().map(|row| argmax(row) as f32));
        self.vertex_data.push(vertex_argmax.insert_axis(Axis(1)));
    }

    /// Append a layer that computes output using the given function.
    /// The function should be linear, without a bias term.
    /// Then `add_bias` should add the bias term
    pub fn generalized_linear_at(&mut self, layer: usize, fun: impl Fn(ArrayView2<f32>) -> Array2<f32>, add_bias: impl Fn(ArrayViewMut2<f32>)) {
        // Compute vertex data (x' = x*w + b)
        let mut vertex_data = fun(self.vertex_data[layer].view());
        add_bias(vertex_data.view_mut());
        self.vertex_data.push(vertex_data);
        // Compute new face_data:
        // If old was x = p*u + v,
        //  new is x' = x*w + b = p*(u*w) + (v*w + b)
        // Where v is face_data[..,2,..]
        let face_data = &self.face_data[layer];
        let (a,b,c) = face_data.dim();
        let face_data2 = face_data.to_shape((a*b, c)).unwrap();
        let new_face_data = fun(face_data2.view());
        let (_,d) = new_face_data.dim();
        let mut new_face_data = new_face_data.into_shape_with_order((a,b,d)).unwrap();
        add_bias(new_face_data.index_axis_mut(Axis(1), 2));
        self.face_data.push(new_face_data);
    }

    /// Append a layer that computes output using a linear transformation:
    ///   x_{l} = w*x_l' + b.
    pub fn linear_at(&mut self, layer: usize, weight: &ArrayView2<f32>, bias: &ArrayView1<f32>) {
        self.generalized_linear_at(layer, |x| x.dot(weight), |mut x| x += bias)
    }
    /// Append a layer that computes output using a linear transformation:
    ///   x_{l+1} = w*x_l + b.
    pub fn linear(&mut self, weight: &ArrayView2<f32>, bias: &ArrayView1<f32>) {
        self.linear_at(self.last_layer(), weight, bias);
    }

    /// Append a layer that adds the output of two existing layers
    pub fn sum(&mut self, layer1: usize, layer2: usize) {
        self.vertex_data.push(&self.vertex_data[layer1] +&self.vertex_data[layer2]);
        self.face_data.push(&self.face_data[layer1] + &self.face_data[layer2]);
    }
    /// Append a layer that adds an earlier layer to the output
    pub fn residual(&mut self, layer: usize) {
        self.sum(layer, self.last_layer());
    }
}

/// Trait for deciding to split an edge
trait EdgeSplitter {
    fn split_point(&self, partition: &Partition, a: Vertex, b: Vertex) -> Option<f32>;
    fn apply_split(&mut self, partition: &mut Partition, new_vertex: Vertex, t: f32);
}

/// Iterator for all halfedges that form a face
pub struct HalfedgesOnFace<'a> {
    start: Halfedge,
    item: Option<Halfedge>,
    he_next: &'a [Halfedge],
}
impl<'a> HalfedgesOnFace<'a> {
    fn new(start: Halfedge, he_next: &'a [Halfedge]) -> Self {
        Self { start, item: Some(start), he_next }
    }
}
impl<'a> Iterator for HalfedgesOnFace<'a> {
    type Item = Halfedge;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.item;
        if let Some(pos) = item {
            let next = self.he_next[pos.0];
            self.item = if next == self.start { None } else { Some(next) }
        }
        item
    }
}


#[cfg(test)]
mod test {
    use ndarray::{stack, Array1};
    use rand::{rngs::SmallRng, SeedableRng};

    use crate::{nn::NNModule, svg::SvgOptions};

    use super::*;

    #[test]
    fn constructor_invariants() {
        Partition::square(2.0).check_invariants();
    }

    #[test]
    fn iterators() {
        let p = Partition::square(2.0);
        assert_eq!(p.vertices().len(), p.num_vertices());
        assert_eq!(p.halfedges().len(), p.num_halfedges());
        assert_eq!(p.edges().len(), p.num_edges());
        assert_eq!(p.num_halfedges(), p.num_edges() * 2);
    }

    #[test]
    fn split_edge() {
        fn check_split_edge(mut p: Partition, edge: Edge, t: f32) {
            let faces = p.edge_optfaces(edge);
            let (was_split, v, he1, he2) = p.split_edge(edge, t);
            //println!("split_edge {edge:?} at {t}\n{p:?}");
            p.check_invariants();
            assert_eq!(was_split, t != 0.0 && t != 1.0);
            assert_ne!(he1, he2);
            assert_eq!(faces, (p.optface(he1), p.optface(he2)));
            assert_eq!(p.start(he1), v);
            assert_eq!(p.start(he2), v);
        }
        for p in [Partition::square(2.0)] {
            for edge in p.edges() {
                for t in [0.0, 0.5, 1.0] {
                    check_split_edge(p.clone(), edge, t);
                }
            }
        }
    }

    #[test]
    fn split_face() {
        for p in [Partition::square(2.0)] {
            let mut p = p.clone();
            let a = p.halfedge_on_face(Face(0));
            let b = p.next(p.next(a));
            println!("Before split {a:?}-{b:?}\n{p:?}");
            p.split_face(a, b, EdgeLabel::default());
            println!("split_face {a:?}-{b:?}\n{p:?}");
            p.check_invariants();
        }
    }

    #[test]
    fn simple_relu() {
        let mut p = Partition::square(2.0);
        p.relu();
        println!("Relu\n{p:?}");
        p.check_invariants();
        // applying relu again should keep everything the same (except adding a layer)
        let mut p2 = p.clone();
        p2.relu();
        p2.check_invariants();
        assert_eq!(p.num_vertices(), p2.num_vertices());
        assert_eq!(p.num_faces(), p2.num_faces());
        assert_eq!(p.num_halfedges(), p2.num_halfedges());
    }

    #[test]
    fn triangles() {
        // make weights and biases
        let n = 5;
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
        println!("weight: {weight}");
        println!("bias: {bias}");
        // partition
        let mut p = Partition::square(2.0);
        p.linear(&weight.view(), &bias.view());
        p.check_invariants();
        p.relu();
        p.check_invariants();
        println!("{} faces", p.num_faces());
        // hack: write to file
        use std::fs::File;
        let mut file = File::create("triangles.svg").expect("Can't create file");
        SvgOptions::new().write_svg(&p, &mut file).unwrap();
    }

    #[test]
    fn random_nn() {
        let mut rng = SmallRng::seed_from_u64(42);
        let dim_in = 2;
        //let dim_out = 1000;
        let dim_out = 100;
        let layer = crate::nn::Linear::new_uniform(dim_in, dim_out, &mut rng);
        // partition
        let mut p = Partition::square(1.0);
        layer.apply(&mut p);
        p.check_invariants();
        p.relu();
        p.check_invariants();
        println!("{} faces", p.num_faces());
        // hack: write to file
        if p.num_faces() < 10000 {
            use std::fs::File;
            let mut file = File::create("random_nn.svg").expect("Can't create file");
            SvgOptions::new().write_svg(&p, &mut file).unwrap();
        }
    }

    #[test]
    fn max_pool() {
        let mut rng = SmallRng::seed_from_u64(42);
        let dim_in = 2;
        let dim_hidden = 10;
        let dim_out = 10;
        //let dim_hidden = 3;
        //let dim_out = 4;
        // two layer network
        let mut p = Partition::square(1.0);
        crate::nn::Linear::new_uniform(dim_in, dim_hidden, &mut rng).apply(&mut p);
        p.relu();
        p.check_invariants();
        println!("{} faces after relu", p.num_faces());
        crate::nn::Linear::new_uniform(dim_hidden, dim_out, &mut rng).apply(&mut p);
        if true {
            use std::fs::File;
            let mut file = File::create("max_pool_pre.svg").expect("Can't create file");
            SvgOptions::new().write_svg(&p, &mut file).unwrap();
        }
        p.check_invariants();
        p.max_pool();
        p.check_invariants();
        println!("{} faces", p.num_faces());
        if true {
            use std::fs::File;
            let mut file = File::create("max_pool.svg").expect("Can't create file");
            SvgOptions::new().write_svg(&p, &mut file).unwrap();
        }
    }
}

