// This module defines a halfedge data structure for partitioning 2d space.

use std::fmt::Debug;
use ndarray::{array, s, Array, Array2, Array3, ArrayView, ArrayView1, ArrayView2, ArrayViewMut2, Axis, Dimension, NewAxis};

/*
// A partition of 2d space
struct Partition {
  vertices: Vec<Float[2]>,
  vertices_out: Array2<f32>,
  face_verts: Vec<usize>,
  face_starts: Vec<usize>,
}

type Vertex   = usize;
type Halfedge = usize;
type Edge     = usize; // even
const NONE = Halfedge::MAX;
type OptHalfedge = Option<NonMaxUsize>;
*/

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Vertex(usize);

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Halfedge(usize);

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Edge(usize);

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Face(usize);

#[derive(Copy, Clone, PartialEq, Eq)]
struct OptFace(usize);

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
        face.0.into()
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

/// A partition of 2d space.
/// Stored as a half-edge data structure
#[derive(Debug, Clone, PartialEq)]
pub struct Partition {
    //vertex_in:   Array2<f32>, // vertex -> f32[2]
    //vertex_out:  Array2<f32>, // vertex -> f32[D]
    //vertex_data: Array2<f32>, // vertex -> f32[D]
    /// Data associated with vertices.
    /// This is the output of layer l in the neural network.
    vertex_data: Vec<Array2<f32>>, // layer -> vertex -> f32[D_l]
    /// For each ReLU layer, for each vertex: a mask indicating which dimensions are non-negative
    // vertex_mask_data: Vec<Array2<u64>>, // layer -> vertex -> mask of non-negative dimensions
    // grouped by face
    /*
    he_vertex:   Vec<Vertex>,    // halfedge -> vertex
    face_start:  Vec<Halfedge>,  // face -> first halfedge
    face_end:    Vec<Halfedge>,  // face -> after last halfedge
    opposite:    Vec<Option<Halfedge>>, // halfedge -> halfedge
    face_split:  Vec<usize>, // 
    */
    // grouped by halfedge
    face_data:  Vec<Array3<f32>>,  // layer -> face -> f32[D_in, D_F]
    face_he:    Vec<Halfedge>,     // face -> halfedge
    he_vertex:  Vec<Vertex>,       // halfedge -> vertex
    he_face:    Vec<OptFace>,      // halfedge -> face or NONE
    he_next:    Vec<Halfedge>,     // halfedge -> halfedge
    he_prev:    Vec<Halfedge>,     // halfedge -> halfedge
    //edge_label: Vec<Label>, // edge -> label
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
    return k;
}

/// Find the point t at which a+t*(b-a) == 0
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

pub(crate) fn relu<D: Dimension>(arr: &ArrayView<f32, D>) -> Array<f32, D> {
    arr.map(|x| x.max(0.0))
}

impl Partition {

    // Constructors

    /// Construct a 
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
            face_he:   vec![Halfedge(0)],
            he_face:   Vec::from_iter((0..n).flat_map(|_| [OptFace(0), OptFace::NONE])),
            he_vertex: Vec::from_iter((0..n).flat_map(|i| [Vertex(i), Vertex((i+1) % n)])),
            he_next:   Vec::from_iter((0..n).flat_map(|i| [Halfedge(2*((i+1)%n)), Halfedge(2*((i+n-1)%n)+1)])),
            he_prev:   Vec::from_iter((0..n).flat_map(|i| [Halfedge(2*((i+n-1)%n)), Halfedge(2*((i+1)%n)+1)])),
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
            (true, new_v, new_edge.halfedge(0), he2)
        }
    }

    /// Split a face by adding an edge between the start vertices of a and b.
    /// A new face is inserted for the loop start(a)...start(b).
    pub fn split_face(&mut self, a: Halfedge, b: Halfedge) -> Face {
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
    fn split_by<Split>(&mut self, split: Split)
        where Split: Fn(&Self, Vertex, Vertex) -> Option<f32>
    {
        // Split edges such that all zeros of the function go through vertices.
        let mut zero_vertices = vec![false; self.num_vertices()];
        // Note: splitting can add more edges, only iterate up to number of edges we had before we started
        let num_edges = self.num_edges();
        for edge in (0..num_edges).map(Edge) {
            let (a,b) = self.endpoints(edge);
            if zero_vertices[a.0] || zero_vertices[b.0] {
                // already have a zero crossing on this edge
            } else if let Some(t) = split(&self, a, b) {
                // split at t
                let (was_split, new_vertex, _he1, _he2) = self.split_edge(edge, t);
                if was_split {
                    zero_vertices.push(true);
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
                        self.split_face(he1, he2);
                    }
                }
            }
        }
    }

    // Neural network operations
    
    /// Update partition by adding a ReLU layer
    /// This means:
    ///  * split all faces at activation[l,_,d]=0 for any dim d
    ///  * set activation[l+1] = max(0,activation[l])
    pub fn relu(&mut self) {
        // Split faces on activation = 0
        let acts: ndarray::ArrayBase<ndarray::ViewRepr<&f32>, ndarray::Dim<[usize; 2]>> = self.activations_last();
        for dim in 0..acts.len_of(Axis(1)) {
            self.split_by(|slf, a, b| {
                let acts = slf.activations_last();
                find_zero(acts[[a.0, dim]], acts[[b.0, dim]])
            });
        }
        // Compute mask of non-negative dimensions for each face,
        // the mask is true iff all vertices are non-negative
        let acts = self.activations_last();
        let mut mask = Array2::<bool>::default((self.num_faces(), acts.ncols()));
        for he in self.halfedges() {
            if let Some(face) = self.face(he) {
                mask.row_mut(face.0).zip_mut_with(&acts.row(self.start(he).0),
                    |m, x| *m &= *x >= 0.0
                );
            }
        }
        // Compute relu output for vertices
        let new_vertex_data = relu(&acts);
        self.vertex_data.push(new_vertex_data);
        // Apply mask to compute per-face activation
        let mut face_data = self.face_data.last().unwrap().clone();
        face_data.zip_mut_with(&mask.slice(s![.., NewAxis, ..]), |x, m| if !m { *x = 0.0; });
        self.face_data.push(face_data);
    }

    /// Append a layer that computes output using the given function.
    /// The function should be linear, without a bias term.
    /// Then `add_bias` should add the bias term
    pub fn add_layer(&mut self, fun: impl Fn(ArrayView2<f32>) -> Array2<f32>, add_bias: impl Fn(ArrayViewMut2<f32>)) {
        // Compute vertex data (x' = x*w + b)
        let mut vertex_data = fun(self.activations_last());
        add_bias(vertex_data.view_mut());
        self.vertex_data.push(vertex_data);
        // Compute new face_data:
        // If old was x = p*u + v,
        //  new is x' = x*w + b = p*(u*w) + (v*w + b)
        // Where v is face_data[..,2,..]
        let face_data = self.face_data.last().unwrap();
        let (a,b,c) = face_data.dim();
        let face_data2 = face_data.to_shape((a*b, c)).unwrap();
        let new_face_data = fun(face_data2.view());
        let (_,d) = new_face_data.dim();
        let mut new_face_data = new_face_data.into_shape_with_order((a,b,d)).unwrap();
        add_bias(new_face_data.index_axis_mut(Axis(1), 2));
        self.face_data.push(new_face_data);
    }

    /// Append a layer that computes output using a linear transformation:
    ///   x_{l+1} = w*x_l + b.
    pub fn linear_transform(&mut self, weight: &ArrayView2<f32>, bias: &ArrayView1<f32>) {
        self.add_layer(|x| x.dot(weight), |mut x| x += bias)

    }

    /// Append a layer that adds an earlier layer to the output
    pub fn residual(&mut self, layer: usize) {
        self.vertex_data.push(&self.vertex_data[layer] + self.vertex_data.last().unwrap());
        self.face_data.push(&self.face_data[layer] + self.face_data.last().unwrap());
    }
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

    use crate::{nn::IsModule, svg::SvgOptions};

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
            p.split_face(a, b);
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
        p.linear_transform(&weight.view(), &bias.view());
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
}

