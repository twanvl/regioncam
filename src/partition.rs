// This module defines a halfedge data structure for partitioning 2d space.

use std::fmt::Debug;

// Index types

//type Index = usize;
pub(crate) type Index = u32;

/// Declare a type to be used for indexes
macro_rules! declare_index_type {
    ($vis:vis $ty:ident) => {
        #[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
        $vis struct $ty(Index);

        impl $ty {
            pub fn index(self) -> usize {
                self.0 as usize
            }
        }
        impl From<usize> for $ty {
            fn from(index: usize) -> Self {
                $ty(index as Index)
            }
        }
        impl From<$ty> for usize {
            fn from(index: $ty) -> Self {
                index.index()
            }
        }
        impl<T> std::ops::Index<$ty> for [T] {
            type Output = T;
            fn index(&self, index: $ty) -> &Self::Output {
                &self[index.0 as usize]
            }
        }
    }
}
pub(crate) use declare_index_type;

/// Declare an OptA type that works like Option<A> but takes no extra space
/// It would be nice if we could use NonMax for the indices, but that makes conversion to integers non-trivial
macro_rules! declare_opt_index_type {
    ($vis:vis $optty:ident = Option<$ty:ident>) => {
        declare_index_type!($vis $optty);

        impl $optty {
            const NONE: $optty = $optty(Index::MAX);
            pub fn some(index: $ty) -> Self {
                $optty(index.0)
            }
            pub fn view(self) -> Option<$ty> {
                if self == $optty::NONE {
                    None
                } else {
                    Some($ty(self.0))
                }
            }
        }
        impl From<$ty> for $optty {
            fn from(index: $ty) -> Self {
                $optty::some(index)
            }
        }
        impl From<$optty> for Option<$ty> {
            fn from(opt_index: $optty) -> Self {
                opt_index.view()
            }
        }
        impl Debug for $optty {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self.view() {
                    None => write!(f, "None"),
                    Some(index) => write!(f, "{:?}", index),
                }
            }
        }
    }
}


declare_index_type!(pub Vertex);
declare_index_type!(pub Halfedge);
declare_index_type!(pub Edge);
declare_index_type!(pub Face);

declare_opt_index_type!(OptVertex = Option<Vertex>);
declare_opt_index_type!(OptHalfedge = Option<Halfedge>);
declare_opt_index_type!(OptFace = Option<Face>);

impl Debug for Vertex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

impl Halfedge {
    pub fn edge(self) -> Edge {
        Edge(self.0 / 2)
    }
    pub fn side(self) -> Index {
        self.0 & 1
    }
    pub fn new(edge: Edge, side: Index) -> Halfedge {
        Halfedge(edge.0 * 2 + (side & 1))
    }
    pub fn opposite(self) -> Self {
        Halfedge(self.0 ^ 1)
    }
}
impl Debug for Halfedge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "h{}", self.0)
        /*if self.side() == 0 {
            write!(f, "h{}", self.edge().0)
        } else {
            write!(f, "H{}", self.edge().0)
        }*/
    }
}

impl Edge {
    #[inline]
    pub fn halfedge(self, side: Index) -> Halfedge {
        Halfedge::new(self, side)
    }
    #[inline]
    pub fn halfedges(self) -> (Halfedge, Halfedge) {
        (self.halfedge(0), self.halfedge(1))
    }
}
impl Debug for Edge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "e{}", self.0)
    }
}

impl Debug for Face {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "f{}", self.0)
    }
}


/// A partition of 2d space.
/// Stored as a half-edge data structure / doubly connected edge list
#[derive(Debug, Clone, PartialEq)]
pub struct Partition {
    vertex_he:  Vec<OptHalfedge>,  // vertex -> halfedge or NONE
    
    face_he:    Vec<Halfedge>,     // face -> halfedge

    he_vertex:  Vec<Vertex>,       // halfedge -> vertex
    he_face:    Vec<OptFace>,      // halfedge -> face or NONE
    he_next:    Vec<Halfedge>,     // halfedge -> halfedge
    he_prev:    Vec<Halfedge>,     // halfedge -> halfedge
}

impl Partition {

    // Constructors

    /// Construct a Partition with a single n-sided region
    pub fn polygon(n: usize) -> Self {
        assert!(n >= 3);
        assert!(n <= Index::MAX as usize);
        let n = n as Index;
        Self {
            vertex_he: Vec::from_iter((0..n).map(|i| OptHalfedge(2*i))),
            face_he:   vec![Halfedge(0)],
            he_face:   Vec::from_iter((0..n).flat_map(|_| [OptFace(0), OptFace::NONE])),
            he_vertex: Vec::from_iter((0..n).flat_map(|i| [Vertex(i), Vertex((i+1) % n)])),
            he_next:   Vec::from_iter((0..n).flat_map(|i| [Halfedge(2*((i+1)%n)), Halfedge(2*((i+n-1)%n)+1)])),
            he_prev:   Vec::from_iter((0..n).flat_map(|i| [Halfedge(2*((i+n-1)%n)), Halfedge(2*((i+1)%n)+1)])),
        }
    }

    // Accessors

    pub fn num_vertices(&self) -> usize {
        self.vertex_he.len()
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

    pub fn opposite(&self, he: Halfedge) -> Halfedge {
        he.opposite()
    }
    pub fn next(&self, he: Halfedge) -> Halfedge {
        self.he_next[he.index()]
    }
    pub fn prev(&self, he: Halfedge) -> Halfedge {
        self.he_prev[he.index()]
    }
    fn optface(&self, he: Halfedge) -> OptFace {
        self.he_face[he.index()]
    }
    pub fn face(&self, he: Halfedge) -> Option<Face> {
        self.optface(he).into()
    }
    pub fn start(&self, he: Halfedge) -> Vertex {
        self.he_vertex[he.index()]
    }
    pub fn end(&self, he: Halfedge) -> Vertex {
        self.start(he.opposite())
    }
    pub fn endpoints(&self, edge: Edge) -> (Vertex, Vertex) {
        (self.start(edge.halfedge(0)), self.start(edge.halfedge(1)))
    }
    pub fn edge_faces(&self, edge: Edge) -> (Option<Face>, Option<Face>) {
        (self.face(edge.halfedge(0)), self.face(edge.halfedge(1)))
    }
    pub fn is_boundary(&self, edge: Edge) -> bool {
        let (f1, f2) = self.edge_faces(edge);
        f1.is_none() || f2.is_none()
    }
    pub fn halfedge_on_face(&self, face: Face) -> Halfedge {
        self.face_he[face.index()]
    }
    pub fn halfedge_leaving_vertex(&self, vertex: Vertex) -> Option<Halfedge> {
        self.vertex_he[vertex.index()].view()
    }

    // Iterators

    pub fn vertices(&self) -> impl ExactSizeIterator<Item=Vertex> {
        (0..self.num_vertices() as Index).map(Vertex)
    }
    pub fn faces(&self) -> impl ExactSizeIterator<Item=Face> {
        (0..self.num_faces() as Index).map(Face)
    }
    pub fn halfedges(&self) -> impl ExactSizeIterator<Item=Halfedge> {
        (0..self.num_halfedges() as Index).map(Halfedge)
    }
    pub fn edges(&self) -> impl ExactSizeIterator<Item=Edge> {
        (0..self.num_edges() as Index).map(Edge)
    }
    pub fn halfedges_on_face(&self, face: Face) -> HalfedgesOnFace<'_> {
        HalfedgesOnFace::new(self.halfedge_on_face(face), &self.he_next)
    }
    pub fn vertices_on_face(&self, face: Face) -> impl '_ + Iterator<Item=Vertex> {
        self.halfedges_on_face(face).map(|he| self.start(he))
    }
    pub fn halfedges_leaving_vertex(&self, vertex: Vertex) -> HalfedgesLeavingVertex<'_> {
        HalfedgesLeavingVertex::new(self.vertex_he[vertex.index()], &self.he_next)
    }

    // Invariants

    /// Check the class invariants
    pub fn check_invariants(&self) {
        // vectors have the right size
        assert_eq!(self.vertex_he.len(), self.num_vertices());
        assert_eq!(self.face_he.len(), self.num_faces());
        assert_eq!(self.he_vertex.len(), self.num_halfedges());
        assert_eq!(self.he_face.len(), self.num_halfedges());
        assert_eq!(self.he_next.len(), self.num_halfedges());
        assert_eq!(self.he_prev.len(), self.num_halfedges());
        // all indices in the vectors are valid
        for he in &self.face_he {
            assert!(he.index() < self.num_halfedges(), "face_he: Invalid halfedge in face: {} out of bounds {}", he.index(), self.num_halfedges());
        }
        for face in &self.he_face {
            if let Some(face) = face.view() {
                assert!(face.index() < self.num_faces());
            }
        }
        for vertex in &self.he_vertex {
            assert!(vertex.index() < self.num_vertices());
        }
        for he in &self.he_next {
            assert!(he.index() < self.num_halfedges(), "he_next: Invalid halfedge: {} out of bounds {}", he.index(), self.num_halfedges());
        }
        for he in &self.he_prev {
            assert!(he.index() < self.num_halfedges(), "he_prev: Invalid halfedge: {} out of bounds {}", he.index(), self.num_halfedges());
        }
        // previous/next of halfedges line up
        for he in self.halfedges() {
            assert_eq!(self.next(self.prev(he)), he, "he_next and he_prev should agree for {he:?}");
            assert_eq!(self.prev(self.next(he)), he, "he_prev and he_next should agree for {he:?}");
        }
        // next halfedge has correct start
        assert!(self.halfedges().all(|he| self.start(self.next(he)) == self.end(he)));
        // next halfedge has same face
        assert!(self.halfedges().all(|he| self.face(self.next(he)) == self.face(he)));
        // he_face and face_he agree
        assert!(self.faces().all(|f| self.face(self.halfedge_on_face(f)) == Some(f)));
        // he_vertex and vertex_he agree
        assert!(self.vertices().all(|v| if let Some(he) = self.halfedge_leaving_vertex(v) { self.start(he) == v } else { true }));
        // if a vertex is used by a halfedge, then it must have an associated he_vertex
        assert!(self.halfedges().all(|he| self.halfedge_leaving_vertex(self.start(he)).is_some() || self.next(he) == he.opposite()));
        // no loops
        assert!(self.halfedges().all(|he| self.next(he) != he));
        // there are no degenerate faces (faces with 2 vertices)
        // except that we allow degenerate faces not connected to anything (for now!)
        assert!(self.halfedges().all(|he| self.next(self.next(he)) != he || self.next(he) == he.opposite()));
        // there are no degenerate edges (edges that end in a vertex with no other connections)
        assert!(self.halfedges().all(|he| self.next(he) != he.opposite() || self.next(self.next(he)) == he));
        // Note: we *do* allow edges with the same face on both sides
    }

    // Mutations

    /// Split an edge by inserting a new vertex.
    /// Returns new vertex and new edge
    pub fn split_edge(&mut self, edge: Edge) -> (Vertex, Edge) {
        let (_a, b) = self.endpoints(edge);
        // add new vertex & edge
        let new_v = Vertex::from(self.num_vertices());
        let new_edge = Edge::from(self.num_edges());
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
        let (face1, face2) = (self.he_face[he1.index()], self.he_face[he2.index()]);
        let next1 = self.next(he1);
        let prev2 = self.prev(he2);
        self.he_vertex.extend_from_slice(&[new_v, b]);
        self.he_vertex[he2.index()] = new_v;
        self.he_face.extend_from_slice(&[face1, face2]);
        self.he_next.extend_from_slice(&[next1, he2]);
        self.he_next[he1.index()] = new_edge.halfedge(0);
        self.he_next[prev2.index()] = new_edge.halfedge(1);
        self.he_prev.extend_from_slice(&[he1, prev2]);
        self.he_prev[he2.index()] = new_edge.halfedge(1);
        self.he_prev[next1.index()] = new_edge.halfedge(0);
        self.vertex_he.push(new_edge.halfedge(0).into());
        if self.vertex_he[b.index()] == he2.into() {
            self.vertex_he[b.index()] = new_edge.halfedge(1).into();
        }
        (new_v, new_edge)
    }

    /// Split a face by adding an edge between the start vertices of a and b.
    /// A new face is inserted for the loop start(a)...start(b).
    pub fn split_face(&mut self, a: Halfedge, b: Halfedge) -> (Edge, Face) {
        let face = self.face(a).expect("Cannot split the outside face");
        assert_eq!(self.face(a), self.face(b));
        assert_ne!(a, b, "Must give different halfedges to split face");
        assert_ne!(a, self.prev(b), "Must give non-adjacent halfedges to split face");
        assert_ne!(b, self.prev(a), "Must give non-adjacent halfedges to split face");
        // add edge
        let (as_, _ae) = (self.start(a), self.end(a));
        let (bs,  _be) = (self.start(b), self.end(b));
        let (ap, bp) = (self.prev(a), self.prev(b));
        // old situation:
        //    aps --ap--> as --a--> ae  --...-->  bps --bp--> bs --b--> be
        // new situation:
        //    aps --ap--> as --new_edge[0]--> bs --b--> be
        //    bps --bp--> bs --new_edge[1]--> as --a--> ae
        let new_edge = Edge::from(self.num_edges());
        /*
        println!("Split face: --{ap:?}--> {as_:?} --{a:?}--> {ae:?} --> .. --{bp:?}--> {bs:?} --{b:?}--> {be:?}");
        println!("Into:");
        println!("  --{ap:?}--> {as_:?} --{:?}--> {bs:?} --{b:?}--> {be:?}", new_edge.halfedge(0));
        println!("  --{bp:?}--> {bs:?} --{:?}--> {as_:?} --{a:?}--> {ae:?}", new_edge.halfedge(1));
        */
        self.he_vertex.extend_from_slice(&[as_, bs]);
        self.he_face.extend_from_slice(&[face.into(), face.into()]);
        self.he_next.extend_from_slice(&[b, a]);
        self.he_next[ap.index()] = new_edge.halfedge(0);
        self.he_next[bp.index()] = new_edge.halfedge(1);
        self.he_prev.extend_from_slice(&[ap, bp]);
        self.he_prev[a.index()] = new_edge.halfedge(1);
        self.he_prev[b.index()] = new_edge.halfedge(0);
        // add face
        let new_face = Face::from(self.num_faces());
        self.face_he.push(new_edge.halfedge(1));
        self.face_he[face.index()] = new_edge.halfedge(0);
        // assign halfedges to new face
        self.assign_halfedge_in_loop_to_face(new_edge.halfedge(1), new_face.into());
        (new_edge, new_face)
    }

    /// Assign all halfedges in a loop starting at he to the given face
    fn assign_halfedge_in_loop_to_face(&mut self, mut he: Halfedge, face: OptFace) {
        while self.he_face[he.index()] != face {
            self.he_face[he.index()] = face;
            he = self.next(he);
        }
    }

    /// Merge two faces by removing the edge between them.
    /// This function only marks the edge as invalid, without actually removing it, because that would mess up the order of edges.
    ///
    /// Returns: removed face.
    /// After this call the removed face is replaced by the last face as in Vec::swap_remove
    /// Removed edges are marked invalid but not actually removed from the data structure
    pub fn merge_faces(&mut self, edge: Edge) -> Face {
        let face0 = self.face(edge.halfedge(0)).expect("Can't merge with outside face");
        let face1 = self.face(edge.halfedge(1)).expect("Can't merge with outside face");
        assert_ne!(face0, face1, "Can't merge edges within a face");
        // Assign halfedges to same face
        self.assign_halfedge_in_loop_to_face(self.halfedge_on_face(face1), face0.into());
        // Update prev/next
        // old situation:
        //    --p0--> a --he0--> b --n0-->
        //    --p1--> b --he1--> a --n1-->
        // new situation:
        //    --p0--> a --n1-->
        //    --p1--> b --n0-->
        let (he0, he1) = edge.halfedges();
        let (n0, n1) = (self.next(he0), self.next(he1));
        let (p0, p1) = (self.prev(he0), self.prev(he1));
        let (a, b) = (self.start(he0), self.start(he1));
        self.he_next[p0.index()] = n1;
        self.he_next[p1.index()] = n0;
        self.he_prev[n0.index()] = p1;
        self.he_prev[n1.index()] = p0;
        // The delted edge can't be the face_he
        if self.face_he[face0.index()].edge() == edge {
            self.face_he[face0.index()] = n0;
        }
        // The deleted edge can't be the vertex_he
        if self.vertex_he[a.index()] == he0.into() {
            self.vertex_he[a.index()] = n1.into();
        }
        if self.vertex_he[b.index()] == he1.into() {
            self.vertex_he[b.index()] = n0.into();
        }
        // mark edge as removed
        self.mark_edge_invalid(edge);

        // We might have made a degenerate edge. We should remove it now
        // This happens when there were two or more edges between the faces
        // situation:  --p0--> a --n1/p0.opposite()-->
        //let mut edges_to_remove = vec![edge]
        fn remove_degenerate_edges(slf: &mut Partition, mut he: Halfedge, face: Face) {
            while slf.next(he) == he.opposite() {
                // old situation: --p--> v1 --he--> v2 --he.opposite--> v1 --n-->
                // new situation: --p--> v1 --n-->
                let op = he.opposite();
                let p = slf.prev(he);
                let n = slf.next(op);
                let v1 = slf.start(he);
                let v2 = slf.start(op);
                slf.he_next[p.index()] = n;
                slf.he_prev[n.index()] = p;
                if slf.face_he[face.index()].edge() == he.edge() {
                    slf.face_he[face.index()] = p;
                }
                if slf.vertex_he[v1.index()] == he.into() {
                    slf.vertex_he[v1.index()] = n.into();
                }
                debug_assert_ne!(slf.vertex_he[v1.index()], op.into());
                // Make isolated loop
                slf.mark_edge_invalid(he.edge());
                // Make/check isolated vertex?
                // Note: it is possible that we try to remove the same edge twice with the second call of remove_degenerate_edges
                debug_assert!(slf.vertex_he[v2.index()] == op.into() || slf.vertex_he[v2.index()] == OptHalfedge::NONE, "Degenerate edge should have isolated vertex");
                slf.vertex_he[v2.index()] = OptHalfedge::NONE;
                // TODO: the edge he and the vertex he.end() can be deleted.
                // It is possible that he was already in a degenerate loop before this call.
                // We don't want to end up in an infinite loop in that case.
                // This can happen with weird non-convex topologies
                if p == op {
                    break;
                }
                he = p;
            }
        }
        remove_degenerate_edges(self, p0, face0);
        remove_degenerate_edges(self, p1, face0);

        // Remove the unused face, do this only at the end, because this might also invalidate face0
        self.unchecked_remove_face(face1);

        face1
    }

    /// Mark an edge as removed/invalid
    fn mark_edge_invalid(&mut self, edge: Edge) {
        let (he0, he1) = edge.halfedges();
        self.he_next[he0.index()] = he1;
        self.he_next[he1.index()] = he0;
        self.he_prev[he0.index()] = he1;
        self.he_prev[he1.index()] = he0;
        self.he_face[he0.index()] = OptFace::NONE;
        self.he_face[he1.index()] = OptFace::NONE;
    }
    /// Is the given edge invalid?
    pub fn is_invalid(&self, edge: Edge) -> bool {
        self.is_invalid_halfedge(edge.halfedge(0))
    }
    pub fn is_invalid_halfedge(&self, he: Halfedge) -> bool {
        self.next(he) == he.opposite()
    }

    /// Remove all invalid edges. This renumbers existing edges.
    pub(crate) fn remove_invalid_edges(&mut self, mut before_remove: impl FnMut(Edge)) {
        for edge in (0..self.num_edges()).rev().map(Edge::from) {
            if self.is_invalid(edge) {
                before_remove(edge);
                self.unchecked_remove_edge(edge);
            }
        }
    }

    // Remove an edge that is not being used. Should only be called on invalid edges
    fn unchecked_remove_edge(&mut self, unused_edge: Edge) {
        debug_assert!(self.is_invalid(unused_edge), "Edge should be unused");

        // detach from vertices
        for side in [0,1] {
            let unused_he = unused_edge.halfedge(side);
            let v = self.start(unused_he);
            if self.vertex_he[v.index()] == unused_he.into() {
                self.vertex_he[v.index()] = OptHalfedge::NONE;
            }
        }
        
        // Make the last edge unused
        let last_edge = Edge::from(self.num_edges() - 1);
        if last_edge != unused_edge && !self.is_invalid(last_edge) {
            for side in [0,1] {
                let unused_he = unused_edge.halfedge(side);
                let last_he = last_edge.halfedge(side);
                // update references to last_he
                let last_he_vertex = self.start(last_he);
                let last_he_prev = self.prev(last_he);
                let last_he_next = self.next(last_he);
                self.he_next[last_he_prev.index()] = unused_he;
                self.he_prev[last_he_next.index()] = unused_he;
                if let Some(face) = self.face(last_he) {
                    if self.face_he[face.index()] == last_he {
                        self.face_he[face.index()] = unused_he;
                    }
                }
                if self.vertex_he[last_he_vertex.index()] == last_he.into() {
                    self.vertex_he[last_he_vertex.index()] = unused_he.into();
                }
                // assign halfedge properties
                self.he_next[unused_he.index()] = self.he_next[last_he.index()];
                self.he_prev[unused_he.index()] = self.he_prev[last_he.index()];
                self.he_vertex[unused_he.index()] = self.he_vertex[last_he.index()];
                self.he_face[unused_he.index()] = self.he_face[last_he.index()];
            }
        }

        // Remove the last edge
        self.he_next.truncate(self.he_next.len() - 2);
        self.he_prev.truncate(self.he_prev.len() - 2);
        self.he_vertex.truncate(self.he_vertex.len() - 2);
        self.he_face.truncate(self.he_face.len() - 2);
    }

    // Remove a face that is not being used
    fn unchecked_remove_face(&mut self, unused_face: Face) {
        debug_assert!(!self.he_face.contains(&unused_face.into()), "Face should be unused");

        let last_face = Face::from(self.num_faces() - 1);
        if unused_face != last_face {
            // Move the content of last_face into face
            // Update references from halfedges in last_face to point to face
            self.assign_halfedge_in_loop_to_face(self.halfedge_on_face(last_face), unused_face.into());
            self.face_he[unused_face.index()] = self.face_he[last_face.index()];
        }

        // Now remove the last face
        self.face_he.pop();
    }

    // Remove a vertex that is not being used
    #[allow(unreachable_code)]
    fn unchecked_remove_vertex(&mut self, unused_vertex: Vertex) {
        debug_assert!(self.halfedges().all(|he| self.is_invalid(he.edge()) || self.start(he) != unused_vertex), "Vertex should be unused");

        let last_vertex = Vertex::from(self.num_vertices() - 1);
        if unused_vertex != last_vertex {
            // Update all references to last_vertex
            todo!("This needs a way to get incident halfedges");
        }
    }

    /// Remove a face by swapping it with the last face (as in Vec::swap_remove)
    /// Detaches all halfedges from that face
    pub fn swap_remove_face(&mut self, face: Face) {
        self.assign_halfedge_in_loop_to_face(self.halfedge_on_face(face), OptFace::NONE);
        self.unchecked_remove_face(face);
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
    pub fn peek(&self) -> Option<Halfedge> {
        self.item
    }
}
impl<'a> Iterator for HalfedgesOnFace<'a> {
    type Item = Halfedge;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.item;
        if let Some(pos) = item {
            let next = self.he_next[pos.index()];
            self.item = if next == self.start { None } else { Some(next) }
        }
        item
    }
}


/// Iterator for all halfedges that leave a vertex
pub struct HalfedgesLeavingVertex<'a> {
    start: OptHalfedge,
    item: OptHalfedge,
    he_next: &'a [Halfedge],
}
impl<'a> HalfedgesLeavingVertex<'a> {
    fn new(start: OptHalfedge, he_next: &'a [Halfedge]) -> Self {
        Self { start, item: start, he_next }
    }
    pub fn peek(&self) -> Option<Halfedge> {
        self.item.view()
    }
}
impl<'a> Iterator for HalfedgesLeavingVertex<'a> {
    type Item = Halfedge;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.item;
        if let Some(pos) = item.view() {
            let next: Halfedge = self.he_next[pos.opposite().index()];
            self.item = if OptHalfedge::from(next) == self.start { OptHalfedge::NONE } else { OptHalfedge::some(next) }
        }
        item.view()
    }
}


#[cfg(test)]
mod test {
    use super::*;

    // Partitions to use for tests
    fn test_cases() -> Vec<Partition> {
        let mut out = vec![];
        // simple polygons
        out.push(Partition::polygon(3));
        out.push(Partition::polygon(4));
        out.push(Partition::polygon(6));
        // a square cut into two triangles
        out.push(from_halfedges(
            &[0,1, 1,2, 2,3, 3,0, 0,2],
            &[1,0, 1,0, 2,0, 2,0, 2,1],
            &[2,7, 9,1, 6,3, 8,5, 4,0],
        ));
        // two nested triangles, connected with an edge
        out.push(from_halfedges(
            &[0,1, 1,2, 2,0, 0,3, 3,4, 4,5, 5,3],
            &[1,0, 1,0, 1,0, 1,1, 1,2, 1,2, 1,2],
            &[2,5, 4,1, 6,3, 8,0, 10,13, 12,9, 7,11],
        ));
        out
    }
    
    fn from_halfedges(he_vertex: &[usize], he_face: &[usize], he_next: &[usize]) -> Partition {
        let vertex_he = {
            let num_vertices = he_vertex.iter().max().unwrap() + 1;
            let mut vertex_he = vec![OptHalfedge::NONE; num_vertices];
            for (he, v) in he_vertex.iter().copied().enumerate() {
                vertex_he[v] = OptHalfedge::from(he);
            }
            vertex_he
        };
        let face_he = {
            let num_faces = *he_face.iter().max().unwrap();
            let mut face_he = vec![Halfedge(0); num_faces];
            for (he, f) in he_face.iter().copied().enumerate() {
                if f > 0 {
                    face_he[f - 1] = Halfedge::from(he);
                }
            }
            face_he
        };
        let he_prev = {
            let mut he_prev = vec![Halfedge(0); he_next.len()];
            for (i, j) in he_next.iter().copied().enumerate() {
                he_prev[j] = Halfedge::from(i);
            }
            he_prev
        };
        Partition {
            vertex_he: vertex_he,
            face_he:   face_he,
            he_vertex: he_vertex.iter().copied().map(Vertex::from).collect(),
            he_face:   he_face.iter().copied().map(|i| if i > 0 { OptFace::from(i-1) } else { OptFace::NONE }).collect(),
            he_next:   he_next.iter().copied().map(Halfedge::from).collect(),
            he_prev:   he_prev,
        }
    }

    #[test]
    fn test_cases_invariants() {
        for p in test_cases() {
            p.check_invariants();
        }
    }

    #[test]
    fn iterators() {
        let p = Partition::polygon(4);
        assert_eq!(p.vertices().len(), p.num_vertices());
        assert_eq!(p.halfedges().len(), p.num_halfedges());
        assert_eq!(p.edges().len(), p.num_edges());
        assert_eq!(p.num_halfedges(), p.num_edges() * 2);
        assert_eq!(p.halfedges_on_face(Face(0)).count(), 4);
        for vertex in p.vertices() {
            assert_eq!(p.halfedges_leaving_vertex(vertex).count(), 2);
        }
    }

    fn test_split_edge(mut p: Partition, edge: Edge) {
        p.split_edge(edge);
        p.check_invariants();
    }
    
    #[test]
    fn split_edge() {
        // Test all possible edge splits
        for p in test_cases() {
            for edge in p.edges() {
                if !p.is_boundary(edge) {
                    test_split_edge(p.clone(), edge);
                }
            }
        }
    }

    fn test_split_face(mut p: Partition, he1: Halfedge, he2: Halfedge) {
        let face = p.face(he1);
        let (new_edge, new_face) = p.split_face(he1, he2);
        p.check_invariants();
        assert_eq!(p.edge_faces(new_edge), (face, Some(new_face)));
    }

    #[test]
    fn split_face() {
        for p in test_cases() {
            // Test all possible face splits
            for face in p.faces() {
                for he1 in p.halfedges_on_face(face) {
                    for he2 in p.halfedges_on_face(face) {
                        if ![he1, p.next(he1), p.prev(he1)].contains(&he2) {
                            test_split_face(p.clone(), he1, he2);
                        }
                    }
                }
            }
        }
    }

    fn test_merge_face(mut p: Partition, edge: Edge) {
        p.merge_faces(edge);
        p.check_invariants();
        p.remove_invalid_edges(|_|{});
        p.check_invariants();
    }

    #[test]
    fn merge_face() {
        for p in test_cases() {
            for edge in p.edges() {
                let (face0, face1) = p.edge_faces(edge);
                if face0.is_some() && face1.is_some() && face0 != face1 {
                    test_merge_face(p.clone(), edge);
                }
            }
        }
    }
}

