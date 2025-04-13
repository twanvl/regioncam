// Python wrapper
use pyo3::prelude::*;
use pyo3::types::PyList;
use numpy::*;
use std::fs::File;
use std::fmt::Write;

use ndarray::{Axis, Dim, Dimension};
use pyo3::exceptions::{PyAttributeError, PyIndexError, PyValueError};
use pyo3::types::{PyDict, PyString, PyTuple};
use pyo3::{intern, PyClass, DowncastError, PyTraverseError, PyVisit};

use ::regioncam::*;

#[pymodule(name = "regioncam")]
mod regioncam {

    use super::*;

    #[pyclass]
    #[pyo3(name="Regioncam")]
    struct PyRegioncam {
        regioncam: Regioncam,
        plane: Option<Py<PyPlane>>,
        // options to use for svg visualization
        svg_options: SvgOptions,
        // points to include in visualizatoin
        marked_points: Vec<MarkedPoint>,
    }

    #[pymethods]
    impl PyRegioncam {
        /// Create a Regioncam object.
        ///
        /// Parameters:
        ///  * size: Size of the rectangle to cover.
        #[new]
        #[pyo3(signature=(size=1.0))]
        fn new(size: f32) -> Self {
            Self { regioncam: Regioncam::square(size), plane: None, svg_options: SvgOptions::default(), marked_points: vec![] }
        }
        /// Create a Regioncam object with a circular region
        #[staticmethod]
        fn circle(radius: f32) -> Self {
            Self { regioncam: Regioncam::circle(radius), plane: None, svg_options: SvgOptions::default(), marked_points: vec![] }
        }
        /// Create a Regioncam object for a plane through the given points.
        #[staticmethod]
        #[pyo3(signature=(points, labels=vec![]))]
        fn through_points<'py>(#[pyo3(from_py_with="downcast_array")] points: Bound<'py, PyArray2<f32>>, mut labels: Vec<String>) -> PyResult<Self> {
            let plane = Plane::through_points(&points.readonly().as_array());
            let regioncam = Regioncam::from_plane(&plane);
            labels.resize(plane.points.nrows(), String::new());
            let marked_points = plane.points.rows().into_iter().zip(labels.into_iter()).map(
                    |(point,label)| {
                        let position = point.as_slice().unwrap().try_into().unwrap();
                        let color = None;
                        MarkedPoint { position, label, color }
                    }
                ).collect();
            let plane = Py::new(points.py(), PyPlane(plane))?;
            Ok(Self { regioncam, plane: Some(plane), svg_options: SvgOptions::default(), marked_points })
        }

        /// Mark points in the input space.
        /// 
        /// If this Regioncam was constructed with a plane through points,
        ///  then these points are projected onto that plane.
        #[pyo3(signature=(points, labels=vec![], project_to_plane=true))]
        fn mark_points<'py>(&mut self, #[pyo3(from_py_with="downcast_array")] points: Bound<'py, PyArray2<f32>>, mut labels: Vec<String>, project_to_plane: bool) -> PyResult<()> {
            let py = points.py();
            let points = points.readonly();
            let points = points.as_array();
            labels.resize(points.nrows(), String::new());
            let projected_points =
                if let Some(plane) = &self.plane {
                    if project_to_plane {
                        Some(plane.borrow(py).0.inverse(&points.view()))
                    } else { None }
                } else { None };
            let points = projected_points.as_ref().map_or(points, |x| x.view());
            self.marked_points.extend(points.rows().into_iter().zip(labels.into_iter()).map(
                    |(point,label)| {
                        let position = point.as_slice().unwrap().try_into().unwrap();
                        let color = None;
                        MarkedPoint { position, label, color }
                    }
                ));
            Ok(())
        }
        
        fn remove_markers(&mut self) {
            self.marked_points.clear();
        }

        /// The number of vertices in the partition.
        /// Equivalent to `len(self.vertices)`
        #[getter]
        fn num_vertices(&self) -> usize {
            self.regioncam.num_vertices()
        }
        /// The number of edges in the partition.
        /// Equivalent to `len(self.edges)`
        #[getter]
        fn num_edges(&self) -> usize {
            self.regioncam.num_edges()
        }
        /// The number of faces in the partition.
        /// Equivalent to `len(self.faces)`
        #[getter]
        fn num_faces(&self) -> usize {
            self.regioncam.num_faces()
        }
        /// The number of layers
        /// Equivalent to `len(self.layers)`
        #[getter]
        fn num_layers(&self) -> usize {
            self.regioncam.num_layers()
        }
        /// The last layer
        /// Equivalent to `self.layers[-1]`
        #[getter]
        fn last_layer<'py>(slf: Bound<'py, PyRegioncam>) -> PyLayer {
            let layer = slf.borrow().regioncam.last_layer();
            PyLayer { rc: slf.unbind(), layer }
        }

        /// The 2D input coordinates for all vertices.
        /// Equivalent to `self.layers[0].vertex_activations`
        #[getter]
        fn vertex_inputs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
            self.regioncam.inputs().to_pyarray(py)
        }
        /// The output values / activations in the last layer, for all vertices.
        /// Equivalent to `self.layers[-1].vertex_activations`
        #[getter]
        fn vertex_outputs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
            self.regioncam.activations_last().to_pyarray(py)
        }
        /// The output values /  for all vertices, at a given layer
        fn vertex_activations<'py>(&self, py: Python<'py>, layer: usize) -> Bound<'py, PyArray2<f32>> {
            self.regioncam.activations(layer).to_pyarray(py)
        }

        /// The output values for all faces.
        /// A point x in face f has output value  [x[0],x[1],1] @ face_outputs[f]
        #[getter]
        fn face_outputs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f32>> {
            self.regioncam.face_activations().to_pyarray(py)
        }
        fn face_activations<'py>(&self, py: Python<'py>, layer: usize) -> Bound<'py, PyArray3<f32>> {
            self.regioncam.face_activations_at(layer).to_pyarray(py)
        }

        /// Sequence of all faces / regions
        #[getter]
        fn faces(slf: Py<PyRegioncam>) -> Faces {
            Faces(slf)
        }
        /// Sequence of all vertices
        #[getter]
        fn vertices(slf: Py<PyRegioncam>) -> Vertices {
            Vertices(slf)
        }
        /// Sequence of all edges
        #[getter]
        fn edges(slf: Py<PyRegioncam>) -> Edges {
            Edges(slf)
        }
        /// Sequence of all layers
        #[getter]
        fn layers(slf: Py<PyRegioncam>) -> Layers {
            Layers(slf)
        }

        /// Mapping from 2d space
        #[getter]
        fn plane<'py>(&self, py: Python<'py>) -> PyResult<Py<PyPlane>> {
            match &self.plane {
                Some(plane) => Ok(plane.clone_ref(py)),
                None => Err(PyAttributeError::new_err("Not constructed from a plane")),
            }
        }
        
        /// Visualize the regions, and write this to an svg file.
        ///
        /// Parameters:
        ///  * path:        path of the file to write to
        ///  * size:        width and height of the image in pixels
        ///  * line_width:  line width to use for edges
        #[pyo3(signature=(path, **kwargs))]
        fn write_svg(&self, path: &str, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
            let svg_opts = parse_svg_options(&self.svg_options, kwargs)?;
            let mut file = File::create(path)?;
            self.do_write_svg(svg_opts, &mut file)?;
            Ok(())
        }

        /// Get an svg representation of the regions.
        ///
        /// Parameters:
        ///  * size:        width and height of the image in pixels
        ///  * line_width:  line width to use for edges
        #[pyo3(signature=(**kwargs))]
        fn _repr_svg_(&self, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<String> {
            let svg_opts = parse_svg_options(&self.svg_options, kwargs)?;
            let mut out = vec![];
            self.do_write_svg(svg_opts, &mut out)?;
            Ok(String::from_utf8(out)?)
        }

        /// Set svg format options
        #[pyo3(signature=(**kwargs))]
        fn set_format(&mut self, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
            self.svg_options = parse_svg_options(&self.svg_options, kwargs)?;
            Ok(())
        }

        // layers

        /// Add a new layer that applies a ReLU operation.
        ///
        /// Parameters:
        ///  * `input_layer``: use the output of the given layer as input (default: last layer).
        #[pyo3(signature=(input_layer=None))]
        fn relu(&mut self, input_layer: Option<usize>) -> usize {
            let input_layer = input_layer.unwrap_or(self.regioncam.last_layer());
            self.regioncam.relu_at(input_layer)
        }

        /// Add a new layer that applies a LeakyReLU operation.
        ///
        /// Parameters:
        ///  * `negative_slope`: multiplicative factor for negative inputs.
        ///  * `input_layer``: use the output of the given layer as input (default: last layer).
        #[pyo3(signature=(negative_slope, input_layer=None))]
        fn leaky_relu(&mut self, negative_slope: f32, input_layer: Option<usize>) -> usize {
            let input_layer = input_layer.unwrap_or(self.regioncam.last_layer());
            self.regioncam.leaky_relu_at(input_layer, negative_slope)
        }

        /// Add a max pooling layer.
        ///
        /// Parameters:
        ///  * `input_layer``: use the output of the given layer as input (default: last layer).
        #[pyo3(signature=(input_layer=None))]
        fn max_pool(&mut self, input_layer: Option<usize>) -> usize {
            let input_layer = input_layer.unwrap_or(self.regioncam.last_layer());
            self.regioncam.max_pool_at(input_layer)
        }

        /// Add an argmax pooling layer. This can be used as a classification like softmax.
        ///
        /// Parameters:
        ///  * `input_layer``: use the output of the given layer as input (default: last layer).
        #[pyo3(signature=(input_layer=None))]
        fn argmax_pool(&mut self, input_layer: Option<usize>) -> usize {
            let input_layer = input_layer.unwrap_or(self.regioncam.last_layer());
            self.regioncam.argmax_pool_at(input_layer)
        }

        /// Add an argmax pooling layer. This can be used as a classification like softmax.
        ///
        /// Parameters:
        ///  * `input_layer``: use the output of the given layer as input (default: last layer).
        #[pyo3(signature=(input_layer=None))]
        fn sign(&mut self, input_layer: Option<usize>) -> usize {
            let input_layer = input_layer.unwrap_or(self.regioncam.last_layer());
            self.regioncam.sign_at(input_layer)
        }

        /// Add a decision boundary.
        /// If the last layer has 1 ouptut use a sign classifier (hard sigmoid).
        /// If the last layer has >1 output use an argmax classifier (hard softmax).
        ///
        /// Parameters:
        ///  * `input_layer``: use the output of the given layer as input (default: last layer).
        #[pyo3(signature=(input_layer=None))]
        fn decision_boundary(&mut self, input_layer: Option<usize>) -> usize {
            let input_layer = input_layer.unwrap_or(self.regioncam.last_layer());
            self.regioncam.decision_boundary_at(input_layer)
        }

        /// Add a new layer that applies a linear transformation,
        ///   x_out = x_in @ weight + bias
        ///
        /// Parameters:
        ///  * `weight`:       weight matrix
        ///  * `bias`:         bias vector
        ///  * `input_layer``: use the output of the given layer as input (default: last layer).
        #[pyo3(signature=(weight, bias, input_layer=None))]
        /*fn linear<'py>(&mut self, #[pyo3(from_py_with="downcast_array")] weight: PyReadonlyArray2<'py, f32>, bias: PyReadonlyArray1<'py, f32>, input_layer: Option<usize>) {
            let input_layer = input_layer.unwrap_or(self.partition.last_layer());
            self.partition.linear_at(input_layer, &weight.as_array(), &bias.as_array());
        }*/
        fn linear<'py>(&mut self,
                #[pyo3(from_py_with="downcast_array")] weight: Bound<'py, PyArray2<f32>>,
                #[pyo3(from_py_with="downcast_array")] bias: Bound<'py, PyArray1<f32>>,
                input_layer: Option<usize>
            ) -> usize {
            let input_layer = input_layer.unwrap_or(self.regioncam.last_layer());
            self.regioncam.linear_at(input_layer, &weight.readonly().as_array(), &bias.readonly().as_array())
        }

        /// Add a neural network or network layer.
        /// This function accepts layers from `regioncam.nn`, and lists of layers.
        ///
        /// Parameters:
        ///  * `layer`:        network or layer(s) to apply
        ///  * `input_layer``: use the output of the given layer as input (default: last layer).
        /// Returns:
        ///  * layer number of the output
        #[pyo3(signature=(layer, input_layer=None))]
        fn add<'a,'py>(&mut self, py: Python<'py>, #[pyo3(from_py_with="nn::to_layer")] layer: PyCow<'a, 'py, PyNNModule>, input_layer: Option<usize>) -> PyResult<usize> {
            self.add_layer(py, &layer.borrow(), input_layer)
        }
    }

    impl PyRegioncam {
        fn do_write_svg(&self, svg_opts: SvgOptions, mut w: &mut dyn std::io::Write) -> std::io::Result<()> {
            let mut writer = SvgWriter::new(&self.regioncam, &svg_opts);
            writer.points = &self.marked_points;
            writer.write_svg(&mut w)
        }

        fn add_layer<'py>(&mut self, py: Python<'py>, module: &PyNNModule, input_layer: Option<usize>) -> PyResult<usize> {
            let input_layer = input_layer.unwrap_or(self.regioncam.last_layer());
            match module {
                PyNNModule::Linear { weight, bias } => {
                    let weight = weight.bind(py).readonly();
                    let bias   = bias.bind(py).readonly();
                    Ok(self.regioncam.linear_at(input_layer, &weight.as_array(), &bias.as_array()))
                }
                PyNNModule::ReLU() => {
                    self.regioncam.relu_at(input_layer);
                    Ok(self.regioncam.last_layer())
                }
                PyNNModule::LeakyReLU { negative_slope } => {
                    self.regioncam.leaky_relu_at(input_layer, *negative_slope);
                    Ok(self.regioncam.last_layer())
                }
                PyNNModule::Residual { layer } => {
                    let after_layer = self.add_layer(py, layer.get(), Some(input_layer))?;
                    self.regioncam.sum(input_layer, after_layer);
                    Ok(self.regioncam.last_layer())
                }
                PyNNModule::Sequential { layers } => {
                    let mut input_layer = input_layer;
                    for item in layers {
                        input_layer = self.add_layer(py, item.get(), Some(input_layer))?;
                    }
                    Ok(input_layer)
                }
            }
        }
    }


    /// A face in a regioncam
    #[pyclass(name="Face")]
    pub struct PyFace {
        rc: Py<PyRegioncam>,
        face: Face,
    }
    #[pymethods]
    impl PyFace {
        fn __index__(&self) -> usize {
            self.face.index()
        }
        fn __repr__(&self) -> String {
            format!("Face({})", self.face.index())
        }
        /// The list of vertices that make up this face
        #[getter]
        fn vertices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
            let pyrc = self.rc.bind(py);
            let rc = pyrc.borrow();
            let vertices = rc.regioncam.vertices_on_face(self.face);
            //let vertices = vertices.map(|vertex| PyVertex{ rc: pyrc.clone().unbind(), vertex});
            let vertices = vertices.map(|vertex| PyVertex{ rc: self.rc.clone_ref(py), vertex});
            PyList::new(py, vertices.collect::<Vec<_>>())
        }
        /// Ids of all vertices that make up this face
        #[getter]
        fn vertex_ids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<usize>> {
            let rc = self.rc.borrow(py);
            let vertex_ids = rc.regioncam.vertices_on_face(self.face).map(usize::from);
            PyArray1::from_iter(py, vertex_ids)
        }
        /// Values at a given layer
        fn activations<'py>(&self, py: Python<'py>, layer: usize) -> Bound<'py, PyArray2<f32>> {
            let rc = self.rc.borrow(py);
            let row = rc.regioncam.face_activations_at(layer).index_axis(Axis(0), self.face.index());
            row.to_pyarray(py)
        }
        /// Input values
        #[getter]
        fn inputs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
            self.activations(py, 0)
        }
        /// Output values
        #[getter]
        fn outputs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
            let rc = self.rc.borrow(py);
            self.activations(py, rc.regioncam.last_layer())
        }
        fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
            visit.call(&self.rc)
        }
    }

    /// A vertex in a regioncam
    #[pyclass]
    pub struct PyVertex {
        rc: Py<PyRegioncam>,
        vertex: Vertex,
    }
    #[pymethods]
    impl PyVertex {
        fn __index__(&self) -> usize {
            self.vertex.index()
        }
        fn __repr__(&self) -> String {
            format!("Vertex({})", self.vertex.index())
        }
        /// Values at a given layer
        fn activations<'py>(&self, py: Python<'py>, layer: usize) -> Bound<'py, PyArray1<f32>> {
            let rc = self.rc.borrow(py);
            let row = rc.regioncam.activations(layer).row(self.vertex.index());
            row.to_pyarray(py)
        }
        /// Input values
        #[getter]
        fn inputs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
            self.activations(py, 0)
        }
        /// Output values
        #[getter]
        fn outputs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
            let rc = self.rc.borrow(py);
            self.activations(py, rc.regioncam.last_layer())
        }
        /// Is this vertex on the outer boundary of the regioncam?
        #[getter]
        fn on_image_boundary<'py>(&self, py: Python<'py>) -> bool {
            let rc: PyRef<'_, PyRegioncam> = self.rc.borrow(py);
            rc.regioncam.halfedges_leaving_vertex(self.vertex).any(
                |he| rc.regioncam.is_boundary(he.edge())
            )
        }
        /// Is this vertex on the decision boundary?
        #[getter]
        fn on_decision_boundary<'py>(&self, py: Python<'py>) -> bool {
            let rc: PyRef<'_, PyRegioncam> = self.rc.borrow(py);
            let decision_boundary_layer = rc.regioncam.last_layer();
            rc.regioncam.halfedges_leaving_vertex(self.vertex).any(
                |he| rc.regioncam.edge_label(he.edge()).layer == decision_boundary_layer
            )
        }
        fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
            visit.call(&self.rc)
        }
    }

    /// A vertex in a regioncam
    #[pyclass]
    pub struct PyEdge {
        rc: Py<PyRegioncam>,
        edge: Edge,
    }
    #[pymethods]
    impl PyEdge {
        fn __index__(&self) -> usize {
            self.edge.index()
        }
        fn __repr__(&self) -> String {
            format!("Edge({})", self.edge.index())
        }
        #[getter]
        fn vertices<'py>(&self, py: Python<'py>) -> [PyVertex; 2] {
            let rc = self.rc.borrow(py);
            let endpoints = rc.regioncam.endpoints(self.edge);
            [PyVertex{ rc: self.rc.clone_ref(py), vertex: endpoints.0 },
             PyVertex{ rc: self.rc.clone_ref(py), vertex: endpoints.1 }]
        }
        #[getter]
        fn faces<'py>(&self, py: Python<'py>) -> [Option<PyFace>; 2] {
            let rc: PyRef<'_, PyRegioncam> = self.rc.borrow(py);
            let faces = rc.regioncam.edge_faces(self.edge);
            [faces.0.map(|face| PyFace{ rc: self.rc.clone_ref(py), face }),
             faces.1.map(|face| PyFace{ rc: self.rc.clone_ref(py), face })]
        }
        #[getter]
        fn is_image_boundary<'py>(&self, py: Python<'py>) -> bool {
            let rc: PyRef<'_, PyRegioncam> = self.rc.borrow(py);
            rc.regioncam.is_boundary(self.edge)
        }
        #[getter]
        fn is_decision_boundary<'py>(&self, py: Python<'py>) -> bool {
            let rc: PyRef<'_, PyRegioncam> = self.rc.borrow(py);
            let decision_boundary_layer = rc.regioncam.last_layer();
            rc.regioncam.edge_label(self.edge).layer == decision_boundary_layer
        }
        /// In which layer was this edge added?
        #[getter]
        fn layer_nr<'py>(&self, py: Python<'py>) -> usize {
            let rc: PyRef<'_, PyRegioncam> = self.rc.borrow(py);
            rc.regioncam.edge_label(self.edge).layer
        }
        #[getter]
        fn layer<'py>(&self, py: Python<'py>) -> PyLayer {
            let layer = self.layer_nr(py);
            PyLayer{ rc: self.rc.clone_ref(py), layer }
        }
        fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
            visit.call(&self.rc)
        }
    }

    /// A layer in a regioncam
    #[pyclass]
    pub struct PyLayer {
        rc: Py<PyRegioncam>,
        layer: LayerNr,
    }
    #[pymethods]
    impl PyLayer {
        fn __index__(&self) -> usize {
            self.layer
        }
        fn __repr__(&self) -> String {
            format!("Layer({})", self.layer)
        }
        /// Vertex values
        #[getter]
        fn vertex_activations<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
            let rc = self.rc.borrow(py);
            rc.regioncam.activations(self.layer).to_pyarray(py)
        }
        #[getter]
        fn face_activations<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f32>> {
            let rc = self.rc.borrow(py);
            rc.regioncam.face_activations_at(self.layer).to_pyarray(py)
        }
        fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
            visit.call(&self.rc)
        }
    }
    
    // Collections

    /// Wrap an indexable collection as a Python sequence
    macro_rules! declare_pysequence {
        ($list:ident, $num_items:ident, $type:ty, $ctor:expr) => {
            #[pyclass]
            pub struct $list(Py<PyRegioncam>);
            #[pymethods]
            impl $list {
                fn __len__<'py>(&self, py: Python<'py>) -> usize {
                    self.0.borrow(py).$num_items()
                }
                fn __getitem__<'py>(&self, py: Python<'py>, index: isize) -> PyResult<$type> {
                    let len = self.__len__(py);
                    let index = if index < 0 {
                        (len as isize + index) as usize
                    } else {
                        index as usize
                    };
                    if index < len {
                        let rc = self.0.clone_ref(py);
                        Ok($ctor(rc, index))
                    } else {
                        Err(PyIndexError::new_err("index out of range"))
                    }
                }
            }
        };
    }

    declare_pysequence!(Faces, num_faces, PyFace, (|rc, index| PyFace { rc, face: Face::from(index) }));
    declare_pysequence!(Vertices, num_vertices, PyVertex, (|rc, index| PyVertex { rc, vertex: Vertex::from(index) }));
    declare_pysequence!(Edges, num_edges, PyEdge, (|rc, index| PyEdge { rc, edge: Edge::from(index) }));
    declare_pysequence!(Layers, num_layers, PyLayer, (|rc, index| PyLayer { rc, layer: index }));


    /// A plane through points in a high dimensional space
    #[pyclass]
    #[pyo3(frozen, name="Plane")]
    struct PyPlane(Plane);
    #[pymethods]
    impl PyPlane {
        #[new]
        fn new<'py>(#[pyo3(from_py_with="downcast_array")] points: Bound<'py, PyArray2<f32>>) -> Self {
            let plane = Plane::through_points(&points.readonly().as_array());
            PyPlane(plane)
        }
        #[staticmethod]
        fn from_linear<'py>(#[pyo3(from_py_with="downcast_array")] weight: Bound<'py, PyArray2<f32>>, #[pyo3(from_py_with="downcast_array")] bias: Bound<'py, PyArray1<f32>>) -> Self {
            let plane = Plane::from(::regioncam::nn::Linear{ weight: weight.to_owned_array(), bias: bias.to_owned_array() });
            PyPlane(plane)
        }

        fn forward<'py>(&self, #[pyo3(from_py_with="downcast_array")] points: Bound<'py, PyArray2<f32>>) -> Bound<'py, PyArray2<f32>> {
            self.0.forward(&points.readonly().as_array()).to_pyarray(points.py())
        }
        fn inverse<'py>(&self, #[pyo3(from_py_with="downcast_array")] points: Bound<'py, PyArray2<f32>>) -> Bound<'py, PyArray2<f32>> {
            self.0.inverse(&points.readonly().as_array()).to_pyarray(points.py())
        }
        #[getter]
        fn weight<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
            self.0.mapping.weight.to_pyarray(py)
        }
        #[getter]
        fn bias<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
            self.0.mapping.bias.to_pyarray(py)
        }
    }
    
    /// A layer in a neural network
    #[pyclass]
    #[pyo3(frozen, name="Module")]
    pub enum PyNNModule {
        /// A linear transformation
        Linear {
            weight: Py<PyArray2<f32>>,
            bias: Py<PyArray1<f32>>,
        },
        /// A rectified linear activation function
        ReLU(),
        /// A leaky relu activation function
        LeakyReLU {
            negative_slope: f32,
        },
        /// A residual connection around a network layer
        Residual {
            layer: Py<PyNNModule>,
        },
        /// A sequence of network layers
        Sequential {
            layers: Vec<Py<PyNNModule>>,
        },
    }

    #[pymethods]
    impl PyNNModule {
        fn __repr__<'py>(&self, py: Python<'py>) -> PyResult<String> {
            let mut out = String::new();
            self.to_string(py, 0, &mut out)?;
            Ok(out)
        }
        /// Numer of input features
        #[getter]
        fn in_features<'py>(&self, py: Python<'py>) -> PyResult<usize> {
            Ok(self.shape(py)?[0])
        }
        /// Numer of output features
        #[getter]
        fn out_features<'py>(&self, py: Python<'py>) -> PyResult<usize> {
            Ok(self.shape(py)?[1])
        }
        // GC traversal
        fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
            match self {
                PyNNModule::Linear {weight, bias} => {
                    visit.call(weight)?;
                    visit.call(bias)?;
                },
                PyNNModule::Residual {layer} => {
                    visit.call(layer)?;
                },
                PyNNModule::Sequential {layers} => {
                    for layer in layers {
                        visit.call(layer)?;
                    }
                }
                _ => ()
            }
            Ok(())
        }
    }

    impl PyNNModule {
        fn shape<'py>(&self, py: Python<'py>) -> PyResult<Dim<[usize;2]>> {
            match self {
                PyNNModule::Linear { weight, .. } => Ok(weight.bind(py).dims()),
                _ => Err(PyAttributeError::new_err("shape undefined for this layer type"))
            }
        }
        fn to_string<'py>(&self, py: Python<'py>, indent: usize, w: &mut String) -> PyResult<()> {
            fn write_indent(w: &mut String, indent: usize) {
                for _ in 0..indent {
                    write!(w, "  ").unwrap();
                }
            }
            match self {
                PyNNModule::Linear { .. } => {
                    let shape = self.shape(py)?;
                    writeln!(w, "Linear(in_features={}, out_features={})", shape[0], shape[1]).unwrap();
                }
                PyNNModule::ReLU() => {
                    writeln!(w, "ReLU").unwrap();
                }
                PyNNModule::LeakyReLU { negative_slope } => {
                    writeln!(w,"LeakyReLU({negative_slope})").unwrap();
                }
                PyNNModule::Residual { layer } => {
                    writeln!(w, "Residual(").unwrap();
                    write_indent(w, indent+1);
                    layer.get().to_string(py, indent + 1, w)?;
                    write_indent(w, indent);
                    writeln!(w, ")").unwrap();
                }
                PyNNModule::Sequential { layers } => {
                    writeln!(w, "Sequential(").unwrap();
                    for (layer, i) in layers.iter().zip(0..) {
                        write_indent(w, indent + 1);
                        write!(w, "({i}): ").unwrap();
                        layer.get().to_string(py, indent + 1, w)?;
                    }
                    write_indent(w, indent);
                    writeln!(w, ")").unwrap();
                }
            }
            Ok(())
        }
    }

    #[pymodule]
    mod nn {
        use super::*;

        /// A linear transformation
        #[pyfunction]
        #[pyo3(name="Linear")]
        fn linear<'py>(
                #[pyo3(from_py_with="downcast_array")] weight: Bound<'py, PyArray2<f32>>,
                #[pyo3(from_py_with="downcast_array")] bias: Bound<'py, PyArray1<f32>>,
            ) -> PyNNModule {
            assert_eq!(weight.dims()[1], bias.dims()[0]);
            PyNNModule::Linear { weight: weight.unbind(), bias: bias.unbind() }
        }

        /// A rectified linear activation function
        #[pyfunction]
        #[pyo3(name="ReLU")]
        fn relu() -> PyNNModule {
            PyNNModule::ReLU()
        }

        /// A leaky relu activation function
        #[pyfunction]
        #[pyo3(name="LeakyReLU", signature=(negative_slope=1e-2))]
        fn leaky_relu(negative_slope: f32) -> PyNNModule {
            PyNNModule::LeakyReLU { negative_slope }
        }

        /// A leaky relu activation function
        #[pyfunction]
        #[pyo3(name="Residual")]
        fn residual(layer: Py<PyNNModule>) -> PyNNModule {
            PyNNModule::Residual { layer }
        }

        /// A sequence of network layers
        #[pyfunction]
        #[pyo3(name="Sequential", signature=(*layers))]
        fn sequential(layers: Vec<Py<PyNNModule>>) -> PyNNModule {
            PyNNModule::Sequential { layers }
        }

        /// An identity layer
        #[pyfunction]
        #[pyo3(name="Identity")]
        fn identity() -> PyNNModule {
            PyNNModule::Sequential { layers: vec![] }
        }

        /// Convert a torch layer into a Regioncam layer
        #[pyfunction]
        fn convert<'py>(arg: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyNNModule>> {
            to_layer(arg)?.into_bound(arg.py())
        }

        pub fn to_layer<'a,'py>(arg: &'a Bound<'py, PyAny>) -> PyResult<PyCow<'a, 'py, PyNNModule>> {
            if let Ok(layer) = arg.downcast() {
                Ok(PyCow::Bound(layer))
            } else if qualname(arg) == "torch.nn.modules.linear.Linear" {
                let weight = arg.getattr(intern!(arg.py(), "weight"))?;
                let weight = downcast_array(&weight)?;
                let bias = arg.getattr(intern!(arg.py(), "bias"))?;
                let bias = downcast_array(&bias)?;
                // pytorch has transposed weight matrix
                let weight = weight.transpose()?;
                Ok(PyCow::Owned(linear(weight, bias)))
            } else if qualname(arg) == "torch.nn.modules.activation.ReLU" {
                Ok(PyCow::Owned(relu()))
            } else if qualname(arg) == "torch.nn.modules.activation.LeakyReLU" {
                let negative_slope = arg.getattr(intern!(arg.py(), "negative_slope"))?.extract()?;
                Ok(PyCow::Owned(leaky_relu(negative_slope)))
            } else if qualname(arg) == "torch.nn.modules.linear.Identity" {
                Ok(PyCow::Owned(identity()))
            } else if qualname(arg) == "torch.nn.modules.dropout.Dropout" ||
                      qualname(arg) == "torch.nn.modules.dropout.Dropout1d" ||
                      qualname(arg) == "torch.nn.modules.dropout.Dropout2d" {
                // Torch dropout behaves like a linear layer when not training,
                // So replace dropout layers with identity layers.
                Ok(PyCow::Owned(identity()))
            } else if qualname(arg) == "torch.nn.modules.container.Sequential" {
                let mut layers = Vec::new();
                for item in arg.try_iter()? {
                    let item = item?;
                    let layer = to_layer(&item)?;
                    layers.push(layer.into_bound(arg.py())?.unbind());
                }
                Ok(PyCow::Owned(sequential(layers)))
            } else if let Ok(list) = arg.downcast::<PyList>() {
                // convert a list to a sequential layer
                let mut layers = Vec::new();
                for item in list {
                    let layer = to_layer(&item)?;
                    layers.push(layer.into_bound(arg.py())?.unbind());
                }
                Ok(PyCow::Owned(sequential(layers)))
            } else {
                Err(DowncastError::new(arg, "Layer").into())
            }
        }
    }

    #[pymodule_init]
    fn init<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {
        // Re-export layer constructors
        let layer_cls = m.getattr("nn")?;
        for name in ["Linear","ReLU","LeakyReLU","Residual","Sequential"] {
            m.add(name, layer_cls.getattr(name)?)?;
        }
        Ok(())
    }

    enum PyCow<'a, 'py, T> {
        Bound(&'a Bound<'py, T>),
        Owned(T),
    }
    impl<'a, 'py, T: PyClass + Into<PyClassInitializer<T>>> PyCow<'a, 'py, T> {
        fn into_bound(self, py: Python<'py>) -> PyResult<Bound<'py, T>> {
            match self {
                PyCow::Bound(value) => Ok(value.clone()),
                PyCow::Owned(value) => Bound::new(py, value),
            }
        }
        fn borrow(&'a self) -> PyCowRef<'a,T> {
            match self {
                PyCow::Bound(value) => PyCowRef::Bound(value.borrow()),
                PyCow::Owned(value) => PyCowRef::Owned(value),
            }
        }
    }
    enum PyCowRef<'a, T: PyClass> {
        Bound(PyRef<'a, T>),
        Owned(&'a T),
    }
    impl<'a, T: PyClass> std::ops::Deref for PyCowRef<'a, T> {
        type Target = T;
        fn deref(&self) -> &Self::Target {
            match self {
                PyCowRef::Bound(value) => value,
                PyCowRef::Owned(value) => value,
            }
        }
    }

    fn qualname<'py>(arg: &'py Bound<'py, PyAny>) -> Bound<'py, PyString> {
        arg.get_type().fully_qualified_name().unwrap_or_else(|_| PyString::new(arg.py(), ""))
    }
    // Downcast a python object to an array.
    // Supports torch tensors in addition to numpy arrays
    fn downcast_array<'py,A,Ix>(arg: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyArray<A,Ix>>>
        where
        A: Element, Ix: Dimension
    {
        if let Ok(arr) = arg.downcast() {
            Ok(arr.clone())
        } else if qualname(&arg) == "torch.nn.parameter.Parameter" {
            let arg = arg.getattr(intern!(arg.py(), "data"))?;
            downcast_array(&arg)
        } else if qualname(&arg) == "torch.Tensor" {
            // convert from torch to numpy
            let arg = arg.call_method0(intern!(arg.py(), "detach"))?;
            let arg = arg.call_method0(intern!(arg.py(), "cpu"))?;
            let arg = arg.call_method0(intern!(arg.py(), "numpy"))?;
            downcast_array(&arg)
        } else {
            let dtype = numpy::dtype::<A>(arg.py()).str();
            let dtype = dtype.as_ref().unwrap_or(intern!(arg.py(),"unknown"));
            let typename = match Ix::NDIM {
                Some(ndim) => format!("PyArray{ndim}(dtype={})", dtype),
                None => format!("PyArrayAny(dtype={})", dtype),
            };
            Err(DowncastError::new(arg, typename).into())
        }
    }
    
    fn parse_svg_options<'py>(opts: &SvgOptions, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<SvgOptions> {
        let mut opts = opts.clone();
        if let Some(kwargs) = kwargs {
            for (k, v) in kwargs {
                let k = k.downcast::<PyString>()?.to_str()?;
                if k == "line_width" {
                    opts.line_width = v.extract()?;
                } else if k == "draw_boundary" {
                    opts.draw_boundary = v.extract()?;
                } else if k == "line_width_decision_boundary" || k == "decision_boundary_line_width" {
                    opts.line_width_decision_boundary = v.extract()?;
                } else if k == "image_size" || k == "size" {
                    if let Ok(v) = v.downcast::<PyTuple>() {
                        opts.image_size = v.extract()?;
                    } else {
                        let size = v.extract()?;
                        opts.image_size = (size, size);
                    }
                } else if k == "point_size" || k == "marker_size" {
                    opts.point_size = v.extract()?;
                } else if k == "label_size" || k == "text_size" || k == "font_size" {
                    opts.font_size = v.extract()?;
                } else if k == "color_edges" || k == "line_color_amount" {
                    opts.line_color_amount = v.extract()?;
                } else if k == "line_color_by_layer" {
                    opts.line_color_by_layer = v.extract()?;
                } else if k == "line_color_by_neuron" {
                    opts.line_color_by_neuron = v.extract()?;
                } else if k == "face_color_by_layer" {
                    opts.face_color_by_layer = v.extract()?;
                } else if k == "line_color" {
                    opts.line_color = extract_color(&v)?;
                } else if k == "line_color_decision_boundary" || k == "decision_boundary_color" {
                    opts.line_color = extract_color(&v)?;
                } else if k == "layer_line_colors" || k == "line_colors" {
                    opts.layer_line_colors = extract_colors(&v)?;
                    opts.line_color_by_layer = true;
                } else {
                    return Err(PyValueError::new_err(format!("Unexpected argument: '{k}'")));
                }
            }
        }
        Ok(opts)
    }
    
    fn extract_colors<'py>(arg: &Bound<'py, PyAny>) -> PyResult<Vec<ColorF32>> {
        let list = arg.extract::<Vec<Bound<'py, PyAny>>>()?;
        list.iter().map(extract_color).collect()
    }
    fn extract_color<'py>(arg: &Bound<'py, PyAny>) -> PyResult<ColorF32> {
        if let Ok((r, g, b)) = arg.extract() {
            Ok(ColorF32::new(r, g, b))
        } else if let Some(color) = arg.extract().ok().and_then(|s: String| named_color(&s)) {
            Ok(color)
        } else {
            Err(DowncastError::new(arg, "Tuple[Float,Float,Float] | String").into())
        }
    }
    fn named_color(name: &str) -> Option<ColorF32> {
        match name {
            "black"   => Some(ColorF32::new(0.0, 0.0, 0.0)),
            "red"     => Some(ColorF32::new(1.0, 0.0, 0.0)),
            "green"   => Some(ColorF32::new(0.0, 1.0, 0.0)),
            "blue"    => Some(ColorF32::new(0.0, 0.0, 1.0)),
            "yellow"  => Some(ColorF32::new(1.0, 1.0, 0.0)),
            "magenta" => Some(ColorF32::new(1.0, 0.0, 1.0)),
            "cyan"    => Some(ColorF32::new(0.0, 1.0, 1.0)),
            "white"   => Some(ColorF32::new(1.0, 1.0, 1.0)),
            _ => None,
        }
    }
}