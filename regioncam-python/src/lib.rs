// Python wrapper
use std::ops::Range;
use std::fs::File;
use std::fmt::Write;

use ndarray::{Axis, Dim, Dimension, CowArray};
use numpy::*;
use pyo3::prelude::*;
use pyo3::exceptions::{PyAttributeError, PyIndexError, PyValueError};
use pyo3::types::{PyDict, PyList, PyString, PyTuple, PyBytes};
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
        render_options: RenderOptions,
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
            Self { regioncam: Regioncam::square(size), plane: None, render_options: RenderOptions::default(), marked_points: vec![] }
        }
        /// Create a Regioncam object with a circular region
        #[staticmethod]
        fn circle(radius: f32) -> Self {
            Self { regioncam: Regioncam::circle(radius), plane: None, render_options: RenderOptions::default(), marked_points: vec![] }
        }
        /// Create a Regioncam for a plane through the given points.
        #[staticmethod]
        #[pyo3(signature=(points, labels=vec![], colors=vec![], size=1.5))]
        fn through_points<'py>(
                #[pyo3(from_py_with="downcast_array")] points: Bound<'py, PyArray2<f32>>,
                labels: Vec<String>,
                #[pyo3(from_py_with="extract_colors")] colors: Vec<Color>,
                size: f32,
        ) -> PyResult<Self> {
            let plane = Plane::through_points(&points.readonly().as_array());
            let regioncam = Regioncam::from_plane(&plane, size);
            let labels = labels.into_iter().chain(std::iter::repeat(String::new()));
            let colors = colors.into_iter().map(Some).chain(std::iter::repeat(None));
            let marked_points = plane.points.rows().into_iter().zip(labels).zip(colors).map(
                    |((point, label), color)| {
                        let position = point.as_slice().unwrap().try_into().unwrap();
                        MarkedPoint { position, label, color }
                    }
                ).collect();
            let plane = Py::new(points.py(), PyPlane(plane))?;
            Ok(Self { regioncam, plane: Some(plane), render_options: RenderOptions::default(), marked_points })
        }

        /// Mark points in the input space.
        /// 
        /// Parameters:
        ///  * points: Point to mark
        ///  * labels: List of string labels for the points
        ///  * colors: List of colors for the points (default: no color)
        ///  * project_to_plane: If this Regioncam was constructed on a plane, then the points are projected onto that plane (default: true)
        #[pyo3(signature=(points, labels=vec![], colors=vec![], project_to_plane=true))]
        fn mark_points<'py>(
                &mut self,
                #[pyo3(from_py_with="downcast_array")] points: Bound<'py, PyArray2<f32>>,
                labels: Vec<String>,
                #[pyo3(from_py_with="extract_colors")] colors: Vec<Color>,
                project_to_plane: bool
        ) -> PyResult<()> {
            let py = points.py();
            let points = points.readonly();
            let points = points.as_array();
            let points = 
                if let Some(plane) = self.plane.as_ref().filter(|_| project_to_plane) {
                    CowArray::from(plane.borrow(py).0.project(&points.view()))
                } else {
                    CowArray::from(points)
                };
            let labels = labels.into_iter().chain(std::iter::repeat(String::new()));
            let colors = colors.into_iter().map(Some).chain(std::iter::repeat(None));
            self.marked_points.extend(points.rows().into_iter().zip(labels).zip(colors).map(
                    |((point, label), color)| {
                        let position = point.as_slice().unwrap().try_into().unwrap();
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
        
        /// Extract a 1D slice of this 2D regioncam
        /// 
        /// Parameters:
        ///  * line:             A line in 2d space, or
        ///  * points:           A 2d array with 2 points
        ///  * project_to_plane: If this Regioncam was constructed on a plane, then the points are projected onto that plane (default: true)
        #[pyo3(signature=(line_or_points, project_to_plane=true))]
        fn slice<'py>(&self, line_or_points: LineOrPoints<'py>, project_to_plane: bool) -> PyRegioncam1D {
            let line = match line_or_points {
                LineOrPoints::Line(line) => line.get().0.clone(),
                LineOrPoints::Points(points) => {
                    let py = points.py();
                    let points = points.readonly();
                    let points = points.as_array();
                    let points = 
                        if let Some(plane) = self.plane.as_ref().filter(|_| project_to_plane) {
                            CowArray::from(plane.borrow(py).0.project(&points.view()))
                        } else {
                            CowArray::from(points)
                        };
                    Plane1D::through_points(&points.view())
                }
            };
            PyRegioncam1D {
                regioncam: self.regioncam.slice(&line)
            }
        }
        
        /// Visualize the regions, and write this to an svg file.
        ///
        /// Parameters:
        ///  * path:              path of the file to write to
        ///  * *format options*:  See set_format.
        #[cfg(feature = "svg")]
        #[pyo3(signature=(path, **kwargs))]
        fn write_svg(&self, path: &str, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
            let mut file = File::create(path)?;
            self.render(kwargs, |renderer| {
                renderer.write_svg(&mut file)?;
                Ok(())
            })
        }

        /// Visualize the regions, and write this to an png file.
        ///
        /// Parameters:
        ///  * path:              path of the file to write to
        ///  * *format options*:  See set_format.
        #[cfg(feature = "png")]
        #[pyo3(signature=(path, **kwargs))]
        fn write_png(&self, path: &str, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
            self.render(kwargs, |renderer| {
                renderer.write_png(path).map_err(png_error_to_py_error)
            })
        }

        /// Get an svg representation of the regions, as a string.
        ///
        /// Parameters:
        ///  * *format options*:   See set_format.
        #[cfg(feature = "svg")]
        #[pyo3(signature=(**kwargs))]
        fn svg(&self, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<String> {
            let mut out = vec![];
            self.render(kwargs, |renderer| {
                renderer.write_svg(&mut out)?;
                Ok(String::from_utf8(out)?)
            })
        }

        #[cfg(feature = "repr_svg")]
        #[pyo3(signature=(**kwargs))]
        fn _repr_svg_(&self, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<String> {
            self.svg(kwargs)
        }

        #[cfg(feature = "repr_png")]
        #[pyo3(signature=(**kwargs))]
        fn _repr_png_(&self, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Vec<u8>> {
            self.render(kwargs, |renderer| {
                renderer.encode_png().map_err(png_error_to_py_error)
            })
        }

        /// Visualize the regions, returns an object that can be shown in a jupyter notebook.
        /// 
        /// In jupyter notebooks it is important to use `rc.show()` or `IPython.display.display(rc)`,
        /// instead of ending a cell with a regioncam object,
        /// because the latter prevents the regioncam from ever being garbage collected.
        ///
        /// Parameters:
        ///  * *format options*:   See set_format.
        #[cfg(feature = "repr_png")]
        #[pyo3(signature=(**kwargs))]
        fn show(&self, kwargs: Option<&Bound<'_, PyDict>>, py: Python<'_>) -> PyResult<PngImage> {
            self.render(kwargs, |renderer| {
                let data = renderer.encode_png().map_err(png_error_to_py_error)?;
                let data = PyBytes::new(py, &data).unbind();
                Ok(PngImage { data })
            })
        }

        /// Set rendering options
        /// 
        /// Parameters:
        ///  * size:           Width and height of the image in pixels
        ///                    Either a single number or a tuple of (width, height)
        ///  * draw_boundary:  Draw edges on the region boundary? Default: false
        ///  * draw_faces:     Draw regions by filling them with a random color? Default: true
        ///  * draw_edges:     Draw edges between regions? Default: true
        ///  * draw_vertices:  Draw verteices where edges meet? Default: false
        ///  * line_width:     Line width to use for edges
        ///  * line_color:     Color to use for edges (except the decision boundary).
        ///                    Either a (r,g,b) tuple of numbers in range(0,1) or a string name
        ///  * line_color_decision_boundary:
        ///                    Color to use for decision boundary edges.
        ///  * line_width_decision_boundary:
        ///                    Line width for the decision boundary edges
        ///  * marker_size:    Circle diameter of marked points
        ///  * font_size:      Font size for marked points
        ///  * line_color_by_layer:
        ///                    Use a different color for the edges from each layer.
        ///                    Default: true
        //   * line_color_by_neuron:
        ///                    Use a different color for the edges from each neuron.
        ///                    Default: true
        ///  * line_color_amount:
        ///                    How much to use the layer/neuron line color, as opposed to the base line_color.
        ///                    Default: 0.333
        ///  * layer_line_colors:
        ///                    List of colors to use for each layer.
        ///  * face_color_by_layer:
        ///                    Determine face colors per layer, mixing with colors from earlier layers.
        ///                    In particular has the effect of coloring regions by class label.
        ///                    Default: true
        #[pyo3(signature=(**kwargs))]
        fn set_format(&mut self, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
            self.render_options = parse_render_options(&self.render_options, kwargs)?;
            Ok(())
        }
    }

    #[derive(FromPyObject)]
    enum LineOrPoints<'py> {
        Line(Bound<'py, PyPlane1D>),
        Points(#[pyo3(from_py_with="downcast_array")] Bound<'py, PyArray2<f32>>),
    }

    impl PyRegioncam {
        /// Make a Renderer for rendering the regioncam to svg or png
        fn render<T>(&self, kwargs: Option<&Bound<'_, PyDict>>, fun: impl FnOnce(Renderer<'_>) -> PyResult<T>) -> PyResult<T> {
            let svg_opts = parse_render_options(&self.render_options, kwargs)?;
            let renderer = Renderer::new(&self.regioncam, &svg_opts).with_points(&self.marked_points);
            fun(renderer)
        }
    }

    /// Expose NNBuilder methods to python interface
    macro_rules! impl_py_nn_builder {
        ($class: ident) => {
        #[pymethods]
        impl $class {
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
            layer.borrow().add_to(py, &mut self.regioncam, input_layer)
        }
    }}}

    impl_py_nn_builder!(PyRegioncam);


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
            let vertices = vertices.map(|vertex| PyVertex{ rc: self.rc.clone_ref(py), vertex});
            PyList::new(py, vertices.collect::<Vec<_>>())
        }
        /// The list of edges that make up this face
        #[getter]
        fn edges<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
            let pyrc = self.rc.bind(py);
            let rc = pyrc.borrow();
            let halfedges = rc.regioncam.halfedges_on_face(self.face);
            let edges = halfedges.map(|he| PyEdge{ rc: self.rc.clone_ref(py), edge: he.edge() });
            PyList::new(py, edges.collect::<Vec<_>>())
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
        /// Area of this face in the input space
        #[getter]
        fn area<'py>(&self, py: Python<'py>) -> f32 {
            let rc: PyRef<'_, PyRegioncam> = self.rc.borrow(py);
            rc.regioncam.face_area(self.face)
        }
        fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
            visit.call(&self.rc)
        }
    }

    /// A vertex in a regioncam
    #[pyclass(name="Vertex")]
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
        #[getter]
        fn incident_edges<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
            let pyrc = self.rc.bind(py);
            let rc = pyrc.borrow();
            let halfedges = rc.regioncam.halfedges_leaving_vertex(self.vertex);
            let edges = halfedges.map(|he| PyEdge{ rc: self.rc.clone_ref(py), edge: he.edge() });
            PyList::new(py, edges.collect::<Vec<_>>())
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
    #[pyclass(name="Edge")]
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
        /// Length of this edge in the input space
        #[getter]
        fn length<'py>(&self, py: Python<'py>) -> f32 {
            let rc: PyRef<'_, PyRegioncam> = self.rc.borrow(py);
            rc.regioncam.edge_length(self.edge)
        }
        fn length_at<'py>(&self, py: Python<'py>, layer: usize) -> f32 {
            let rc: PyRef<'_, PyRegioncam> = self.rc.borrow(py);
            rc.regioncam.edge_length_at(layer, self.edge)
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
    #[pyclass(name="Layer")]
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
    
    /// A png image.
    /// 
    /// This is used for the output of Regioncam.show(), so the regioncam can be garbage collected.
    #[cfg(feature = "repr_png")]
    #[pyclass]
    pub struct PngImage {
        data: Py<PyBytes>,
    }
    #[pymethods]
    impl PngImage {
        fn _repr_png_(&self, py: Python<'_>) -> Py<PyBytes> {
            self.data.clone_ref(py)
        }
        #[getter]
        fn data(&self, py: Python<'_>) -> Py<PyBytes> {
            self.data.clone_ref(py)
        }
        /// Write this image to an png file.
        ///
        /// Parameters:
        ///  * path:              path of the file to write to
        #[cfg(feature = "png")]
        fn write_png(&self, py: Python<'_>, path: &str) -> PyResult<()> {
            let data = self.data.bind(py);
            std::fs::write(path, data.as_bytes())?;
            Ok(())
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

    
    /// A 1D regioncam
    #[pyclass(name="Regioncam1D")]
    struct PyRegioncam1D {
        regioncam: Regioncam1D,
        //plane: Option<Py<PyPlane1D>>,
    }
    #[derive(FromPyObject)]
    enum SizeArg1D {
        Range(f32,f32),
        Size(f32),
    }
    impl From<SizeArg1D> for Range<f32> {
        fn from(size: SizeArg1D) -> Self {
            match size {
                SizeArg1D::Range(a, b) => a..b,
                SizeArg1D::Size(size) => -size*0.5 .. size*0.5,
            }
        }
    }

    #[pymethods]
    impl PyRegioncam1D {
        #[new]
        fn new(size: SizeArg1D) -> Self {
            Self {
                regioncam: Regioncam1D::new(size.into())
            }
        }
        /// Create a Regioncam1D for a line through the given points.
        #[staticmethod]
        #[pyo3(signature=(points, size=1.5))]
        fn through_points<'py>(
                #[pyo3(from_py_with="downcast_array")] points: Bound<'py, PyArray2<f32>>,
                size: f32,
        ) -> PyResult<Self> {
            let plane = Plane1D::through_points(&points.readonly().as_array());
            let regioncam = Regioncam1D::from_plane(&plane, size);
            Ok(Self { regioncam })
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
        /// The number of layers
        /// Equivalent to `len(self.layers)`
        #[getter]
        fn num_layers(&self) -> usize {
            self.regioncam.num_layers()
        }
        
        #[getter]
        fn inputs<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
            self.regioncam.sort();
            self.regioncam.activations(0).column(0).to_pyarray(py)
        }
        #[getter]
        fn outputs<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
            self.activations(py, self.regioncam.last_layer())
        }
        // Values of vertices at the given layer
        fn activations<'py>(&mut self, py: Python<'py>, layer: usize) -> Bound<'py, PyArray2<f32>> {
            self.regioncam.sort();
            self.regioncam.activations(layer).to_pyarray(py)
        }
    }

    impl_py_nn_builder!(PyRegioncam1D);

    //declare_pysequence!(PyRegioncam1D, Vertices1D, num_vertices, PyVertex1D, (|rc, index| PyVertex1D { rc, vertex: Vertex::from(index) }));

    macro_rules! declare_py_plane {
        ($PyPlane: ident, $Plane: ident) => {
            /// A plane through points in a high dimensional space
            #[pyclass]
            #[pyo3(frozen, name="Plane")]
            struct $PyPlane($Plane);
            #[pymethods]
            impl $PyPlane {
                #[new]
                fn new<'py>(#[pyo3(from_py_with="downcast_array")] points: Bound<'py, PyArray2<f32>>) -> Self {
                    let plane = $Plane::through_points(&points.readonly().as_array());
                    Self(plane)
                }
                #[staticmethod]
                fn from_linear<'py>(#[pyo3(from_py_with="downcast_array")] weight: Bound<'py, PyArray2<f32>>, #[pyo3(from_py_with="downcast_array")] bias: Bound<'py, PyArray1<f32>>) -> Self {
                    let plane = $Plane::from(::regioncam::nn::Linear{ weight: weight.to_owned_array(), bias: bias.to_owned_array() });
                    Self(plane)
                }

                fn forward<'py>(&self, #[pyo3(from_py_with="downcast_array")] points: Bound<'py, PyArray2<f32>>) -> Bound<'py, PyArray2<f32>> {
                    self.0.forward(&points.readonly().as_array()).to_pyarray(points.py())
                }
                fn inverse<'py>(&self, #[pyo3(from_py_with="downcast_array")] points: Bound<'py, PyArray2<f32>>) -> Bound<'py, PyArray2<f32>> {
                    self.0.project(&points.readonly().as_array()).to_pyarray(points.py())
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
        }
    }
    declare_py_plane!(PyPlane, Plane);
    declare_py_plane!(PyPlane1D, Plane1D);
    
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
    
        fn add_to<'py, B : NNBuilder>(&self, py: Python<'py>, net: &mut B, input_layer: Option<B::LayerNr>) -> PyResult<B::LayerNr> {
            let input_layer = input_layer.unwrap_or(net.last_layer());
            match self {
                PyNNModule::Linear { weight, bias } => {
                    let weight = weight.bind(py).readonly();
                    let bias   = bias.bind(py).readonly();
                    Ok(net.linear_at(input_layer, &weight.as_array(), &bias.as_array()))
                }
                PyNNModule::ReLU() => {
                    net.relu_at(input_layer);
                    Ok(net.last_layer())
                }
                PyNNModule::LeakyReLU { negative_slope } => {
                    net.leaky_relu_at(input_layer, *negative_slope);
                    Ok(net.last_layer())
                }
                PyNNModule::Residual { layer } => {
                    let after_layer = layer.get().add_to(py, net, Some(input_layer))?;
                    net.sum(input_layer, after_layer);
                    Ok(net.last_layer())
                }
                PyNNModule::Sequential { layers } => {
                    let mut input_layer = input_layer;
                    for item in layers {
                        input_layer = item.get().add_to(py, net, Some(input_layer))?;
                    }
                    Ok(input_layer)
                }
            }
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
    
    fn parse_render_options<'py>(opts: &RenderOptions, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<RenderOptions> {
        let mut opts = opts.clone();
        if let Some(kwargs) = kwargs {
            for (k, v) in kwargs {
                let k = k.downcast::<PyString>()?.to_str()?;
                if k == "line_width" {
                    opts.line_width = v.extract()?;
                } else if k == "line_width_decision_boundary" || k == "decision_boundary_line_width" {
                    opts.line_width_decision_boundary = v.extract()?;
                } else if k == "draw_boundary" {
                    opts.draw_boundary = v.extract()?;
                } else if k == "draw_faces" || k == "draw_regions" {
                    opts.draw_faces = v.extract()?;
                } else if k == "draw_edges" {
                    opts.draw_edges = v.extract()?;
                } else if k == "draw_vertices" {
                    opts.draw_vertices = v.extract()?;
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
                } else if k == "vertex_size" {
                    opts.vertex_size = v.extract()?;
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
                } else if k == "text_only_markers" {
                    opts.text_only_markers = v.extract()?;
                    if opts.text_only_markers {
                        opts.point_size = 0.0;
                    }
                } else {
                    return Err(PyValueError::new_err(format!("Unexpected argument: '{k}'")));
                }
            }
        }
        Ok(opts)
    }
    
    fn extract_colors<'py>(arg: &Bound<'py, PyAny>) -> PyResult<Vec<Color>> {
        let list = arg.extract::<Vec<Bound<'py, PyAny>>>()?;
        list.iter().map(extract_color).collect()
    }
    fn extract_color<'py>(arg: &Bound<'py, PyAny>) -> PyResult<Color> {
        if let Ok((r, g, b)) = arg.extract() {
            Ok(Color::new(r, g, b))
        } else if let Some(color) = arg.extract().ok().and_then(|s: String| named_color(&s)) {
            Ok(color)
        } else {
            Err(DowncastError::new(arg, "Tuple[Float,Float,Float] | String").into())
        }
    }
    fn named_color(name: &str) -> Option<Color> {
        match name {
            "black"   => Some(Color::new(0.0, 0.0, 0.0)),
            "red"     => Some(Color::new(1.0, 0.0, 0.0)),
            "green"   => Some(Color::new(0.0, 1.0, 0.0)),
            "blue"    => Some(Color::new(0.0, 0.0, 1.0)),
            "yellow"  => Some(Color::new(1.0, 1.0, 0.0)),
            "magenta" => Some(Color::new(1.0, 0.0, 1.0)),
            "cyan"    => Some(Color::new(0.0, 1.0, 1.0)),
            "white"   => Some(Color::new(1.0, 1.0, 1.0)),
            _ => None,
        }
    }
    
    #[cfg(feature = "png")]
    fn png_error_to_py_error(err: png::EncodingError) -> PyErr {
        use pyo3::exceptions::PyException;
        match err {
            png::EncodingError::IoError(err) => err.into(),
            _ => PyException::new_err(format!("{err}")),
        }
    }
}