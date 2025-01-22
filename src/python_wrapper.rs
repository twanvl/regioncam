// Python wrapper
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::exceptions::PyTypeError;
use numpy::*;
use std::{fs::File, ops::Range};

use crate::partition::{Face, Partition};
use crate::svg::SvgOptions;

#[pymodule(name = "regioncam")]
mod regioncam {
    use super::*;

    #[pyclass]
    #[derive(Clone)]
    struct Regioncam {
        partition: Partition,
    }

    #[pymethods]
    impl Regioncam {
        #[new]
        fn new(size: f32) -> Self {
            Self{ partition: Partition::square(size) }
        }

        /// The number of vertices in the partition.
        #[getter]
        fn num_vertices(&self) -> usize {
            self.partition.num_vertices()
        }
        /// The number of edges in the partition.
        #[getter]
        fn num_edges(&self) -> usize {
            self.partition.num_edges()
        }
        /// The number of faces in the partition.
        #[getter]
        fn num_faces(&self) -> usize {
            self.partition.num_faces()
        }
        /// The number of layers
        #[getter]
        fn num_layers(&self) -> usize {
            self.partition.num_layers()
        }
        /// The index of the last layer
        #[getter]
        fn last_layer(&self) -> usize {
            self.partition.last_layer()
        }

        /// The 2D input coordinates for all vertices.
        #[getter]
        fn vertices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
            self.partition.inputs().to_pyarray(py)
        }
        /// The output values / activations in the last layer, for all vertices.
        #[getter]
        fn vertex_outputs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
            self.partition.activations_last().to_pyarray(py)
        }
        /// The output values /  for all vertices, at a given layer
        fn vertex_outputs_at<'py>(&self, py: Python<'py>, layer: usize) -> Bound<'py, PyArray2<f32>> {
            self.partition.activations(layer).to_pyarray(py)
        }

        /// The output values for all faces.
        /// A point x in face f has output value  [x[0],x[1],1] @ face_outputs[f]
        #[getter]
        fn face_outputs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f32>> {
            self.partition.face_activations().to_pyarray(py)
        }
        fn face_outputs_at<'py>(&self, py: Python<'py>, layer: usize) -> Bound<'py, PyArray3<f32>> {
            self.partition.face_activations_at(layer).to_pyarray(py)
        }

        /// List of faces / regions
        #[getter]
        fn faces<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
            PyList::new(py, self.partition.faces().map(PyFace))
        }
        /// List of vertex ids in a face
        fn face_vertex_ids<'py>(&self, face: Bound<'py, PyFace>) -> Vec<usize> {
            let face = face.borrow().0;
            self.partition.vertices_on_face(face).map(usize::from).collect()
        }

        /// Visualize the regions, and write this to an svg file.
        /// 
        /// Parameters:
        ///  * path:        path of the file to write to
        ///  * size:        width and height of the image in pixels
        ///  * line_width:  line width to use for edges
        #[pyo3(signature=(path, *, size=None, line_width=None))]
        fn write_svg(&self, path: &str, size: Option<f32>, line_width: Option<f32>) -> PyResult<()> {
            let mut file = File::create(path)?;
            let mut svg = SvgOptions::new();
            if let Some(size) = size { svg.image_size = (size,size); }
            if let Some(line_width) = line_width { svg.line_width = line_width; }
            svg.write_svg(&self.partition, &mut file)?;
            Ok(())
        }

        /// Get an svg representation of the regions.
        /// 
        /// Parameters:
        ///  * size:        width and height of the image in pixels
        ///  * line_width:  line width to use for edges
        #[pyo3(signature=(*, size=None, line_width=None))]
        fn _repr_svg_(&self, size: Option<f32>, line_width: Option<f32>) -> PyResult<String> {
            let mut out = vec![];
            let mut svg = SvgOptions::new();
            if let Some(size) = size { svg.image_size = (size,size); }
            if let Some(line_width) = line_width { svg.line_width = line_width; }
            svg.write_svg(&self.partition, &mut out)?;
            Ok(String::from_utf8(out)?)
        }

        // layers

        /// Add a new layer that applies a ReLU operation.
        /// 
        /// Parameters:
        ///  * `input_layer``: use the output of the given layer as input (default: last layer).
        #[pyo3(signature=(input_layer=None))]
        fn relu(&mut self, input_layer: Option<usize>) {
            let input_layer = input_layer.unwrap_or(self.partition.last_layer());
            self.partition.relu_at(input_layer)
        }

        /// Add a new layer that applies a LeakyReLU operation.
        /// 
        /// Parameters:
        ///  * `factor`: multiplicative factor for negative inputs.
        ///  * `input_layer``: use the output of the given layer as input (default: last layer).
        #[pyo3(signature=(factor, input_layer=None))]
        fn leaky_relu(&mut self, factor: f32, input_layer: Option<usize>) {
            let input_layer = input_layer.unwrap_or(self.partition.last_layer());
            self.partition.leaky_relu_at(input_layer, factor)
        }

        /// Add a new layer that applies a linear transformation,
        ///   x_out = x_in @ weight + bias
        /// 
        /// Parameters:
        ///  * `weight`:       weight matrix
        ///  * `bias`:         bias vector
        ///  * `input_layer``: use the output of the given layer as input (default: last layer).
        #[pyo3(signature=(weight, bias, input_layer=None))]
        fn linear<'py>(&mut self, weight: PyReadonlyArray2<'py, f32>, bias: PyReadonlyArray1<'py, f32>, input_layer: Option<usize>) {
            let input_layer = input_layer.unwrap_or(self.partition.last_layer());
            self.partition.linear_at(input_layer, &weight.as_array(), &bias.as_array());
        }

        /// Add a neural network or network layer.
        /// This function accepts layers from `regioncam.nn`, and lists of layers.
        /// 
        /// Parameters:
        ///  * `net`:          network or layer(s) to apply
        ///  * `input_layer``: use the output of the given layer as input (default: last layer).
        #[pyo3(signature=(net, input_layer=None))]
        fn add<'py>(&mut self, net: &Bound<'py, PyAny>, input_layer: Option<usize>) -> PyResult<()> {
            let input_layer = input_layer.unwrap_or(self.partition.last_layer());
            if let Ok(net) = net.downcast::<nn::Linear2>() {
                let net = net.borrow();
                let weight = net.weight.bind(net.py()).readonly();
                let bias   = net.bias.bind(net.py()).readonly();
                self.partition.linear_at(input_layer, &weight.as_array(), &bias.as_array());
            } else if let Ok(_net) = net.downcast::<nn::ReLU>() {
                self.partition.relu_at(input_layer);
            } else if let Ok(net) = net.downcast::<nn::LeakyReLU>() {
                self.partition.leaky_relu_at(input_layer, net.borrow().factor);
            } else if let Ok(net) = net.downcast::<nn::Residual>() {
                self.add(net.borrow().0.as_any().bind(net.py()), Some(input_layer))?;
                self.partition.residual(input_layer);
            } else if let Ok(net) = net.downcast::<nn::Sequential>() {
                let net = net.borrow();
                let list = net.0.bind(net.py());
                self.add_list(list, input_layer)?;
            } else if let Ok(list) = net.downcast::<PyList>() {
                self.add_list(list, input_layer)?;
            } else {
                return Err(PyTypeError::new_err("argument 'net': object is not a supported layer type."));
            }
            Ok(())
        }
    }

    impl Regioncam {
        fn add_list<'py>(&mut self, list: &Bound<'py, PyList>, mut input_layer: usize) -> PyResult<()> {
            for item in list {
                self.add(&item, Some(input_layer))?;
                input_layer = self.partition.last_layer();
            }
            Ok(())
        }
    }

    #[pyclass(name="Face")]
    pub struct PyFace(Face);

    #[pyclass(name="Face")]
    pub struct PyFace2(Py<Regioncam>, Face);

    #[pymodule]
    pub mod nn {
        use super::*;
        /// A linear neural network layer
        #[pyclass(name="Linear")]
        pub struct Linear(crate::nn::Linear);
        #[pyclass(name="Linear", get_all, set_all)]
        pub struct Linear2 {
            pub weight: Py<PyArray2<f32>>,
            pub bias: Py<PyArray1<f32>>,
        }
        #[pymethods]
        impl Linear2 {
            #[new]
            fn new(weight: Py<PyArray2<f32>>, bias: Py<PyArray1<f32>>) -> Self {
                Linear2 { weight, bias }
            }
            fn forward<'py>(&self, x: Bound<'py, PyArray2<f32>>) -> PyResult<Bound<'py, PyAny>> {
                x.matmul(self.weight.clone_ref(x.py()))?.add(self.bias.clone_ref(x.py()))
            }
            fn __call__<'py>(&self, x: Bound<'py, PyArray2<f32>>) -> PyResult<Bound<'py, PyAny>> {
                self.forward(x)
            }
        }

        /// A ReLU activation function
        #[pyclass]
        pub struct ReLU();
        #[pymethods]
        impl ReLU {
            #[new]
            fn new() -> Self {
                ReLU()
            }
        }

        /// A LeakyReLU activation function
        #[pyclass(get_all, set_all)]
        pub struct LeakyReLU {
            pub factor: f32
        }
        #[pymethods]
        impl LeakyReLU {
            #[new]
            fn new(factor: f32) -> Self {
                LeakyReLU { factor }
            }
        }

        /// A residual connection around one or more network layers
        #[pyclass]
        pub struct Residual(pub Py<PyAny>);
        #[pymethods]
        impl Residual {
            #[new]
            fn new(net: Py<PyAny>) -> Self {
                Residual(net)
            }
        }

        /// A list of neural network layers
        #[pyclass]
        pub struct Sequential(pub Py<PyList>);
        #[pymethods]
        impl Sequential {
            #[new]
            fn new(nets: Py<PyList>) -> Self {
                Sequential(nets)
            }
        }
    }

}