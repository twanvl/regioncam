// Python wrapper
use pyo3::prelude::*;
use pyo3::types::PyList;
use numpy::*;
use std::fs::File;
use std::fmt::Write;

use ndarray::{Dim, Dimension};
use pyo3::exceptions::PyAttributeError;
use pyo3::PyClass;
use pyo3::{intern, types::PyString, DowncastError};

use ::regioncam::*;

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
        /// Create a Regioncam object.
        ///
        /// Parameters:
        ///  * size: Size of the rectangle to cover.
        #[new]
        fn new(size: f32) -> Self {
            Self{ partition: Partition::square(size) }
        }
        /// Create a Regioncam object with a circular region
        #[staticmethod]
        fn circle(radius: f32) -> Self {
            Self{ partition: Partition::circle(radius) }
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
        fn faces2<'py>(slf: &Bound<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
            PyList::new(py, slf.borrow().partition.faces().map(|f| PyFace2(slf.clone().unbind(), f)))
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
        ///  * `negative_slope`: multiplicative factor for negative inputs.
        ///  * `input_layer``: use the output of the given layer as input (default: last layer).
        #[pyo3(signature=(negative_slope, input_layer=None))]
        fn leaky_relu(&mut self, negative_slope: f32, input_layer: Option<usize>) {
            let input_layer = input_layer.unwrap_or(self.partition.last_layer());
            self.partition.leaky_relu_at(input_layer, negative_slope)
        }

        /// Add a max pooling layer.
        ///
        /// Parameters:
        ///  * `input_layer``: use the output of the given layer as input (default: last layer).
        #[pyo3(signature=(input_layer=None))]
        fn max_pool(&mut self, input_layer: Option<usize>) {
            let input_layer = input_layer.unwrap_or(self.partition.last_layer());
            self.partition.max_pool_at(input_layer)
        }

        /// Add an argmax pooling layer. This can be used as a classification like softmax.
        ///
        /// Parameters:
        ///  * `input_layer``: use the output of the given layer as input (default: last layer).
        #[pyo3(signature=(input_layer=None))]
        fn argmax_pool(&mut self, input_layer: Option<usize>) {
            let input_layer = input_layer.unwrap_or(self.partition.last_layer());
            self.partition.argmax_pool_at(input_layer)
        }

        /// Add an argmax pooling layer. This can be used as a classification like softmax.
        ///
        /// Parameters:
        ///  * `input_layer``: use the output of the given layer as input (default: last layer).
        #[pyo3(signature=(input_layer=None))]
        fn sign(&mut self, input_layer: Option<usize>) {
            let input_layer = input_layer.unwrap_or(self.partition.last_layer());
            self.partition.sign_at(input_layer)
        }

        /// Add a decision boundary.
        /// If the last layer has 1 ouptut use a sign classifier (hard sigmoid).
        /// If the last layer has >1 output use an argmax classifier (hard softmax).
        ///
        /// Parameters:
        ///  * `input_layer``: use the output of the given layer as input (default: last layer).
        #[pyo3(signature=(input_layer=None))]
        fn decision_boundary(&mut self, input_layer: Option<usize>) {
            let input_layer = input_layer.unwrap_or(self.partition.last_layer());
            self.partition.decision_boundary_at(input_layer)
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
            ) {
            let input_layer = input_layer.unwrap_or(self.partition.last_layer());
            self.partition.linear_at(input_layer, &weight.readonly().as_array(), &bias.readonly().as_array());
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
        fn add<'a,'py>(&mut self, py: Python<'py>, #[pyo3(from_py_with="nn::to_layer")] layer: PyCow<'a, 'py, Layer>, input_layer: Option<usize>) -> PyResult<usize> {
            self.add_layer(py, &layer.borrow(), input_layer)
        }
    }

    impl Regioncam {
        fn add_layer<'py>(&mut self, py: Python<'py>, layer: &Layer, input_layer: Option<usize>) -> PyResult<usize> {
            let input_layer = input_layer.unwrap_or(self.partition.last_layer());
            match layer {
                Layer::Linear { weight, bias } => {
                    let weight = weight.bind(py).readonly();
                    let bias   = bias.bind(py).readonly();
                    self.partition.linear_at(input_layer, &weight.as_array(), &bias.as_array());
                    Ok(self.partition.last_layer())
                }
                Layer::ReLU() => {
                    self.partition.relu_at(input_layer);
                    Ok(self.partition.last_layer())
                }
                Layer::LeakyReLU { negative_slope } => {
                    self.partition.leaky_relu_at(input_layer, *negative_slope);
                    Ok(self.partition.last_layer())
                }
                Layer::Residual { layer } => {
                    let after_layer = self.add_layer(py, layer.get(), Some(input_layer))?;
                    self.partition.sum(input_layer, after_layer);
                    Ok(self.partition.last_layer())
                }
                Layer::Sequential { layers } => {
                    let mut input_layer = input_layer;
                    for item in layers {
                        input_layer = self.add_layer(py, item.get(), Some(input_layer))?;
                    }
                    Ok(input_layer)
                }
            }
        }
    }

    #[pyclass(name="Face")]
    pub struct PyFace(Face);

    #[pyclass(name="Face")]
    pub struct PyFace2(Py<Regioncam>, Face);
    #[pymethods]
    impl PyFace2 {
        /// Ids of all vertices that make up this face
        #[getter]
        fn vertex_ids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<usize>> {
            let rc = self.0.borrow(py);
            let vertex_ids = rc.partition.vertices_on_face(self.1).map(usize::from);
            PyArray1::from_iter(py, vertex_ids)
        }
    }

    /// A layer in a neural network
    #[pyclass]
    #[pyo3(frozen)]
    pub enum Layer {
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
            layer: Py<Layer>,
        },
        /// A sequence of network layers
        Sequential {
            layers: Vec<Py<Layer>>,
        },
    }

    #[pymethods]
    impl Layer {
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
    }
    impl Layer {
        fn shape<'py>(&self, py: Python<'py>) -> PyResult<Dim<[usize;2]>> {
            match self {
                Layer::Linear { weight, .. } => Ok(weight.bind(py).dims()),
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
                Layer::Linear { .. } => {
                    let shape = self.shape(py)?;
                    writeln!(w, "Linear(in_features={}, out_features={})", shape[0], shape[1]).unwrap();
                }
                Layer::ReLU() => {
                    writeln!(w, "ReLU").unwrap();
                }
                Layer::LeakyReLU { negative_slope } => {
                    writeln!(w,"LeakyReLU({negative_slope})").unwrap();
                }
                Layer::Residual { layer } => {
                    writeln!(w, "Residual(").unwrap();
                    write_indent(w, indent+1);
                    layer.get().to_string(py, indent + 1, w)?;
                    write_indent(w, indent);
                    writeln!(w, ")").unwrap();
                }
                Layer::Sequential { layers } => {
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
            ) -> Layer {
            assert_eq!(weight.dims()[1], bias.dims()[0]);
            Layer::Linear { weight: weight.unbind(), bias: bias.unbind() }
        }

        /// A rectified linear activation function
        #[pyfunction]
        #[pyo3(name="ReLU")]
        fn relu() -> Layer {
            Layer::ReLU()
        }

        /// A leaky relu activation function
        #[pyfunction]
        #[pyo3(name="LeakyReLU", signature=(negative_slope=1e-2))]
        fn leaky_relu(negative_slope: f32) -> Layer {
            Layer::LeakyReLU { negative_slope }
        }

        /// A leaky relu activation function
        #[pyfunction]
        #[pyo3(name="Residual")]
        fn residual(layer: Py<Layer>) -> Layer {
            Layer::Residual { layer }
        }

        /// A sequence of network layers
        #[pyfunction]
        #[pyo3(name="Sequential", signature=(*layers))]
        fn sequential(layers: Vec<Py<Layer>>) -> Layer {
            Layer::Sequential { layers }
        }

        /// Convert a torch layer into a Regioncam layer
        #[pyfunction]
        fn convert<'py>(arg: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Layer>> {
            to_layer(arg)?.into_bound(arg.py())
        }

        pub fn to_layer<'a,'py>(arg: &'a Bound<'py, PyAny>) -> PyResult<PyCow<'a, 'py, Layer>> {
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

    /*
    /// Utility: find a plane through 3 points
    /// returns a linear transformation that puts the farthest point
    #[pyfunction]
    fn plane_through_points(points: PyArray2<f32>) -> PyResult<nn::Linear> {
        if points.shape() != [3,2] {
            return Err(PyErr::new_err("Expected shape [3,2]"))
        }
        bias = points.mean();
        points -= bias;

        todo!()
    }
*/
}