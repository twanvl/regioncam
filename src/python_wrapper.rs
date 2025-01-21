// Python wrapper
use pyo3::prelude::*;
use numpy::*;

use crate::partition::Partition;

#[pymodule(name = "regioncam")]
mod regioncam {
    use std::{fs::File, ops::Range};

    use pyo3::types::PyList;

    use crate::{partition::Face, svg::SvgOptions};

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

        /// List of face objects
        #[getter]
        fn faces<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
            PyList::new(py, self.partition.faces().map(PyFace))
        }
        /// List of vertex ids in a face
        fn face_vertex_ids<'py>(&self, face: Bound<'py, PyFace>) -> Vec<usize> {
            let face = face.borrow().0;
            self.partition.vertices_on_face(face).map(usize::from).collect()
        }

        #[pyo3(signature=(path, *, size=None, line_width=None))]
        fn write_svg(&self, path: &str, size: Option<f32>, line_width: Option<f32>) -> PyResult<()> {
            let mut file = File::create(path)?;
            let mut svg = SvgOptions::new();
            if let Some(size) = size { svg.image_size = (size,size); }
            if let Some(line_width) = line_width { svg.line_width = line_width; }
            svg.write_svg(&self.partition, &mut file)?;
            Ok(())
        }

        // layers

        /// Add a new layer that applies a ReLU operation to the last layer's output
        #[pyo3(signature=(layer=None))]
        fn relu(&mut self, layer: Option<usize>) {
            let layer = layer.unwrap_or(self.partition.last_layer());
            self.partition.relu_at(layer)
        }

        /// Add a new layer that applies a LeakyReLU operation to the last layer's output
        #[pyo3(signature=(factor, layer=None))]
        fn leaky_relu(&mut self, factor: f32, layer: Option<usize>) {
            let layer = layer.unwrap_or(self.partition.last_layer());
            self.partition.leaky_relu_at(layer, factor)
        }

        /// Add a new layer that applies a linear transformation to the last layer's output
        #[pyo3(signature=(weight, bias, layer=None))]
        fn linear<'py>(&mut self, weight: PyReadonlyArray2<'py, f32>, bias: PyReadonlyArray1<'py, f32>, layer: Option<usize>) {
            let layer = layer.unwrap_or(self.partition.last_layer());
            self.partition.linear_at(layer, &weight.as_array(), &bias.as_array());
        }
    }

    #[pyclass(name="Face")]
    struct PyFace(Face);
}