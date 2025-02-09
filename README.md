Regioncam - visualize linear regions in neural networks
=========

Regioncam is a rust library and python package for visualizing linear regions in a neural network.
Regioncam works by tracking the output of all neural network layers in the regions where these outputs are linear. The inputs are in a 2 dimensional space.

Compiling
---------

Compiling the Regioncam python bindings requires rust and [`maturin`](https://github.com/PyO3/maturin).
```sh
cd regioncam-python
maturin develop --release
python
```

Usage of python interface
-----

```python
import regioncam
import numpy as np

# Create a regioncam object, with the region [-1..1]^2
rc = regioncam.Regioncam(size=1)
# Apply a linear layer
rng = np.random.default_rng()
weight = rng.standard_normal((2, 30), dtype=np.float32)
bias = rng.standard_normal((30,), dtype=np.float32)
rc.linear(weight, bias)
# Apply a relu activation function
rc.relu()
# Write to svg
rc.write_svg("example.svg")
# Inspect regions
print(f"Created {rc.num_faces} regions")
for face in rc.faces:
    print(face.vertex_ids)
```

Produces:  
<img src="example.svg" alt="drawing" width="300"/>

Note: Weights and biases must be numpy arrays or torch tensors with dtype float32.

Regioncam has limited support for torch nn layers:

```python
net = torch.nn.Sequential(
    torch.nn.Linear(2,30),
    torch.nn.ReLU(),
    torch.nn.Linear(30,30),
    torch.nn.ReLU(),
)
rc.add(net)
```

See `examples/` for an example of visualizing a trained torch network.

The following layer types are supported:
* ReLU
* LeakyReLU
* Linear
* Sequential
* Identity
* Dropout (treated as Identity)
* residual

Algorithm
---------

Regioncam similar to [Splinecam](https://github.com/AhmedImtiazPrio/splinecam/), but the algorithm is different.

Regioncam maintains a [halfedge datastructure](https://en.wikipedia.org/wiki/Doubly_connected_edge_list) of linear regions, which is updated when a piecewise activation function is applied.
It also stores the activations $x^{(l)}$ for every vertex on every layer.
The activations for faces are stored as a $\mathbb{R}^{3\times D}$ matrix $F$, where the activation of an input point $u$ in that face is given by 
 $x^{(l)} = f(u) = (u_1, u_2, 1) F$.

A ReLU activation is applied one dimension at a time:
 * Split each edge $(u,v)$ with $x_u^{(l)} < 0$ and $x_v^{(l)} > 0$, by adding a new vertex $w$ where $x_w^{(l)} = 0$. This is a simple linear interpolation between $u$ and $v$.
 * Then all regions/faces are split by adding edges between vertices with a 0 activation.

For max pooling activations:
 * Split each edge where the argmax changes, such that all points at which the argmax changes are on a vertex.
   In general the argmax is the set of pooled dimensions that are equal to the maximum value. In this new vertex this set has at least two elements, that is, if $\text{argmax} x_u = \{ k_u \}$ and $\text{argmax} x_v = \{ k_v \}$, then we add a new vertex $w$ on the edge with $\text{argmax} x_w = \{ k_u, k_v \}$.
 * Split faces by adding interior edges between vertices with the same argmax.
 * We again do this one dimension at a time, keeping track of the maximum for all pooled dimensions so far.
 * This can result in adding unecessary vertices and edges. These are then cleaned up afterwards by merging faces with the same argmax.
 