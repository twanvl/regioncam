Regioncam - visualize linear regions in neural networks
=========

Regioncam is a rust library and python package for visualizing linear regions in a neural network.
Regioncam works by tracking the output of all neural network layers in the regions where these outputs are linear. The inputs are in a 2 dimensional space.

Compiling
---------

Compiling Regioncam python bindings requires rust and [`maturin`](https://github.com/PyO3/maturin).
```sh
cd regioncam-python
maturin develop --release --strip
python
```

Usage
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
    print(rc.face_vertex_ids(face))
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

Algorithm
---------

Regioncam similar to [Splinecam](https://github.com/AhmedImtiazPrio/splinecam/), but the algorithm is different.

Regioncam maintains a [halfedge datastructure](https://en.wikipedia.org/wiki/Doubly_connected_edge_list) of linear regions, as well as the activations $x_l$ for every vertex on every layer.
When applying a ReLU non-linearity, each edge is split by adding vertices where $x_l=0$. Then regions/faces are split to connect these zero vertices.