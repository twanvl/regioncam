Regioncam - visualize linear regions in neural networks
=========

Regioncam is a rust library and python package for visualizing linear regions in a neural network.

This is similar to [Splinecam](https://github.com/AhmedImtiazPrio/splinecam/), but the algorithm is different.

Compiling
---------

Compiling Regioncam requires rust and `maturin`.
```
maturin develop --release
python
```

Usage
-----

Regioncam works by tracking the output of all neural network layers in the regions where these outputs are linear. The inputs are in a 2 dimensional space.

```python
import regioncam
import numpy as np
rng = 
rc = regioncam.Regioncam(size = 1) # Create a regioncam object, with the region [-1..1]^2
rc.linear(weight, bias) # Apply a linear layer
rc.relu() # Apply a relu activation function
rc.write_svg("file.svg")
```

Weights and biases must be numpy arrays with dtype float32.

Algorithm
---------

Regioncam maintains a [halfedge datastructure](https://en.wikipedia.org/wiki/Doubly_connected_edge_list) of linear regions, as well as the activations $x_l$ for every vertex on every layer.
When applying a ReLU non-linearity, each edge is split by adding vertices where $x_l=0$. Then faces are split to connect these zero vertices.