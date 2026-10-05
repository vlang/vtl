# `vtl.nn.layers`

Layers transform autograd variables and expose trainable parameters to
optimizers. Available implementations include linear and convolution layers,
pooling, normalization, dropout, embeddings, LSTM, attention, positional
encoding, flattening, and activation layers such as ReLU, sigmoid, tanh, GELU,
Swish, Mish, ELU, and Leaky ReLU.

The common low-level constructor returns a `types.Layer[T]`:

```v ignore
import vtl.autograd
import vtl.nn.layers

ctx := autograd.ctx[f32]()
layer := layers.linear_layer[f32](ctx, 4, 2)
```

Inputs and outputs use the shapes documented by each layer. For sequential
composition, prefer the [models API](../models/README.md). CUDA and Vulkan
implementations are conditional and experimental; see their config types and
the [device notes](../../docs/DEVICE_MEMORY.md).
