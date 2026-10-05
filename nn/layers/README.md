# `vtl.nn.layers`

Layers transform autograd variables and expose trainable parameters to
optimizers. Implementations on `main` include:

| Category | Layers |
| --- | --- |
| Core | Input, Linear, Flatten, Embedding |
| Convolution and pooling | Conv2D, MaxPool2D, Pool2D |
| Normalization and regularization | BatchNorm, LayerNorm, Dropout |
| Recurrent and attention | GRU, LSTM, multi-head attention, positional encoding |
| Activations | ReLU, Sigmoid, Tanh, Softmax, Leaky ReLU, ELU, GELU, Swish, Mish, Softplus, SELU, HardSwish |

The common low-level constructor returns a `types.Layer[T]`:

```v ignore
import vtl.autograd
import vtl.nn.layers

ctx := autograd.ctx[f32]()
layer := layers.linear_layer[f32](ctx, 4, 2)
```

Shapes, parameters, and supported input ranks are layer-specific; follow each
constructor's source comments and tests. Sequential composition is described
in the [models reference](../models/README.md). CUDA and Vulkan paths are
conditional and experimental; see the [device notes](../../docs/DEVICE_MEMORY.md).
