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

`gru_layer(ctx, input_size, hidden_size)` returns this interface and starts
each sequence from a zero hidden state. Use `new_gru_layer` when you need the
typed `GRULayer[T]` API: `forward_with_state(input, h0)` returns both the full
sequence and final hidden state, with autograd support for either output and
the initial state.

`lstm_layer` uses batch-first `[batch, sequence, features]` inputs and returns
the output at every timestep. It supports stacked layers and full CPU BPTT for
input and parameters, with gate matrices ordered input, forget, cell, output.
Use `new_lstm_layer` and `forward_with_state(input, h0, c0)` to provide initial
states and receive final hidden and cell states shaped
`[num_layers, batch, hidden_size]`; autograd propagates gradients through each
of the three outputs and back to the initial states.

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
