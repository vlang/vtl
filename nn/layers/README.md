# `vtl.nn.layers`

Layers transform autograd variables and expose trainable parameters to
optimizers. Implementations on `main` include:

| Category | Layers |
| --- | --- |
| Core | Input, Linear, Flatten, Embedding |
| Convolution and pooling | Conv2D, MaxPool2D, Pool2D |
| Normalization and regularization | BatchNorm, GroupNorm, LayerNorm, RMSNorm, Dropout |
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

## GroupNorm

GroupNorm splits channels into `num_groups` and normalizes each group
independently for every batch item, including trailing spatial dimensions. Its
input layout is channel-first (`[batch, channels, ...]`). The constructor's
`input_shape` excludes batch; with affine enabled (the default), it learns one
scale and bias per channel.

```v ignore
import vtl.autograd
import vtl.nn.layers

ctx := autograd.ctx[f32]()
norm := layers.group_norm_layer[f32](ctx, [8, 16, 16], 4, layers.GroupNormConfig{})
```

`num_groups` must be positive and divide the channel count. GroupNorm uses
current input statistics in both training and evaluation, so it has no running
statistics state.

## RMSNorm

RMSNorm divides each trailing `normalized_shape` block by its root mean square,
then applies a learnable per-element weight when `elementwise_affine` is true.
Unlike LayerNorm, it does not subtract the mean or learn a bias. If `eps` is
omitted, VTL uses the machine epsilon for `f32` or `f64` computations, matching
PyTorch's default for these dtypes. See the
[PyTorch RMSNorm reference](https://docs.pytorch.org/docs/2.14/generated/torch.nn.RMSNorm.html)
for the formula and public API.

```v ignore
import vtl.autograd
import vtl.nn.layers

ctx := autograd.ctx[f32]()
norm := layers.rms_norm_layer[f32](ctx, [128], layers.RMSNormConfig{})
```
