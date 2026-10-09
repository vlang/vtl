# Tutorial: Neural Networks

VTL provides a high-level `Sequential` model API in `vtl.nn.models` that
builds on the autograd engine.  This tutorial walks through building,
training, and evaluating a neural network from scratch.

## Overview

The training workflow has four steps:

1. **Build** — define the model architecture with `Sequential`
2. **Forward** — run input through `model.forward(x)`
3. **Loss** — compute `model.loss(y_pred, y_target)`
4. **Update** — call `loss.backprop()` then `optimizer.update()`

## Building a model

```v
import vtl.autograd
import vtl.nn.models

ctx := autograd.ctx[f64]()

mut model := models.sequential_from_ctx[f64](ctx)
model.input([2]) // 2 input features
model.linear(8) // hidden layer: 2 -> 8
model.relu() // non-linearity
model.linear(1) // output layer: 8 -> 1
model.mse_loss() // loss function
```

### All available layers

| Layer method | Description |
|-------------|-------------|
| `linear(n)` | Linear transformation `y = x·Wᵀ + b` |
| `relu()` | Rectified Linear Unit |
| `leaky_relu()` | Leaky ReLU |
| `elu()` | Exponential Linear Unit |
| `sigmoid()` | Logistic sigmoid |
| `tanh()` | Hyperbolic tangent |
| `softmax()` | Softmax over the last dimension |
| `gelu()` | Gaussian Error Linear Unit |
| `swish()` | Swish activation |
| `mish()` | Mish activation |
| `softplus()` | Smooth approximation to ReLU, `log(1 + exp(x))` |
| `selu()` | Scaled Exponential Linear Unit |
| `hardswish()` | Efficient piecewise approximation to Swish |
| `flatten()` | Flatten all non-batch dimensions |
| `maxpool2d(kernel, padding, stride)` | 2D max pooling |
| `avgpool2d(kernel, padding, stride)` | 2D average pooling |
| `global_avgpool2d()` | Global average pooling (per channel) |
| `conv2d(in, out, kernel_size, config)` | 2D convolution |
| `conv1d(out, kernel_size, config)` | Grouped 1D convolution over channel-first sequences |
| `batchnorm1d(num_features, config)` | 1D batch normalisation |
| `layer_norm(normalized_shape, config)` | Layer normalisation |
| `embedding(vocab_size, embed_dim)` | Token embedding (integer indices → vectors) |
| `lstm(input_size, hidden_size, num_layers)` | Long Short-Term Memory layer |
| `multihead_attention(embed_dim, num_heads)` | Multi-head self-attention |
| `positional_encoding(embed_dim, max_len)` | Sinusoidal positional encoding |
| `dropout()` | Dropout (eval mode: no-op) |

## LSTM sequences

`lstm_layer` accepts batch-first input shaped `[batch, sequence, features]`
and returns `[batch, sequence, hidden_size]`. It supports stacked layers and
computes gradients for the input, each gate matrix, and each bias through
backpropagation through time. Gate matrices use input, forget, cell, output
order.

```v
import vtl
import vtl.autograd
import vtl.nn.layers

ctx := autograd.ctx[f64]()
recurrent := layers.lstm_layer[f64](ctx, 2, 4, 2)
sequence := ctx.variable(vtl.ones[f64]([3, 5, 2]))
prediction := recurrent.forward(sequence)!
assert prediction.value.shape == [3, 5, 4]
assert recurrent.variables().len == 8
prediction.backprop()!
assert sequence.grad.shape == sequence.value.shape
```

`maxpool2d` saves each selected input's flat index for backpropagation. If
overlapping windows select the same input, their gradients are added at that
input; ties keep the first maximum in scan order.

`avgpool2d` distributes each output gradient evenly across its window. With
padding, padded positions receive no input gradient, while the divisor remains
the full kernel area. The [average pooling example](../examples/nn_average_pooling/README.md)
prints both the pooled values and the input gradients.

The three smooth/efficient activations can be selected directly in a Sequential
model:

```v
import vtl.nn.models

mut model := models.sequential[f64]()
model.input([8])
model.linear(16)
model.softplus()
model.linear(4)
model.selu()
```

### All available losses

| Loss method | Class | Description |
|-------------|-------|-------------|
| `mse_loss()` | `MSELoss` | Mean Squared Error |
| `bce_loss()` | `BCELoss` | Binary Cross-Entropy (per-element) |
| `sigmoid_cross_entropy_loss()` | `SigmoidCrossEntropyLoss` | Sigmoid CE, numerically stable |
| `softmax_cross_entropy_loss()` | `SoftmaxCrossEntropyLoss` | Softmax CE for multi-class |
| `cross_entropy_loss()` | `CrossEntropyLoss` | Cross-Entropy |
| `huber_loss(delta: 1.0)` | `HuberLoss` | Huber (smooth L1) loss |
| `l1_loss()` | `L1Loss` | Mean Absolute Error (MAE); zero subgradient at exact matches |
| `hinge_loss()` | `HingeLoss` | Binary SVM hinge loss; targets must be -1 or +1 |
| `focal_loss()` | `FocalLoss` | Binary focal loss for imbalanced classes; default logits, alpha 0.25, gamma 2 |
| `nll_loss(weight)` | `NLLLoss` | Negative Log Likelihood |
| `kl_div_loss()` | `KLDivLoss` | KL Divergence D_KL(P‖Q) |

### Binary focal loss with probabilities

For precomputed probabilities, configure the standalone loss with `from_logits: false`.
Targets are binary values in `[0, 1]`. The `Sequential.focal_loss()` builder uses
logits and the default `alpha: 0.25`, `gamma: 2` configuration.

```v
import vtl
import vtl.autograd
import vtl.nn.loss

ctx := autograd.ctx[f64]()
prediction := ctx.variable(vtl.from_array([0.8], [1])!)
target := vtl.from_array([1.0], [1])!
criterion := loss.focal_loss[f64](from_logits: false, alpha: 0.25, gamma: 2.0)
mut loss_value := criterion.loss(prediction, target)!
loss_value.backprop()!
```

### All available optimizers

| Optimizer | Module | Key feature |
|----------|--------|-------------|
| `adam_optimizer(lr, ...)` | `optimizers` | Adaptive moment estimation |
| `adamw(...)` | `optimizers` | Adam + decoupled weight decay |
| `rmsprop(...)` | `optimizers` | Per-parameter learning rates |
| `adagrad(...)` | `optimizers` | Accumulates squared grads |
| `nadam_optimizer(...)` | `optimizers` | Adam with Nesterov momentum and momentum scheduling |
| `radam_optimizer(...)` | `optimizers` | Rectifies Adam variance during early steps |
| `sgd(...)` | `optimizers` | Vanilla stochastic gradient descent |

See [TUTORIAL_OPTIMIZERS.md](./TUTORIAL_OPTIMIZERS.md) for full optimizer details
and scheduler usage.

## Conv1D for sequence data

`Sequential.conv1d` consumes channel-first tensors shaped
`[batch, channels, length]`. The layer supports stride, padding, dilation,
and grouped channels. Its CPU backward computes gradients for the input,
kernel, and bias.

```v
import vtl
import vtl.autograd
import vtl.nn.layers
import vtl.nn.models

ctx := autograd.ctx[f64]()
mut model := models.sequential_from_ctx[f64](ctx)
model.input([1, 5])
model.conv1d(2, 3, layers.Conv1DConfig{ padding: 1 })
sequence := vtl.from_array([0.1, 0.2, 0.3, 0.4, 0.5], [1, 1, 5])!
mut input := ctx.variable(sequence)
mut output := model.forward(input)!
println(output.value.shape) // [1, 2, 5]
output.backprop()!
```

See the runnable [Conv1D example](../examples/nn_conv1d/) for a complete forward and backward pass.

## Recurrent GRU layer

`Sequential.gru(input_size, hidden_size)` adds a single-layer, unidirectional
GRU with a zero initial hidden state. It accepts
`[sequence, batch, input_features]` and returns
`[sequence, batch, hidden_size]`. Its reset, update, and candidate weights use
PyTorch's `[reset, update, new]` gate order. The lower-level
`vtl.nn.internal.gru_forward_single` also accepts an explicit initial state and
returns the final state. CPU backpropagation computes gradients for the input,
initial hidden state, and all four parameter tensors.

For stateful inference or truncated backpropagation, construct a typed layer
with `new_gru_layer` and call `forward_with_state`. It returns both the output
sequence and final hidden state; gradients from either result flow through the
same GRU recurrence and back to `h0`.

```v
import vtl
import vtl.autograd
import vtl.nn.layers

ctx := autograd.ctx[f64]()
layer := layers.new_gru_layer[f64](ctx, 2, 4)
sequence := ctx.variable(vtl.from_array([0.1, 0.2, 0.3, 0.4], [2, 1, 2])!)
h0 := ctx.variable(vtl.zeros[f64]([1, 4]))
mut output, mut final_state := layer.forward_with_state(sequence, h0)!
loss := output.sum()!.add(final_state.sum()!)!
loss.backprop()!
println(final_state.value.shape) // [1, 4]
println(h0.grad.shape) // [1, 4]
```

`Sequential.gru` continues to initialize the hidden state to zero and returns
the output sequence. Use `new_gru_layer` when the caller needs to provide or
carry the hidden state between batches.

```v
import vtl
import vtl.autograd
import vtl.nn.models

ctx := autograd.ctx[f64]()
mut model := models.sequential_from_ctx[f64](ctx)
model.input([3, 1, 2])
model.gru(2, 4)
sequence := vtl.from_array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [3, 1, 2])!
mut input := ctx.variable(sequence)
mut output := model.forward(input)!
println(output.value.shape) // [3, 1, 4]
output.backprop()!
println(input.grad.shape) // [3, 1, 2]
```

See the runnable [GRU example](../examples/nn_gru/) for a complete
`Sequential` forward and backward smoke test. Sequential checkpoints save and
restore all four GRU parameter tensors.

## Training loop

```v ignore
// assumes model, x, y_target defined above
mut optimizer := optimizers.adam_optimizer[f64](learning_rate: 0.001)
optimizer.build_params(model.info.layers)

for epoch in 0 .. 100 {
    y_pred := model.forward(x)!
    mut loss := model.loss(y_pred, y_target)!

    println('Epoch ${epoch}: loss = ${loss.value.get([0]):.4f}')

    loss.backprop()!
    optimizer.update()!
}
```

## Mini-batch training

For large datasets, split the data into mini-batches and loop over them
inside each epoch.  Use `.slice()` to extract a batch:

```v ignore
// assumes model, x_all, y_tensor, n_batches, batch_size defined above
for b in 0 .. n_batches {
    offset := b * batch_size
    mut x_batch := x_all.slice([offset, offset + batch_size])!
    y_batch := y_tensor.slice([offset, offset + batch_size])!

    y_pred := model.forward(x_batch)!
    mut loss := model.loss(y_pred, y_batch)!

    loss.backprop()!
    optimizer.update()!
}
```

## Reproducibility

VTL initialises weights using V's global random generator. Call
`vtl.random_seed` before creating the model to get reproducible results:

```v
import vtl
import vtl.autograd

vtl.random_seed(42)
ctx := autograd.ctx[f64]()
_ = ctx
```

## Practical tips

### Softmax cross-entropy (multi-class classification)

- Use **interleaved class ordering** (`class_id = i % n_classes`) so every
  mini-batch sees all classes.
- Keep `batch_size >= n_classes`.
- Use `learning_rate = 0.01` and `batch_size = 6+` for stable training.

### MSE regression

- Full-batch gradient descent (all samples at once) works well for small
  datasets.
- Use `learning_rate = 0.001` and at least 60 epochs.

### Sigmoid cross-entropy (binary classification)

- Large batch sizes (32+) help — see the XOR example.
- `learning_rate = 0.01` is a reliable starting point.

## Examples

| Example | Task | Loss |
|---------|------|------|
| [`nn_xor`](../examples/nn_xor/) | XOR binary classification | Sigmoid CE |
| [`nn_regression_sine`](../examples/nn_regression_sine/) | sin(x) regression | MSE |
| [`nn_simple_two_layer`](../examples/nn_simple_two_layer/) | Random target fitting | MSE |
| [`nn_multiclass_iris`](../examples/nn_multiclass_iris/) | 3-class classification | Softmax CE |
| [`nn_autoencoder_simple`](../examples/nn_autoencoder_simple/) | Reconstruction | MSE |

## See also

- [Autograd Tutorial](./TUTORIAL_AUTOGRAD.md) — how gradients are computed
- [Optimizers Tutorial](./TUTORIAL_OPTIMIZERS.md) — Adam/AdamW/RMSProp/AdaGrad/SGD + schedulers
- [First Steps](./TUTORIAL_FIRST_STEPS.md) — tensor creation and properties
