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
| `flatten()` | Flatten all non-batch dimensions |
| `maxpool2d(kernel, padding, stride)` | 2D max pooling |
| `avgpool2d(kernel, padding, stride)` | 2D average pooling |
| `global_avgpool2d()` | Global average pooling (per channel) |
| `conv2d(in, out, kernel_size, config)` | 2D convolution |
| `batchnorm1d(num_features, config)` | 1D batch normalisation |
| `layer_norm(normalized_shape, config)` | Layer normalisation |
| `embedding(vocab_size, embed_dim)` | Token embedding (integer indices → vectors) |
| `lstm(input_size, hidden_size, num_layers)` | Long Short-Term Memory layer |
| `multihead_attention(embed_dim, num_heads)` | Multi-head self-attention |
| `positional_encoding(embed_dim, max_len)` | Sinusoidal positional encoding |
| `dropout()` | Dropout (eval mode: no-op) |

### All available losses

| Loss method | Class | Description |
|-------------|-------|-------------|
| `mse_loss()` | `MSELoss` | Mean Squared Error |
| `bce_loss()` | `BCELoss` | Binary Cross-Entropy (per-element) |
| `sigmoid_cross_entropy_loss()` | `SigmoidCrossEntropyLoss` | Sigmoid CE, numerically stable |
| `softmax_cross_entropy_loss()` | `SoftmaxCrossEntropyLoss` | Softmax CE for multi-class |
| `cross_entropy_loss()` | `CrossEntropyLoss` | Cross-Entropy |
| `huber_loss(delta: 1.0)` | `HuberLoss` | Huber (smooth L1) loss |
| `nll_loss(weight)` | `NLLLoss` | Negative Log Likelihood |
| `kl_div_loss()` | `KLDivLoss` | KL Divergence D_KL(P‖Q) |

### All available optimizers

| Optimizer | Module | Key feature |
|----------|--------|-------------|
| `adam_optimizer(lr, ...)` | `optimizers` | Adaptive moment estimation |
| `adamw(...)` | `optimizers` | Adam + decoupled weight decay |
| `rmsprop(...)` | `optimizers` | Per-parameter learning rates |
| `adagrad(...)` | `optimizers` | Accumulates squared grads |
| `sgd(...)` | `optimizers` | Vanilla stochastic gradient descent |

See [TUTORIAL_OPTIMIZERS.md](./TUTORIAL_OPTIMIZERS.md) for full optimizer details
and scheduler usage.

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

VTL initialises weights using the global random seed.  Call `rand.seed` before
creating the autograd context to get reproducible results:

```v
import rand
import vtl.autograd

rand.seed([u32(42), u32(0)])
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

## Visualizing Training with vsl.plot

VTL integrates with [vsl.plot](https://github.com/vlang/vsl) to produce
interactive training visualizations directly from V code. This is the
recommended workflow for understanding model convergence and debugging.

### Collecting metrics during training

```v ignore
import vsl.plot

mut losses := []f64{cap: epochs}
mut accuracies := []f64{cap: epochs}

for epoch in 0 .. epochs {
	y_pred := model.forward(x)!
	mut loss := model.loss(y_pred, y_tensor)!
	losses << loss.value.get([0])

	// Compute accuracy (for classification tasks)
	mut correct := 0
	for i in 0 .. n_samples {
		pred_class := if y_pred.value.get([i, 0]) >= 0.5 { 1.0 } else { 0.0 }
		if pred_class == targets[i] { correct++ }
	}
	accuracies << f64(correct) / f64(n_samples)

	loss.backprop()!
	optimizer.update()!
}
```

### Plotting the results

```v ignore
mut plt := plot.Plot.new()
epoch_x := []f64{len: epochs, init: f64(index)}

plt.scatter(
    x:    epoch_x
    y:    losses
    mode: 'lines'
    line: plot.Line{ color: '#F44336', width: 2.0 }
    name: 'Loss'
)
plt.layout(
    title: 'Training Loss'
    xaxis: plot.Axis{ title: plot.AxisTitle{ text: 'Epoch' } }
    yaxis: plot.Axis{ title: plot.AxisTitle{ text: 'Loss' } }
)
plt.show()!
```

### Comparing predictions vs ground truth

For regression tasks, overlay the true function with model predictions:

```v ignore
// After training, evaluate at each point
mut y_predicted := []f64{len: n_samples}
for i in 0 .. n_samples {
	x_check := vtl.from_array([x_data[i]], [1, 1])!
	mut xv := ctx.variable(x_check, requires_grad: false)
	pred := model.forward(xv)!
	y_predicted[i] = pred.value.get([0, 0])
}

mut plt := plot.Plot.new()
plt.scatter(x: x_data, y: y_true, mode: 'lines', name: 'True')
plt.scatter(x: x_data, y: y_predicted, mode: 'lines', name: 'Predicted')
plt.layout(title: 'Regression: True vs Predicted')
plt.show()!
```

See the full examples: [`nn_regression_sine_plot`](../examples/nn_regression_sine_plot/)
and [`nn_training_metrics_plot`](../examples/nn_training_metrics_plot/).

## Recurrent layers: LSTM and GRU

VTL provides two recurrent layer types for sequence modeling:

| Layer | Gates | Parameters | Cell state |
|-------|-------|------------|------------|
| LSTM  | 4 (i, f, g, o) | `4 * hidden_size * (input_size + hidden_size)` | Yes |
| GRU   | 3 (r, z, n) | `3 * hidden_size * (input_size + hidden_size)` | No |

GRU is simpler and often trains faster, while LSTM can model longer
dependencies thanks to its explicit cell state.

```v ignore
import vtl.nn.layers

// LSTM layer: input_size=10, hidden_size=32
lstm := layers.lstm_layer[f64](ctx, 10, 32)

// GRU layer: input_size=10, hidden_size=32
gru := layers.gru_layer[f64](ctx, 10, 32)
```

Both expect input shape `[batch, seq_len, input_size]` and produce output
shape `[batch, hidden_size]` (the final hidden state).

## Examples

| Example | Task | Loss |
|---------|------|------|
| [`nn_xor`](../examples/nn_xor/) | XOR binary classification | Sigmoid CE |
| [`nn_regression_sine`](../examples/nn_regression_sine/) | sin(x) regression | MSE |
| [`nn_regression_sine_plot`](../examples/nn_regression_sine_plot/) | sin(x) regression + plot | MSE |
| [`nn_training_metrics_plot`](../examples/nn_training_metrics_plot/) | XOR + metrics visualization | MSE |
| [`nn_simple_two_layer`](../examples/nn_simple_two_layer/) | Random target fitting | MSE |
| [`nn_multiclass_iris`](../examples/nn_multiclass_iris/) | 3-class classification | Softmax CE |
| [`nn_autoencoder_simple`](../examples/nn_autoencoder_simple/) | Reconstruction | MSE |

## See also

- [Autograd Tutorial](./TUTORIAL_AUTOGRAD.md) — how gradients are computed
- [Optimizers Tutorial](./TUTORIAL_OPTIMIZERS.md) — Adam/AdamW/RMSProp/AdaGrad/SGD + schedulers
- [First Steps](./TUTORIAL_FIRST_STEPS.md) — tensor creation and properties