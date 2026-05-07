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

Available layers: `linear(n)`, `relu()`, `leaky_relu()`, `elu()`, `sigmod()`,
`flatten()`, `maxpool2d(...)`.

Available losses: `mse_loss()`, `sigmoid_cross_entropy_loss()`,
`softmax_cross_entropy_loss()`.

## Choosing a loss function

| Task | Loss | Output |
|------|------|--------|
| Regression | `mse_loss()` | Any real value |
| Binary classification | `sigmoid_cross_entropy_loss()` | Single logit |
| Multi-class classification | `softmax_cross_entropy_loss()` | One logit per class |

## Training loop

```v ignore
// assumes model, x, y_target defined above
mut optimizer := optimizers.sgd[f64](learning_rate: 0.001)
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
- [First Steps](./TUTORIAL_FIRST_STEPS.md) — tensor creation and properties
