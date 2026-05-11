# Optimizers

VTL provides several gradient-based optimizers for training neural networks.
All optimizers share the same interface:

1. Call `build_params(layers)` once after constructing the model
2. After each loss `backward()`, call `update()!`

All optimizers also support learning rate schedulers — see the last section below.

## Adam

```v ignore
import vtl
import vtl.autograd as ag
import vtl.nn.layers
import vtl.nn.loss
import vtl.nn.optimizers

mut ctx := ag.ctx[f64]()

// Build a two-layer network
lin1 := layers.linear_layer[f64](ctx, 784, 256)
lin2 := layers.linear_layer[f64](ctx, 256, 10)
model_layers := [layers.Layer[f64](lin1), layers.Layer[f64](lin2)]

// Create Adam optimizer
mut opt := optimizers.adam_optimizer[f64](learning_rate: 0.001)
opt.build_params(model_layers)

// Dummy data — replace with real training data
input_vals := vtl.zeros[f64]([64, 784])
target_vals := vtl.zeros[f64]([64, 10])

mut x := ctx.variable(input_vals)
for layer in model_layers {
	x = layer.forward(x)!
}
mut y_target := ctx.variable(target_vals)

l := loss.mse_loss[f64]()
mut loss_val := l.loss(x, target_vals)!
loss_val.backward()!
opt.update()!
```

Config options:

| Field | Default | Description |
|-------|---------|-------------|
| `learning_rate` | `0.001` | Step size α |
| `beta1` | `0.9` | First-moment decay rate |
| `beta2` | `0.999` | Second-moment decay rate |
| `epsilon` | `1e-8` | Numerical stability constant |

## AdamW

AdamW is Adam with decoupled weight decay — typically gives better
regularisation than Adam with L2 penalty.

```v ignore
import vtl.nn.optimizers

mut opt := optimizers.adamw[f64](learning_rate: 0.001, weight_decay: 0.01)
opt.build_params(model_layers)
```

Config options (same as Adam, plus `weight_decay` with default `0.01`).

## RMSProp

```v ignore
import vtl.nn.optimizers

mut opt := optimizers.rmsprop[f64](learning_rate: 0.001, alpha: 0.99)
opt.build_params(model_layers)
```

| Field | Default | Description |
|-------|---------|-------------|
| `learning_rate` | `0.001` | Step size |
| `alpha` | `0.99` | Smoothing constant for squared-gradient average |
| `epsilon` | `1e-8` | Stability constant |
| `weight_decay` | `0.0` | L2 regularisation coefficient |

## AdaGrad

```v ignore
import vtl.nn.optimizers

mut opt := optimizers.adagrad[f64](learning_rate: 0.01)
opt.build_params(model_layers)
```

Accumulates squared gradients; effective learning rate decreases for
frequently updated parameters. Good for sparse gradients.

| Field | Default | Description |
|-------|---------|-------------|
| `learning_rate` | `0.01` | Initial step size |
| `epsilon` | `1e-8` | Stability constant |
| `weight_decay` | `0.0` | L2 regularisation coefficient |

## SGD

Vanilla Stochastic Gradient Descent:

```v ignore
import vtl.nn.optimizers

mut opt := optimizers.sgd[f64](learning_rate: 0.01)
opt.build_params(model_layers)
```

## Learning Rate Schedulers

Schedulers adjust the learning rate during training. Create a scheduler,
then pass the current step and (optionally) a metric delta to `next_lr()`:

```v ignore
import vtl.nn.optimizers

// StepLR: reduce LR by gamma every step_size steps
mut scheduler := optimizers.step_lr[f64](step_size: 30, gamma: 0.1)

// ExponentialLR: multiply LR by gamma every step
mut scheduler2 := optimizers.exponential_lr[f64](gamma: 0.95)

// CosineAnnealingLR: cosine decay from initial_lr to lrd
mut scheduler3 := optimizers.cosine_annealing_lr[f64](t_max: 100, lrd: 1e-5)

// ReduceLROnPlateau: reduce when metric stops improving
mut scheduler4 := optimizers.reduce_lr_on_plateau[f64](patience: 10, factor: 0.1)

// Inside the training loop:
for step in 0 .. 100 {
	opt.update()!
	current_lr := scheduler.next_lr(0.001, step)
	// or for ReduceLROnPlateau:
	// current_lr := scheduler4.next_lr(0.001, step, metric_delta: val_loss - prev_loss)
}
```

## Complete Training Loop Example

```v ignore
import vtl
import vtl.autograd as ag
import vtl.nn.layers
import vtl.nn.loss
import vtl.nn.optimizers

mut ctx := ag.ctx[f64]()

model_layers := [
	layers.Layer[f64](layers.linear_layer[f64](ctx, 784, 256)),
	layers.Layer[f64](layers.relu_layer[f64](ctx)),
	layers.Layer[f64](layers.linear_layer[f64](ctx, 256, 10)),
]

mut opt := optimizers.adam_optimizer[f64](learning_rate: 0.001)
opt.build_params(model_layers)

mut scheduler := optimizers.step_lr[f64](step_size: 30, gamma: 0.1)

for epoch in 0 .. 10 {
	// Replace with real data batches
	input_batch := vtl.zeros[f64]([64, 784])
	target_batch := vtl.zeros[f64]([64, 10])

	mut x := ctx.variable(input_batch)
	for layer in model_layers {
		x = layer.forward(x)!
	}
	mut target := ctx.variable(target_batch)

	l := loss.mse_loss[f64]()
	mut loss_val := l.loss(x, target_batch)!
	loss_val.backward()!
	opt.update()!

	current_lr := scheduler.next_lr(0.001, epoch)
	println('Epoch ${epoch}: lr = ${current_lr}')
}
```
