# Optimizers

VTL provides several gradient-based optimizers for training neural networks.
All optimizers share the same interface:

1. Call `build_params(model)` where model is `[]types.Layer[T]`
2. After each loss `backprop()`, call `update()!`

All optimizers also support learning rate schedulers — see the last section below.

## Adam

```v
import vtl
import vtl.autograd
import vtl.nn.layers
import vtl.nn.optimizers
import vtl.nn.types

mut ctx := autograd.ctx[f64]()

lin1 := layers.linear_layer[f64](ctx, 784, 256)
lin2 := layers.linear_layer[f64](ctx, 256, 10)
model := [types.Layer[f64](lin1), types.Layer[f64](lin2)]

mut opt := optimizers.adam_optimizer[f64](optimizers.AdamOptimizerConfig{
	learning_rate: 0.001
})
opt.build_params(model)

// Dummy data — replace with real training data
input_vals := vtl.zeros[f64]([64, 784])
target_vals := vtl.zeros[f64]([64, 10])

mut x := ctx.variable(input_vals)
for layer in model {
	x = layer.forward(x)!
}
target := ctx.variable(target_vals)
_ = target

mut loss_val := model[1].forward(x)!
loss_val.backprop()!
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

```v
import vtl
import vtl.autograd
import vtl.nn.layers
import vtl.nn.optimizers
import vtl.nn.types

mut ctx := autograd.ctx[f64]()
lin1 := layers.linear_layer[f64](ctx, 784, 256)
lin2 := layers.linear_layer[f64](ctx, 256, 10)
model := [types.Layer[f64](lin1), types.Layer[f64](lin2)]

mut opt := optimizers.adamw[f64](optimizers.AdamWOptimizerConfig{
	learning_rate: 0.001
	weight_decay:  0.01
})
opt.build_params(model)
```

Config options (same as Adam, plus `weight_decay` with default `0.01`).

## RMSProp

```v
import vtl
import vtl.autograd
import vtl.nn.layers
import vtl.nn.optimizers
import vtl.nn.types

mut ctx := autograd.ctx[f64]()
lin1 := layers.linear_layer[f64](ctx, 784, 256)
lin2 := layers.linear_layer[f64](ctx, 256, 10)
model := [types.Layer[f64](lin1), types.Layer[f64](lin2)]

mut opt := optimizers.rmsprop[f64](optimizers.RMSPropOptimizerConfig{
	learning_rate: 0.001
	alpha:         0.99
})
opt.build_params(model)
```

| Field | Default | Description |
|-------|---------|-------------|
| `learning_rate` | `0.001` | Step size |
| `alpha` | `0.99` | Smoothing constant for squared-gradient average |
| `epsilon` | `1e-8` | Stability constant |
| `weight_decay` | `0.0` | L2 regularisation coefficient |

## AdaGrad

```v
import vtl
import vtl.autograd
import vtl.nn.layers
import vtl.nn.optimizers
import vtl.nn.types

mut ctx := autograd.ctx[f64]()
lin1 := layers.linear_layer[f64](ctx, 784, 256)
lin2 := layers.linear_layer[f64](ctx, 256, 10)
model := [types.Layer[f64](lin1), types.Layer[f64](lin2)]

mut opt := optimizers.adagrad[f64](optimizers.AdaGradOptimizerConfig{
	learning_rate: 0.01
})
opt.build_params(model)
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

```v
import vtl
import vtl.autograd
import vtl.nn.layers
import vtl.nn.optimizers
import vtl.nn.types

mut ctx := autograd.ctx[f64]()
lin1 := layers.linear_layer[f64](ctx, 784, 256)
lin2 := layers.linear_layer[f64](ctx, 256, 10)
model := [types.Layer[f64](lin1), types.Layer[f64](lin2)]

mut opt := optimizers.sgd[f64](optimizers.SgdOptimizerConfig{
	learning_rate: 0.01
})
opt.build_params(model)
```

## Learning Rate Schedulers

Schedulers adjust the learning rate during training. Create a scheduler,
then pass the current step and (optionally) a metric delta to `next_lr()`:

```v
import vtl
import vtl.autograd
import vtl.nn.layers
import vtl.nn.optimizers
import vtl.nn.types

mut ctx := autograd.ctx[f64]()
lin1 := layers.linear_layer[f64](ctx, 784, 256)
lin2 := layers.linear_layer[f64](ctx, 256, 10)
model := [types.Layer[f64](lin1), types.Layer[f64](lin2)]
mut opt := optimizers.sgd[f64](optimizers.SgdOptimizerConfig{ learning_rate: 0.01 })
opt.build_params(model)

// StepLR: reduce LR by gamma every step_size steps
mut scheduler := optimizers.step_lr[f64](30, 0.1)

// ExponentialLR: multiply LR by gamma every step
mut scheduler2 := optimizers.exponential_lr[f64](0.95)

// CosineAnnealingLR: cosine decay from initial_lr to lrd
mut scheduler3 := optimizers.cosine_annealing_lr[f64](100, 1e-5)

// ReduceLROnPlateau: reduce when metric stops improving
mut scheduler4 := optimizers.reduce_lr_on_plateau[f64](optimizers.ReduceLROnPlateauConfig{
	patience: 10
	factor:   0.1
})

// Inside the training loop:
for step := 0; step < 100; step++ {
	opt.update()!
	current_lr := scheduler.next_lr(0.001, step)
	_ = current_lr
}
```

## Complete Training Loop Example

```v
import vtl
import vtl.autograd
import vtl.nn.layers
import vtl.nn.optimizers
import vtl.nn.types

mut ctx := autograd.ctx[f64]()

lin1 := layers.linear_layer[f64](ctx, 784, 256)
lin2 := layers.linear_layer[f64](ctx, 256, 10)
model := [types.Layer[f64](lin1), types.Layer[f64](lin2)]

mut opt := optimizers.adam_optimizer[f64](optimizers.AdamOptimizerConfig{
	learning_rate: 0.001
})
opt.build_params(model)

mut scheduler := optimizers.step_lr[f64](30, 0.1)

for epoch := 0; epoch < 10; epoch++ {
	// Replace with real data batches
	input_batch := vtl.zeros[f64]([64, 784])
	target_batch := vtl.zeros[f64]([64, 10])

	mut x := ctx.variable(input_batch)
	for layer in model {
		x = layer.forward(x)!
	}
	target := ctx.variable(target_batch)
	_ = target

	x.backprop()!
	opt.update()!

	current_lr := scheduler.next_lr(0.001, epoch)
	println('Epoch ${epoch}: lr = ${current_lr}')
}
```
