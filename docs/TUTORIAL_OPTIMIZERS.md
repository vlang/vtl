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
import vtl.nn.models
import vtl.nn.optimizers

mut ctx := autograd.ctx[f32]()
mut model := models.sequential_from_ctx[f32](ctx)
model.input([784])
model.linear(256)
model.linear(10)
model.mse_loss()

mut opt := optimizers.adam_optimizer[f32](optimizers.AdamOptimizerConfig{
	learning_rate: 0.001
})
opt.build_params(model.info.layers)

// Dummy data — replace with real training data
input_vals := vtl.zeros[f32]([64, 784])
target_vals := vtl.zeros[f32]([64, 10])

mut x := ctx.variable(input_vals)
pred := model.forward(x)!
mut loss_val := model.loss(pred, target_vals)!
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

```v ignore
import vtl
import vtl.autograd
import vtl.nn.layers
import vtl.nn.optimizers
import vtl.nn.types

mut ctx := autograd.ctx[f32]()
lin1 := layers.linear_layer[f32](ctx, 784, 256)
lin2 := layers.linear_layer[f32](ctx, 256, 10)
model := [types.Layer[f32](lin1), types.Layer[f32](lin2)]

mut opt := optimizers.adamw[f32](optimizers.AdamWOptimizerConfig{
	learning_rate: 0.001
	weight_decay:  0.01
})
opt.build_params(model)
```

Config options (same as Adam, plus `weight_decay` with default `0.01`).

## RMSProp

```v ignore
import vtl
import vtl.autograd
import vtl.nn.layers
import vtl.nn.optimizers
import vtl.nn.types

mut ctx := autograd.ctx[f32]()
lin1 := layers.linear_layer[f32](ctx, 784, 256)
lin2 := layers.linear_layer[f32](ctx, 256, 10)
model := [types.Layer[f32](lin1), types.Layer[f32](lin2)]

mut opt := optimizers.rmsprop[f32](optimizers.RMSPropOptimizerConfig{
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

```v ignore
import vtl
import vtl.autograd
import vtl.nn.layers
import vtl.nn.optimizers
import vtl.nn.types

mut ctx := autograd.ctx[f32]()
lin1 := layers.linear_layer[f32](ctx, 784, 256)
lin2 := layers.linear_layer[f32](ctx, 256, 10)
model := [types.Layer[f32](lin1), types.Layer[f32](lin2)]

mut opt := optimizers.adagrad[f32](optimizers.AdaGradOptimizerConfig{
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

```v ignore
import vtl
import vtl.autograd
import vtl.nn.layers
import vtl.nn.optimizers
import vtl.nn.types

mut ctx := autograd.ctx[f32]()
lin1 := layers.linear_layer[f32](ctx, 784, 256)
lin2 := layers.linear_layer[f32](ctx, 256, 10)
model := [types.Layer[f32](lin1), types.Layer[f32](lin2)]

mut opt := optimizers.sgd[f32](optimizers.SgdOptimizerConfig{
	learning_rate: 0.01
})
opt.build_params(model)
```

SGD also supports coupled weight decay, momentum, dampening, and Nesterov
momentum. For momentum, register parameters with `build_params` so the optimizer
can create its per-parameter buffers. The first buffer uses the regularized
gradient directly; `dampening` applies to subsequent updates.

```v ignore
mut opt := optimizers.sgd[f32](optimizers.SgdOptimizerConfig{
	learning_rate: 0.01
	momentum:      0.9
	weight_decay:  0.0001
	nesterov:      true
})
opt.build_params(model)
```

## NAdam

NAdam adds Nesterov momentum to Adam. VTL follows the momentum schedule and
bias corrections used by PyTorch; defaults are `lr: 0.002`, `beta1: 0.9`,
`beta2: 0.999`, `epsilon: 1e-8`, and `momentum_decay: 0.004`.

```v ignore
import vtl.nn.optimizers

mut opt := optimizers.nadam_optimizer[f32](learning_rate: 0.002, momentum_decay: 0.004)
```

The configuration also supports `weight_decay` and
`decoupled_weight_decay`. The latter applies decay to parameters separately
from the gradient and moment estimates.

## RAdam

RAdam rectifies Adam's adaptive variance after its running estimate becomes
reliable, while using a bias-corrected momentum update during its early steps.
VTL uses the PyTorch rectification threshold and epsilon placement. Defaults
are `lr: 0.001`, `beta1: 0.9`, `beta2: 0.999`, and `epsilon: 1e-8`.

```v ignore
import vtl.nn.optimizers

mut opt := optimizers.radam_optimizer[f32](learning_rate: 0.001)
```

The configuration supports coupled `weight_decay` by default, with an optional
`decoupled_weight_decay` mode.

## Learning Rate Schedulers

Schedulers adjust the learning rate during training. Create a scheduler,
then pass the current step and (optionally) a metric delta to `next_lr()`:

```v ignore
import vtl
import vtl.autograd
import vtl.nn.layers
import vtl.nn.optimizers
import vtl.nn.types

mut ctx := autograd.ctx[f32]()
lin1 := layers.linear_layer[f32](ctx, 784, 256)
lin2 := layers.linear_layer[f32](ctx, 256, 10)
model := [types.Layer[f32](lin1), types.Layer[f32](lin2)]
mut opt := optimizers.sgd[f32](optimizers.SgdOptimizerConfig{ learning_rate: 0.01 })
opt.build_params(model)

// StepLR: reduce LR by gamma every step_size steps
mut scheduler := optimizers.step_lr[f32](30, 0.1)

// MultiStepLR: decay at selected epoch indices
mut scheduler_multi := optimizers.multi_step_lr[f32](optimizers.MultiStepLRConfig{
	milestones: [30, 80]
	gamma:      0.1
}) or { panic(err) }

// ExponentialLR: multiply LR by gamma every step
mut scheduler2 := optimizers.exponential_lr[f32](0.95)

// CosineAnnealingLR: cosine decay from initial_lr to lrd
mut scheduler3 := optimizers.cosine_annealing_lr[f32](100, 1e-5)

// CosineAnnealingWarmRestarts: restart after 10 steps, then double each cycle
mut scheduler_cosine_restarts := optimizers.cosine_annealing_warm_restarts[f32](
	optimizers.CosineAnnealingWarmRestartsConfig{
		t_0:     10
		t_mult:  2
		eta_min: 1e-5
	}
) or { panic(err) }

// LinearLR: linear warmup from 10% to 100% of the base learning rate
mut scheduler4 := optimizers.linear_lr[f32](0.1, 1.0, 10)

// PolynomialLR and ConstantLR: polynomial decay and temporary rate scaling
mut scheduler_poly := optimizers.polynomial_lr[f32](optimizers.PolynomialLRConfig{
	total_iters: 100
	power:       0.9
}) or { panic(err) }
mut scheduler_constant := optimizers.constant_lr[f32](optimizers.ConstantLRConfig{
	factor:      0.25
	total_iters: 5
}) or { panic(err) }

// OneCycleLR: minibatch schedule from max_lr/div_factor to max_lr and back
mut scheduler5 := optimizers.one_cycle_lr[f32](optimizers.OneCycleLRConfig{
	max_lr:         0.01
	total_steps:    100
	cycle_momentum: true
	base_momentum:  0.85
	max_momentum:   0.95
}) or { panic(err) }

// CyclicLR: triangular cycles between base_lr and max_lr every minibatch
mut cyclic := optimizers.cyclic_lr[f32](optimizers.CyclicLRConfig{
	base_lr:        0.0001
	max_lr:         0.01
	step_size_up:   100
	cycle_momentum: true
	base_momentum:  0.8
	max_momentum:   0.9
}) or { panic(err) }

// ReduceLROnPlateau: reduce when metric stops improving
mut scheduler6 := optimizers.reduce_lr_on_plateau[f32](optimizers.ReduceLROnPlateauConfig{
	patience:       10
	factor:         0.1
	mode:           .min
	threshold_mode: .rel
	min_lr:         1e-6
})

// Inside the training loop:
for step := 0; step < 100; step++ {
	opt.update()!
	current_lr := scheduler.next_lr(0.001, step)
	_ = current_lr
}
```

`OneCycleLR` advances once per minibatch. Its default curve ramps to `max_lr`
and then anneals below the initial rate using cosine interpolation. Choose
`anneal_strategy: 'linear'` for a linear curve or `three_phase: true` to return
to the initial rate before the final decay. By default, momentum cycles
inversely from `max_momentum` to `base_momentum` and back; `next_momentum(step)`
returns the value for the caller to apply to its optimizer. Set
`cycle_momentum: false` to disable the momentum curve. The defaults (`0.85` and
`0.95`) match PyTorch. The constructor returns an error for invalid cycle or
momentum bounds.

`CyclicLR` also advances once per minibatch. By default it uses equal rise and
fall durations and a repeating triangular amplitude. Select `.triangular2` to
halve the amplitude after each cycle or `.exp_range` to scale it by `gamma` on
each step. Set `step_size_down` for an asymmetric cycle; leave it at its default
value of zero to match `step_size_up`. A custom `scale_fn` can define another
amplitude policy; `scale_mode: .cycle` passes its one-based cycle number, while
`.iterations` passes the zero-based minibatch step. By default, momentum follows
an inverse cycle between `max_momentum` and `base_momentum`. Read it through
`next_momentum(step)` and apply the returned option to the optimizer yourself;
set `cycle_momentum: false` when inverse momentum cycling is not wanted. The
scheduler computes both curves without mutating optimizer state.

`MultiStepLR` decays once at each milestone; repeated milestones cause repeated
decays on that epoch. `PolynomialLR` follows
`base_lr * (1 - step / total_iters)^power` and clamps to zero after its final
step. `ConstantLR` applies `factor` before `total_iters`, then returns the
original base rate. These schedulers calculate each value from the base rate
and step supplied by the caller.

`CosineAnnealingWarmRestarts` starts each cycle at the base rate and anneals to
`eta_min`. `t_0` sets the first cycle length; `t_mult` multiplies the length
after each restart. The step can count minibatches or epochs, depending on your
training loop.

`ReduceLROnPlateau` consumes the observed metric and remembers the best value.
Use `mode: .min` for losses or `.max` for scores, and choose `.rel` or `.abs`
thresholds. Call `step(current_lr, validation_metric)` after validation; it
returns the next rate for the caller to apply. It supports patience, cooldown,
`min_lr`, and epsilon. Use `try_reduce_lr_on_plateau` when configuration errors
should be returned instead of panicking. The original delta-based
`next_lr(current_lr, step, metric_delta)` remains available for compatibility.

```v ignore
mut plateau := optimizers.reduce_lr_on_plateau[f32](optimizers.ReduceLROnPlateauConfig{
	patience: 2
	factor:   0.5
	min_lr:   1e-6
})
mut learning_rate := 0.001
for validation_loss in [0.8, 0.7, 0.71, 0.73, 0.74] {
	learning_rate = plateau.step(learning_rate, validation_loss)
}
```

```v ignore
mut one_cycle := optimizers.one_cycle_lr[f32](optimizers.OneCycleLRConfig{
	max_lr:      0.01
	total_steps: 100
}) or { panic(err) }
for step in 0 .. 100 {
	learning_rate := one_cycle.next_lr(0.001, step)
	// Apply `learning_rate` to the optimizer before the next batch.
}
```

```v ignore
mut cyclic := optimizers.cyclic_lr[f32](optimizers.CyclicLRConfig{
	base_lr:        0.0001
	max_lr:         0.01
	step_size_up:   100
	step_size_down: 200
	mode:           .triangular2
}) or { panic(err) }
for step in 0 .. 300 {
	learning_rate := cyclic.next_lr(0.001, step)
	// Apply `learning_rate` before the next batch.
}
```

## Complete Training Loop Example

```v ignore
import vtl
import vtl.autograd
import vtl.nn.models
import vtl.nn.optimizers

mut ctx := autograd.ctx[f32]()
mut model := models.sequential_from_ctx[f32](ctx)
model.input([784])
model.linear(256)
model.linear(10)
model.mse_loss()

mut opt := optimizers.adam_optimizer[f32](optimizers.AdamOptimizerConfig{
	learning_rate: 0.001
})
opt.build_params(model.info.layers)

mut scheduler := optimizers.step_lr[f32](30, 0.1)

for epoch := 0; epoch < 10; epoch++ {
	// Replace with real data batches
	input_batch := vtl.zeros[f32]([64, 784])
	target_batch := vtl.zeros[f32]([64, 10])

	mut x := ctx.variable(input_batch)
	pred := model.forward(x)!
	mut loss_val := model.loss(pred, target_batch)!
	loss_val.backprop()!
	opt.update()!

	current_lr := scheduler.next_lr(0.001, epoch)
	println('Epoch ${epoch}: lr = ${current_lr}')
}
```

## Gradient Clipping

Clip the global L2 norm after backpropagation and before the optimizer update.
The function scales trainable parameter gradients in place and returns the norm
before clipping. Gradients below the limit are left unchanged. The default
`clip_grad_norm` uses L2; `clip_grad_norm_with_type` accepts finite positive
p-norms, the zero-count norm, and the infinity norm.

```v ignore
import vtl
import vtl.autograd
import vtl.nn.optimizers
import math

context := autograd.ctx[f64]()
parameter := context.variable(vtl.from_1d([0.0, 0.0])!)
mut parameters := [parameter]

// Usually populated by loss.backprop().
parameter.grad.set_nth(0, 3.0)
parameter.grad.set_nth(1, 4.0)

original_norm := optimizers.clip_grad_norm[f64](mut parameters, 1.0)!
assert original_norm == 5.0
assert math.abs(parameter.grad.get_nth(0) - 0.6) < 1e-12
assert math.abs(parameter.grad.get_nth(1) - 0.8) < 1e-12

// Use the L1 norm instead of the default L2 norm.
l1_norm := optimizers.clip_grad_norm_with_type[f64](mut parameters, 1.0, 1.0)!

// Alternatively, clip each element independently to [-0.5, 0.5].
optimizers.clip_grad_value[f64](mut parameters, 0.5)!
```

To inspect a norm without changing gradients, use `get_total_grad_norm` (or
`get_total_grad_norm_with_type`). If the norm is already available, pass it to
`clip_grads_with_norm` to apply clipping without calculating it again.

The norm and value limits must be finite and non-negative. Norm clipping rejects
non-finite gradient norms instead of
changing gradients to NaN.

The autograd `Variable.clamp(min, max)` operation also supports element-wise
clamping in a forward graph. Its backward pass passes gradients through values
inside the inclusive bounds and zeroes gradients outside them.
