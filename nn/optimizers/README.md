# Neural network optimizers

VTL optimizers update `autograd.Variable[T]` parameters after a loss has been
backpropagated. Construct an optimizer, register the model layers once with
`build_params`, then call `update()` after each `backprop()`.

```v ignore
mut optimizer := optimizers.radam_optimizer[f32](learning_rate: 0.001)
optimizer.build_params(model.info.layers)
loss.backprop()!
optimizer.update()!
```

See the [optimizer tutorial](../../docs/TUTORIAL_OPTIMIZERS.md) for end-to-end
training examples and [neural network tutorial](../../docs/TUTORIAL_NEURAL_NETWORKS.md)
for `Sequential` models.

## Optimizers

| Constructor | Defaults | Notes |
|-------------|----------|-------|
| `sgd[T](learning_rate: 0.01)` | Learning rate 0.01 | Stochastic gradient descent |
| `adagrad[T](...)` | Learning rate 0.01 | Accumulates squared gradients |
| `rmsprop[T](...)` | Learning rate 0.001 | Exponential squared-gradient average |
| `adam_optimizer[T](...)` | LR 0.001, β₁ 0.9, β₂ 0.999 | Adam with bias correction; existing CUDA/Vulkan acceleration |
| `adamw[T](...)` | LR 0.001, decay 0.01 | Adam with decoupled weight decay |
| `nadam_optimizer[T](...)` | LR 0.002, β₁ 0.9, β₂ 0.999, momentum decay 0.004 | Dozat momentum schedule; f32/f64 CPU paths |
| `radam_optimizer[T](...)` | LR 0.001, β₁ 0.9, β₂ 0.999 | Rectified Adam variance estimate; f32 and f64 CPU paths |

NAdam and RAdam configuration structs accept `weight_decay` (default `0`).
Set `decoupled_weight_decay: true` to apply decay directly to parameter values,
without adding it to the gradient or moment estimates. NAdam additionally accepts
`momentum_decay` (default `0.004`).

## Learning-rate schedulers

`StepLR`, `MultiStepLR`, `ExponentialLR`, `CosineAnnealingLR`,
`CosineAnnealingWarmRestarts`, `LinearLR`, `PolynomialLR`, `ConstantLR`,
`CyclicLR`, `OneCycleLR`, and
`ReduceLROnPlateau` update learning rates by epoch or minibatch step.
`MultiStepLR` applies one decay for each milestone, including duplicate
milestones. `LinearLR` interpolates between two factors, while `PolynomialLR`
decays to zero over a fixed number of steps. `ConstantLR` scales the base rate
for a fixed number of steps, then restores it.
`CosineAnnealingWarmRestarts` resets to the base rate after each cosine cycle,
with optional geometric growth of cycle lengths.
`OneCycleLR` uses minibatch steps
to ramp from `max_lr / div_factor` to `max_lr`, then anneal to
`max_lr / (div_factor * final_div_factor)`. It supports cosine or linear
annealing and an optional three-phase curve. It computes learning rates only;
cycling optimizer momentum and parameter groups are not implemented. Its
constructor returns an error for negative `max_lr`, nonpositive `total_steps`
or division factors, and `pct_start` outside `(0, 1)`.
`ReduceLROnPlateau.step(current_lr, metric)` consumes the monitored value and
tracks the best metric, with `min` or `max` mode and absolute or relative
thresholds. It supports `patience`, `cooldown`, `min_lr`, and `epsilon` and
returns the rate for the caller to apply. `try_reduce_lr_on_plateau` reports
invalid configuration; the original `next_lr(current_lr, step, metric_delta)`
remains available for callers that already calculate metric deltas.

`CyclicLR` provides triangular, triangular2, and exp_range amplitude policies
with independently configurable rise and fall durations. It also computes the
inverse momentum curve through `next_momentum(step)` by default; set
`cycle_momentum: false` to disable it. The returned momentum is a value for the
caller to apply. Supply `scale_fn` to define a custom cycle amplitude and set
`scale_mode` to `.cycle` (the default) or `.iterations` to choose whether the
callback receives a one-based cycle number or the zero-based minibatch step,
respectively. Optimizer mutation and parameter groups are not implemented.

```v ignore
import vtl.nn.optimizers

fn decay_by_cycle(cycle int) f64 {
	return 1.0 / f64(cycle + 1)
}

scheduler := optimizers.cyclic_lr[f64](optimizers.CyclicLRConfig{
	base_lr:      0.001
	max_lr:       0.01
	step_size_up: 100
	scale_fn:     decay_by_cycle
})!
lr := scheduler.next_lr(0.001, 150)
```

## NAdam

NAdam incorporates Nesterov momentum into Adam. It uses the scheduled momentum
coefficient `μ_t = β₁(1 - 0.5 × 0.96^(t × momentum_decay))` and the corresponding
bias-corrected look-ahead estimate. Its defaults follow PyTorch's NAdam API.

```v ignore
mut optimizer := optimizers.nadam_optimizer[f64](
	learning_rate: 0.002
	momentum_decay: 0.004
)
```

## RAdam

RAdam rectifies the adaptive variance once its running estimate is reliable. It
uses the unrectified bias-corrected momentum during early steps and the
rectification factor after `rho_t > 5`, matching PyTorch's threshold and
epsilon placement.

```v ignore
mut optimizer := optimizers.radam_optimizer[f64](
	learning_rate: 0.001
	weight_decay: 0.01
	decoupled_weight_decay: true
)
```

## Validation

Run commands from the V module workspace, outside this repository:

```sh
cd ~/.vmodules
systemd-run --user --scope --quiet --property=MemoryMax=768M \
  --property=MemorySwapMax=0 --setenv=VJOBS=2 -- \
  v test ./vtl/nn/optimizers
systemd-run --user --scope --quiet --property=MemoryMax=768M \
  --property=MemorySwapMax=0 --setenv=VJOBS=2 -- \
  v -prod test ./vtl/nn/optimizers
```

In constrained environments, run each command under a systemd scope with an
appropriate `MemoryMax`.
