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
systemd-run --user --scope --quiet --property=MemoryMax=768M --property=MemorySwapMax=0 --setenv=VJOBS=2 -- v test ./vtl/nn/optimizers
systemd-run --user --scope --quiet --property=MemoryMax=768M --property=MemorySwapMax=0 --setenv=VJOBS=2 -- v -prod test ./vtl/nn/optimizers
```

In constrained environments, run each command under a systemd scope with an
appropriate `MemoryMax`.
