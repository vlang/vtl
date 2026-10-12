# `vtl.nn.loss`

Differentiable objectives consume a prediction `autograd.Variable[T]` and a
target `vtl.Tensor[T]`, returning a scalar-like variable for backpropagation.
This module currently provides:

| Constructor | Objective and input convention |
| --- | --- |
| `mse_loss` | Mean squared error: `mean((x - y)^2)`. |
| `bce_loss` | Binary cross entropy: `-mean(y log(p) + (1-y) log(1-p))`; applies sigmoid by default (`from_logits: true`). |
| `sigmoid_cross_entropy_loss` | Binary cross entropy from logits, using a sigmoid internally. |
| `cross_entropy_loss` | `-sum(y * log_softmax(logits))`; raw logits with one-hot/soft targets. |
| `softmax_cross_entropy_loss` | Multiclass cross entropy with softmax over logits and one-hot targets. |
| `nll_loss` | Negative log likelihood from log probabilities after log-softmax; accepts class indices or one-hot/soft targets and optional per-class weights. |
| `kl_div_loss` | `sum(P * log(P / Q))`; input is `log(Q)` and target is probability distribution `P`. |
| `huber_loss` | `0.5 d^2` when `|d| <= delta`, otherwise `delta (|d| - 0.5 delta)`, averaged. |

Constructor signatures and reduction behavior are defined in the corresponding
source files. In particular, binary cross entropy's `from_logits` setting and
the expected representation for multiclass losses must match the model output.

`cross_entropy_loss_with_options` supports optional class weights,
`ignore_index`, `label_smoothing` in `[0, 1]`, and `none`, `mean`, or `sum` reductions. Use
`cross_entropy_loss_indices` for `Tensor[int]` targets; it accepts a scalar
target with logits shaped `[C]` or targets shaped `[N, ...]` with logits shaped
`[N, C, ...]`. The generic `cross_entropy_loss` also accepts same-type
integer-valued class-index targets and one-hot/soft targets matching the logits.
For `mean`, weighted class-index targets divide by the sum of their
non-ignored class weights; probability targets divide by the number of samples.

`nll_loss(weight)` uses mean reduction by default. Use `nll_loss_with_options`
to select sum or per-sample reduction, or ignore a class-index target, while
keeping the weight tensor optional:

```v ignore
objective := loss.nll_loss_with_options[f32](class_weights,
	ignore_index: -100,
	reduction: 'sum',
)
```

Use `nll_loss_indices` when class indices are stored in `Tensor[int]`; the
regular `nll_loss` accepts a target tensor with the same element type as input.
Both accept unbatched log-probabilities with shape `[C]` and a scalar
class-index target, or batched log-probabilities with shape `[N, C, ...]` and
class-index targets with shape `[N, ...]`.

NLL `none` reduction returns one loss per target position (a scalar for an
unbatched input). `mean` reduction divides by the total target weight when
class weights are provided, and otherwise averages over non-ignored class-index
targets. Matrix targets are an extra VTL extension; their weighted mean divides
by the number of target positions.

```v ignore
import vtl.nn.loss

objective := loss.mse_loss[f32]()
loss_value := objective.loss(prediction, target)!
loss_value.backprop()!
```

See the [neural-network tutorial](../../docs/TUTORIAL_NEURAL_NETWORKS.md) for a
training loop and [loss tests](loss_test.v) for concrete input conventions.
