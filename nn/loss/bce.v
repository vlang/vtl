module loss

import vtl
import vtl.autograd
import vtl.nn.gates.loss as loss_gates
import vtl.nn.internal

// BCELoss computes Binary Cross-Entropy loss per element and returns the mean.
//
// Formula (when `from_logits = false`):
//   `L = -mean(y·log(p) + (1-y)·log(1-p))`
//
// When `from_logits = true` (default), a sigmoid is applied to `input` first,
// which is numerically more stable than computing sigmoid outside the loss.
//
// Shape: `input` and `target` are both `[batch, ...]` with values in (0, 1) for target.
//
// Example:
// ```v
// import vtl.nn.loss
// l := loss.bce_loss[f64]()          // from_logits: true
// out := l.loss(logits, labels)!
// ```
pub struct BCELoss[T] {
	from_logits bool
}

// BCELossConfig configures BCELoss.
//
// Fields:
//   - `from_logits` — when `true` (default) a sigmoid is applied to the raw model output
//     before computing the loss. Recommended for numerical stability.
@[params]
pub struct BCELossConfig {
	from_logits bool = true // apply sigmoid to input (recommended for numerical stability)
}

// bce_loss creates a new BCELoss instance.
pub fn bce_loss[T](config BCELossConfig) &BCELoss[T] {
	return &BCELoss[T]{
		from_logits: config.from_logits
	}
}

pub fn (l &BCELoss[T]) loss(input &autograd.Variable[T], target &vtl.Tensor[T]) !&autograd.Variable[T] {
	mut input_val := input.value
	if l.from_logits {
		input_val = internal.sigmoid[T](input_val)
	}
	output := internal.bce[T](input_val, target)!
	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := loss_gates.bce_gate[T](input.value, target, l.from_logits)
		gate.cache(mut result, input)!
	}
	return result
}
