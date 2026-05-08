module loss

import vtl
import vtl.autograd
import vtl.nn.gates.loss as loss_gates
import vtl.nn.internal

// NLLLoss computes the Negative Log Likelihood loss.
//
// Intended to be used after a log-softmax layer for multi-class classification.
//
// Input shape:  `[batch, n_classes]` — log-probabilities (log-softmax output)
// Target shape: `[batch, n_classes]` — one-hot class labels
//
// Fields:
//   - `weight`       — optional per-class weight tensor (`nil` = uniform weights)
//   - `ignore_index` — class index to ignore in the loss computation (default: -1 = none)
//   - `reduction`    — `'mean'` | `'sum'` | `'none'` (default: `'mean'`)
//
// Example:
// ```v
// import vtl.nn.loss
// l := loss.nll_loss[f64](unsafe { nil })
// log_probs := log_softmax_output  // shape [batch, n_classes]
// out := l.loss(log_probs, one_hot_target)!
// ```
pub struct NLLLoss[T] {
pub:
	weight       &vtl.Tensor[T] = unsafe { nil }
	ignore_index int            = -1
	reduction    string         = 'mean'
}

// nll_loss creates a new NLLLoss instance.
//
// `weight` — optional per-class weight tensor; pass `unsafe { nil }` for uniform weights.
pub fn nll_loss[T](weight &vtl.Tensor[T]) &NLLLoss[T] {
	return &NLLLoss[T]{
		weight:    weight
		reduction: 'mean'
	}
}

pub fn (_ &NLLLoss[T]) loss(input &autograd.Variable[T], target &vtl.Tensor[T]) !&autograd.Variable[T] {
	output := internal.nll[T](input.value, target)!
	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := loss_gates.nll_loss_gate[T](input.value, target)
		gate.cache(mut result, input)!
	}
	return result
}
