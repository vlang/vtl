module loss

import vtl
import vtl.autograd
import vtl.nn.types
import vtl.nn.gates.loss as loss_gates
import vtl.nn.internal

// NLLLoss computes the Negative Log Likelihood loss.
//
// Intended to be used after a log-softmax layer for multi-class classification.
//
// Input shape: `[batch, n_classes]` — log-probabilities (log-softmax output).
// Target shape: `[batch]` class indices or `[batch, n_classes]` one-hot/soft labels.
//
// Fields:
//   - `weight`       — optional per-class weight tensor (`nil` = uniform weights)
//   - `ignore_index` — class index to ignore in class-index targets (default: -1 = none)
//   - `reduction`    — `'none'` | `'mean'` | `'sum'` (default: `'mean'`)
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

// NLLLossOptions configures ignored class indices and scalar reduction.
@[params]
pub struct NLLLossOptions {
pub:
	ignore_index int    = -1
	reduction    string = 'mean'
}

// nll_loss creates a new NLLLoss instance.
//
// `weight` — optional per-class weight tensor; pass `unsafe { nil }` for uniform weights.
pub fn nll_loss[T](weight &vtl.Tensor[T]) types.Loss[T] {
	return nll_loss_with_options[T](weight, NLLLossOptions{})
}

// nll_loss_with_options creates an NLLLoss with a configurable ignore index and
// scalar (`mean` or `sum`) or per-sample (`none`) reduction. `ignore_index`
// applies to class-index targets.
pub fn nll_loss_with_options[T](weight &vtl.Tensor[T], options NLLLossOptions) types.Loss[T] {
	concrete := &NLLLoss[T]{
		weight:       weight
		ignore_index: options.ignore_index
		reduction:    options.reduction
	}
	return types.loss[T](voidptr(concrete), nll_loss_loss_dispatch[T])
}

// loss exposes this operation as part of the public API.
pub fn (l &NLLLoss[T]) loss(input &autograd.Variable[T], target &vtl.Tensor[T]) !&autograd.Variable[T] {
	output := internal.nll[T](input.value, target, l.weight, l.ignore_index, l.reduction)!
	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := loss_gates.nll_loss_weighted_gate_with_options[T](input.value, target, l.weight,
			l.ignore_index, l.reduction)
		gate.cache(mut result, input)!
	}
	return result
}

fn nll_loss_loss_dispatch[T](loss_ptr voidptr, input voidptr, target voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	typed_target := unsafe { &vtl.Tensor[T](target) }
	result := unsafe { (&NLLLoss[T](loss_ptr)).loss(typed_input, typed_target)! }
	return voidptr(result)
}
