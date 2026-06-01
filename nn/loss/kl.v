module loss

import vtl
import vtl.autograd
import vtl.nn.types
import vtl.nn.gates.loss as loss_gates
import vtl.nn.internal

// KLDivLoss computes the Kullback-Leibler divergence loss.
//
// Formula: `D_KL(P ‖ Q) = sum(P · log(P / Q))`
//
// Input:  `[batch, n_classes]` — Q log-probabilities (e.g. log-softmax output)
// Target: `[batch, n_classes]` — P probabilities (non-negative, sum to 1 per sample)
//
// Fields:
//   - `reduction` — `'mean'` | `'sum'` | `'none'` (default: `'mean'`)
//
// Example:
// ```v
// import vtl.nn.loss
// l := loss.kl_div_loss[f64]()
// out := l.loss(log_q, p_target)!
// ```
pub struct KLDivLoss[T] {
pub:
	reduction string = 'mean'
}

// kl_div_loss creates a new KLDivLoss instance with `reduction = 'mean'`.
pub fn kl_div_loss[T]() types.Loss[T] {
	concrete := &KLDivLoss[T]{
		reduction: 'mean'
	}
	return types.loss[T](voidptr(concrete), kl_div_loss_loss_dispatch[T])
}

// loss exposes this operation as part of the public API.
pub fn (_ &KLDivLoss[T]) loss(input &autograd.Variable[T], target &vtl.Tensor[T]) !&autograd.Variable[T] {
	output := internal.kl_div[T](input.value, target)!
	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := loss_gates.kl_div_loss_gate[T](input.value, target)
		gate.cache(mut result, input)!
	}
	return result
}

fn kl_div_loss_loss_dispatch[T](loss_ptr voidptr, input voidptr, target voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	typed_target := unsafe { &vtl.Tensor[T](target) }
	result := unsafe { (&KLDivLoss[T](loss_ptr)).loss(typed_input, typed_target)! }
	return voidptr(result)
}
