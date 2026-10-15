module loss

import vtl
import vtl.autograd
import vtl.nn.types
import vtl.nn.gates.loss as loss_gates
import vtl.nn.internal

// SmoothL1Loss computes PyTorch-style Smooth L1 with configurable reduction.
//
// For error `d`, the loss is `0.5*d*d/beta` when `|d| < beta`, and
// `|d| - 0.5*beta` otherwise. A beta of zero selects absolute error. This is
// distinct from Huber loss, whose outer linear region is scaled by `delta`.
//
// Example:
// ```v ignore
// import vtl.nn.loss
// objective := loss.smooth_l1_loss[f32](beta: 0.5)
// result := objective.loss(prediction, target)!
// ```
pub struct SmoothL1Loss[T] {
	beta      T
	reduction string
}

// SmoothL1LossConfig configures the transition point between quadratic and
// linear loss regions. Beta must be finite and non-negative; zero selects
// absolute error. The default beta is 1.
@[params]
pub struct SmoothL1LossConfig {
	beta      f64    = 1.0
	reduction string = 'mean'
}

// smooth_l1_loss creates a SmoothL1Loss with the configured reduction.
pub fn smooth_l1_loss[T](config SmoothL1LossConfig) types.Loss[T] {
	concrete := &SmoothL1Loss[T]{
		beta:      vtl.cast[T](config.beta)
		reduction: config.reduction
	}
	return types.loss[T](voidptr(concrete), smooth_l1_loss_dispatch[T])
}

// loss computes the configured Smooth L1 reduction and attaches its gradient.
pub fn (l &SmoothL1Loss[T]) loss(input &autograd.Variable[T], target &vtl.Tensor[T]) !&autograd.Variable[T] {
	output := internal.smooth_l1[T](input.value, target, l.beta, l.reduction)!
	mut result := input.context.variable(output)
	if input.requires_grad {
		gate := loss_gates.smooth_l1_loss_gate[T](input.value, target, l.beta, l.reduction)
		gate.cache(mut result, input)!
	}
	return result
}

fn smooth_l1_loss_dispatch[T](loss_ptr voidptr, input voidptr, target voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	typed_target := unsafe { &vtl.Tensor[T](target) }
	result := unsafe { (&SmoothL1Loss[T](loss_ptr)).loss(typed_input, typed_target)! }
	return voidptr(result)
}
