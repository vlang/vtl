module loss

import vtl
import vtl.autograd
import vtl.nn.types
import vtl.nn.gates.loss
import vtl.nn.internal

// SigmoidCrossEntropyLoss
pub struct SigmoidCrossEntropyLoss[T] {}

// sigmoid_cross_entropy_loss exposes this operation as part of the public API.
pub fn sigmoid_cross_entropy_loss[T]() types.Loss[T] {
	concrete := &SigmoidCrossEntropyLoss[T]{}
	return types.loss[T](voidptr(concrete), sigmoid_cross_entropy_loss_loss_dispatch[T])
}

// loss exposes this operation as part of the public API.
pub fn (_ &SigmoidCrossEntropyLoss[T]) loss(input &autograd.Variable[T], target &vtl.Tensor[T]) !&autograd.Variable[T] {
	output := internal.sigmoid_cross_entropy[T](input.value, target)!

	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := loss.sigmoid_cross_entropy_gate[T](input, target)
		gate.cache(mut result, input, target)!
	}

	return result
}

fn sigmoid_cross_entropy_loss_loss_dispatch[T](loss_ptr voidptr, input voidptr, target voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	typed_target := unsafe { &vtl.Tensor[T](target) }
	result := unsafe { (&SigmoidCrossEntropyLoss[T](loss_ptr)).loss(typed_input, typed_target)! }
	return voidptr(result)
}
