module loss

import vtl
import vtl.autograd
import vtl.nn.gates.loss as loss_gates
import vtl.nn.internal

// SoftmaxCrossEntropyLoss defines a public data structure for this module.
pub struct SoftmaxCrossEntropyLoss[T] {}

// softmax_cross_entropy_loss exposes this operation as part of the public API.
pub fn softmax_cross_entropy_loss[T]() &SoftmaxCrossEntropyLoss[T] {
	return &SoftmaxCrossEntropyLoss[T]{}
}

// loss exposes this operation as part of the public API.
pub fn (_ &SoftmaxCrossEntropyLoss[T]) loss(input &autograd.Variable[T], target &vtl.Tensor[T]) !&autograd.Variable[T] {
	output := internal.softmax_cross_entropy[T](input.value, target)!

	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := loss_gates.softmax_cross_entropy_gate[T](input, target)
		gate.cache(mut result, input)!
	}

	return result
}
