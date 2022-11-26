module loss

import vtl
import vtl.autograd

// SoftmaxCrossEntropyLoss
pub struct SoftmaxCrossEntropyLoss[T] {}

pub fn softmax_cross_entropy_loss[T]() &SoftmaxCrossEntropyLoss[T] {
	return &SoftmaxCrossEntropyLoss[T]{}
}

pub fn (_ &SoftmaxCrossEntropyLoss[T]) loss(input &autograd.Variable[T], target &vtl.Tensor[T]) ?&autograd.Variable[T] {
	return error('Not implemented')

	// output := internal.softmax_cross_entropy[T](input.value, target)?

	// mut result := input.context.variable(output)

	// if input.requires_grad {
	// 	gate := loss.softmax_cross_entropy_gate[T](input, target)
	// 	gate.cache(mut result, input, target)?
	// }

	// return result
}
