module loss

import vtl
import vtl.autograd
import vtl.nn.gates.loss
import vtl.nn.internal

// SigmoidCrossEntropyLoss
pub struct SigmoidCrossEntropyLoss[T] {}

pub fn sigmoid_cross_entropy_loss[T]() &SigmoidCrossEntropyLoss[T] {
	return &SigmoidCrossEntropyLoss[T]{}
}

pub fn (_ &SigmoidCrossEntropyLoss[T]) loss(input &autograd.Variable[T], target &vtl.Tensor[T]) !&autograd.Variable[T] {
	output := internal.sigmoid_cross_entropy[T](input.value, target)!

	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := loss.sigmoid_cross_entropy_gate[T](input, target)
		gate.cache(mut result, input, target)!
	}

	return result
}
