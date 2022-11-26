module loss

import vtl
import vtl.autograd
import vtl.nn.gates.loss
import vtl.nn.internal

// MSELoss
pub struct MSELoss[T] {}

pub fn mse_loss[T]() &MSELoss[T] {
	return &MSELoss[T]{}
}

pub fn (_ &MSELoss[T]) loss(input &autograd.Variable[T], target &vtl.Tensor[T]) ?&autograd.Variable[T] {
	output := internal.mse[T](input.value, target)?

	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := loss.mse_gate[T](input, target)
		gate.cache(mut result, input, target)?
	}

	return result
}
