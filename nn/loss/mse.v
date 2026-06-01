module loss

import vtl
import vtl.autograd
import vtl.nn.gates.loss
import vtl.nn.internal

// MSELoss computes the Mean Squared Error between the model output and the target.
//
// Formula: `L = mean((input - target)²)`
//
// Typically used for regression tasks.
//
// Example:
// ```v
// import vtl.nn.loss
// l := loss.mse_loss[f64]()
// out  := l.loss(prediction, target)!
// ```
pub struct MSELoss[T] {}

// mse_loss creates a new MSELoss instance.
pub fn mse_loss[T]() &MSELoss[T] {
	return &MSELoss[T]{}
}

// loss exposes this operation as part of the public API.
pub fn (_ &MSELoss[T]) loss(input &autograd.Variable[T], target &vtl.Tensor[T]) !&autograd.Variable[T] {
	output := internal.mse[T](input.value, target)!

	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := loss.mse_gate[T](input, target)
		gate.cache(mut result, input, target)!
	}

	return result
}
