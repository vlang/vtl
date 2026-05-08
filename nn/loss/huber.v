module loss

import vtl
import vtl.autograd
import vtl.nn.gates.loss as loss_gates
import vtl.nn.internal

// HuberLoss computes the Huber loss (smooth L1 loss):
//   L = mean(delta * (|x - y| - 0.5 * delta)) if |x-y| > delta
//       mean(0.5 * (x - y)^2)                    otherwise
// delta is a threshold that controls the transition between L1 and L2.
pub struct HuberLoss[T] {
	delta T
}

@[params]
pub struct HuberLossConfig {
	delta f64 = 1.0
}

pub fn huber_loss[T](config HuberLossConfig) &HuberLoss[T] {
	return &HuberLoss[T]{
		delta: vtl.cast[T](config.delta)
	}
}

pub fn (l &HuberLoss[T]) loss(input &autograd.Variable[T], target &vtl.Tensor[T]) !&autograd.Variable[T] {
	output := internal.huber[T](input.value, target, l.delta)!
	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := loss_gates.huber_loss_gate[T](input.value, target, l.delta)
		gate.cache(mut result, input)!
	}
	return result
}
