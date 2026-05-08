module loss

import vtl
import vtl.autograd
import vtl.nn.gates.loss as loss_gates
import vtl.nn.internal

// HuberLoss (also known as Smooth L1 Loss) combines L1 and L2 loss, controlled by `delta`.
//
// Formula:
//   `L = mean(0.5·(x-y)²)               if |x-y| ≤ delta`
//   `L = mean(delta·(|x-y| - 0.5·delta)) if |x-y| > delta`
//
// Behaves like MSE for small errors and like MAE for large errors,
// making it more robust to outliers than pure MSE.
//
// Example:
// ```v
// import vtl.nn.loss
// l := loss.huber_loss[f64](delta: 1.0)
// out := l.loss(prediction, target)!
// ```
pub struct HuberLoss[T] {
	delta T
}

// HuberLossConfig configures HuberLoss.
//
// Fields:
//   - `delta` — transition threshold between L2 (below) and L1 (above) behaviour. Default: 1.0.
@[params]
pub struct HuberLossConfig {
	delta f64 = 1.0
}

// huber_loss creates a new HuberLoss instance.
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
