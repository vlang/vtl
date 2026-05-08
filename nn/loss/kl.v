module loss

import vtl
import vtl.autograd
import vtl.nn.gates.loss as loss_gates
import vtl.nn.internal

// KLDivLoss computes KL Divergence loss: D_KL(P || Q) = sum(P * log(P/Q))
// Input: [..., n_classes] — log-probabilities Q
// Target: [..., n_classes] — target probabilities P
pub struct KLDivLoss[T] {
pub:
	reduction string = 'mean'
}

pub fn kl_div_loss[T]() &KLDivLoss[T] {
	return &KLDivLoss[T]{reduction: 'mean'}
}

pub fn (_ &KLDivLoss[T]) loss(input &autograd.Variable[T], target &vtl.Tensor[T]) !&autograd.Variable[T] {
	output := internal.kl_div[T](input.value, target)!
	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := loss_gates.kl_div_loss_gate[T](input.value, target)
		gate.cache(mut result, input)!
	}
	return result
}