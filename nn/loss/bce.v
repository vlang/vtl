module loss

import vtl
import vtl.autograd
import vtl.nn.gates.loss as loss_gates
import vtl.nn.internal

// BCELoss computes Binary Cross Entropy loss: -mean(y * log(p) + (1-y) * log(1-p))
// Input shape: [batch, ...] (binary classification per element)
// Target shape: same as input, values in (0, 1)
pub struct BCELoss[T] {
	from_logits bool
}

@[params]
pub struct BCELossConfig {
	from_logits bool = true // apply sigmoid to input (recommended for numerical stability)
}

pub fn bce_loss[T](config BCELossConfig) &BCELoss[T] {
	return &BCELoss[T]{from_logits: config.from_logits}
}

pub fn (l &BCELoss[T]) loss(input &autograd.Variable[T], target &vtl.Tensor[T]) !&autograd.Variable[T] {
	mut input_val := input.value
	if l.from_logits {
		input_val = internal.sigmoid[T](input_val)
	}
	output := internal.bce[T](input_val, target)!
	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := loss_gates.bce_gate[T](input.value, target, l.from_logits)
		gate.cache(mut result, input)!
	}
	return result
}
