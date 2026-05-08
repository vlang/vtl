module loss

import vtl
import vtl.autograd
import vtl.nn.gates.loss as loss_gates
import vtl.nn.internal

// NLLLoss computes Negative Log Likelihood loss.
// Input: [..., n_classes] — log-probabilities (log-softmax output)
// Target: [..., n_classes] — one-hot or class indices (if class indices, use reduction='none' or weight)
// reduction: 'mean' | 'sum' | 'none'
pub struct NLLLoss[T] {
pub:
	weight       &vtl.Tensor[T] = unsafe { nil }
	ignore_index int = -1
	reduction    string = 'mean'
}

pub fn nll_loss[T](weight &vtl.Tensor[T]) &NLLLoss[T] {
	return &NLLLoss[T]{
		weight:    weight
		reduction: 'mean'
	}
}

pub fn (_ &NLLLoss[T]) loss(input &autograd.Variable[T], target &vtl.Tensor[T]) !&autograd.Variable[T] {
	output := internal.nll[T](input.value, target)!
	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := loss_gates.nll_loss_gate[T](input.value, target)
		gate.cache(mut result, input)!
	}
	return result
}