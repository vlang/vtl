module loss

import vtl
import vtl.autograd
import vtl.nn.internal

// CrossEntropyLoss combines LogSoftmax + NLLLoss in one forward pass.
// This is more numerically stable than applying softmax then log separately.
// input: [batch_size, n_classes] raw logits
// target: [batch_size, n_classes] one-hot targets OR [batch_size] class indices
pub struct CrossEntropyLoss[T] {
pub:
	weight       &vtl.Tensor[T] = unsafe { nil }
	ignore_index int            = -1
	reduction    string         = 'mean' // 'mean' | 'sum' | 'none'
}

pub fn cross_entropy_loss[T]() &CrossEntropyLoss[T] {
	return &CrossEntropyLoss[T]{
		reduction: 'mean'
	}
}

pub fn (_ &CrossEntropyLoss[T]) loss(input &autograd.Variable[T], target &vtl.Tensor[T]) !&autograd.Variable[T] {
	output := internal.cross_entropy[T](input.value, target)!
	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := cross_entropy_loss_gate[T](input.value, target)
		gate.cache(mut result, input)!
	}
	return result
}

pub struct CrossEntropyLossGate[T] {
pub:
	target &vtl.Tensor[T] = unsafe { nil }
}

pub fn cross_entropy_loss_gate[T](input &vtl.Tensor[T], target &vtl.Tensor[T]) &CrossEntropyLossGate[T] {
	return &CrossEntropyLossGate[T]{
		target: target
	}
}

pub fn (g &CrossEntropyLossGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := internal.cross_entropy_backward[T](gradient, payload.variable.value, g.target)!
	return [r0]
}

pub fn (g &CrossEntropyLossGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			autograd.register[T]('CrossEntropy', g, result, [a])!
		}
		else {
			return error('CrossEntropy: cache: invalid argument')
		}
	}
}
