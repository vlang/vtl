module loss

import vtl
import vtl.autograd
import vtl.nn.types
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

// cross_entropy_loss exposes this operation as part of the public API.
pub fn cross_entropy_loss[T]() types.Loss[T] {
	concrete := &CrossEntropyLoss[T]{
		reduction: 'mean'
	}
	return types.loss[T](voidptr(concrete), cross_entropy_loss_loss_dispatch[T])
}

// loss exposes this operation as part of the public API.
pub fn (_ &CrossEntropyLoss[T]) loss(input &autograd.Variable[T], target &vtl.Tensor[T]) !&autograd.Variable[T] {
	output := internal.cross_entropy[T](input.value, target)!
	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := cross_entropy_loss_gate[T](input.value, target)
		gate.cache(mut result, input)!
	}
	return result
}

fn cross_entropy_loss_loss_dispatch[T](loss_ptr voidptr, input voidptr, target voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	typed_target := unsafe { &vtl.Tensor[T](target) }
	result := unsafe { (&CrossEntropyLoss[T](loss_ptr)).loss(typed_input, typed_target)! }
	return voidptr(result)
}

// CrossEntropyLossGate defines a public data structure for this module.
pub struct CrossEntropyLossGate[T] {
pub:
	target &vtl.Tensor[T] = unsafe { nil }
}

// cross_entropy_loss_gate exposes this operation as part of the public API.
pub fn cross_entropy_loss_gate[T](input &vtl.Tensor[T], target &vtl.Tensor[T]) &CrossEntropyLossGate[T] {
	return &CrossEntropyLossGate[T]{
		target: target
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &CrossEntropyLossGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := internal.cross_entropy_backward[T](gradient, payload.variable.value, g.target)!
	return [r0]
}

fn cross_entropy_loss_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&CrossEntropyLossGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &CrossEntropyLossGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			autograd.register[T]('CrossEntropy', voidptr(g),
				cross_entropy_loss_gate_backward_dispatch[T], result, [a])!
		}
		else {
			return error('CrossEntropy: cache: invalid argument')
		}
	}
}
