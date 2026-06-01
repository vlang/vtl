module loss

import vtl
import vtl.autograd
import vtl.nn.internal

// SoftmaxCrossEntropyGate defines a public data structure for this module.
pub struct SoftmaxCrossEntropyGate[T] {
pub:
	cache  &autograd.Variable[T] = unsafe { nil }
	target &vtl.Tensor[T]        = unsafe { nil }
}

// softmax_cross_entropy_gate exposes this operation as part of the public API.
pub fn softmax_cross_entropy_gate[T](cache &autograd.Variable[T], target &vtl.Tensor[T]) &SoftmaxCrossEntropyGate[T] {
	return &SoftmaxCrossEntropyGate[T]{
		cache:  cache
		target: target
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &SoftmaxCrossEntropyGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	return internal.softmax_cross_entropy_backward[T](gradient, g.cache.value, g.target)
}

fn softmax_cross_entropy_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&SoftmaxCrossEntropyGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &SoftmaxCrossEntropyGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]

	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true

			autograd.register[T]('SoftmaxCrossEntropy', voidptr(g),
				softmax_cross_entropy_gate_backward_dispatch[T], result, [a])!
		}
		else {
			return error('SoftmaxCrossEntropyGate: cache: invalid argument')
		}
	}
}
