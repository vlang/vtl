module loss

import vtl
import vtl.autograd
import vtl.nn.internal

// SigmoidCrossEntropyGate defines a public data structure for this module.
pub struct SigmoidCrossEntropyGate[T] {
pub:
	cache  &autograd.Variable[T] = unsafe { nil }
	target &vtl.Tensor[T]        = unsafe { nil }
}

// sigmoid_cross_entropy_gate exposes this operation as part of the public API.
pub fn sigmoid_cross_entropy_gate[T](cache &autograd.Variable[T], target &vtl.Tensor[T]) &SigmoidCrossEntropyGate[T] {
	return &SigmoidCrossEntropyGate[T]{
		cache:  cache
		target: target
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &SigmoidCrossEntropyGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	return internal.sigmoid_cross_entropy_backward[T](gradient, g.cache.value, g.target)
}

fn sigmoid_cross_entropy_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&SigmoidCrossEntropyGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &SigmoidCrossEntropyGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]

	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true

			autograd.register[T]('SigmoidCrossEntropy', voidptr(g),
				sigmoid_cross_entropy_gate_backward_dispatch[T], result, [a])!
		}
		else {
			return error('SigmoidCrossEntropyGate: cache: invalid argument')
		}
	}
}
