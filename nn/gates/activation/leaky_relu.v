module activation

import vtl
import vtl.autograd
import vtl.nn.internal

// LeakyReluGate defines a public data structure for this module.
pub struct LeakyReluGate[T] {
pub:
	cache &vtl.Tensor[T] = unsafe { nil }
	slope T
}

// leaky_relu_gate exposes this operation as part of the public API.
pub fn leaky_relu_gate[T](cache &vtl.Tensor[T], slope T) &LeakyReluGate[T] {
	return &LeakyReluGate[T]{
		cache: cache
		slope: slope
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &LeakyReluGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := internal.deriv_leaky_relu[T](gradient, g.cache, g.slope)!
	return [r0]
}

fn leaky_relu_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&LeakyReluGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &LeakyReluGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]

	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true

			autograd.register[T]('LeakyRelu', voidptr(g), leaky_relu_gate_backward_dispatch[T],
				result, [a])!
		}
		else {
			return error('LeakyRelu: cache: invalid argument')
		}
	}
}
