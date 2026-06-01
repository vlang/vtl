module activation

import vtl
import vtl.autograd
import vtl.nn.internal

// EluGate defines a public data structure for this module.
pub struct EluGate[T] {
pub:
	cache &vtl.Tensor[T] = unsafe { nil }
	alpha T
}

// elu_gate exposes this operation as part of the public API.
pub fn elu_gate[T](cache &vtl.Tensor[T], alpha T) &EluGate[T] {
	return &EluGate[T]{
		cache: cache
		alpha: alpha
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &EluGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := internal.deriv_elu[T](gradient, g.cache, g.alpha)!
	return [r0]
}

fn elu_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&EluGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &EluGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]

	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true

			autograd.register[T]('Elu', voidptr(g), elu_gate_backward_dispatch[T], result, [
				a,
			])!
		}
		else {
			return error('Elu: cache: invalid argument')
		}
	}
}
