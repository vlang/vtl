module activation

import vtl
import vtl.autograd
import vtl.nn.internal

// TanhGate defines a public data structure for this module.
pub struct TanhGate[T] {
pub:
	cache &vtl.Tensor[T] = unsafe { nil }
}

// tanh_gate exposes this operation as part of the public API.
pub fn tanh_gate[T](cache &vtl.Tensor[T]) &TanhGate[T] {
	return &TanhGate[T]{
		cache: cache
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &TanhGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := internal.deriv_tanh[T](gradient, g.cache)!
	return [r0]
}

fn tanh_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&TanhGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &TanhGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			autograd.register[T]('Tanh', voidptr(g), tanh_gate_backward_dispatch[T], result, [
				a,
			])!
		}
		else {
			return error('Tanh: cache: invalid argument')
		}
	}
}
