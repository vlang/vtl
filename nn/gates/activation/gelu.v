module activation

import vtl
import vtl.autograd
import vtl.nn.internal

// GELUGate defines a public data structure for this module.
pub struct GELUGate[T] {
pub:
	cache &vtl.Tensor[T] = unsafe { nil }
}

// gelu_gate exposes this operation as part of the public API.
pub fn gelu_gate[T](cache &vtl.Tensor[T]) &GELUGate[T] {
	return &GELUGate[T]{
		cache: cache
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &GELUGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := internal.deriv_gelu[T](gradient, g.cache)!
	return [r0]
}

fn gelu_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&GELUGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &GELUGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			autograd.register[T]('GELU', voidptr(g), gelu_gate_backward_dispatch[T], result, [
				a,
			])!
		}
		else {
			return error('GELU: cache: invalid argument')
		}
	}
}
