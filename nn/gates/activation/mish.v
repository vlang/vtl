module activation

import vtl
import vtl.autograd
import vtl.nn.internal

// MishGate defines a public data structure for this module.
pub struct MishGate[T] {
pub:
	cache &vtl.Tensor[T] = unsafe { nil }
}

// mish_gate exposes this operation as part of the public API.
pub fn mish_gate[T](cache &vtl.Tensor[T]) &MishGate[T] {
	return &MishGate[T]{
		cache: cache
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &MishGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := internal.deriv_mish[T](gradient, g.cache)!
	return [r0]
}

fn mish_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&MishGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &MishGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			autograd.register[T]('Mish', voidptr(g), mish_gate_backward_dispatch[T], result, [
				a,
			])!
		}
		else {
			return error('Mish: cache: invalid argument')
		}
	}
}
