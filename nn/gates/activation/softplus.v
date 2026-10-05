module activation

import vtl
import vtl.autograd
import vtl.nn.internal

pub struct SoftplusGate[T] {
pub:
	cache &vtl.Tensor[T] = unsafe { nil }
}

pub fn softplus_gate[T](cache &vtl.Tensor[T]) &SoftplusGate[T] {
	return &SoftplusGate[T]{
		cache: cache
	}
}

pub fn (g &SoftplusGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	return [internal.deriv_softplus[T](payload.variable.grad, g.cache)!]
}

fn softplus_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&SoftplusGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

pub fn (g &SoftplusGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			autograd.register[T]('Softplus', voidptr(g), softplus_gate_backward_dispatch[T], result, [a])!
		}
		else {
			return error('Softplus: cache: invalid argument')
		}
	}
}
