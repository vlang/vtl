module activation

import vtl
import vtl.autograd
import vtl.nn.internal

pub struct SeluGate[T] {
pub:
	cache &vtl.Tensor[T] = unsafe { nil }
}

pub fn selu_gate[T](cache &vtl.Tensor[T]) &SeluGate[T] {
	return &SeluGate[T]{
		cache: cache
	}
}

pub fn (g &SeluGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	return [internal.deriv_selu[T](payload.variable.grad, g.cache)!]
}

fn selu_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&SeluGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

pub fn (g &SeluGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			autograd.register[T]('SELU', voidptr(g), selu_gate_backward_dispatch[T], result, [a])!
		}
		else {
			return error('SELU: cache: invalid argument')
		}
	}
}
