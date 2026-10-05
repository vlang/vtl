module activation

import vtl
import vtl.autograd
import vtl.nn.internal

pub struct HardSwishGate[T] {
pub:
	cache &vtl.Tensor[T] = unsafe { nil }
}

pub fn hardswish_gate[T](cache &vtl.Tensor[T]) &HardSwishGate[T] {
	return &HardSwishGate[T]{
		cache: cache
	}
}

pub fn (g &HardSwishGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	return [internal.deriv_hardswish[T](payload.variable.grad, g.cache)!]
}

fn hardswish_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&HardSwishGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

pub fn (g &HardSwishGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			autograd.register[T]('HardSwish', voidptr(g), hardswish_gate_backward_dispatch[T], result,
				[a])!
		}
		else {
			return error('HardSwish: cache: invalid argument')
		}
	}
}
