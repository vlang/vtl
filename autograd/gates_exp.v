module autograd

import vtl

// ExpGate defines a public data structure for this module.
pub struct ExpGate[T] {
pub:
	a &Variable[T] = unsafe { nil }
}

// exp_gate exposes this operation as part of the public API.
pub fn exp_gate[T](a &Variable[T]) &ExpGate[T] {
	return &ExpGate[T]{
		a: a
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &ExpGate[T]) backward(payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := gradient.multiply[T](g.a.value.exp[T]())!
	return [r0]
}

fn exp_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &Payload[T](payload) }
	tensors := unsafe { (&ExpGate[T](gate)).backward(typed_payload)! }
	return tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &ExpGate[T]) cache(mut result Variable[T], args ...CacheParam) ! {
	a := args[0]

	match a {
		Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true

			register[T]('Exp', voidptr(g), exp_gate_backward_dispatch[T], result, [a])!
		}
		else {
			return error('ExpGate: a must be a Variable')
		}
	}
}
