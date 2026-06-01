module autograd

import vtl

// SinGate defines a public data structure for this module.
pub struct SinGate[T] {
pub:
	a &Variable[T] = unsafe { nil }
}

// sin_gate exposes this operation as part of the public API.
pub fn sin_gate[T](a &Variable[T]) &SinGate[T] {
	return &SinGate[T]{
		a: a
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &SinGate[T]) backward(payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := gradient.multiply[T](g.a.value.cos[T]())!
	return [r0]
}

fn sin_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &Payload[T](payload) }
	tensors := unsafe { (&SinGate[T](gate)).backward(typed_payload)! }
	return tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &SinGate[T]) cache(mut result Variable[T], args ...CacheParam) ! {
	a := args[0]

	match a {
		Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true

			register[T]('Sin', voidptr(g), sin_gate_backward_dispatch[T], result, [a])!
		}
		else {
			return error('SinGate: a must be a Variable')
		}
	}
}

// CosGate defines a public data structure for this module.
pub struct CosGate[T] {
pub:
	a &Variable[T] = unsafe { nil }
}

// cos_gate exposes this operation as part of the public API.
pub fn cos_gate[T](a &Variable[T]) &CosGate[T] {
	return &CosGate[T]{
		a: a
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &CosGate[T]) backward(payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := gradient.multiply[T](g.a.value.sin[T]().multiply_scalar[T](vtl.cast[T](-1))!)!
	return [r0]
}

fn cos_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &Payload[T](payload) }
	tensors := unsafe { (&CosGate[T](gate)).backward(typed_payload)! }
	return tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &CosGate[T]) cache(mut result Variable[T], args ...CacheParam) ! {
	a := args[0]

	match a {
		Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true

			register[T]('Cos', voidptr(g), cos_gate_backward_dispatch[T], result, [a])!
		}
		else {
			return error('CosGate: a must be a Variable')
		}
	}
}

// TanGate defines a public data structure for this module.
pub struct TanGate[T] {
pub:
	a &Variable[T] = unsafe { nil }
}

// tan_gate exposes this operation as part of the public API.
pub fn tan_gate[T](a &Variable[T]) &TanGate[T] {
	return &TanGate[T]{
		a: a
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &TanGate[T]) backward(payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	cos := g.a.value.cos[T]()
	r0 := gradient.divide[T](cos.multiply[T](cos)!)!
	return [r0]
}

fn tan_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &Payload[T](payload) }
	tensors := unsafe { (&TanGate[T](gate)).backward(typed_payload)! }
	return tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &TanGate[T]) cache(mut result Variable[T], args ...CacheParam) ! {
	a := args[0]

	match a {
		Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true

			register[T]('Tan', voidptr(g), tan_gate_backward_dispatch[T], result, [a])!
		}
		else {
			return error('TanGate: a must be a Variable')
		}
	}
}
