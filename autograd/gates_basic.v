module autograd

import vtl

enum BinaryGateBackwardOperation {
	multiply
	divide
}

// fused_binary_gate_backward calculates both input gradients in one traversal
// for same-shaped contiguous tensors. Broadcasted and strided inputs retain the
// general tensor-operation path below.
@[direct_array_access]
fn fused_binary_gate_backward[T](gradient &vtl.Tensor[T], left &vtl.Tensor[T], right &vtl.Tensor[T],
	operation BinaryGateBackwardOperation) ?[]&vtl.Tensor[T] {
	if gradient.shape != left.shape || gradient.shape != right.shape || !gradient.is_row_major_contiguous()
		|| !left.is_row_major_contiguous() || !right.is_row_major_contiguous()
		|| gradient.data.data.len != gradient.size || left.data.data.len != left.size
		|| right.data.data.len != right.size {
		return none
	}
	mut left_gradient := vtl.tensor_like[T](gradient)
	mut right_gradient := vtl.tensor_like[T](gradient)
	for i in 0 .. gradient.size {
		grad := gradient.data.data[i]
		left_value := left.data.data[i]
		right_value := right.data.data[i]
		match operation {
			.multiply {
				left_gradient.data.data[i] = grad * right_value
				right_gradient.data.data[i] = grad * left_value
			}
			.divide {
				left_gradient.data.data[i] = grad / right_value
				right_gradient.data.data[i] = -(grad * left_value) / (right_value * right_value)
			}
		}
	}
	return [left_gradient, right_gradient]
}

// AddGate defines a public data structure for this module.
pub struct AddGate[T] {}

// add_gate exposes this operation as part of the public API.
pub fn add_gate[T]() &AddGate[T] {
	return &AddGate[T]{}
}

// backward exposes this operation as part of the public API.
pub fn (g &AddGate[T]) backward(payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	return [gradient, gradient]
}

fn add_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &Payload[T](payload) }
	tensors := unsafe { (&AddGate[T](gate)).backward(typed_payload)! }
	return tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &AddGate[T]) cache(mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	b := args[1]

	match a {
		Variable[T] {
			match b {
				Variable[T] {
					register[T]('Add', voidptr(g), add_gate_backward_dispatch[T], result, [
						a,
						b,
					])!
				}
				else {
					return error('AddGate: b must be a Variable')
				}
			}
		}
		else {
			return error('AddGate: a must be a Variable')
		}
	}
}

// SubtractGate defines a public data structure for this module.
pub struct SubtractGate[T] {}

// subtract_gate exposes this operation as part of the public API.
pub fn subtract_gate[T]() &SubtractGate[T] {
	return &SubtractGate[T]{}
}

// backward exposes this operation as part of the public API.
pub fn (g &SubtractGate[T]) backward(payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	opposite := gradient.multiply_scalar[T](vtl.cast[T](-1))!
	return [gradient, opposite]
}

fn subtract_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &Payload[T](payload) }
	tensors := unsafe { (&SubtractGate[T](gate)).backward(typed_payload)! }
	return tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &SubtractGate[T]) cache(mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	b := args[1]

	match a {
		Variable[T] {
			match b {
				Variable[T] {
					register[T]('Sub', voidptr(g), subtract_gate_backward_dispatch[T], result, [
						a,
						b,
					])!
				}
				else {
					return error('SubtractGate: b must be a Variable')
				}
			}
		}
		else {
			return error('SubtractGate: a must be a Variable')
		}
	}
}

// MultiplyGate defines a public data structure for this module.
pub struct MultiplyGate[T] {
pub:
	a &Variable[T] = unsafe { nil }
	b &Variable[T] = unsafe { nil }
}

// multiply_gate exposes this operation as part of the public API.
pub fn multiply_gate[T](a &Variable[T], b &Variable[T]) &MultiplyGate[T] {
	return &MultiplyGate[T]{
		a: a
		b: b
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &MultiplyGate[T]) backward(payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	if fused := fused_binary_gate_backward[T](gradient, g.a.value, g.b.value, .multiply) {
		return fused
	}
	r0 := gradient.multiply[T](g.b.value)!
	r1 := gradient.multiply[T](g.a.value)!
	return [r0, r1]
}

fn multiply_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &Payload[T](payload) }
	tensors := unsafe { (&MultiplyGate[T](gate)).backward(typed_payload)! }
	return tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &MultiplyGate[T]) cache(mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	b := args[1]

	match a {
		Variable[T] {
			match b {
				Variable[T] {
					register[T]('Multiply', voidptr(g), multiply_gate_backward_dispatch[T], result, [
						a,
						b,
					])!
				}
				else {
					return error('MultiplyGate: b must be a Variable')
				}
			}
		}
		else {
			return error('MultiplyGate: a must be a Variable')
		}
	}
}

// DivideGate defines a public data structure for this module.
pub struct DivideGate[T] {
pub:
	a &Variable[T] = unsafe { nil }
	b &Variable[T] = unsafe { nil }
}

// divide_gate exposes this operation as part of the public API.
pub fn divide_gate[T](a &Variable[T], b &Variable[T]) &DivideGate[T] {
	return &DivideGate[T]{
		a: a
		b: b
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &DivideGate[T]) backward(payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	if fused := fused_binary_gate_backward[T](gradient, g.a.value, g.b.value, .divide) {
		return fused
	}
	r0 := gradient.divide[T](g.b.value)!
	b_squared := g.b.value.multiply[T](g.b.value)!
	opposite := gradient.multiply_scalar[T](vtl.cast[T](-1))!
	mut r1 := opposite.multiply[T](g.a.value)!
	r1 = r1.divide[T](b_squared)!
	return [r0, r1]
}

fn divide_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &Payload[T](payload) }
	tensors := unsafe { (&DivideGate[T](gate)).backward(typed_payload)! }
	return tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &DivideGate[T]) cache(mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	b := args[1]

	match a {
		Variable[T] {
			match b {
				Variable[T] {
					register[T]('Divide', voidptr(g), divide_gate_backward_dispatch[T], result, [
						a,
						b,
					])!
				}
				else {
					return error('DivideGate: b must be a Variable')
				}
			}
		}
		else {
			return error('DivideGate: a must be a Variable')
		}
	}
}
