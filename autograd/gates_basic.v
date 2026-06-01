module autograd

import vtl

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

// cache exposes this operation as part of the public API.
pub fn (g &AddGate[T]) cache(mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	b := args[1]

	match a {
		Variable[T] {
			match b {
				Variable[T] {
					result.grad = vtl.zeros_like[T](result.value)
					result.requires_grad = true

					register[T]('Add', g, result, [a, b])!
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

// cache exposes this operation as part of the public API.
pub fn (g &SubtractGate[T]) cache(mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	b := args[1]

	match a {
		Variable[T] {
			match b {
				Variable[T] {
					result.grad = vtl.zeros_like[T](result.value)
					result.requires_grad = true

					register[T]('Sub', g, result, [a, b])!
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
	r0 := gradient.multiply[T](g.b.value)!
	r1 := gradient.multiply[T](g.a.value)!
	return [r0, r1]
}

// cache exposes this operation as part of the public API.
pub fn (g &MultiplyGate[T]) cache(mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	b := args[1]

	match a {
		Variable[T] {
			match b {
				Variable[T] {
					result.grad = vtl.zeros_like[T](result.value)
					result.requires_grad = true

					register[T]('Multiply', g, result, [a, b])!
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
	r0 := gradient.divide[T](g.b.value)!
	bx2 := g.b.value.multiply_scalar[T](vtl.cast[T](2))!
	opposite := gradient.multiply_scalar[T](vtl.cast[T](-1))!
	mut r1 := opposite.multiply[T](g.a.value)!
	r1 = r1.divide[T](bx2)!
	return [r0, r1]
}

// cache exposes this operation as part of the public API.
pub fn (g &DivideGate[T]) cache(mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	b := args[1]

	match a {
		Variable[T] {
			match b {
				Variable[T] {
					result.grad = vtl.zeros_like[T](result.value)
					result.requires_grad = true

					register[T]('Divide', g, result, [a, b])!
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
