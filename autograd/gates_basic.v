module autograd

import vtl

pub struct AddGate[T] {}

pub fn add_gate[T]() &AddGate[T] {
	return &AddGate[T]{}
}

pub fn (g &AddGate[T]) backward[T](payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	return [gradient, gradient]
}

pub fn (g &AddGate[T]) cache[T](mut result Variable[T], args ...CacheParam) ! {
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

pub struct SubstractGate[T] {}

pub fn subtract_gate[T]() &SubstractGate[T] {
	return &SubstractGate[T]{}
}

pub fn (g &SubstractGate[T]) backward[T](payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	oposite := gradient.multiply_scalar[T](vtl.cast[T](-1))!
	return [gradient, oposite]
}

pub fn (g &SubstractGate[T]) cache[T](mut result Variable[T], args ...CacheParam) ! {
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
					panic('SubGate: b must be a Variable')
				}
			}
		}
		else {
			panic('SubGate: a must be a Variable')
		}
	}
}

pub struct MultiplyGate[T] {
pub:
	a &Variable[T]
	b &Variable[T]
}

pub fn multiply_gate[T](a &Variable[T], b &Variable[T]) &MultiplyGate[T] {
	return &MultiplyGate[T]{
		a: a
		b: b
	}
}

pub fn (g &MultiplyGate[T]) backward[T](payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := gradient.multiply[T](g.b.value)!
	r1 := gradient.multiply[T](g.a.value)!
	return [r0, r1]
}

pub fn (g &MultiplyGate[T]) cache[T](mut result Variable[T], args ...CacheParam) ! {
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
					panic('MultiplyGate: b must be a Variable')
				}
			}
		}
		else {
			panic('MultiplyGate: a must be a Variable')
		}
	}
}

pub struct DivideGate[T] {
pub:
	a &Variable[T]
	b &Variable[T]
}

pub fn divide_gate[T](a &Variable[T], b &Variable[T]) &DivideGate[T] {
	return &DivideGate[T]{
		a: a
		b: b
	}
}

pub fn (g &DivideGate[T]) backward[T](payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := gradient.divide[T](g.b.value)!
	bx2 := g.b.value.multiply_scalar[T](vtl.cast[T](2))!
	oposite := gradient.multiply_scalar[T](vtl.cast[T](-1))!
	mut r1 := oposite.multiply[T](g.a.value)!
	r1 = r1.divide[T](bx2)!
	return [r0, r1]
}

pub fn (g &DivideGate[T]) cache[T](mut result Variable[T], args ...CacheParam) ! {
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
					panic('DivideGate: b must be a Variable')
				}
			}
		}
		else {
			panic('DivideGate: a must be a Variable')
		}
	}
}
