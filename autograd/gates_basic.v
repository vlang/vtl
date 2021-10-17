module autograd

import vtl

pub struct AddGate<T> {}

pub fn new_add_gate<T>() &AddGate<T> {
	return &AddGate<T>{}
}

pub fn (g &AddGate<T>) backward<T>(payload &Payload<T>) []&vtl.Tensor<T> {
	gradient := payload.variable.grad
	return [gradient, gradient]
}

pub fn (g &AddGate<T>) cache<T>(mut result Variable<T>, args ...CacheParam) {
	a := args[0]
	b := args[1]

	if a is Variable<T> && b is Variable<T> {
		result.grad = vtl.zeros_like<T>(result.value)
		result.requires_grad = true

		register<T>('Add', g, result, a, b)
	}
}

pub struct SubstractGate<T> {}

pub fn new_substract_gate<T>() &AddGate<T> {
	return &AddGate<T>{}
}

pub fn (g &SubstractGate<T>) backward<T>(payload &Payload<T>) []&vtl.Tensor<T> {
	gradient := payload.variable.grad
	oposite := vtl.multiply_scalar<T>(gradient, T(-1))
	return [gradient, oposite]
}

pub fn (g &SubstractGate<T>) cache<T>(mut result Variable<T>, args ...CacheParam) {
	a := args[0]
	b := args[1]

	if a is Variable<T> && b is Variable<T> {
		result.grad = vtl.zeros_like<T>(result.value)
		result.requires_grad = true

		register<T>('Sub', g, result, a, b)
	}
}

pub struct MultiplyGate<T> {
pub:
	a &Variable<T>
	b &Variable<T>
}

pub fn new_multiply_gate<T>(a &Variable<T>, b &Variable<T>) &MultiplyGate<T> {
	return &MultiplyGate<T>{
		a: a
		b: b
	}
}

pub fn (g &MultiplyGate<T>) backward<T>(payload &Payload<T>) []&vtl.Tensor<T> {
	gradient := payload.variable.grad
	gradient_a := vtl.multiply<T>(gradient, g.a.value)
	gradient_b := vtl.multiply<T>(gradient, g.b.value)
	return [b, a]
}

pub fn (g &MultiplyGate<T>) cache<T>(mut result Variable<T>, args ...CacheParam) {
	a := args[0]
	b := args[1]

	if a is Variable<T> && b is Variable<T> {
		result.grad = vtl.zeros_like<T>(result.value)
		result.requires_grad = true

		register<T>('Multiply', g, result, a, b)
	}
}

pub struct DivideGate<T> {
pub:
	a &Variable<T>
	b &Variable<T>
}

pub fn new_divide_gate<T>(a &Variable<T>, b &Variable<T>) &DivideGate<T> {
	return &DivideGate<T>{
		a: a
		b: b
	}
}

pub fn (g &DivideGate<T>) backward<T>(payload &Payload<T>) []&vtl.Tensor<T> {
	gradient := payload.variable.grad
	r0 := vtl.divide<T>(gradient, g.b.value)
	bx2 := vtl.multiply_scalar<T>(g.b.value, T(2))
	oposite := vtl.multiply_scalar<T>(gradient, T(-1))
	mut r1 := vtl.multiply<T>(oposite, g.a.value)
	r1 = vtl.divide(r1, bx2)
	return [r0, r1]
}

pub fn (g &DivideGate<T>) cache<T>(mut result Variable<T>, args ...CacheParam) {
	a := args[0]
	b := args[1]

	if a is Variable<T> && b is Variable<T> {
		result.grad = vtl.zeros_like<T>(result.value)
		result.requires_grad = true

		register<T>('Divide', g, result, a, b)
	}
}
