module autograd

import vtl

pub struct ExpGate<T> {
pub:
	a &Variable<T>
}

pub fn new_exp_gate<T>(a &Variable<T>) &ExpGate<T> {
	return &ExpGate<T>{
		a: a
	}
}

pub fn (g &ExpGate<T>) backward<T>(payload &Payload<T>) []&vtl.Tensor<T> {
	gradient := payload.variable.grad
	r0 := vtl.multiply(gradient, vtl.exp(g.a.value))
	return [r0]
}

pub fn (g &ExpGate<T>) cache<T>(mut result Variable<T>, args ...CacheParam) {
	a := args[0]

	if a is Variable<T> {
		result.grad = vtl.zeros_like<T>(result.value)
		result.requires_grad = true

		register<T>('Exp', g, result, a)
	}
}
