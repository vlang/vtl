module autograd

import vtl

pub struct MatMulGate<T> {
pub:
	a &Variable<T>
	b &Variable<T>
}

pub fn new_multiply_gate<T>(a &Variable<T>, b &Variable<T>) &MatMulGate<T> {
	return &MatMulGate<T>{
		a: a
		b: b
	}
}

pub fn (g &MatMulGate<T>) backward<T>(payload &Payload<T>) []&vtl.Tensor<T> {
	gradient := payload.variable.grad
	// r0 := gradient.matmul(g.b.value.t())
        // r1 := g.a.value.t().matmul(gradient)
	// return [r0, r1]
        return [gradient, gradient]
}

pub fn (g &MatMulGate<T>) cache<T>(mut result Variable<T>, args ...CacheParam) {
	a := args[0]
	b := args[1]

	if a is Variable<T> && b is Variable<T> {
		result.grad = vtl.zeros_like<T>(result.value)
		result.requires_grad = true

		register<T>('MatMul', g, result, a, b)
	}
}
