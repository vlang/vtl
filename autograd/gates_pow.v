module autograd

import math
import vtl

pub struct PowGate<T> {
        a &Variable<T>
	b &Variable<T>
}

pub fn new_pow_gate<T>(a &Variable<T>, b &Variable<T>) &PowGate<T> {
	return &PowGate<T>{
		a: a
		b: b
	}
}

pub fn (g &PowGate<T>) backward<T>(payload &Payload<T>) []&vtl.Tensor<T> {
	gradient := payload.variable.grad
        mut r0 := vtl.new_tensor_like<T>(gradient)
        mut r1 := vtl.new_tensor_like<T>(gradient)
	mut iter := vtl.iterators<T>([gradient, g.a.value, g.b.value])
	for {
		vals, pos := vtl.iterators_next<T>(mut iter) or { break }
                val0 := vals[0] * vals[2] * T(math.pow(f64(vals[1]), f64(vals[2]) - 1))
                val1 := vals[0] * T(math.pow(f64(vals[1]), f64(vals[2]))) * T(math.log(f64(vals[1])))
		r0.data.set<T>(pos, val0)
		r1.data.set<T>(pos, val1)
	}
	return [r0, r1]
}

pub fn (g &PowGate<T>) cache<T>(mut result Variable<T>, args ...CacheParam) {
	a := args[0]
	b := args[1]

	if a is Variable<T> && b is Variable<T> {
		result.grad = vtl.zeros_like<T>(result.value)
		result.requires_grad = true

		register<T>('Pow', g, result, a, b)
	}
}
