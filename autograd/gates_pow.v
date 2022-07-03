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
	mut iters := vtl.iterators<T>([gradient, g.a.value, g.b.value])
	for {
		vals, pos := vtl.iterators_next<T>(mut iters) or { break }
		val0 := vals[0] * vals[2] * T(math.pow(f64(vals[1]), f64(vals[2]) - 1))
		val1 := vals[0] * T(math.pow(f64(vals[1]), f64(vals[2]))) * T(math.log(f64(vals[1])))
		r0.set_nth(pos, val0)
		r1.set_nth(pos, val1)
	}
	return [r0, r1]
}

pub fn (g &PowGate<T>) cache<T>(mut result Variable<T>, args ...CacheParam) {
	a := args[0]
	b := args[1]

	match a {
		Variable<T> {
			match b {
				Variable<T> {
					result.grad = vtl.zeros_like<T>(result.value)
					result.requires_grad = true

					register<T>('Pow', g, result, [a, b])
				}
				else {
					panic('PowGate: b must be a Variable')
				}
			}
		}
		else {
			panic('PowGate: a must be a Variable')
		}
	}
}
