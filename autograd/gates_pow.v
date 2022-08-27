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

pub fn (g &PowGate<T>) backward<T>(payload &Payload<T>) ?[]&vtl.Tensor<T> {
	gradient := payload.variable.grad
	mut iters, shape := gradient.iterators<T>([g.a.value, g.b.value])?
	mut r0 := vtl.new_tensor_like_with_shape<T>(gradient, shape)
	mut r1 := vtl.new_tensor_like_with_shape<T>(gradient, shape)
	for {
		vals, i := vtl.iterators_next<T>(mut iters) or { break }
		val0 := vals[0] * vals[2] * vtl.new_t<T>(math.pow(f64(vals[1]), f64(vals[2]) - 1))
		val1 := vals[0] * vtl.new_t<T>(math.pow(f64(vals[1]), f64(vals[2]))) * vtl.new_t<T>(math.log(f64(vals[1])))
		r0.set(i, val0)
		r1.set(i, val1)
	}
	return [r0, r1]
}

pub fn (g &PowGate<T>) cache<T>(mut result Variable<T>, args ...CacheParam) ? {
	a := args[0]
	b := args[1]

	match a {
		Variable<T> {
			match b {
				Variable<T> {
					result.grad = vtl.zeros_like<T>(result.value)
					result.requires_grad = true

					register<T>('Pow', g, result, [a, b])?
				}
				else {
					return error('PowGate: b must be a Variable')
				}
			}
		}
		else {
			return error('PowGate: a must be a Variable')
		}
	}
}
