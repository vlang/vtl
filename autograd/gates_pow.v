module autograd

import math
import vtl

// PowGate defines a public data structure for this module.
pub struct PowGate[T] {
	a &Variable[T] = unsafe { nil }
	b &Variable[T] = unsafe { nil }
}

// pow_gate exposes this operation as part of the public API.
pub fn pow_gate[T](a &Variable[T], b &Variable[T]) &PowGate[T] {
	return &PowGate[T]{
		a: a
		b: b
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &PowGate[T]) backward(payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	mut iters, shape := gradient.iterators[T]([g.a.value, g.b.value])!
	mut r0 := vtl.tensor_like_with_shape[T](gradient, shape)
	mut r1 := vtl.tensor_like_with_shape[T](gradient, shape)
	for {
		vals, i := iters.next() or { break }
		val0 := vals[0] * vals[2] * vtl.cast[T](math.pow(f64(vals[1]), f64(vals[2]) - 1))
		val1 := vals[0] * vtl.cast[T](math.pow(f64(vals[1]), f64(vals[2]))) * vtl.cast[T](math.log(f64(vals[1])))
		r0.set(i, val0)
		r1.set(i, val1)
	}
	return [r0, r1]
}

fn pow_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &Payload[T](payload) }
	tensors := unsafe { (&PowGate[T](gate)).backward(typed_payload)! }
	return tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &PowGate[T]) cache(mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	b := args[1]

	match a {
		Variable[T] {
			match b {
				Variable[T] {
					result.grad = vtl.zeros_like[T](result.value)
					result.requires_grad = true

					register[T]('Pow', voidptr(g), pow_gate_backward_dispatch[T], result, [
						a,
						b,
					])!
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
