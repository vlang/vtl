module layers

import vtl
import vtl.autograd

// DropoutGate defines a public data structure for this module.
pub struct DropoutGate[T] {
pub:
	prob f64
	mask &vtl.Tensor[T] = unsafe { nil }
}

// dropout_gate exposes this operation as part of the public API.
pub fn dropout_gate[T](mask &vtl.Tensor[T], prob f64) &DropoutGate[T] {
	return &DropoutGate[T]{
		mask: mask
		prob: prob
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &DropoutGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	prob := g.prob
	result := gradient.nmap([g.mask], fn [prob] [T](xs []T, i []int) {
		return xs[0] * xs[1] / prob
	})!

	return [result]
}

// cache exposes this operation as part of the public API.
pub fn (g &DropoutGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]

	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true

			autograd.register[T]('Dropout', g, result, [a])!
		}
		else {
			return error('DropoutGate: cache: invalid argument')
		}
	}
}
