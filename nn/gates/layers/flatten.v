module layers

import vtl
import vtl.autograd

// FlattenGate defines a public data structure for this module.
pub struct FlattenGate[T] {
pub:
	input        &autograd.Variable[T] = unsafe { nil }
	cached_shape []int
}

// flatten_gate exposes this operation as part of the public API.
pub fn flatten_gate[T](input &autograd.Variable[T], cached_shape []int) &FlattenGate[T] {
	return &FlattenGate[T]{
		input:        input
		cached_shape: cached_shape
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &FlattenGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	mut next_shape := [gradient.shape[0]]
	next_shape << g.cached_shape
	return [gradient.reshape(next_shape)!]
}

// cache exposes this operation as part of the public API.
pub fn (g &FlattenGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]

	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true

			autograd.register[T]('Flatten', g, result, [a])!
		}
		else {
			return error('FlattenGate: cache: invalid argument')
		}
	}
}
