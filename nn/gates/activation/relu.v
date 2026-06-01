module activation

import vtl
import vtl.autograd
import vtl.nn.internal

// ReLUGate defines a public data structure for this module.
pub struct ReLUGate[T] {
pub:
	cache &vtl.Tensor[T] = unsafe { nil }
}

// relu_gate exposes this operation as part of the public API.
pub fn relu_gate[T](cache &vtl.Tensor[T]) &ReLUGate[T] {
	return &ReLUGate[T]{
		cache: cache
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &ReLUGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := internal.deriv_relu[T](gradient, g.cache)!
	return [r0]
}

// cache exposes this operation as part of the public API.
pub fn (g &ReLUGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]

	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true

			autograd.register[T]('Relu', g, result, [a])!
		}
		else {
			return error('Relu: cache: invalid argument')
		}
	}
}
