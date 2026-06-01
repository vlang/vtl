module activation

import vtl
import vtl.autograd
import vtl.nn.internal

// SwishGate defines a public data structure for this module.
pub struct SwishGate[T] {
pub:
	cache &vtl.Tensor[T] = unsafe { nil }
}

// swish_gate exposes this operation as part of the public API.
pub fn swish_gate[T](cache &vtl.Tensor[T]) &SwishGate[T] {
	return &SwishGate[T]{
		cache: cache
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &SwishGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := internal.deriv_swish[T](gradient, g.cache)!
	return [r0]
}

// cache exposes this operation as part of the public API.
pub fn (g &SwishGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			autograd.register[T]('Swish', g, result, [a])!
		}
		else {
			return error('Swish: cache: invalid argument')
		}
	}
}
