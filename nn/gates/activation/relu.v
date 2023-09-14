module activation

import vtl
import vtl.autograd
import vtl.nn.internal

pub struct ReLUGate[T] {
pub:
	cache &vtl.Tensor[T]
}

pub fn relu_gate[T](cache &vtl.Tensor[T]) &ReLUGate[T] {
	return &ReLUGate[T]{
		cache: cache
	}
}

pub fn (g &ReLUGate[T]) backward[T](payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := internal.deriv_relu[T](gradient, g.cache)!
	return [r0]
}

pub fn (g &ReLUGate[T]) cache[T](mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]

	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true

			autograd.register[T]('Relu', g, result, [args[0]])!
		}
		else {
			return error('Relu: cache: invalid argument')
		}
	}
}
