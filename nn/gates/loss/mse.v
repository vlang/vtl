module loss

import vtl
import vtl.autograd
import vtl.nn.internal

pub struct MseGate[T] {
pub:
	cache  &autograd.Variable[T] = unsafe { nil }
	target &vtl.Tensor[T]        = unsafe { nil }
}

pub fn mse_gate[T](cache &autograd.Variable[T], target &vtl.Tensor[T]) &MseGate[T] {
	return &MseGate[T]{
		cache: cache
		target: target
	}
}

pub fn (g &MseGate[T]) backward[T](payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	return internal.mse_backward[T](gradient, g.cache.value, g.target)
}

pub fn (g &MseGate[T]) cache[T](mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]

	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true

			autograd.register[T]('MSE', g, result, [a])!
		}
		else {
			return error('MSEGate: cache: invalid argument')
		}
	}
}
