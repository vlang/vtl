module activation

import vtl
import vtl.autograd
import vtl.nn.internal

pub struct TanhGate[T] {
pub:
	cache &vtl.Tensor[T] = unsafe { nil }
}

pub fn tanh_gate[T](cache &vtl.Tensor[T]) &TanhGate[T] {
	return &TanhGate[T]{
		cache: cache
	}
}

pub fn (g &TanhGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := internal.deriv_tanh[T](gradient, g.cache)!
	return [r0]
}

pub fn (g &TanhGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			autograd.register[T]('Tanh', g, result, [a])!
		}
		else {
			return error('Tanh: cache: invalid argument')
		}
	}
}
