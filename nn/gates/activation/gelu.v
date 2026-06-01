module activation

import vtl
import vtl.autograd
import vtl.nn.internal

pub struct GELUGate[T] {
pub:
	cache &vtl.Tensor[T] = unsafe { nil }
}

pub fn gelu_gate[T](cache &vtl.Tensor[T]) &GELUGate[T] {
	return &GELUGate[T]{
		cache: cache
	}
}

pub fn (g &GELUGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := internal.deriv_gelu[T](gradient, g.cache)!
	return [r0]
}

pub fn (g &GELUGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			autograd.register[T]('GELU', g, result, [a])!
		}
		else {
			return error('GELU: cache: invalid argument')
		}
	}
}
