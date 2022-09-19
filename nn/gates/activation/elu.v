module activation

import vtl
import vtl.autograd
import vtl.nn.internal

pub struct EluGate<T> {
pub:
	cache &vtl.Tensor<T>
}

pub fn elu_gate<T>(cache &vtl.Tensor<T>) &EluGate<T> {
	return &EluGate<T>{
		cache: cache
	}
}

pub fn (g &EluGate<T>) backward<T>(payload &autograd.Payload<T>) ?[]&vtl.Tensor<T> {
	gradient := payload.variable.grad
	r0 := internal.deriv_elu<T>(gradient, g.cache, vtl.new_t<T>(0))?
	return [r0]
}

pub fn (g &EluGate<T>) cache<T>(mut result autograd.Variable<T>, args ...autograd.CacheParam) ? {
	a := args[0]

	match a {
		autograd.Variable<T> {
			result.grad = vtl.zeros_like<T>(result.value)
			result.requires_grad = true

			autograd.register<T>('Elu', g, result, [a])?
		}
		else {
			return error('Elu: cache: invalid argument')
		}
	}
}
