module activation

import vtl
import vtl.autograd
import vtl.nn.internal

pub struct LeakyReluGate<T> {
pub:
	cache &vtl.Tensor<T>
}

pub fn leaky_relu_gate<T>(cache &vtl.Tensor<T>) &LeakyReluGate<T> {
	return &LeakyReluGate<T>{
		cache: cache
	}
}

pub fn (g &LeakyReluGate<T>) backward<T>(payload &autograd.Payload<T>) ?[]&vtl.Tensor<T> {
	gradient := payload.variable.grad
	r0 := internal.deriv_leaky_relu<T>(gradient, g.cache, vtl.cast<T>(0))?
	return [r0]
}

pub fn (g &LeakyReluGate<T>) cache<T>(mut result autograd.Variable<T>, args ...autograd.CacheParam) ? {
	a := args[0]

	match a {
		autograd.Variable<T> {
			result.grad = vtl.zeros_like<T>(result.value)
			result.requires_grad = true

			autograd.register<T>('LeakyRelu', g, result, [a])?
		}
		else {
			return error('LeakyRelu: cache: invalid argument')
		}
	}
}
