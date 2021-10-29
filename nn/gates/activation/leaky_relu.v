module activation

import vtl
import vtl.autograd
import vtl.nn.internal

pub struct LeakyReluGate<T> {
pub:
	cache &vtl.Tensor<T>
}

pub fn new_leaky_relu_gate<T>(cache &vtl.Tensor<T>) &LeakyReluGate<T> {
	return &LeakyReluGate<T>{
		cache: cache
	}
}

pub fn (g &LeakyReluGate<T>) backward<T>(payload &autograd.Payload<T>) []&vtl.Tensor<T> {
	gradient := payload.variable.grad
	return [internal.deriv_leaky_relu(gradient, g.cache)]
}

pub fn (g &LeakyReluGate<T>) cache<T>(mut result autograd.Variable<T>, args ...autograd.CacheParam) {
	result.grad = vtl.zeros_like<T>(result.value)
	result.requires_grad = true

	register<T>('LeakyRelu', g, result, ...args)
}
