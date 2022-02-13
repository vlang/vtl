module activation

import vtl
import vtl.autograd
import vtl.nn.internal

pub struct LeakyEluGate<T> {
pub:
	cache &vtl.Tensor<T>
}

pub fn new_leaky_relu_gate<T>(cache &vtl.Tensor<T>) &LeakyEluGate<T> {
	return &LeakyEluGate<T>{
		cache: cache
	}
}

pub fn (g &LeakyEluGate<T>) backward<T>(payload &autograd.Payload<T>) []&vtl.Tensor<T> {
	gradient := payload.variable.grad
	return [internal.deriv_leaky_relu(gradient, g.cache)]
}

pub fn (g &LeakyEluGate<T>) cache<T>(mut result autograd.Variable<T>, args ...autograd.CacheParam) {
	result.grad = vtl.zeros_like<T>(result.value)
	result.requires_grad = true

	register<T>('LeakyElu', g, result, ...args)
}
