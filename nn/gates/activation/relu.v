module activation

import vtl
import vtl.autograd
import vtl.nn.internal

pub struct ReLUGate<T> {
pub:
	cache &vtl.Tensor<T>
}

pub fn new_relu_gate<T>(cache &vtl.Tensor<T>) &ReLUGate<T> {
	return &ReLUGate<T>{
		cache: cache
	}
}

pub fn (g &ReLUGate<T>) backward<T>(payload &autograd.Payload<T>) []&vtl.Tensor<T> {
	gradient := payload.variable.grad
	return [internal.deriv_relu(gradient, g.cache)]
}

pub fn (g &ReLUGate<T>) cache<T>(mut result autograd.Variable<T>, args ...autograd.CacheParam) {
	result.grad = vtl.zeros_like<T>(result.value)
	result.requires_grad = true

	register<T>('Softmax', g, result, ...args)
}
