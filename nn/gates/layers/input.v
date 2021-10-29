module layers

import vtl
import vtl.autograd

pub struct InputGate<T> {}

pub fn (g &InputGate<T>) backward<T>(payload &autograd.Payload<T>) []&vtl.Tensor<T> {
	gradient := payload.variable.grad
	return [gradient]
}

pub fn (g &InputGate<T>) cache<T>(mut result autograd.Variable<T>, args ...autograd.CacheParam) {
	result.grad = vtl.zeros_like<T>(result.value)
	result.requires_grad = true

	register<T>('Input', g, result, ...args)
}
