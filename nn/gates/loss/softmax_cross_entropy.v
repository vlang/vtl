module loss

import vtl
import vtl.autograd
import vtl.nn.internal

pub struct SoftmaxCrossEntropyGate<T> {
pub:
	cache  &autograd.Variable<T>
	target &vtl.Tensor<T>
}

pub fn new_softmax_cross_entropy_gate<T>(cache &autograd.Variable<T>, target &vtl.Tensor<T>) &SoftmaxCrossEntropyGate<T> {
	return &SoftmaxCrossEntropyGate<T>{
		cache: cache
		target: target
	}
}

pub fn (g &SoftmaxCrossEntropyGate<T>) backward<T>(payload &autograd.Payload<T>) []&vtl.Tensor<T> {
	gradient := payload.variable.grad
	return internal.softmax_cross_entropy_backward<T>(gradient, g.cache.value, g.target)
}

pub fn (g &SoftmaxCrossEntropyGate<T>) cache<T>(mut result autograd.Variable<T>, args ...autograd.CacheParam) {
	result.grad = vtl.zeros_like<T>(result.value)
	result.requires_grad = true

	register<T>('SoftmaxCrossEntropy', g, result, ...args)
}
