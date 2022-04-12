module loss

import vtl
import vtl.autograd
import vtl.nn.internal

pub struct SigmoidCrossEntropyGate<T> {
pub:
	cache  &autograd.Variable<T>
	target &vtl.Tensor<T>
}

pub fn new_sigmoid_cross_entropy_gate<T>(cache &autograd.Variable<T>, target &vtl.Tensor<T>) &SigmoidCrossEntropyGate<T> {
	return &SigmoidCrossEntropyGate<T>{
		cache: cache
		target: target
	}
}

pub fn (g &SigmoidCrossEntropyGate<T>) backward<T>(payload &autograd.Payload<T>) []&vtl.Tensor<T> {
	gradient := payload.variable.grad
	return internal.sigmoid_cross_entropy_backward<T>(gradient, g.cache.value, g.target)
}

pub fn (g &SigmoidCrossEntropyGate<T>) cache<T>(mut result autograd.Variable<T>, args ...autograd.CacheParam) {
	result.grad = vtl.zeros_like<T>(result.value)
	result.requires_grad = true

	register<T>('SigmoidCrossEntropy', g, result, []&autograd.Variable<T>{})
}
