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

pub fn (g &SigmoidCrossEntropyGate<T>) backward<T>(payload &autograd.Payload<T>) ?[]&vtl.Tensor<T> {
	gradient := payload.variable.grad
	return internal.sigmoid_cross_entropy_backward<T>(gradient, g.cache.value, g.target)
}

pub fn (g &SigmoidCrossEntropyGate<T>) cache<T>(mut result autograd.Variable<T>, args ...autograd.CacheParam) ? {
	a := args[0]
	b := args[1]

	match a {
		autograd.Variable<T> {
			match b {
				autograd.Variable<T> {
					result.grad = vtl.zeros_like<T>(result.value)
					result.requires_grad = true

					autograd.register<T>('SigmoidCrossEntropy', g, result, [a, b])?
				}
				else {
					return error('SigmoidCrossEntropyGate: cache: invalid argument')
				}
			}
		}
		else {
			return error('SigmoidCrossEntropyGate: cache: invalid argument')
		}
	}
}
