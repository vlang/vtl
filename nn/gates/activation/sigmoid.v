module activation

import vtl
import vtl.autograd
import vtl.nn.internal

pub struct SigmoidGate<T> {
pub:
	cache &vtl.Tensor<T>
}

pub fn new_sigmoid_gate<T>(cache &vtl.Tensor<T>) &SigmoidGate<T> {
	return &SigmoidGate<T>{
		cache: cache
	}
}

pub fn (g &SigmoidGate<T>) backward<T>(payload &autograd.Payload<T>) ?[]&vtl.Tensor<T> {
	gradient := payload.variable.grad
	r0 := internal.deriv_sigmoid<T>(gradient, g.cache)?
	return [r0]
}

pub fn (g &SigmoidGate<T>) cache<T>(mut result autograd.Variable<T>, args ...autograd.CacheParam) ? {
	a := args[0]

	match a {
		autograd.Variable<T> {
			result.grad = vtl.zeros_like<T>(result.value)
			result.requires_grad = true

			autograd.register<T>('Sigmoid', g, result, [a])?
		}
		else {
			return error('Sigmoid: cache: invalid argument')
		}
	}
}
