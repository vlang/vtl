module layers

import vtl
import vtl.autograd

pub struct DropoutGate<T> {
pub:
	prob f64
	mask &vtl.Tensor<T>
}

pub fn new_dropout_gate<T>(mask &vtl.Tensor<T>, prob f64) &DropoutGate<T> {
	return &DropoutGate<T>{
		mask: mask
		prob: prob
	}
}

pub fn (g &DropoutGate<T>) backward<T>(payload &autograd.Payload<T>) ?[]&vtl.Tensor<T> {
	gradient := payload.variable.grad
	return [vtl.divide_scalar(vtl.multiply(gradient, g.mask), T(g.prob))]
}

pub fn (g &DropoutGate<T>) cache<T>(mut result autograd.Variable<T>, args ...autograd.CacheParam) ? {
	a := args[0]

	match a {
		autograd.Variable<T> {
			result.grad = vtl.zeros_like<T>(result.value)
			result.requires_grad = true

			autograd.register<T>('Dropout', g, result, [a])?
		}
		else {
			return error('DropoutGate: cache: invalid argument')
		}
	}
}
