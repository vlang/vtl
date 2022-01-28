module layers

import vtl
import vtl.autograd

pub struct DropoutGate<T> {
pub:
	mask &vtl.Tensor<T>
}

pub fn new_dropout_gate<T>(mask &vtl.Tensor<T>) &DropoutGate<T> {
	return &DropoutGate<T>{
		mask: mask
	}
}

pub fn (g &DropoutGate<T>) backward<T>(payload &autograd.Payload<T>) []&vtl.Tensor<T> {
	gradient := payload.variable.grad
	return [vtl.multiply(gradient, g.mask)]
}

pub fn (g &DropoutGate<T>) cache<T>(mut result autograd.Variable<T>, args ...autograd.CacheParam) {
	a := args[0]

	match a {
		autograd.Variable<T> {
			result.grad = vtl.zeros_like<T>(result.value)
			result.requires_grad = true

			autograd.register<T>('Dropout', g, result, [a])
		}
		else {
			panic('DropoutGate: cache: invalid argument')
		}
	}
}
