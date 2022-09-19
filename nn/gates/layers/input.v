module layers

import vtl
import vtl.autograd

pub struct InputGate<T> {}

pub fn input_gate<T>() &InputGate<T> {
	return &InputGate<T>{}
}

pub fn (g &InputGate<T>) backward<T>(payload &autograd.Payload<T>) ?[]&vtl.Tensor<T> {
	gradient := payload.variable.grad
	return [gradient]
}

pub fn (g &InputGate<T>) cache<T>(mut result autograd.Variable<T>, args ...autograd.CacheParam) ? {
	a := args[0]

	match a {
		autograd.Variable<T> {
			result.grad = vtl.zeros_like<T>(result.value)
			result.requires_grad = true

			autograd.register<T>('Input', g, result, [a])?
		}
		else {
			return error('InputGate: cache: invalid argument')
		}
	}
}
