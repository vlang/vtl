module layers

import vtl
import vtl.autograd

// InputGate defines a public data structure for this module.
pub struct InputGate[T] {}

// input_gate exposes this operation as part of the public API.
pub fn input_gate[T]() &InputGate[T] {
	return &InputGate[T]{}
}

// backward exposes this operation as part of the public API.
pub fn (g &InputGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	return [gradient]
}

// cache exposes this operation as part of the public API.
pub fn (g &InputGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]

	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true

			autograd.register[T]('Input', g, result, [a])!
		}
		else {
			return error('InputGate: cache: invalid argument')
		}
	}
}
