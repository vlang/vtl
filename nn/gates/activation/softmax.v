module activation

import vtl
import vtl.autograd
import vtl.nn.internal

// SoftmaxGate stores the input tensor and dim to compute jacobian on backward.
pub struct SoftmaxGate[T] {
pub:
	input &vtl.Tensor[T] = unsafe { nil }
	dim   int
}

pub fn softmax_gate[T](input &vtl.Tensor[T], dim int) &SoftmaxGate[T] {
	return &SoftmaxGate[T]{
		input: input
		dim:   dim
	}
}

pub fn (g &SoftmaxGate[T]) backward[T](payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := internal.deriv_softmax[T](gradient, g.input, g.dim)!
	return [r0]
}

pub fn (g &SoftmaxGate[T]) cache[T](mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			autograd.register[T]('Softmax', g, result, [a])!
		}
		else {
			return error('Softmax: cache: invalid argument')
		}
	}
}
