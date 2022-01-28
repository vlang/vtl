module layers

import vtl
import vtl.autograd
import vtl.nn.internal

pub struct MaxpoolGate<T> {
pub:
	max_indices &vtl.Tensor<int>
	kernel      []int
	shape       []int
	stride      []int
	padding     []int
}

pub fn new_maxpool_gate<T>(max_indices &vtl.Tensor<int>, kernel []int, shape []int, stride []int, padding []int) &MaxpoolGate<T> {
	return &MaxpoolGate<T>{
		max_indices: max_indices
		kernel: kernel
		shape: shape
		stride: stride
		padding: padding
	}
}

pub fn (g &MaxpoolGate<T>) backward<T>(payload &autograd.Payload<T>) []&vtl.Tensor<T> {
	gradient := payload.variable.grad

	r0 := internal.maxpool_backward<T>(g.shape, g.max_indices, gradient)

	return [r0]
}

pub fn (g &MaxpoolGate<T>) cache<T>(mut result autograd.Variable<T>, args ...autograd.CacheParam) {
	a := args[0]

	match a {
		autograd.Variable<T> {
			result.grad = vtl.zeros_like<T>(result.value)
			result.requires_grad = true

			autograd.register<T>('Maxpool', g, result, [a])
		}
		else {
			panic('MaxpoolGate: cache: invalid argument')
		}
	}
}
