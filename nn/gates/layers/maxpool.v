module layers

import vtl
import vtl.autograd
import vtl.nn.internal

pub struct MaxPool2DGate[T] {
pub:
	max_indices &vtl.Tensor[int]
	kernel      []int
	shape       []int
	stride      []int
	padding     []int
}

pub fn maxpool2d_gate[T](max_indices &vtl.Tensor[int], kernel []int, shape []int, stride []int, padding []int) &MaxPool2DGate[T] {
	return &MaxPool2DGate[T]{
		max_indices: max_indices
		kernel: kernel
		shape: shape
		stride: stride
		padding: padding
	}
}

pub fn (g &MaxPool2DGate[T]) backward[T](payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad

	r0 := internal.maxpool2d_backward[T](g.shape, g.max_indices, gradient)

	return [r0]
}

pub fn (g &MaxPool2DGate[T]) cache[T](mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]

	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true

			autograd.register[T]('MaxPool2D', g, result, [args[0]])!
		}
		else {
			return error('MaxPool2DGate: cache: invalid argument')
		}
	}
}
