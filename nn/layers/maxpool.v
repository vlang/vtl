module layers

import vtl
import vtl.la
import vtl.autograd
import vtl.nn.internal
import vtl.nn.gates.layers

// MaxpoolLayer is a layer that implements the maxpooling operation.
pub struct MaxpoolLayer<T> {
	input_shape []int
	kernel      []int
	padding     []int
	stride      []int
}

pub fn new_maxpool_layer<T>(ctx &autograd.Context<T>, input_shape []int, kernel []int, padding []int, stride []int) &MaxpoolLayer<T> {
	return &MaxpoolLayer<T>{
		input_shape: input_shape
		kernel: kernel
		padding: padding
		stride: stride
	}
}

pub fn (layer MaxpoolLayer<T>) output_shape() []int {
	return [layer.weights.value.shape[0]]
}

pub fn (layer MaxpoolLayer<T>) variables() []&autograd.Variable<T> {
	return []&autograd.Variable<T>{}
}

pub fn (layer MaxpoolLayer<T>) forward(mut input autograd.Variable<T>) &autograd.Variable<T> {
	max_indices, output := internal.maxpool(input.value, layer.kernel, layer.padding,
		layer.stride)
	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := layers.new_maxpool_gate<T>(input.value.shape, max_indices, layer.kernel,
			layer.padding, layer.stride)
		gate.cache(mut result, input, layer.kernel, layer.padding, layer.stride)
	}

	return result
}
