module layers

import vtl
// import vtl.la
import vtl.autograd
import vtl.nn.internal
import vtl.nn.gates.layers

// MaxPool2DLayer is a layer that implements the maxpooling operation.
pub struct MaxPool2DLayer<T> {
	input_shape []int
	kernel      []int
	padding     []int
	stride      []int
}

pub fn new_maxpool2d_layer<T>(ctx &autograd.Context<T>, input_shape []int, kernel []int, padding []int, stride []int) &MaxPool2DLayer<T> {
	return &MaxPool2DLayer<T>{
		input_shape: input_shape.clone()
		kernel: kernel.clone()
		padding: padding.clone()
		stride: stride.clone()
	}
}

pub fn (layer &MaxPool2DLayer<T>) output_shape() []int {
	c := layer.input_shape[0]
	h := layer.input_shape[1]
	w := layer.input_shape[2]
	kh := layer.kernel[0]
	kw := layer.kernel[1]
	ph := layer.padding[0]
	pw := layer.padding[1]
	sh := layer.stride[0]
	sw := layer.stride[1]

	return [c, (h - kh + 2 * ph) / sh + 1, (w - kw + 2 * pw) / sw + 1]
}

pub fn (layer &MaxPool2DLayer<T>) variables() []&autograd.Variable<T> {
	return []&autograd.Variable<T>{}
}

pub fn (layer &MaxPool2DLayer<T>) forward(mut input autograd.Variable<T>) ?&autograd.Variable<T> {
	max_indices, output := internal.maxpool2d<T>(input.value, layer.kernel, layer.padding,
		layer.stride)
	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := layers.new_maxpool2d_gate<T>(max_indices, layer.kernel, input.value.shape,
			layer.padding, layer.stride)
		gate.cache(mut result, input, layer.kernel, layer.padding, layer.stride)?
	}

	return result
}
