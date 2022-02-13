module layers

import vtl
import vtl.autograd
import vtl.nn.internal
import vtl.nn.gates.layers

// ReluLayer is a layer that applies the rectified linear unit function element-wise.
pub struct ReluLayer<T> {
	output_shape []int
}

pub fn new_relu_layer<T>(ctx &autograd.Context<T>, output_shape []int) &ReluLayer<T> {
	return &ReluLayer<T>{
		output_shape: output_shape.clone()
	}
}

pub fn (layer &ReluLayer<T>) output_shape() []int {
	return layer.output_shape
}

pub fn (_ &ReluLayer<T>) variables() []&autograd.Variable<T> {
	return []&autograd.Variable<T>{}
}

pub fn (layer &ReluLayer<T>) forward(mut input autograd.Variable<T>) &autograd.Variable<T> {
	output := internal.relu<T>(input.value)
	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := layers.new_relu_gate<T>(input.value)
		gate.cache(mut result, input)
	}
	return result
}

pub fn relu<T>(v &autograd.Variable<T>) &autograd.Variable<T> {
	output := internal.relu<T>(v.value)
	mut result := v.context.variable(output)

	if v.requires_grad {
		gate := layers.new_relu_gate<T>(v.value)
		gate.cache(mut result, v)
	}
	return result
}
