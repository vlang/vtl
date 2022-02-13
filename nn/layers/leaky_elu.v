module layers

import vtl
import vtl.autograd
import vtl.nn.internal
import vtl.nn.gates.layers

// LeakyEluLayer is an activation layer that applies the leaky elu function to the input.
pub struct LeakyEluLayer<T> {
	output_shape []int
}

pub fn new_leaky_elu_layer<T>(ctx &autograd.Context<T>, output_shape []int) &LeakyEluLayer<T> {
	return &LeakyEluLayer<T>{
		output_shape: output_shape.clone()
	}
}

pub fn (layer &LeakyEluLayer<T>) output_shape() []int {
	return layer.output_shape
}

pub fn (_ &LeakyEluLayer<T>) variables() []&autograd.Variable<T> {
	return []&autograd.Variable<T>{}
}

pub fn (layer &LeakyEluLayer<T>) forward(mut input autograd.Variable<T>) &autograd.Variable<T> {
	output := internal.leaky_elu<T>(input.value)
	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := layers.new_leaky_elu_gate<T>(input.value)
		gate.cache(mut result, input)
	}
	return result
}

pub fn leaky_elu<T>(v &autograd.Variable<T>) &autograd.Variable<T> {
	output := internal.leaky_elu<T>(v.value)
	mut result := v.context.variable(output)

	if v.requires_grad {
		gate := layers.new_leaky_elu_gate<T>(v.value)
		gate.cache(mut result, v)
	}
	return result
}
